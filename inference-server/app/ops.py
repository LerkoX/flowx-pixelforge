"""算子实现：纯 Python 对象（管道/张量/PIL 图像）进出的函数，不感知对象 ID 与网络。
每个函数对应注册表中的一个算子，是"能力"的最小单元——新增能力就是在这里加一个函数
并在 main.py 注册一行。"""
import os
import random
import time

import torch
from PIL import Image

from .samplers import SAMPLERS, DEFAULT_SAMPLER, make_scheduler


class ModelRef:
    """打过 LoRA 补丁的 MODEL 视图，对应 ComfyUI ModelPatcher.clone + add_patches：
    不改原管道，只记录补丁列表，采样时才按需启用。可串联叠加多个 LoRA。"""

    def __init__(self, pipe, patches=()):
        self.pipe = pipe
        self.patches = tuple(patches)  # ((adapter_name, strength), ...)


def resolve_pipe(model):
    """MODEL 对象统一解析：裸 pipe → (pipe, [])；ModelRef → (pipe, 补丁列表)。"""
    if isinstance(model, ModelRef):
        return model.pipe, list(model.patches)
    return model, []


def exec_device_of(pipe):
    """取执行设备：offload（model/sequential）模式下子模块驻留 meta/cpu，
    pipe.device 属性会返回 meta（取第一个模块的设备），必须用 diffusers 记录的
    _execution_device（offload 钩子的实际计算设备）。无 offload 时回退 pipe.device。"""
    dev = getattr(pipe, "_execution_device", None)
    if dev is not None and str(dev) != "meta":
        return dev
    dev = pipe.device
    if str(dev) == "meta":  # 双保险：meta 无意义，落到可用设备
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev


def _enable_adapters(pipe, patches):
    """采样前启用补丁；无补丁时确保 LoRA 全部关闭（管道是常驻共享的）。"""
    if patches:
        pipe.set_adapters([n for n, _ in patches], [w for _, w in patches])
    else:
        try:
            pipe.disable_lora()
        except Exception:
            pass  # 未加载过任何 LoRA


def checkpoint_load(models, ckpt):
    """Checkpoint Loader：加载 checkpoint，返回同一 workflow 的 model/clip/vae 三个视图。"""
    key, _newly = models.load(ckpt)
    pipe = models.get(key)
    return {"model": pipe, "clip": pipe, "vae": pipe}


def clip_encode(pipe, text):
    """CLIP Text Encode：文本 → conditioning 张量。"""
    tokens = pipe.tokenizer(
        text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        cond = pipe.text_encoder(tokens.input_ids.to(exec_device_of(pipe)))[0]
    return {"cond": cond}


def latent_empty(width=512, height=512, batch_size=1):
    """Empty Latent Image：按尺寸创建零 latent（噪声在采样时注入）。"""
    if width % 8 or height % 8:
        raise ValueError("width/height must be multiples of 8")
    latent = torch.zeros((batch_size, 4, height // 8, width // 8),
                         dtype=torch.float16, device="cuda")
    return {"latent": latent}


def sample(model, pos, neg, base, seed=-1, steps=20, cfg=7.0,
           sampler_name=DEFAULT_SAMPLER, denoise=1.0, preview_cb=None,
           interrupt_check=None):
    """KSampler：手动采样循环，返回 {'latent': ..., 'seed': 实际种子}。
    model 可为裸 pipe 或带 LoRA 补丁的 ModelRef（采样前启用、采样后关闭）。
    preview_cb 可选：签名 preview_cb(latents, step_index, total)，在每一步
    去噪后回调（由调用方注入预览录制，如 app.preview.PreviewRecorder），
    本层不感知网络。
    interrupt_check 可选：无参回调，每步采样前调用，抛异常即中断
    （由调用方注入取消检查，如 app.execution.check_cancelled）。"""
    if sampler_name not in SAMPLERS:
        raise ValueError(f"unknown sampler '{sampler_name}', available: {sorted(SAMPLERS)}")

    pipe, patches = resolve_pipe(model)
    if patches:
        print("[sample] lora patches: "
              + ", ".join(f"{n}@{w}" for n, w in patches), flush=True)
    _enable_adapters(pipe, patches)

    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    device = exec_device_of(pipe)
    sched = make_scheduler(sampler_name, pipe.scheduler.config)
    sched.set_timesteps(steps, device=device)
    timesteps = sched.timesteps

    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(base.shape, generator=gen, device=device, dtype=base.dtype)

    if denoise >= 0.999:
        latents = (noise * sched.init_noise_sigma).to(base.dtype)
        ts = timesteps
    else:
        start = min(int(round(steps * (1 - denoise))), steps - 1)
        ts = timesteps[start:]
        latents = sched.add_noise(base, noise, timesteps[start:start + 1]).to(base.dtype)
        print(f"[sample] denoise={denoise} -> skip first {start} steps", flush=True)

    b = base.shape[0]
    hidden = torch.cat([neg.expand(b, -1, -1), pos.expand(b, -1, -1)])

    t0 = time.time()
    try:
        with torch.no_grad():
            for i, t in enumerate(ts):
                if interrupt_check is not None:
                    interrupt_check()  # 取消检查点：抛异常即中断采样
                inp = sched.scale_model_input(torch.cat([latents] * 2), t)
                noise_pred = pipe.unet(inp, t, encoder_hidden_states=hidden).sample
                uncond, cond = noise_pred.chunk(2)
                guided = uncond + cfg * (cond - uncond)
                latents = sched.step(guided, t, latents).prev_sample
                print(f"[sample] step {i+1}/{len(ts)}", flush=True)
                if preview_cb is not None:
                    try:
                        preview_cb(latents, i, len(ts))
                    except Exception as e:
                        # 预览推送绝不影响采样主流程
                        print(f"[sample] preview callback failed (ignored): {e}",
                              flush=True)
    finally:
        if patches:
            try:
                pipe.disable_lora()
            except Exception:
                pass
    print(f"[sample] done in {time.time()-t0:.1f}s seed={seed}", flush=True)

    return {"latent": latents, "seed": seed}


def lora_apply(models, model, lora, strength=1.0):
    """LoRA 加载：权重按 adapter 名缓存进管道，返回带补丁的新 MODEL 视图。"""
    pipe, patches = resolve_pipe(model)
    adapter = models.load_lora(models.key_of(pipe), lora)
    return {"model": ModelRef(pipe, patches + [(adapter, strength)]),
            "clip": pipe}


def motion_load(models, ckpt, motion):
    """Motion 加载：SD1.x checkpoint + MotionAdapter → AnimateDiffPipeline（文生视频）。
    直接收底模名（非 MODEL 引用）：组合过程需要全新实例化底模，先 checkpoint.load
    只会让旧底模被对象仓库钉在内存里（8GB WSL 会 OOM）。
    返回与 checkpoint_load 同构的 model/clip/vae 三视图。"""
    key, _ = models.load_motion(ckpt, motion)
    ad_pipe = models.get(key)
    return {"model": ad_pipe, "clip": ad_pipe, "vae": ad_pipe}


def vae_decode(pipe, latents):
    """VAE Decode：latent → PIL 图像（batch>1 时为列表）。
    latent 按 VAE 实际 dtype 转换（M3：VAE 可能已独立转 fp32 防黑图，
    而采样链路 latent 保持 fp16）。"""
    with torch.no_grad():
        img = pipe.vae.decode(
            latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor,
            return_dict=False)[0]
    pil = pipe.image_processor.postprocess(img, output_type="pil")
    return {"image": pil[0] if len(pil) == 1 else pil}


def vae_encode(pipe, image):
    """VAE Encode：PIL 图像（或列表）→ latent，供图生图（denoise<1）重采样。
    乘 scaling_factor 与 vae_decode 的除法互逆（ComfyUI VAEEncode 同语义）。"""
    images = image if isinstance(image, list) else [image]
    tensors = [pipe.image_processor.preprocess(img) for img in images]
    img_t = torch.cat(tensors).to(device=exec_device_of(pipe), dtype=pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(img_t).latent_dist.sample()
    return {"latent": (latent * pipe.vae.config.scaling_factor).to(torch.float16)}


def video_sample(model, prompt="", neg_prompt="", image=None,
                 width=832, height=480, num_frames=121, fps=24,
                 steps=50, cfg=5.0, seed=-1, decode_chunk_size=0,
                 preview_cb=None, interrupt_check=None):
    """Video Sample：文/图生视频，一个算子兼容多种视频管道。
    返回 {'video': {'frames': [PIL...], 'fps': n}, 'seed': 实际种子}。

    管道差异适配（不堆 if-else，按 __call__ 签名过滤参数）：
    - Wan TI2V 系：prompt 文本条件，image 可选（首帧）
    - SVD 系：image 必填（图生视频），无文本条件；cfg 映射到 min/max_guidance_scale，
      fps 是采样条件参数（SVD 的 __call__ 接受 fps）
    - decode_chunk_size>0 且管道支持时透传（VAE 分块解码省显存）

    分钟级任务——调用方应经异步 job（POST /jobs）执行；preview_cb /
    interrupt_check 语义同 sample()，经 callback_on_step_end 每步回调。"""
    import inspect

    pipe, patches = resolve_pipe(model)
    if patches:
        print("[video.sample] lora patches: "
              + ", ".join(f"{n}@{w}" for n, w in patches), flush=True)
    _enable_adapters(pipe, patches)

    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    gen = torch.Generator(device=exec_device_of(pipe)).manual_seed(seed)

    sig = inspect.signature(pipe.__call__).parameters
    cls_name = type(pipe).__name__

    if "prompt" not in sig and image is None:
        raise ValueError(
            f"{cls_name} 是纯图生视频管道（image 必填），请提供首帧图像")
    if image is not None and "image" not in sig:
        raise ValueError(
            f"{cls_name} 不接受 image 输入（非图生视频管道）；"
            f"纯文生视频请断开 image 端口")

    candidates = {
        "prompt": prompt or None,
        "negative_prompt": neg_prompt or None,
        "image": image,
        "width": width, "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "generator": gen,
        "output_type": "pil",
        "fps": fps,  # SVD 的采样条件参数；Wan 无此参（fps 只进 mp4 元数据）
    }
    if "guidance_scale" not in sig and "min_guidance_scale" in sig:
        # SVD 系：min/max 区间引导，取恒定 cfg
        candidates.pop("guidance_scale")
        candidates["min_guidance_scale"] = cfg
        candidates["max_guidance_scale"] = cfg
    if decode_chunk_size > 0:
        candidates["decode_chunk_size"] = decode_chunk_size

    kwargs = {k: v for k, v in candidates.items() if k in sig and v is not None}
    dropped = sorted(set(candidates) - set(kwargs) - {"prompt", "negative_prompt"})
    if dropped:
        print(f"[video.sample] {cls_name} 不支持的参数已忽略: {dropped}", flush=True)

    def on_step_end(p, i, t, cb_kwargs):
        if interrupt_check is not None:
            interrupt_check()  # 取消检查点：抛 JobCancelled 即中断
        if preview_cb is not None:
            try:
                preview_cb(cb_kwargs.get("latents"), i, steps)
            except Exception as e:
                print(f"[video.sample] preview callback failed (ignored): {e}",
                      flush=True)
        return cb_kwargs

    t0 = time.time()
    try:
        call = dict(**kwargs)
        if "callback_on_step_end" in sig:
            call["callback_on_step_end"] = on_step_end
            if "callback_on_step_end_tensor_inputs" in sig:
                call["callback_on_step_end_tensor_inputs"] = ["latents"]
        out = pipe(**call)
    finally:
        if patches:
            try:
                pipe.disable_lora()
            except Exception:
                pass
    frames = out.frames[0]
    print(f"[video.sample] done in {time.time()-t0:.1f}s "
          f"frames={len(frames)} seed={seed}", flush=True)
    return {"video": {"frames": frames, "fps": fps}, "seed": seed}


def image_load(input_dir, name):
    """Load Image：读服务端 INPUT_DIR 下的本地图片 → PIL（RGB）。
    仅接受纯文件名（防路径穿越）；客户端上传走 POST /images 端点。"""
    if os.path.basename(name) != name or name in ("", ".", ".."):
        raise ValueError(f"image name must be a plain file name under INPUT_DIR, got {name!r}")
    path = os.path.join(input_dir, name)
    if not os.path.isfile(path):
        available = sorted(os.listdir(input_dir)) if os.path.isdir(input_dir) else []
        raise FileNotFoundError(
            f"image '{name}' not found in {input_dir}; available: {available or '(empty)'}")
    return {"image": Image.open(path).convert("RGB")}

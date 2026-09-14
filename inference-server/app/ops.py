"""算子实现：纯 Python 对象（管道/张量/PIL 图像）进出的函数，不感知对象 ID 与网络。
每个函数对应注册表中的一个算子，是"能力"的最小单元——新增能力就是在这里加一个函数
并在 main.py 注册一行。"""
import random
import time

import torch

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
        cond = pipe.text_encoder(tokens.input_ids.to(pipe.device))[0]
    return {"cond": cond}


def latent_empty(width=512, height=512, batch_size=1):
    """Empty Latent Image：按尺寸创建零 latent（噪声在采样时注入）。"""
    if width % 8 or height % 8:
        raise ValueError("width/height must be multiples of 8")
    latent = torch.zeros((batch_size, 4, height // 8, width // 8),
                         dtype=torch.float16, device="cuda")
    return {"latent": latent}


def sample(model, pos, neg, base, seed=-1, steps=20, cfg=7.0,
           sampler_name=DEFAULT_SAMPLER, denoise=1.0, preview_cb=None):
    """KSampler：手动采样循环，返回 {'latent': ..., 'seed': 实际种子}。
    model 可为裸 pipe 或带 LoRA 补丁的 ModelRef（采样前启用、采样后关闭）。
    preview_cb 可选：签名 preview_cb(latents, step_index, total)，在每一步
    去噪后回调（由调用方注入预览推送，如 app.preview.PreviewPusher），
    本层不感知网络。"""
    if sampler_name not in SAMPLERS:
        raise ValueError(f"unknown sampler '{sampler_name}', available: {sorted(SAMPLERS)}")

    pipe, patches = resolve_pipe(model)
    if patches:
        print("[sample] lora patches: "
              + ", ".join(f"{n}@{w}" for n, w in patches), flush=True)
    _enable_adapters(pipe, patches)

    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    device = pipe.device
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


def vae_decode(pipe, latents):
    """VAE Decode：latent → PIL 图像（batch>1 时为列表）。"""
    with torch.no_grad():
        img = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    pil = pipe.image_processor.postprocess(img, output_type="pil")
    return {"image": pil[0] if len(pil) == 1 else pil}

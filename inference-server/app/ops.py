"""算子实现：纯 Python 对象（管道/张量/PIL 图像）进出的函数，不感知对象 ID 与网络。
每个函数对应注册表中的一个算子，是"能力"的最小单元——新增能力就是在这里加一个函数
并在 main.py 注册一行。"""
import os
import random
import time

import torch
from PIL import Image

from .samplers import (DEFAULT_SAMPLER, DEFAULT_SCHEDULER,
                       make_scheduler, resolve_sampler)


class VAEShim:
    """独立加载的 VAE 组件的管道视图（Load VAE，对标 ComfyUI VAELoader）。
    duck-type 对齐 ops.vae_decode / vae_encode 的使用面（.vae /
    .image_processor / .device），算子零改动即可把外挂 VAE（vae-ft-mse 等）
    接入解码/编码链路；与常驻管道解耦（不动管道内置 VAE）。"""

    def __init__(self, vae):
        from diffusers.image_processor import VaeImageProcessor
        self.vae = vae
        scale = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=scale)

    @property
    def device(self):
        return next(self.vae.parameters()).device


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


class LatentBundle:
    """latent.set_noise_mask 的产物（LATENT 对象）：samples + noise_mask。
    noise_mask 为 (1,1,h/8,w/8) 0..1 张量；sample 循环内每步后把 mask 外区域
    混回原始 latent 的当步加噪版（局部重绘核心，对标 ComfyUI SetLatentNoiseMask）。
    其余算子经 resolve_latent 取 samples（mask 仅作用于采样）。"""

    def __init__(self, samples, noise_mask):
        self.samples = samples
        self.noise_mask = noise_mask


def resolve_latent(latent):
    """LATENT 对象统一解包：裸张量 → (tensor, None)；
    LatentBundle → (samples, noise_mask)。"""
    if isinstance(latent, LatentBundle):
        return latent.samples, latent.noise_mask
    return latent, None


class ControlBundle:
    """controlnet.apply 的产物（CONTROL 对象）：ControlNetModel + hint 图 +
    强度 + 步窗口（start/end_percent，按执行步数区间计）。
    hint 存 PIL（apply 时不知采样分辨率），sample 内按 latent 尺寸×8 转张量。"""

    def __init__(self, controlnet, image, strength=1.0,
                 start_percent=0.0, end_percent=1.0):
        self.controlnet = controlnet
        self.image = image
        self.strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent


def controlnet_apply(control_net, image, strength=1.0,
                     start_percent=0.0, end_percent=1.0):
    """ControlNet Apply：把 ControlNetModel 与 hint 图（边缘/姿态等线稿）捆绑，
    输出 CONTROL 对象喂 sample 的 control 端口（对标 ComfyUI ControlNetApply）。
    strength=残差强度（0 ≡ 关闭）；start/end_percent 为生效步窗口
    （按 sample 实际执行步数计，[start,end)），ComfyUI 同名参数语义。"""
    if not hasattr(control_net, "forward") or not hasattr(control_net, "dtype"):
        raise ValueError(
            f"controlnet.apply: control_net 需为 ControlNetModel，"
            f"got {type(control_net).__name__}")
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError(
                f"controlnet.apply: hint 图需单张（batch hint 暂不支持），"
                f"got {len(image)} 张")
        image = image[0]
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"controlnet.apply: image 需为 PIL 图像，got {type(image).__name__}")
    if not 0.0 <= start_percent < 1.0:
        raise ValueError(f"start_percent 需在 [0,1)，got {start_percent}")
    if not 0.0 < end_percent <= 1.0:
        raise ValueError(f"end_percent 需在 (0,1]，got {end_percent}")
    if end_percent <= start_percent:
        raise ValueError(
            f"步窗口为空：[{start_percent},{end_percent})")
    print(f"[controlnet.apply] strength={strength} "
          f"window=[{start_percent},{end_percent}) hint={image.size}", flush=True)
    return {"control": ControlBundle(control_net, image, float(strength),
                                     float(start_percent), float(end_percent))}


def as_cond_segments(cond, name="cond"):
    """COND 归一化为段列表 [(tensor, area|None, strength), ...]：
    裸张量 → 单个整图段；cond.set_area 产物（段列表）原样校验通过；
    SD3 dict 形式明确拒绝。area = (ly, lx, lh, lw) latent 像素坐标。"""
    if isinstance(cond, dict):
        raise ValueError(
            f"{name}: 暂不支持 SD3 dict 形式 COND（embeds+pooled），"
            f"仅支持 SD1.x/SDXL 张量形式")
    if isinstance(cond, list):
        segs = []
        for item in cond:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                raise ValueError(
                    f"{name}: COND 段需为 (tensor, area, strength) 三元组，"
                    f"got {type(item).__name__}")
            t, area, s = item
            if not torch.is_tensor(t):
                raise ValueError(f"{name}: COND 段张量缺失，got {type(t).__name__}")
            segs.append((t, area, float(s)))
        if not segs:
            raise ValueError(f"{name}: COND 段列表为空")
        return segs
    if not torch.is_tensor(cond):
        raise ValueError(f"{name}: COND 需为张量，got {type(cond).__name__}")
    return [(cond, None, 1.0)]


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


def checkpoint_load(models, ckpt, dtype="auto", offload="auto", use_t5="auto"):
    """Checkpoint Loader：加载 checkpoint，返回同一 workflow 的 model/clip/vae 三个视图。
    dtype/offload/use_t5 为性能旋钮（auto 继承服务端环境变量）——旋钮节点化
    （dev-plan 9.6 纪律 #3）：算子参数可逐次覆盖，不同旋钮组合是独立常驻条目。"""
    key, _newly = models.load(ckpt, dtype=dtype, offload=offload, use_t5=use_t5)
    pipe = models.get(key)
    return {"model": pipe, "clip": pipe, "vae": pipe}
    return {"model": pipe, "clip": pipe, "vae": pipe}


def model_unload(models, target=""):
    """显式卸载常驻模型（Unload Model，对标 ComfyUI 的 free/unload 操作）。

    target：空 / all / * = 卸载全部；否则按名字或缓存键匹配（可省略扩展名，
    支持 vae:/cn:/motion 组合键的前缀形式）。返回 unloaded/resident 两个逗号分隔
    字符串（node 输出端口只能传字符串，便于在画布/日志里看结果）。
    走与加载一致的淘汰路径：断对象仓库视图 → gc.collect → empty_cache。
    """
    done = models.unload(target)
    resident = models.resident()
    return {"unloaded": ",".join(done) if done else "(none)",
            "resident": ",".join(resident) if resident else "(empty)"}


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


def resolve_span(steps, denoise=1.0, start_at_step=0, end_at_step=0):
    """KSampler Advanced 步区间解析（纯函数，便于无 GPU 单测）。
    返回采样要执行的时间步区间 [start, end)：
    - start_at_step>0：显式分段起点（封顶 steps-1），优先于 denoise
    - 否则 denoise>=0.999 从头（start=0）；denoise<1 按既有语义推算起点
    - end_at_step<=0 或越界时跑到尾（end=steps）
    区间为空（end<=start）抛 ValueError。"""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if start_at_step > 0:
        start = min(start_at_step, steps - 1)
    elif denoise >= 0.999:
        start = 0
    else:
        start = min(int(round(steps * (1 - denoise))), steps - 1)
    end = steps if end_at_step <= 0 else min(end_at_step, steps)
    if end <= start:
        raise ValueError(
            f"empty sampling span: start={start} end={end} (steps={steps})")
    return start, end


def sample(model, pos, neg, base, seed=-1, steps=20, cfg=7.0,
           sampler_name=DEFAULT_SAMPLER, scheduler=DEFAULT_SCHEDULER,
           denoise=1.0, start_at_step=0, end_at_step=0, add_noise=True,
           control=None, preview_cb=None, interrupt_check=None):
    """KSampler：手动采样循环，返回 {'latent': ..., 'seed': 实际种子}。
    sampler_name（更新公式）× scheduler（sigma 曲线：normal/karras/exponential/beta）
    自由组合（M2 解耦）；旧一体名 dpmpp_2m_karras 兼容（见 app.samplers）。
    model 可为裸 pipe 或带 LoRA 补丁的 ModelRef（采样前启用、采样后关闭）。

    分段采样（KSampler Advanced，对标 ComfyUI）：
    - start_at_step>0：从该步开始（优先于 denoise），输入 latent 按该步
      噪声水平加噪（add_noise=True）或直接当作该步状态接力（add_noise=False，
      用于接上一段 end_at_step 提前停下的输出——此时 latent 天然带残余噪声）
    - end_at_step>0：跑到该步停下（不含），返回带残余噪声的 latent 供接力
    - add_noise=False 且 start_at_step=0/denoise=1：输入 latent 原样起步
      （ComfyUI disable noise 语义，使用者自负）
    - 默认值（0/0/True + denoise）下与历史行为逐字节一致

    注入三类（M4 采样循环改造；均为可选，全缺省时走既有快速路径，逐字节不变）：
    - control（CONTROL 对象，controlnet.apply 产物）：步窗口内逐步跑 ControlNet
      前向，残差×strength 注入 unet（down/mid_block_additional_residuals）；
      整图 cond 且正/负等长时按 diffusers CFG 惯例拼批（单次 cn+unet 前向）
    - COND 段列表（cond.set_area 产物）：每段单独前向，按区域 mask×strength
      加权混合（ComfyUI 同款求和语义：重叠区域影响叠加，不归一化）；
      与 control 互斥（每段都跑 controlnet 成本×N，明确报错）
    - noise_mask（LatentBundle，latent.set_noise_mask 产物）：每步后把 mask
      外区域混回原始 latent 的当步加噪版（局部重绘）
    - ipa（IPABundle，ipadapter.apply 产物）：临时换装 IPAdapter 注意力
      处理器，每步前向带 added_cond_kwargs 图像 embeds（CFG：负向为零向量/
      零图编码），weight×步窗口经处理器 scale 逐步改写；采样后恢复原处理器。
      与 control/area/noise_mask 正交可叠加

    preview_cb 可选：签名 preview_cb(latents, step_index, total)，在每一步
    去噪后回调（由调用方注入预览录制，如 app.preview.PreviewRecorder），
    本层不感知网络。
    interrupt_check 可选：无参回调，每步采样前调用，抛异常即中断
    （由调用方注入取消检查，如 app.execution.check_cancelled）。"""
    sampler_name, scheduler = resolve_sampler(sampler_name, scheduler)

    pipe, patches = resolve_pipe(model)
    ipa = getattr(model, "ipa", None)  # ipadapter.apply 产物（IPABundle）
    if patches:
        print("[sample] lora patches: "
              + ", ".join(f"{n}@{w}" for n, w in patches), flush=True)
    _enable_adapters(pipe, patches)

    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    device = exec_device_of(pipe)
    sched = make_scheduler(sampler_name, scheduler, pipe.scheduler.config)
    print(f"[sample] sampler={sampler_name} scheduler={scheduler} "
          f"steps={steps} cfg={cfg}", flush=True)
    sched.set_timesteps(steps, device=device)
    timesteps = sched.timesteps

    base_samples, noise_mask = resolve_latent(base)
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(base_samples.shape, generator=gen, device=device,
                        dtype=base_samples.dtype)

    start, end = resolve_span(steps, denoise, start_at_step, end_at_step)
    if start_at_step > 0 and denoise < 0.999:
        print(f"[sample] start_at_step={start_at_step} 优先，denoise={denoise} "
              f"被忽略", flush=True)
    ts = timesteps[start:end]
    if not add_noise:
        # 接力模式：输入 latent 已是 start 步的状态（含残余噪声），不再加噪
        latents = base_samples.clone()
        print(f"[sample] add_noise=False -> 接力输入 latent，"
              f"span=[{start},{end})", flush=True)
    elif start_at_step <= 0 and denoise >= 0.999:
        latents = (noise * sched.init_noise_sigma).to(base_samples.dtype)
    else:
        latents = sched.add_noise(base_samples, noise,
                                  timesteps[start:start + 1]).to(base_samples.dtype)
        print(f"[sample] denoise={denoise} start_at_step={start_at_step} "
              f"-> skip first {start} steps", flush=True)
    if end < steps:
        print(f"[sample] end_at_step={end} -> 提前停，latent 带残余噪声可接力",
              flush=True)

    b = base_samples.shape[0]

    # noise_mask（局部重绘）：mask 外区域每步混回原始 latent 的当步加噪版
    if noise_mask is not None:
        noise_mask = noise_mask.to(device=latents.device, dtype=latents.dtype)
        if tuple(noise_mask.shape[-2:]) != tuple(latents.shape[-2:]):
            raise ValueError(
                f"noise_mask 尺寸 {tuple(noise_mask.shape[-2:])} 与 latent "
                f"{tuple(latents.shape[-2:])} 不符（需同分辨率 latent）")
        if noise_mask.shape[0] == 1 and b > 1:
            noise_mask = noise_mask.expand(b, -1, -1, -1)
        print("[sample] noise_mask 生效：mask 外区域每步混回原 latent", flush=True)

    # COND 归一化为段列表；裸张量（无区域、strength=1）为 plain
    pos_segs = as_cond_segments(pos, "pos")
    neg_segs = as_cond_segments(neg, "neg")
    pos_plain = len(pos_segs) == 1 and pos_segs[0][1] is None \
        and pos_segs[0][2] == 1.0
    neg_plain = len(neg_segs) == 1 and neg_segs[0][1] is None \
        and neg_segs[0][2] == 1.0
    if control is not None and not (pos_plain and neg_plain):
        raise ValueError(
            "controlnet 暂不与 cond.set_area 组合（每段都要跑 controlnet，"
            "成本×N）；请仅用整图 cond 或先去掉 control")

    # ControlNet 预准备：hint 按采样分辨率转张量 + 步窗口
    hint = None
    win_lo = win_hi = 0
    if control is not None:
        import numpy as np  # 懒加载：模块级保持纯 torch 面（无 GPU 单测可导入）
        hp, wp = latents.shape[2] * 8, latents.shape[3] * 8
        img = control.image
        if img.size != (wp, hp):
            img = img.resize((wp, hp), Image.BILINEAR)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        hint = torch.from_numpy(arr).permute(2, 0, 1)[None]
        hint = hint.to(device=device, dtype=pipe.unet.dtype)
        hint = hint.expand(b, -1, -1, -1)
        win_lo = round(control.start_percent * len(ts))
        win_hi = round(control.end_percent * len(ts))
        print(f"[sample] controlnet strength={control.strength} "
              f"window=[{win_lo},{win_hi}) of {len(ts)} steps", flush=True)

    def cn_residuals(i, inp, t, hidden):
        """当前步的 controlnet 残差（窗口外/无 control 返回 None）。
        inp/hidden 批数与 unet 输入一致（cat 拼批 2b 时 hint 复制两份）。"""
        if control is None or not (win_lo <= i < win_hi):
            return None
        h = hint if hidden.shape[0] == b else torch.cat([hint, hint])
        down, mid = control.controlnet(
            inp, t, encoder_hidden_states=hidden, controlnet_cond=h,
            conditioning_scale=control.strength, return_dict=False)
        return down, mid

    # IPAdapter 装配（ipadapter.apply 产物，plugins/ipadapter_ops.py）：
    # 临时换装 IPAdapter 注意力处理器 + image 投影层，采样后 try/finally
    # 恢复原样（常驻管道不被污染）。weight/步窗口经逐步改写处理器 scale
    # 实现（0.35.2 的 IPAdapter 处理器读自身 scale 属性；窗口外置 0 →
    # 处理器跳过 IP 注意力，零额外开销）。embeds 为原始 CLIP 输出
    # （未过投影层），unet 的 encoder_hid_proj（ip_image_proj）负责投影。
    ipa_procs = []
    ipa_added_cat = ipa_added_pos = ipa_added_neg = None
    ipa_lo = ipa_hi = 0
    _ipa_old = None
    if ipa is not None:
        _ipa_old = (pipe.unet.attn_processors.copy(),
                    pipe.unet.encoder_hid_proj,
                    pipe.unet.config.get("encoder_hid_dim_type"))
        pipe.load_ip_adapter(ipa["state_dict"], subfolder="", weight_name="",
                             image_encoder_folder=None)
        from diffusers.models.attention_processor import (
            IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
        ipa_procs = [p for p in pipe.unet.attn_processors.values()
                     if isinstance(p, (IPAdapterAttnProcessor,
                                       IPAdapterAttnProcessor2_0))]
        pos_ie = ipa["pos_embeds"].expand(b, *ipa["pos_embeds"].shape[1:])
        neg_ie = ipa["neg_embeds"].expand(b, *ipa["neg_embeds"].shape[1:])
        ipa_added_cat = {"image_embeds": [torch.cat([neg_ie, pos_ie])]}
        ipa_added_pos = {"image_embeds": [pos_ie]}
        ipa_added_neg = {"image_embeds": [neg_ie]}
        ipa_lo = round(ipa["start_percent"] * len(ts))
        ipa_hi = round(ipa["end_percent"] * len(ts))
        print(f"[sample] ipadapter '{ipa.get('name', '?')}' "
              f"weight={ipa['weight']} window=[{ipa_lo},{ipa_hi})/{len(ts)}",
              flush=True)

    def ipa_scale_for(i):
        """按步窗口改写 IPAdapter 处理器强度（窗口外 0 → 处理器跳过）。"""
        if not ipa_procs:
            return
        w = ipa["weight"] if ipa_lo <= i < ipa_hi else 0.0
        for p in ipa_procs:
            p.scale = [w] * len(p.scale)

    def unet_forward(inp, t, hidden, res, added=None):
        """带可选残差/IPAdapter 注入的 unet 前向。"""
        kw = {}
        if res is not None:
            kw["down_block_additional_residuals"] = res[0]
            kw["mid_block_additional_residual"] = res[1]
        if added is not None:
            kw["added_cond_kwargs"] = added
        return pipe.unet(inp, t, encoder_hidden_states=hidden, **kw).sample

    # 前向路径分派：cat（既有拼批，逐字节不变）/ split（既有双前向）/ area（分段混合）
    hidden = None
    seg_mode = "cat"
    if pos_plain and neg_plain:
        pos_e = pos_segs[0][0].expand(b, -1, -1)
        neg_e = neg_segs[0][0].expand(b, -1, -1)
        # 正/负 cond 序列长度不一致（cond.combine 拼接产物）：分两次 UNet 前向，
        # 不做零填充（pad 会作为额外 token 参与注意力污染语义）；长度一致时
        # 保持既有的 neg/pos 拼批单次前向（行为逐字节不变）
        if pos_e.shape[1] != neg_e.shape[1]:
            seg_mode = "split"
            print(f"[sample] cond 长度不一致 pos={pos_e.shape[1]} "
                  f"neg={neg_e.shape[1]}：分两次 UNet 前向", flush=True)
        else:
            hidden = torch.cat([neg_e, pos_e])
    else:
        seg_mode = "area"
        # 区域 mask 预编译（latent 分辨率），裁剪到 latent 范围内
        h_l, w_l = latents.shape[-2:]

        def _seg_mask(area):
            ly, lx, lh, lw = area
            y0, x0 = max(0, ly), max(0, lx)
            y1, x1 = min(h_l, ly + lh), min(w_l, lx + lw)
            if y1 <= y0 or x1 <= x0:
                raise ValueError(
                    f"cond.set_area: 区域 {tuple(area)} 与 latent "
                    f"{h_l}x{w_l} 无交集")
            m = latents.new_zeros(1, 1, h_l, w_l)
            m[0, 0, y0:y1, x0:x1] = 1.0
            return m

        pos_masks = [None if a is None else _seg_mask(a)
                     for _, a, _ in pos_segs]
        neg_masks = [None if a is None else _seg_mask(a)
                     for _, a, _ in neg_segs]
        print(f"[sample] cond 分段：pos={len(pos_segs)} 段 "
              f"neg={len(neg_segs)} 段，按区域 mask×strength 混合", flush=True)

        def blend_side(segs, masks, inp, t, added=None):
            """逐段前向 + 区域加权混合（求和语义，不归一化——与 ComfyUI 一致，
            重叠区域影响叠加；无 area 的段视为整图）。"""
            kw = {"added_cond_kwargs": added} if added is not None else {}
            out = None
            for (tensor, _area, s), m in zip(segs, masks):
                e = tensor.expand(b, -1, -1)
                o = pipe.unet(inp, t, encoder_hidden_states=e, **kw).sample
                if m is not None:
                    o = o * m
                if s != 1.0:
                    o = o * s
                out = o if out is None else out + o
            return out

    t0 = time.time()
    try:
        with torch.no_grad():
            for i, t in enumerate(ts):
                if interrupt_check is not None:
                    interrupt_check()  # 取消检查点：抛异常即中断采样
                ipa_scale_for(i)
                if seg_mode == "cat":
                    inp = sched.scale_model_input(torch.cat([latents] * 2), t)
                    res = cn_residuals(i, inp, t, hidden)
                    noise_pred = unet_forward(inp, t, hidden, res, ipa_added_cat)
                    uncond, cond = noise_pred.chunk(2)
                elif seg_mode == "split":
                    inp = sched.scale_model_input(latents, t)
                    uncond = unet_forward(
                        inp, t, neg_e, cn_residuals(i, inp, t, neg_e),
                        ipa_added_neg)
                    cond = unet_forward(
                        inp, t, pos_e, cn_residuals(i, inp, t, pos_e),
                        ipa_added_pos)
                else:
                    inp = sched.scale_model_input(latents, t)
                    uncond = blend_side(neg_segs, neg_masks, inp, t,
                                        ipa_added_neg)
                    cond = blend_side(pos_segs, pos_masks, inp, t,
                                      ipa_added_pos)
                guided = uncond + cfg * (cond - uncond)
                latents = sched.step(guided, t, latents).prev_sample
                if noise_mask is not None:
                    orig_noisy = sched.add_noise(
                        base_samples, noise, t.reshape(1)).to(latents.dtype)
                    latents = latents * noise_mask + orig_noisy * (1 - noise_mask)
                print(f"[sample] step {i+1}/{len(ts)}", flush=True)
                if preview_cb is not None:
                    try:
                        preview_cb(latents, i, len(ts))
                    except Exception as e:
                        # 预览推送绝不影响采样主流程
                        print(f"[sample] preview callback failed (ignored): {e}",
                              flush=True)
    finally:
        if _ipa_old is not None:
            try:
                pipe.unet.set_attn_processor(_ipa_old[0])
                pipe.unet.encoder_hid_proj = _ipa_old[1]
                pipe.unet.config["encoder_hid_dim_type"] = _ipa_old[2]
            except Exception as e:
                print(f"[sample] ipadapter 处理器恢复失败（常驻管道可能被污染）: "
                      f"{e}", flush=True)
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
    而采样链路 latent 保持 fp16）。LatentBundle（带 noise_mask）取 samples
    解码——mask 仅作用于采样。"""
    latents, _mask = resolve_latent(latents)
    with torch.no_grad():
        img = pipe.vae.decode(
            latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor,
            return_dict=False)[0]
    pil = pipe.image_processor.postprocess(img, output_type="pil")
    return {"image": pil[0] if len(pil) == 1 else pil}


def vae_encode(pipe, image):
    """VAE Encode：PIL 图像（或列表）→ latent，供图生图（denoise<1）重采样。
    乘 scaling_factor 与 vae_decode 的除法互逆（ComfyUI VAEEncode 同语义）。
    输出裸张量；局部重绘请再经 latent.set_noise_mask 包 mask。"""
    images = image if isinstance(image, list) else [image]
    tensors = [pipe.image_processor.preprocess(img) for img in images]
    img_t = torch.cat(tensors).to(device=exec_device_of(pipe), dtype=pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(img_t).latent_dist.sample()
    return {"latent": (latent * pipe.vae.config.scaling_factor).to(torch.float16)}


def video_sample(model, prompt="", neg_prompt="", image=None,
                 width=832, height=480, num_frames=121, fps=24,
                 steps=50, cfg=5.0, seed=-1, decode_chunk_size=0,
                 output_type="pil", preview_cb=None, interrupt_check=None):
    """Video Sample：文/图生视频，一个算子兼容多种视频管道。
    output_type="pil"（默认）：返回 {'video': {'frames': [PIL...], 'fps': n}, 'seed': 实际种子}；
    output_type="latent"：不解码，返回 {'latent': 3D latent, 'seed': 实际种子}，
    交给 vae_decode_video 接力（decode 精度/分块成为流水线可调参数）。

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
        "output_type": output_type,
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
    if output_type == "latent":
        # latent 模式：out.frames 即未解码的 3D latent
        # （SVD: (b,f,c,h,w)；AnimateDiff/Wan: (b,c,f,h,w)，布局交由下游自适应）
        latents = out.frames
        print(f"[video.sample] done in {time.time()-t0:.1f}s "
              f"latent={tuple(latents.shape)} seed={seed}", flush=True)
        return {"latent": latents, "seed": seed}
    frames = out.frames[0]
    print(f"[video.sample] done in {time.time()-t0:.1f}s "
          f"frames={len(frames)} seed={seed}", flush=True)
    return {"video": {"frames": frames, "fps": fps}, "seed": seed}


def vae_decode_video(pipe, latents, num_frames=0, decode_chunk_size=14,
                     force_fp32=True, fps=24):
    """VAE Decode(视频)：3D latent → PIL 帧序列。
    与管道内置 decode 等价，但把两个关键点变成显式参数：
    - force_fp32：VAE 转 fp32 解码（Pascal fp16 解码过曝/亮度漂移的修法，
      与图像管道 VAE_FP32 防黑图同源）；幂等，offload 钩子只搬设备不转 dtype
    - decode_chunk_size：分块解码控显存峰值（SVD 内置默认 14）
    复用管道私有 decode_latents（内部处理 scaling_factor 与帧维布局），
    按签名过滤参数兼容 SVD(num_frames+chunk)/AnimateDiff(无参)差异。
    num_frames<=0 时按 latent 通道维(=4)推断帧维位置。"""
    import inspect

    if force_fp32 and pipe.vae.dtype != torch.float32:
        pipe.vae.to(torch.float32)
        print("[vae.decode_video] vae -> fp32（防过曝漂移）", flush=True)
    dtype = pipe.vae.dtype
    latents = latents.to(device=exec_device_of(pipe), dtype=dtype)
    if num_frames <= 0:
        num_frames = (latents.shape[2] if latents.shape[1] == 4
                      else latents.shape[1])
    sig = inspect.signature(pipe.decode_latents).parameters
    cands = {"num_frames": num_frames, "decode_chunk_size": decode_chunk_size}
    kwargs = {k: v for k, v in cands.items() if k in sig}
    with torch.no_grad():
        video = pipe.decode_latents(latents, **kwargs)
    vp = getattr(pipe, "video_processor", None)
    if vp is None:
        raise ValueError(
            f"{type(pipe).__name__} 无 video_processor，无法后处理视频帧")
    frames = vp.postprocess_video(video, output_type="pil")[0]
    print(f"[vae.decode_video] frames={len(frames)} fp32={force_fp32} "
          f"chunk={decode_chunk_size}", flush=True)
    return {"video": {"frames": frames, "fps": fps}}


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def embedding_load(pipe, embeddings_dir, names):
    """Textual Inversion 加载：把 embedding 文件（badhandv4/EasyNegative 等）
    载进 CLIP 文本编码器，之后在正/反提示词里直接写该词即生效（常用于负面）。
    names 逗号分隔；文件按 <EMBEDDINGS_DIR>/<name>.safetensors/.pt/.bin 查找。
    幂等：token 已在词表则跳过（管道常驻共享，加载一次全局生效）。"""
    loaded = []
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        if name in pipe.tokenizer.get_vocab():
            print(f"[embedding.load] '{name}' 已在词表，跳过", flush=True)
            continue
        base = os.path.join(embeddings_dir, name)
        path = next((base + ext for ext in (".safetensors", ".pt", ".bin")
                     if os.path.isfile(base + ext)), None)
        if path is None:
            available = (sorted(os.listdir(embeddings_dir))
                         if os.path.isdir(embeddings_dir) else [])
            raise FileNotFoundError(
                f"embedding '{name}' not found in {embeddings_dir}; "
                f"available: {available or '(empty)'}")
        pipe.load_textual_inversion(path, token=name)
        loaded.append(name)
        print(f"[embedding.load] loaded {path}", flush=True)
    return {"clip": pipe}


def image_upscale(image, scale=2.0, width=0, height=0, method="lanczos"):
    """Image Upscale：图像放大，hires.fix 的前置（放大 → vae.encode →
    sample(denoise 0.3~0.5) 精修细节）。纯重采样本身不新增细节，细节由后续
    img2img 精修在更高分辨率上补出。
    尺寸二选一：width/height 均 >0 时按目标尺寸，否则按 scale 倍率；
    结果一律向下对齐 8 的倍数（VAE 编码要求尺寸可被 8 整除）。
    method 支持 lanczos（默认，质量最好）/bicubic/bilinear/nearest。
    batch 列表输入逐张处理（与 vae_decode 的 batch 输出对齐）。"""
    if method not in _RESAMPLE:
        raise ValueError(
            f"unknown method '{method}', available: {sorted(_RESAMPLE)}")

    def _up(img):
        w, h = img.size
        if width > 0 and height > 0:
            tw, th = width, height
        else:
            if scale <= 0:
                raise ValueError(f"scale must be > 0, got {scale}")
            tw, th = round(w * scale), round(h * scale)
        tw, th = max(8, tw // 8 * 8), max(8, th // 8 * 8)
        return img.resize((tw, th), _RESAMPLE[method])

    if isinstance(image, list):
        return {"image": [_up(i) for i in image]}
    return {"image": _up(image)}


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


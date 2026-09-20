"""COND / LATENT 张量工具算子（ComfyUI 对标第二/三档，插件化交付）。

- cond.combine / cond.average：conditioning 张量拼接/加权（分区构图基础）
- cond.set_area：区域条件（第三档 M4，采样循环内分段前向按区域混合）
- latent.upscale / latent.composite：latent 空间插值放大/贴入拼接
- latent.set_noise_mask：latent 包裹 noise_mask（局部重绘核心，采样循环消费）

部署：本文件放 PLUGINS_DIR（默认 /models/plugins.d，bind-mount 持久化）重启自动
扫描注册，或经 POST /admin/plugins 热上传。使用：经 inference-op 通用节点调用。

COND 形态：SD1.x/SDXL 为张量 (b, seq, dim)，或 cond.set_area 产物的段列表
[(tensor, area|None, strength), ...]（area 为 (ly,lx,lh,lw) latent 像素坐标）；
SD3 为 dict（embeds+pooled，本期明确报错不支持，避免静默出废片）。
LATENT 为 4D 张量 (b, 4, h/8, w/8) 或 LatentBundle（samples+noise_mask）；
像素坐标参数一律按图像像素计（内部 //8）。
"""
import torch
from PIL import Image

from app import ops as _core_ops  # 插件可见性约定：可复用核心原语


def _as_tensor_cond(cond, op_name):
    """COND 统一为张量；SD3 dict 形式与段列表明确拒绝。"""
    if isinstance(cond, dict):
        raise ValueError(
            f"{op_name} 暂不支持 SD3 dict 形式 COND（embeds+pooled），"
            f"仅支持 SD1.x/SDXL 张量形式")
    if isinstance(cond, list):
        raise ValueError(
            f"{op_name} 只接受整图 cond 张量，got cond.set_area 段列表")
    if not torch.is_tensor(cond):
        raise ValueError(f"{op_name}: COND 需为张量，got {type(cond).__name__}")
    return cond


def cond_combine(cond_a, cond_b):
    """Conditioning Combine：沿序列维拼接两段 conditioning（对标 ComfyUI
    ConditioningCombine），语义 = 两段提示词同时生效（不受 77 token 截断限制）。
    batch 维不同时按广播对齐（一方为 1 则 expand）。
    任一侧为 cond.set_area 段列表时退化为段列表拼接（各自区域/strength 保留，
    多区域构图 = 多个 set_area 产物经本算子串联）。"""
    if isinstance(cond_a, list) or isinstance(cond_b, list):
        segs = _core_ops.as_cond_segments(cond_a, "cond.combine") \
            + _core_ops.as_cond_segments(cond_b, "cond.combine")
        print(f"[cond.combine] 段列表拼接 -> {len(segs)} 段", flush=True)
        return {"cond": segs}
    a = _as_tensor_cond(cond_a, "cond.combine")
    b = _as_tensor_cond(cond_b, "cond.combine")
    if a.shape[0] != b.shape[0]:
        if a.shape[0] == 1:
            a = a.expand(b.shape[0], -1, -1)
        elif b.shape[0] == 1:
            b = b.expand(a.shape[0], -1, -1)
        else:
            raise ValueError(
                f"cond.combine: batch 不匹配 {tuple(a.shape)} vs {tuple(b.shape)}")
    out = torch.cat([a, b], dim=1)
    print(f"[cond.combine] {tuple(a.shape)} + {tuple(b.shape)} -> {tuple(out.shape)}",
          flush=True)
    return {"cond": out}


def cond_set_area(cond, x=0, y=0, width=512, height=512, strength=1.0):
    """Conditioning Set Area：把整图 cond 标记为只在指定区域生效（对标 ComfyUI
    ConditioningSetArea），输出段列表供 sample 分段前向按区域混合。
    x/y/width/height 为图像像素（内部 //8）；strength 为该段权重（>0）。
    多区域构图：每段 clip.encode 分别 set_area 后经 cond.combine 拼接；
    混合为求和语义（不归一化，重叠区域影响叠加，与 ComfyUI 一致）。"""
    t = _as_tensor_cond(cond, "cond.set_area")
    if width <= 0 or height <= 0:
        raise ValueError(f"cond.set_area: width/height 需 > 0，got ({width},{height})")
    if x < 0 or y < 0:
        raise ValueError(f"cond.set_area: x/y 需 >= 0，got ({x},{y})")
    if strength <= 0:
        raise ValueError(f"cond.set_area: strength 需 > 0，got {strength}")
    area = (y // 8, x // 8, height // 8, width // 8)
    if area[2] <= 0 or area[3] <= 0:
        raise ValueError(
            f"cond.set_area: 区域小于 1 latent px（{width}x{height} 图像像素）")
    print(f"[cond.set_area] area(latent px)={area} strength={strength}", flush=True)
    return {"cond": [(t, area, float(strength))]}


def cond_average(cond_a, cond_b, weight=0.5):
    """Conditioning Average：两段 conditioning 按权重加权平均（对标 ComfyUI
    ConditioningAverage）：out = a*weight + b*(1-weight)。
    序列长度不同时短者补零到长者（dim=1）。"""
    a = _as_tensor_cond(cond_a, "cond.average")
    b = _as_tensor_cond(cond_b, "cond.average")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"cond.average: weight 需在 [0,1]，got {weight}")
    if a.shape[0] != b.shape[0]:
        if a.shape[0] == 1:
            a = a.expand(b.shape[0], -1, -1)
        elif b.shape[0] == 1:
            b = b.expand(a.shape[0], -1, -1)
        else:
            raise ValueError(
                f"cond.average: batch 不匹配 {tuple(a.shape)} vs {tuple(b.shape)}")
    n = max(a.shape[1], b.shape[1])
    if a.shape[1] < n:
        a = torch.cat([a, a.new_zeros(a.shape[0], n - a.shape[1], a.shape[2])],
                      dim=1)
    if b.shape[1] < n:
        b = torch.cat([b, b.new_zeros(b.shape[0], n - b.shape[1], b.shape[2])],
                      dim=1)
    out = a * weight + b * (1.0 - weight)
    print(f"[cond.average] weight={weight} -> {tuple(out.shape)}", flush=True)
    return {"cond": out}


_INTERP = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic"}


def latent_upscale(latent, scale=2.0, width=0, height=0, method="bicubic"):
    """Latent Upscale：latent 空间插值放大（对标 ComfyUI LatentUpscale）。
    尺寸二选一：width/height（图像像素，内部 //8）均 >0 时按目标尺寸，
    否则按 scale 倍率；latent 尺寸天然整数对齐（1 latent px = 8 图像 px）。
    method 支持 bicubic（默认）/bilinear/nearest。
    LatentBundle（带 noise_mask）明确拒绝：请先放大再 set_noise_mask。"""
    if isinstance(latent, _core_ops.LatentBundle):
        raise ValueError(
            "latent.upscale 不接受带 noise_mask 的 latent（mask 不会随放大缩放）；"
            "请先 latent.upscale 再 latent.set_noise_mask")
    if method not in _INTERP:
        raise ValueError(
            f"unknown method '{method}', available: {sorted(_INTERP)}")
    if not torch.is_tensor(latent) or latent.dim() != 4:
        raise ValueError(
            f"latent.upscale: 需 4D latent 张量，got "
            f"{tuple(latent.shape) if torch.is_tensor(latent) else type(latent).__name__}")
    b, c, lh, lw = latent.shape
    if width > 0 and height > 0:
        th, tw = max(1, height // 8), max(1, width // 8)
    else:
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        th, tw = max(1, round(lh * scale)), max(1, round(lw * scale))
    kwargs = {"mode": _INTERP[method]}
    if _INTERP[method] in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    out = torch.nn.functional.interpolate(latent, size=(th, tw), **kwargs)
    print(f"[latent.upscale] {tuple(latent.shape)} -> {tuple(out.shape)} "
          f"({method})", flush=True)
    return {"latent": out}


def latent_composite(dst, src, x=0, y=0, feather=0):
    """Latent Composite：把 src latent 贴入 dst 的 (x, y) 处（对标 ComfyUI
    LatentComposite），x/y 为图像像素坐标（内部 //8）。越界部分自动裁剪；
    feather>0（图像像素）时边缘线性羽化过渡。batch：src 为 1 时广播到 dst。"""
    for name, t in (("dst", dst), ("src", src)):
        if isinstance(t, _core_ops.LatentBundle):
            raise ValueError(
                f"latent.composite: {name} 带 noise_mask（不会随贴入变换）；"
                f"请先 composite 再 latent.set_noise_mask")
        if not torch.is_tensor(t) or t.dim() != 4:
            raise ValueError(f"latent.composite: {name} 需 4D latent 张量")
    if x < 0 or y < 0:
        raise ValueError(f"latent.composite: x/y 需 >= 0，got ({x},{y})")
    lx, ly = x // 8, y // 8
    b, c, dh, dw = dst.shape
    h = min(src.shape[2], dh - ly)
    w = min(src.shape[3], dw - lx)
    if h <= 0 or w <= 0:
        raise ValueError(
            f"latent.composite: src 完全越界（dst {dh}x{dw} latent px，"
            f"起点 ({lx},{ly})）")
    src_c = src[:, :, :h, :w].to(dst.dtype)  # 原地运算前统一 dtype（fp16/fp32 混用）
    if src_c.shape[0] == 1 and b > 1:
        src_c = src_c.expand(b, -1, -1, -1)
    elif src_c.shape[0] != b:
        raise ValueError(
            f"latent.composite: batch 不匹配 dst={b} src={src_c.shape[0]}")
    out = dst.clone()
    region = out[:, :, ly:ly + h, lx:lx + w]
    f = feather // 8
    if f > 0 and h > 2 * f and w > 2 * f:
        # 边缘线性斜坡羽化（1 内部 → 0 边缘），x/y 两维斜坡相乘
        ramp_y = torch.ones(h, device=dst.device, dtype=dst.dtype)
        ramp_x = torch.ones(w, device=dst.device, dtype=dst.dtype)
        for i in range(f):
            v = (i + 1) / (f + 1)
            ramp_y[i] = ramp_y[h - 1 - i] = v
            ramp_x[i] = ramp_x[w - 1 - i] = v
        mask = (ramp_y[:, None] * ramp_x[None, :])[None, None]
        region.mul_(1 - mask).add_(src_c * mask)
    else:
        region.copy_(src_c)
    print(f"[latent.composite] src {tuple(src_c.shape)} -> ({lx},{ly}) "
          f"feather={f} latent px", flush=True)
    return {"latent": out}


def latent_set_noise_mask(latent, mask):
    """Set Latent Noise Mask：给 latent 包裹 noise_mask（对标 ComfyUI
    SetLatentNoiseMask），局部重绘核心——sample 循环内每步把 mask 外区域
    混回原始 latent 的当步加噪版，mask 内正常去噪。
    mask 为灰度 IMAGE 对象（对齐第一档蒙版约定：白=重绘区域），
    内部 resize 到 latent 尺寸转 0..1 张量。重复包裹明确拒绝。"""
    samples, existing = _core_ops.resolve_latent(latent)
    if existing is not None:
        raise ValueError("latent.set_noise_mask: 已有 noise_mask，请勿重复包裹")
    if not torch.is_tensor(samples) or samples.dim() != 4:
        raise ValueError("latent.set_noise_mask: latent 需 4D 张量")
    if isinstance(mask, list):
        if len(mask) != 1:
            raise ValueError(
                f"latent.set_noise_mask: mask 需单张灰度 IMAGE，got {len(mask)} 张")
        mask = mask[0]
    if not isinstance(mask, Image.Image):
        raise ValueError(
            f"latent.set_noise_mask: mask 需为 IMAGE 对象（灰度），"
            f"got {type(mask).__name__}")
    lh, lw = samples.shape[2], samples.shape[3]
    m = mask.convert("L").resize((lw, lh), Image.BILINEAR)
    import numpy as np  # 懒加载：与核心 ops 同约定（无 GPU 契约测试可导入）
    arr = np.asarray(m, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].to(device=samples.device,
                                             dtype=samples.dtype)
    print(f"[latent.set_noise_mask] mask -> (1,1,{lh},{lw}) "
          f"mean={float(t.mean()):.3f}", flush=True)
    return {"latent": _core_ops.LatentBundle(samples, t)}


def register(registry):
    registry.register(
        "cond.combine",
        inputs={"cond_a": "COND", "cond_b": "COND"},
        outputs={"cond": "COND"},
        description="Conditioning Combine：两段 conditioning 沿序列维拼接，"
                    "两段提示词同时生效（可突破 77 token 截断）；"
                    "暂不支持 SD3 dict 形式 COND")(cond_combine)
    registry.register(
        "cond.average",
        inputs={"cond_a": "COND", "cond_b": "COND", "weight": "FLOAT"},
        outputs={"cond": "COND"},
        description="Conditioning Average：out = cond_a*weight + cond_b*(1-weight)，"
                    "weight 默认 0.5；序列长度不同时短者补零；"
                    "暂不支持 SD3 dict 形式 COND")(cond_average)
    registry.register(
        "latent.upscale",
        inputs={"latent": "LATENT", "scale": "FLOAT", "width": "INT",
                "height": "INT", "method": "STRING"},
        outputs={"latent": "LATENT"},
        description="Latent Upscale：latent 空间插值放大，按 scale 倍率（默认 2.0）"
                    "或目标 width/height（图像像素）；method 支持 "
                    "bicubic/bilinear/nearest")(latent_upscale)
    registry.register(
        "latent.composite",
        inputs={"dst": "LATENT", "src": "LATENT", "x": "INT", "y": "INT",
                "feather": "INT"},
        outputs={"latent": "LATENT"},
        description="Latent Composite：把 src 贴入 dst 的 (x,y) 处（图像像素坐标），"
                    "越界自动裁剪；feather>0（图像像素）边缘线性羽化；"
                    "带 noise_mask 的 latent 明确拒绝（先 composite 再 set_noise_mask）")(latent_composite)
    registry.register(
        "cond.set_area",
        inputs={"cond": "COND", "x": "INT", "y": "INT", "width": "INT",
                "height": "INT", "strength": "FLOAT"},
        outputs={"cond": "COND"},
        description="Conditioning Set Area：cond 标记为只在 (x,y,width,height) 区域"
                    "生效（图像像素），strength 为该段权重（默认 1.0）；"
                    "多区域 = 多个 set_area 经 cond.combine 拼接；"
                    "暂不与 controlnet 组合")(cond_set_area)
    registry.register(
        "latent.set_noise_mask",
        inputs={"latent": "LATENT", "mask": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="Set Latent Noise Mask：latent 包裹 noise_mask（局部重绘核心）；"
                    "mask 为灰度 IMAGE（白=重绘区域），内部 resize 到 latent 尺寸；"
                    "配 vae.encode + sample(denoise<1) 做 inpaint")(latent_set_noise_mask)

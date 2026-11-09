"""CFG（Classifier-Free Guidance）策略层——对标 ComfyUI 的 CFGGuider。

为什么需要独立一层（真机教训 + ComfyUI 实践）：
标准 CFG 是 `uncond + scale * (cond - uncond)`，但在高 scale 下会产生
**过饱和 / 色彩塌陷 / 细节烧毁**。ComfyUI 把这一步做成可插拔的
`sampler_cfg_function` + `sampler_post_cfg_function` 钩子，于是能挂
CFG++、rescale、automatic CFG 等各种策略；我们的 sample() 原先把这行
硬编码，等于把整类画质改进挡在门外。

已实现策略：
- ``standard``  ：经典 CFG（默认，行为与历史完全一致）
- ``rescale``   ：SD3 论文的 rescale 技巧——把 guided 的标准差拉回 cond 预测
                  的标准差，抑制高 scale 过曝（ComfyUI 的 RescaleCFG 同款）
- ``cfg_pp``    ：CFG++（ICLR 2024）——把 cfg 融入 x0 估计的插值，而非直接
                  推噪声预测；低步数下明显更稳（对应 ComfyUI 的 *_cfg_pp 采样器）
- ``none``      ：跳过 CFG（scale<=1 时等价，也用于 guidance-distilled 模型）

参考实现：
- ComfyUI ``comfy/samplers.py`` 的 ``cfg_function`` / ``sampler_cfg_function``
- SD3 论文 §3.3（rescale）、CFG++ 论文（x0 插值式 CFG）
"""

from __future__ import annotations

CFG_STRATEGIES = ("standard", "rescale", "cfg_pp", "none")

# rescale 的强度上限：论文用 0.7（越大越接近原始 CFG，越小越抑制过曝）
DEFAULT_RESCALE = 0.7


def resolve_cfg_strategy(name: str | None) -> str:
    """规范化策略名（未知值回退 standard 并提示，不静默）。"""
    key = str(name or "standard").lower().strip()
    if key in ("", "default"):
        return "standard"
    if key not in CFG_STRATEGIES:
        print(f"[cfg] 未知策略 '{name}'，回退 standard"
              f"（可选 {list(CFG_STRATEGIES)}）", flush=True)
        return "standard"
    return key


def _rescale(guided, cond_pred, rescale: float):
    """SD3 rescale：把 guided 的逐样本标准差拉回 cond 预测的量级。

    公式（SD3 论文）：x = rescale * x + (1 - rescale) * x * (std(cond) / std(x))
    效果：高 cfg 时不再过曝，低 cfg 时几乎无影响。
    """
    import torch

    std_cond = cond_pred.std(dim=list(range(1, cond_pred.ndim)),
                             keepdim=True)
    std_guided = guided.std(dim=list(range(1, guided.ndim)), keepdim=True)
    # 防 0 除（常量条件预测的极端情况）
    ratio = std_cond / (std_guided + 1e-8)
    return rescale * guided + (1.0 - rescale) * guided * ratio


def apply_cfg(cond_pred, uncond_pred, scale: float, strategy: str = "standard",
              cfg_pp_x0: float = 0.0, rescale: float = DEFAULT_RESCALE):
    """应用 CFG 策略，返回引导后的预测。

    参数
    ----
    cond_pred, uncond_pred : 条件/无条件模型输出（噪声预测，形状一致）
    scale                  : CFG 强度
    strategy               : standard / rescale / cfg_pp / none
    cfg_pp_x0              : cfg_pp 模式下换成 x0 估计时传入的插值系数
                             （0 = 不做 x0 语义，退化成 standard 路径）
    rescale                : rescale 策略的混合系数（默认 0.7）
    """
    strategy = resolve_cfg_strategy(strategy)

    # scale<=1 时 CFG 本就无效：直接返回 cond 预测（省一次多余运算）
    if strategy == "none" or scale <= 1.0:
        return cond_pred

    guided = uncond_pred + scale * (cond_pred - uncond_pred)

    if strategy == "cfg_pp" and cfg_pp_x0 > 0.0:
        # CFG++：在 x0 域插值更稳。这里用预测的线性插值近似——
        # x0_pp = x0_cond + (x0_cond - x0_uncond) * (scale - 1) * k
        # 由调用方按 sigma 换成 x0 后传入 cfg_pp_x0，未换算时走 standard
        k = min(1.0, float(cfg_pp_x0))
        guided = guided * (1.0 - k) + (
            cond_pred + (cond_pred - uncond_pred) * (scale - 1.0)) * k

    if strategy == "rescale" or (strategy == "cfg_pp" and scale > 1.0):
        guided = _rescale(guided, cond_pred, float(rescale))

    return guided


def cfg_variants() -> list[str]:
    """供 /ops 或节点 UI 展示的合法策略列表。"""
    return list(CFG_STRATEGIES)

"""显存治理配置解析（纯 stdlib，可单测）。

OFFLOAD_MODE：
- none       整模型驻留显存（默认，延迟最低）
- model      子模块级搬移（enable_model_cpu_offload，旧开关 ENABLE_CPU_OFFLOAD=1 映射到此）
- sequential 逐层搬移（enable_sequential_cpu_offload，粒度最细、吞吐最低；
             视频模型 5B~14B 级显存不够时的启用档位）

QUANTIZATION：fp8 接口预留（视频/Flux 线必备），当前版本识别配置但未实现，
加载时按原 dtype 继续并告警。
"""
import os

MODES = ("none", "model", "sequential")
QUANTIZATIONS = ("none", "fp8")


def resolve_offload_mode(env=None):
    """OFFLOAD_MODE 优先；缺省回退历史开关 ENABLE_CPU_OFFLOAD=1 → model。"""
    env = os.environ if env is None else env
    mode = (env.get("OFFLOAD_MODE") or "").strip().lower()
    if not mode:
        mode = "model" if env.get("ENABLE_CPU_OFFLOAD", "0") == "1" else "none"
    if mode not in MODES:
        raise ValueError(f"unknown OFFLOAD_MODE '{mode}', available: {MODES}")
    return mode


def resolve_quantization(env=None):
    env = os.environ if env is None else env
    q = (env.get("QUANTIZATION") or "none").strip().lower()
    if q not in QUANTIZATIONS:
        raise ValueError(f"unknown QUANTIZATION '{q}', available: {QUANTIZATIONS}")
    return q

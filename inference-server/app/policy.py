"""显存/淘汰策略运行时旋钮（可读可写）。

为什么单独一个模块：旋钮的实际持有者是 `app.model_manager` 的模块级常量
（`MAX_RESIDENT` / `VRAM_RESERVE_MB`），而 `model_manager` 顶层 `import torch`
——本模块**不 import torch**，由 `app.main` 在启动时用 `attach()` 注入读写器，
于是策略逻辑可以在无 GPU 的开发机上直接单测（`tests/test_policy.py`）。

旋钮语义（详见 model_manager / lease 的注释）：
- `max_resident_models`：条目数兜底上限（显存预算淘汰为主，本项兜底）
- `vram_reserve_mb`：加载时预留给系统的显存（越大越保守、越容易淘汰常驻）
只读观测项：`offload_mode` / `quantization`（env 决定，改需重启）/ `lease_ttl_seconds` /
`lease_wait_timeout_seconds`（租约语义）。

定位：**单流水线串行时的策略实验工具**（dev-plan §P1「显存策略运维节点」）——
并发执行下动这些旋钮会互相干扰，不建议。
"""
from typing import Any, Callable, Dict

READERS: Dict[str, Callable[[], Any]] = {}
WRITERS: Dict[str, Callable[[Any], Any]] = {}

# 可写旋钮的取值约束（防手滑把 reserve 设成 0 导致显存打满）
LIMITS = {
    "max_resident_models": (1, 16),
    "vram_reserve_mb": (0, 8192),
}


def attach(readers: Dict[str, Callable[[], Any]],
           writers: Dict[str, Callable[[Any], Any]]) -> None:
    """由 app.main 注入读写器（key → 取值/写值函数）。重复调用覆盖（测试用）。"""
    READERS.clear(); READERS.update(readers)
    WRITERS.clear(); WRITERS.update(writers)


def snapshot() -> Dict[str, Any]:
    """当前策略快照（可写项 + 只读观测项）。未 attach 时返回空 dict。"""
    out: Dict[str, Any] = {}
    for k in sorted(READERS):
        try:
            out[k] = READERS[k]()
        except Exception as e:  # noqa: BLE001 观测失败不致命
            out[k] = f"<read failed: {type(e).__name__}>"
    return out


def update(patch: Dict[str, Any]) -> Dict[str, Any]:
    """按 patch 写入可写旋钮，返回写后的快照。

    校验：未知键 / 不可写键 / 越界 / 非整数 ⇒ ValueError（端点转 400）。
    写入后立即生效（淘汰逻辑每次调用都读模块级常量）。
    """
    if not isinstance(patch, dict):
        raise ValueError("policy: body 需为对象，如 {\"vram_reserve_mb\": 768}")
    applied: Dict[str, Any] = {}
    for key, raw in patch.items():
        if key not in WRITERS:
            raise ValueError(
                f"policy: {key} 不可写（可写：{sorted(WRITERS)}；"
                f"其余为只读观测项）")
        try:
            val = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"policy: {key} 需为整数，got {raw!r}")
        lo, hi = LIMITS.get(key, (None, None))
        if lo is not None and not (lo <= val <= hi):
            raise ValueError(f"policy: {key}={val} 越界（允许 {lo}~{hi}）")
        WRITERS[key](val)
        applied[key] = val
    return {"applied": applied, "policy": snapshot()}

"""显存策略旋钮单测（无需 torch/GPU：policy 模块不 import torch）。

运行：python3 tests/test_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import policy

FAILED = []


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


def main():
    # 未 attach：快照为空、写入报错（不得静默）
    policy.attach({}, {})
    check("未 attach 时快照为空", policy.snapshot() == {})
    try:
        policy.update({"vram_reserve_mb": 512})
        check("未 attach 时写入必须报错", False)
    except ValueError as e:
        check("未 attach 时写入报错且可读", "不可写" in str(e))

    # attach 假旋钮（模拟 model_manager 模块级常量）
    state = {"max_resident_models": 2, "vram_reserve_mb": 1024}
    policy.attach(
        readers={k: (lambda kk: (lambda: state[kk]))(k) for k in state},
        writers={k: (lambda kk: (lambda v: state.__setitem__(kk, v)))(k) for k in state},
    )
    check("快照读到注入值", policy.snapshot() == {"max_resident_models": 2, "vram_reserve_mb": 1024})

    out = policy.update({"vram_reserve_mb": 768})
    check("写入生效（返回 applied + 新快照）",
          out["applied"] == {"vram_reserve_mb": 768} and state["vram_reserve_mb"] == 768)
    check("写入后快照已更新", policy.snapshot()["vram_reserve_mb"] == 768)

    for bad, why in (({"vram_reserve_mb": 99999}, "越界"),
                     ({"vram_reserve_mb": -1}, "下界"),
                     ({"vram_reserve_mb": "abc"}, "非整数"),
                     ({"offload_mode": "sequential"}, "只读项不可写"),
                     ({"nonexistent": 1}, "未知键"),
                     ("not-a-dict", "body 非对象")):
        try:
            policy.update(bad)
            check(f"{why} 必须报错", False)
        except ValueError:
            check(f"{why} 报错", True)

    # 边界值合法
    for ok in ({"max_resident_models": 1}, {"max_resident_models": 16},
               {"vram_reserve_mb": 0}, {"vram_reserve_mb": 8192}):
        policy.update(ok)
    check("边界值 1/16/0/8192 均被接受",
          state["max_resident_models"] == 16 and state["vram_reserve_mb"] == 8192)

    # 只读观测不因读取失败而崩
    policy.attach(readers={"boom": lambda: (_ for _ in ()).throw(RuntimeError("x"))}, writers={})
    snap = policy.snapshot()
    check("读取失败的观测项降级为说明串", isinstance(snap.get("boom"), str) and "read failed" in snap["boom"])

    print()
    print(f"PASSED = {8 + 6 + 1 + 1} - FAILED = {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

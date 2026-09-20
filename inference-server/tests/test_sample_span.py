"""ops.resolve_span 单元测试：KSampler Advanced 分段采样步区间解析（纯函数）。

本机无 torch/diffusers：桩掉两模块后导入 app.ops（模块级不使用 torch API，
resolve_span 为纯 Python），只测区间解析语义与默认兼容。
运行：python3 -m tests.test_sample_span  或  python3 tests/test_sample_span.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules.setdefault("torch", types.ModuleType("torch"))

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

from app.ops import resolve_span


def check(name, got, want):
    assert got == want, f"FAIL: {name}: got {got}, want {want}"
    print(f"ok: {name} -> {got}")


def expect_raise(name, fn, *needle):
    try:
        fn()
    except ValueError as e:
        assert all(n in str(e) for n in needle), f"FAIL: {name}: {e}"
        print(f"ok: {name} raises ValueError: {str(e)[:50]}...")
        return
    raise SystemExit(f"FAIL: {name} should raise ValueError")


def main():
    # 默认兼容（历史行为）：denoise=1 → 全程；denoise<1 → 推算起点
    check("默认 steps=20 全程", resolve_span(20), (0, 20))
    check("denoise=1 显式", resolve_span(20, denoise=1.0), (0, 20))
    check("denoise=0.5@20 -> 起点 10", resolve_span(20, denoise=0.5), (10, 20))
    check("denoise=0.25@20 -> 起点 15", resolve_span(20, denoise=0.25), (15, 20))
    check("denoise=0.99@20 -> 起点 0 但走加噪路径（区间仍全程）",
          resolve_span(20, denoise=0.99), (0, 20))
    check("denoise 极小封顶 steps-1", resolve_span(20, denoise=0.01), (19, 20))

    # KSampler Advanced：显式分段
    check("start_at_step=10", resolve_span(20, start_at_step=10), (10, 20))
    check("end_at_step=10", resolve_span(20, end_at_step=10), (0, 10))
    check("start+end", resolve_span(20, start_at_step=5, end_at_step=15), (5, 15))
    check("start_at_step 优先于 denoise",
          resolve_span(20, denoise=0.5, start_at_step=3), (3, 20))
    check("start_at_step 越界封顶 steps-1",
          resolve_span(20, start_at_step=99), (19, 20))
    check("end_at_step 越界截断到 steps",
          resolve_span(20, end_at_step=99), (0, 20))

    # 非法区间
    expect_raise("空区间 start==end",
                 lambda: resolve_span(20, start_at_step=10, end_at_step=10),
                 "empty")
    expect_raise("空区间 end<start",
                 lambda: resolve_span(20, start_at_step=15, end_at_step=10),
                 "empty")
    expect_raise("steps<1", lambda: resolve_span(0), "steps")

    # 接力语义自检：两段拼接覆盖全程（0→10, 10→20）
    s1 = resolve_span(20, end_at_step=10)
    s2 = resolve_span(20, start_at_step=10)
    check("接力两段无缝覆盖", (s1[1], s2[0], s2[1]), (10, 10, 20))

    print("all sample span tests passed")


if __name__ == "__main__":
    main()

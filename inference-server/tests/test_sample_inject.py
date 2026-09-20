"""ops 注入逻辑单元测试（第三档 M4）：COND 段归一化 / LatentBundle 解包 /
controlnet_apply 校验。

本机无 torch/diffusers：桩掉两模块后导入 app.ops（torch 桩补 is_tensor；
被测逻辑均为纯 Python 对象/校验语义，不触 torch API）。
采样循环内的注入正确性（controlnet 残差/区域混合/noise_mask 混回）
由真机验收覆盖（抽图核对口径）。
运行：python3 -m tests.test_sample_inject  或  python3 tests/test_sample_inject.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeTensor:
    """最小张量替身：只供 torch.is_tensor 桩识别。"""


_fake_torch = types.ModuleType("torch")
_fake_torch.is_tensor = lambda x: isinstance(x, FakeTensor)
sys.modules.setdefault("torch", _fake_torch)

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

from PIL import Image

from app.ops import (ControlBundle, LatentBundle, as_cond_segments,
                     controlnet_apply, resolve_latent)


def expect_raise(name, fn, args, *needle):
    try:
        fn(*args)
    except ValueError as e:
        assert all(n in str(e) for n in needle), f"FAIL: {name}: {e}"
        print(f"ok: {name} raises ValueError: {str(e)[:60]}...")
        return
    raise SystemExit(f"FAIL: {name} should raise ValueError")


class FakeControlNet:
    dtype = "fp16"

    def forward(self, *a, **k):
        raise NotImplementedError


def main():
    t1, t2 = FakeTensor(), FakeTensor()

    # --- as_cond_segments：裸张量 → 单段整图 ---
    segs = as_cond_segments(t1, "pos")
    assert segs == [(t1, None, 1.0)], segs
    print("ok: 裸张量 -> [(tensor, None, 1.0)]")

    # --- as_cond_segments：段列表原样校验通过（strength 转 float） ---
    area = (0, 0, 32, 32)
    segs = as_cond_segments([(t1, area, 1), (t2, None, 0.5)], "pos")
    assert segs == [(t1, area, 1.0), (t2, None, 0.5)], segs
    print("ok: 段列表通过（strength 归一 float）")

    # --- as_cond_segments：非法输入 ---
    expect_raise("SD3 dict 拒绝", as_cond_segments,
                 ({"cross_attn": t1, "pooled": t2}, "pos"), "SD3")
    expect_raise("空段列表拒绝", as_cond_segments, ([], "pos"), "为空")
    expect_raise("段非三元组拒绝", as_cond_segments, ([(t1, None)], "pos"),
                 "三元组")
    expect_raise("段张量缺失拒绝", as_cond_segments, ([("x", None, 1.0)], "pos"),
                 "张量缺失")
    expect_raise("非标量拒绝", as_cond_segments, (42, "neg"), "张量")

    # --- resolve_latent：裸张量 / Bundle ---
    samples, mask = FakeTensor(), FakeTensor()
    got_s, got_m = resolve_latent(samples)
    assert got_s is samples and got_m is None
    print("ok: resolve_latent 裸张量 -> (samples, None)")
    bundle = LatentBundle(samples, mask)
    got_s, got_m = resolve_latent(bundle)
    assert got_s is samples and got_m is mask
    print("ok: resolve_latent Bundle -> (samples, mask)")

    # --- controlnet_apply：正常 + 校验 ---
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    out = controlnet_apply(FakeControlNet(), img, strength=0.8,
                           start_percent=0.1, end_percent=0.9)
    cb = out["control"]
    assert isinstance(cb, ControlBundle)
    assert cb.strength == 0.8 and cb.start_percent == 0.1 \
        and cb.end_percent == 0.9 and cb.image is img
    print("ok: controlnet_apply 正常 -> ControlBundle")

    out = controlnet_apply(FakeControlNet(), [img])  # 单张列表解包
    assert out["control"].image is img
    print("ok: hint 单张列表解包")

    expect_raise("非 ControlNetModel 拒绝", controlnet_apply, (object(), img),
                 "ControlNetModel")
    expect_raise("batch hint 拒绝", controlnet_apply, (FakeControlNet(),
                 [img, img]), "单张")
    expect_raise("start_percent 越界", controlnet_apply, (FakeControlNet(),
                 img, 1.0, 1.0, 1.0), "[0,1)")
    expect_raise("空窗口拒绝", controlnet_apply, (FakeControlNet(), img,
                 1.0, 0.5, 0.5), "步窗口为空")
    expect_raise("end<=start 拒绝", controlnet_apply, (FakeControlNet(), img,
                 1.0, 0.7, 0.3), "步窗口为空")

    print("\nALL SAMPLE-INJECT TESTS PASSED")


if __name__ == "__main__":
    main()

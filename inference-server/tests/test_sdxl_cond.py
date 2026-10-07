"""M5 SDXL 支持单元测试：COND 形态 / 架构分派编码 / added_cond 构造 / 插件组合。

本机无 torch/diffusers：桩掉两模块（FakeTensor 提供 shape/expand/算术），
被测逻辑均为纯 Python 编排（分派、pooled 传递、time_ids 构造、合并覆盖），
不触真实张量计算。采样循环内的注入正确性（text_embeds/time_ids 进 UNet 前向、
SDXL ControlNet 收 added_cond_kwargs）由真机验收覆盖（抽图核对口径），
SD1.x 路径逐字节不变由回归（accept-m2 sha1 / exec 基线）保证。
运行：python3 -m tests.test_sdxl_cond  或  python3 tests/test_sdxl_cond.py
"""
import os
import sys
import types
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeTensor:
    """最小张量替身：形状 + 广播/算术/拼接 + 字面量记录（够 SDXL COND 编排逻辑用）。"""

    def __init__(self, shape, dtype="float16", device="cuda", values=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.values = values

    def expand(self, *sizes):
        out = list(self.shape)
        for i, s in enumerate(sizes):
            if s < 0:  # -1 = 保持该维
                continue
            out[i] = s
        return FakeTensor(out, self.dtype, self.device)

    def repeat(self, *sizes):
        out = [d * s for d, s in zip(self.shape, sizes)]
        return FakeTensor(out, self.dtype, self.device)

    def to(self, *a, **k):
        return self

    def new_zeros(self, *shape):
        return FakeTensor(shape, self.dtype, self.device)

    def __add__(self, other):
        return FakeTensor(self.shape, self.dtype, self.device)

    __radd__ = __add__

    def __sub__(self, other):
        return FakeTensor(self.shape, self.dtype, self.device)

    def __mul__(self, other):
        return FakeTensor(self.shape, self.dtype, self.device)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return FakeTensor(self.shape, self.dtype, self.device)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"FakeTensor{self.shape}"


def _cat(tensors, dim=0):
    """按 dim 拼接形状（仅用于断言形状，不复制数据）。"""
    shapes = [list(t.shape) for t in tensors]
    out = list(shapes[0])
    out[dim] = sum(s[dim] for s in shapes)
    for s in shapes[1:]:
        for i, v in enumerate(s):
            if i != dim:
                assert v == out[i], f"cat dim {dim} 其余维不一致: {shapes}"
    return FakeTensor(out, tensors[0].dtype, tensors[0].device)


def _tensor(values, dtype="float16", device="cpu"):
    """torch.tensor 替身：一维/二维字面量 → FakeTensor（记录 values 供断言）。"""
    if values and isinstance(values[0], list):
        shape = (len(values), len(values[0]))
    else:
        shape = (1, len(values))
    return FakeTensor(shape, dtype, device, values)


_fake_torch = types.ModuleType("torch")
_fake_torch.is_tensor = lambda x: isinstance(x, FakeTensor)
_fake_torch.tensor = _tensor
_fake_torch.cat = _cat
_fake_torch.no_grad = contextlib.nullcontext
_fake_torch.float16 = "float16"
_fake_torch.float32 = "float32"
sys.modules.setdefault("torch", _fake_torch)

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

from app import ops
from app.ops import (SDXLCond, SDXLSegments, _merge_added, as_cond_segments,
                     build_sdxl_added, check_sdxl_pooled_dim, clip_encode,
                     cond_pooleds, cond_size, sdxl_time_ids)
from app.plugins import load_plugin
from app.registry import Registry

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "..", "nodes",
                      "_common", "server_ops", "cond_latent.py")


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok: {name}")


def expect_raise(name, fn, args, *needle):
    try:
        fn(*args)
    except ValueError as e:
        assert all(n in str(e) for n in needle), f"FAIL: {name}: {e}"
        print(f"ok: {name} raises: {str(e)[:70]}...")
        return
    raise SystemExit(f"FAIL: {name} should raise ValueError")


class FakeUnetCfg:
    addition_time_embed_dim = 8


class FakePipe:
    """SDXL 管道替身：只需 text_encoder_2 / encode_prompt / unet。"""

    def __init__(self, pooled_dim=1280, embeds_dim=2048, ret4=True):
        self.text_encoder_2 = object()
        self._execution_device = "cuda"
        self.ret4 = ret4
        self.calls = []
        unet = types.SimpleNamespace(config=FakeUnetCfg())
        unet.add_embedding = types.SimpleNamespace(
            linear_1=types.SimpleNamespace(in_features=pooled_dim + 8 * 6))
        self.unet = unet

    def encode_prompt(self, **kw):
        self.calls.append(kw)
        e = FakeTensor((1, 77, 2048))
        p = FakeTensor((1, 1280))
        if self.ret4:
            return e, None, p, None
        return e, p


class FakeTok:
    model_max_length = 77

    def __call__(self, text, **kw):
        return types.SimpleNamespace(input_ids=FakeTensor((1, 77)))


class FakeTE:
    def __call__(self, ids):
        return (FakeTensor((1, 77, 768)),)


class FakePipeSD15:
    tokenizer = FakeTok()
    text_encoder = FakeTE()
    _execution_device = "cuda"


def test_cond_shapes():
    emb = FakeTensor((1, 77, 2048))
    pooled = FakeTensor((1, 1280))
    cond = SDXLCond(emb, pooled, 1024, 1024)
    check("SDXLCond -> 单段整图段（embeds 承载）",
          as_cond_segments(cond, "pos") == [(emb, None, 1.0)])
    check("SDXLCond pooled 单段对齐", cond_pooleds(cond) == [pooled])
    check("SDXLCond 尺寸透出", cond_size(cond) == (1024, 1024))

    segs = SDXLSegments([(emb, (0, 0, 64, 64), 1.0)], [pooled], 1024, 1024)
    check("SDXLSegments 走段列表校验（3 元组语义不变）",
          as_cond_segments(segs, "pos") == [(emb, (0, 0, 64, 64), 1.0)])
    check("SDXLSegments pooled 与段对齐", cond_pooleds(segs) == [pooled])
    check("SDXLSegments 尺寸透出", cond_size(segs) == (1024, 1024))

    bad = SDXLSegments([(emb, None, 1.0), (emb, None, 1.0)], [pooled])
    check("段数/pooled 数不一致 -> 视为无 pooled（不静默错配）",
          cond_pooleds(bad) is None)

    check("非 SDXL cond -> cond_pooleds None", cond_pooleds(FakeTensor((1, 77, 768))) is None)
    check("非 SDXL cond -> cond_size (0,0)", cond_size(FakeTensor((1, 77, 768))) == (0, 0))

    check("SD3 dict 仍然明确拒绝",
          _raises(as_cond_segments, {"embeds": emb, "pooled": pooled}, "pos"))


def _raises(fn, *args, needle="SD3"):
    try:
        fn(*args)
    except ValueError as e:
        return needle in str(e)
    return False


def test_time_ids_and_merge():
    tids = sdxl_time_ids(1024, 768, "float16", "cuda")
    check("time_ids 单行六元组 (orig_h,orig_w,crop,crop,target_h,target_w)",
          tids.shape == (1, 6))
    check("time_ids 值 = (h,w,0,0,h,w) 且 dtype/device 随 latent",
          tids.values == [[768, 1024, 0, 0, 768, 1024]]
          and tids.dtype == "float16" and tids.device == "cuda")

    base = {"image_embeds": ["x"]}
    check("_merge_added(None, extra) -> extra", _merge_added(None, base) == base)
    check("_merge_added(base, None) -> base", _merge_added(base, None) == base)
    merged = _merge_added(base, {"text_embeds": "t", "time_ids": "i"})
    check("_merge_added 同名键 extra 覆盖 base",
          merged == {"image_embeds": ["x"], "text_embeds": "t", "time_ids": "i"})
    check("_merge_added(None, None) -> None", _merge_added(None, None) is None)


def test_build_sdxl_added():
    tids = sdxl_time_ids(1024, 1024, "float16", "cuda")
    pa, pb = FakeTensor((1, 1280)), FakeTensor((1, 1280))

    adds, extras = build_sdxl_added("cat", 1, tids, [pa], [pb])
    check("cat：text_embeds 按 CFG 惯序 [neg, pos] 拼批（与 hidden 批序一致）",
          adds["text_embeds"].shape == (2, 1280))
    check("cat：time_ids 复制 2b 行", adds["time_ids"].shape == (2, 6))
    check("cat：无 per-segment extras", extras is None)

    adds2, _ = build_sdxl_added("split", 2, tids, [pa], [pb])
    check("split：(neg_dict, pos_dict) 各 b 行",
          adds2[0]["time_ids"].shape == (2, 6)
          and adds2[1]["text_embeds"].shape == (2, 1280))

    _, extras3 = build_sdxl_added("area", 1, tids, [pa, pa], [pb], 2, 1)
    check("area：(neg 每段, pos 每段) 各带 pooled + 同一 time_ids",
          len(extras3[0]) == 1 and len(extras3[1]) == 2
          and extras3[1][1]["time_ids"].shape == (1, 6))

    expect_raise("SDXL 段缺 pooled -> 报错（不静默零填充）", build_sdxl_added,
                 ("cat", 1, tids, [pa], [None]), "缺 pooled")


def test_pooled_dim_check():
    pipe = FakePipe(pooled_dim=1280)
    check_sdxl_pooled_dim(pipe, FakeTensor((1, 1280)))  # 不抛
    check("pooled 维度校验通过（1280 = 2816-8*6）", True)
    expect_raise("pooled 维度不符报错（给期望值）", check_sdxl_pooled_dim,
                 (pipe, FakeTensor((1, 768))), "1280")


def test_clip_encode_dispatch():
    pipe = FakePipe()
    out = clip_encode(pipe, "a cat", width=1024, height=1024)
    cond = out["cond"]
    check("SDXL: clip.encode -> SDXLCond", isinstance(cond, SDXLCond))
    check("SDXL: embeds/pooled 形态", cond.embeds.shape == (1, 77, 2048)
          and cond.pooled.shape == (1, 1280))
    check("SDXL: 原始尺寸透出（time_ids 用）", (cond.width, cond.height) == (1024, 1024))
    kw = pipe.calls[0]
    check("SDXL: encode_prompt 不用 CFG 双向（正/负各一节点，与 SD1.x 同用法）",
          kw["do_classifier_free_guidance"] is False)
    check("SDXL: prompt_2 未指定 -> 复用 prompt", kw["prompt_2"] is None)
    check("SDXL: clip_skip 0 -> None（diffusers 默认）", kw["clip_skip"] is None)

    pipe2 = FakePipe()
    clip_encode(pipe2, "x", clip_skip=-2)
    check("SDXL: clip_skip=-2 透传", pipe2.calls[0]["clip_skip"] == -2)

    pipe3 = FakePipe(ret4=False)
    check("SDXL: encode_prompt 2 元组返回也兼容",
          isinstance(clip_encode(pipe3, "x")["cond"], SDXLCond))

    sd15 = FakePipeSD15()
    out15 = clip_encode(sd15, "a cat")
    check("SD1.x: clip.encode 仍返回裸张量 COND（行为不变）",
          isinstance(out15["cond"], FakeTensor)
          and out15["cond"].shape == (1, 77, 768))

    out15b = clip_encode(sd15, "a cat", width=1024, height=1024)
    check("SD1.x: 传 width/height 也不改变返回形态（仅 SDXL 生效）",
          isinstance(out15b["cond"], FakeTensor))


def _plugin_ops():
    reg = Registry()
    load_plugin(PLUGIN, reg)
    return reg


def test_plugin_combinations():
    reg = _plugin_ops()
    set_area = reg.get("cond.set_area").fn
    combine = reg.get("cond.combine").fn
    average = reg.get("cond.average").fn

    emb_a = FakeTensor((1, 77, 2048))
    pool_a = FakeTensor((1, 1280))
    a = SDXLCond(emb_a, pool_a, 1024, 1024)

    out = set_area(a, x=0, y=0, width=512, height=512, strength=1.0)["cond"]
    check("插件 cond.set_area(SDXLCond) -> SDXLSegments（保 pooled）",
          isinstance(out, SDXLSegments) and len(out) == 1
          and out[0][0] is emb_a and out[0][1] == (0, 0, 64, 64))
    check("插件 set_area 段带自己的 pooled", cond_pooleds(out) == [pool_a])
    check("插件 set_area 尺寸透传", cond_size(out) == (1024, 1024))

    emb_b = FakeTensor((1, 77, 2048))
    b = SDXLCond(emb_b, FakeTensor((1, 1280)), 0, 0)
    merged = combine(a, b)["cond"]
    check("插件 cond.combine(SDXL, SDXL)：embeds 序列拼接",
          isinstance(merged, SDXLCond) and merged.embeds.shape == (1, 154, 2048))
    check("插件 cond.combine：pooled 取均值（保留 1280 维）",
          merged.pooled.shape == (1, 1280))
    check("插件 cond.combine：尺寸取非 0 一侧", cond_size(merged) == (1024, 1024))

    seg = set_area(a, x=0, y=0, width=512, height=512)["cond"]
    seg2 = combine(b, seg)["cond"]
    check("插件 cond.combine(SDXL, SDXLSegments) -> 段列表（逐段 pooled）",
          isinstance(seg2, SDXLSegments) and len(seg2) == 2
          and len(cond_pooleds(seg2)) == 2)

    avg = average(a, b, 0.25)["cond"]
    check("插件 cond.average(SDXL)：embeds + pooled 双加权平均",
          isinstance(avg, SDXLCond) and avg.pooled.shape == (1, 1280))

    expect_raise("插件拒绝 SDXL 与 SD1.x cond 混用", combine,
                 (a, FakeTensor((1, 77, 768))), "混用")
    expect_raise("插件拒绝分段 cond 的 average", average,
                 (seg, seg, 0.5), "分段")
    expect_raise("插件 set_area 仍拒段列表输入", set_area, (seg, 0, 0, 512, 512, 1.0),
                 "只接受整图")

    t = FakeTensor((1, 77, 768))
    check("SD1.x: cond.set_area 行为不变（裸段列表）",
          isinstance(set_area(t, 0, 0, 512, 512, 1.0)["cond"], list))
    check("SD1.x: cond.combine 行为不变（序列拼接）",
          combine(t, t)["cond"].shape == (1, 154, 768))


def main():
    test_cond_shapes()
    test_time_ids_and_merge()
    test_build_sdxl_added()
    test_pooled_dim_check()
    test_clip_encode_dispatch()
    test_plugin_combinations()
    print("\nALL SDXL COND TESTS PASSED")


if __name__ == "__main__":
    main()

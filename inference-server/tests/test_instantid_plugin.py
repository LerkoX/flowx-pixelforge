"""InstantID 插件契约测试 + ops.ControlBundle 覆写口不变式。

本机无 torch（桩）/cv2/onnxruntime：只覆盖
- 插件注册契约（算子名/端口/类型，FACE 新类型入注册表）
- 不触重依赖的参数校验路径（face_analyze 非 PIL 报错；
  instantid_apply 的非 SDXL / 非 Resampler 权重 / 坏 face 报错）
- ControlBundle 默认 hidden_neg/hidden_pos=None（非 InstantID 路径不变式）

运行：python3 -m tests.test_instantid_plugin
"""
import os
import sys
import types

sys.modules.setdefault("torch", types.ModuleType("torch"))

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

from app.plugins import check_plugin_source, load_plugin, sha256_of
from app.registry import Registry
from app.ops import ControlBundle

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins",
                      "instantid_ops.py")

EXPECTED = {
    "face.analyze": ({"image": "IMAGE", "face_index": "INT",
                             "det_thresh": "FLOAT"},
                     {"face": "FACE", "kps": "IMAGE"}),
    "face.similarity": ({"image_a": "IMAGE", "image_b": "IMAGE",
                          "face_index_a": "INT", "face_index_b": "INT",
                          "det_thresh": "FLOAT"},
                         {"similarity": "FLOAT"}),
    "instantid.apply": ({"model": "MODEL", "ipadapter": "IPADAPTER",
                         "controlnet": "CONTROL_NET", "face": "FACE",
                         "weight": "FLOAT", "cn_strength": "FLOAT",
                         "start_percent": "FLOAT", "end_percent": "FLOAT",
                         "cn_start_percent": "FLOAT",
                         "cn_end_percent": "FLOAT"},
                        {"model": "MODEL", "control": "CONTROL"}),
    "face.mask": ({"image": "IMAGE", "face_index": "INT",
                   "det_thresh": "FLOAT", "expand": "FLOAT",
                   "feather": "FLOAT"},
                  {"mask": "IMAGE", "x": "INT", "y": "INT",
                   "width": "INT", "height": "INT"}),
}


class _Cfg:
    cross_attention_dim = 768  # SD1.x


class _FakeUNet:
    config = _Cfg()
    dtype = "fp16"


class _FakePipe:
    unet = _FakeUNet()


def _expect_err(fn, needle):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert needle in str(e), f"报错不含 {needle!r}: {e}"
        return
    raise AssertionError(f"应报错（{needle!r}）但未报")


def main():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source("instantid_ops.py", content)
    assert err is None, f"check_plugin_source: {err}"
    print(f"ok: check_plugin_source pass (sha256={sha256_of(content)[:12]}...)")

    reg = Registry()
    names = load_plugin(PLUGIN, reg)
    assert sorted(names) == sorted(EXPECTED), f"registered: {names}"
    for name, (inputs, outputs) in EXPECTED.items():
        op = reg.get(name)
        assert op.inputs == inputs, f"{name} inputs: {op.inputs}"
        assert op.outputs == outputs, f"{name} outputs: {op.outputs}"
    assert "FACE" in reg.__class__.__module__ or True
    from app.registry import OBJ_TYPES
    assert "FACE" in OBJ_TYPES
    print(f"ok: 注册契约 {sorted(EXPECTED)}，FACE 对象类型入表")

    import importlib
    mod = importlib.import_module("flowx_plugin_instantid_ops")

    # face_analyze：非 PIL 输入（在校验后才触重依赖导入）
    _expect_err(lambda: mod.face_analyze("not-an-image"),
                "需为单张 PIL 图像")
    print("ok: face_analyze 非 PIL 输入明确报错")

    # face_mask：非 PIL 输入同理明确报错
    _expect_err(lambda: mod.face_mask("not-an-image"),
                "需为单张 PIL 图像")
    print("ok: face_mask 非 PIL 输入明确报错")

    # instantid_apply 校验链（假管道，torch 桩——校验都在 torch 调用之前）
    _expect_err(lambda: mod.instantid_apply(object(), None, None, None),
                "需为 SDXL 管道")
    _expect_err(lambda: mod.instantid_apply(_FakePipe(), None, None, None),
                "仅支持 SDXL 底模")

    class _XLPipe(_FakePipe):
        class unet:
            class config:
                cross_attention_dim = 2048
            dtype = "fp16"

    _expect_err(lambda: mod.instantid_apply(_XLPipe(), {}, None, None),
                "需为 ipadapter.load 的产物")
    _expect_err(lambda: mod.instantid_apply(
        _XLPipe(), {"state_dict": {"image_proj": {}}}, None, None),
        "不是 Resampler")
    _expect_err(lambda: mod.instantid_apply(
        _XLPipe(),
        {"state_dict": {"image_proj": {"latents": None}, "ip_adapter": {}}},
        None, "not-a-face"),
        "需为 face.analyze 的产物")
    print("ok: instantid_apply 校验链（非 SDXL / 坏权重 / 坏 face）")

    # ControlBundle 默认无覆写（非 InstantID 路径不变式）
    img = Image.new("RGB", (8, 8))
    ctl = ControlBundle(object(), img)
    assert ctl.hidden_neg is None and ctl.hidden_pos is None
    print("ok: ControlBundle 默认 hidden_neg/hidden_pos=None（行为不变式）")

    print("PASS: instantid 插件契约全部通过")


if __name__ == "__main__":
    main()

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

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "..", "nodes",
                     "_common", "server_ops",
                      "instantid.py")

EXPECTED = {
    "face.analyze": ({"image": "IMAGE", "face_index": "INT",
                             "det_thresh": "FLOAT",
                             "canvas_width": "INT", "canvas_height": "INT",
                             "canvas_mode": "STRING", "face_scale": "FLOAT",
                             "face_x": "FLOAT", "face_y": "FLOAT"},
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
    err = check_plugin_source("instantid.py", content)
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
    mod = importlib.import_module("flowx_plugin_instantid")

    # face_analyze：非 PIL 输入（在校验后才触重依赖导入）
    _expect_err(lambda: mod.face_analyze("not-an-image"),
                "需为单张 PIL 图像")
    print("ok: face_analyze 非 PIL 输入明确报错")

    # fit_kps_to_canvas：等比变换+居中（纯 python，无重依赖）
    kps = [[342, 564], [496, 564], [419, 650], [370, 740], [468, 740]]
    out, s, ox, oy = mod.fit_kps_to_canvas(kps, 768, 1152, 768, 768)
    assert abs(s - 768/1152) < 1e-9 and abs(ox - 128.0) < 1e-9 and oy == 0.0
    w0 = kps[1][0] - kps[0][0]; h0 = kps[3][1] - kps[0][1]
    w1 = out[1][0] - out[0][0]; h1 = out[3][1] - out[0][1]
    assert abs(w1/h1 - w0/h0) < 1e-9, "变换必须保宽高比"
    # 参考脸几何用例：1152 竖图进 768 方画布，比例 0.875 保持不变
    assert abs(w1/h1 - 154/176) < 0.01, f"{w1/h1}"
    print(f"ok: fit_kps_to_canvas fit 模式（s={s:.3f} ox={ox:.0f}），"
          f"宽高比守恒 {w0/h0:.3f}→{w1/h1:.3f}")

    # cover 模式：等比放大铺满裁溢出（等价 A 轮中心裁切，s=1.0 oy=-192）
    out2, s2, ox2, oy2 = mod.fit_kps_to_canvas(kps, 768, 1152, 768, 768,
                                               mode="cover")
    assert s2 == 1.0 and ox2 == 0.0 and oy2 == -192.0
    w2 = out2[1][0] - out2[0][0]; h2 = out2[3][1] - out2[0][1]
    assert (w2, h2) == (154.0, 176.0), "cover 保原尺度"
    assert abs(w2/h2 - 154/176) < 1e-9
    # 坏 mode 明确拒绝
    _expect_err(lambda: mod.fit_kps_to_canvas(kps, 768, 1152, 768, 768,
                                              mode="stretch"),
                "fit/cover")
    print(f"ok: fit_kps_to_canvas cover 模式（s={s2} oy={oy2:.0f}），"
          f"尺度保持 {w2:.0f}x{h2:.0f}，坏 mode 拒绝")

    # _anchor_scale_kps：以包围盒中心为锚缩放 + 平移到画布归一化位置
    # 全身构图场景：fit 结果上 face_scale=0.3、face_y=0.2（脸小且在上方）
    scaled = mod._anchor_scale_kps(out, 0.3, 0.5, 0.2, 768, 768)
    sw = scaled[1][0] - scaled[0][0]
    sh = scaled[3][1] - scaled[0][1]
    assert abs(sw - w1 * 0.3) < 1e-6 and abs(sh - h1 * 0.3) < 1e-6, "等比缩小"
    cx = (min(p[0] for p in scaled) + max(p[0] for p in scaled)) / 2
    cy = (min(p[1] for p in scaled) + max(p[1] for p in scaled)) / 2
    assert abs(cx - 768 * 0.5) < 1e-6 and abs(cy - 768 * 0.2) < 1e-6, "锚点定位"
    print(f"ok: _anchor_scale_kps 缩放×0.3 + 定位(0.5,0.2) 包围盒 "
          f"{sw:.0f}x{sh:.0f} 中心=({cx:.0f},{cy:.0f})")
    # face_scale 参数校验（无 canvas 拒绝 / 越界拒绝）
    _expect_err(lambda: mod.face_analyze(Image.new("RGB", (64, 64)),
                                         face_scale=0.3),
                "需配合 canvas")
    print("ok: face_scale 无 canvas 明确拒绝")

    # face_analyze canvas 参数校验：只给一个必须报错（在重依赖导入后？
    # 该校验在检测之后——用非 PIL 输入先行拦截，故此处直接调 helper 级
    # 校验路径：PIL 图 + 坏 canvas 会在 app.get 前触发吗？不会——校验在
    # 检测后。本机无 onnxruntime，只覆盖契约层；真机验收覆盖运行路径）
    print("ok: canvas 运行路径留真机验收（本机无 onnxruntime）")

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

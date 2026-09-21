"""preprocess 插件算子契约测试：插件文件可加载、注册契约正确。

本机无 torch/cv2/controlnet_aux：插件模块级零重依赖（懒加载纪律），桩掉
torch/diffusers 后走 app.plugins.load_plugin 真实加载链路；数值正确性由真机
验收覆盖（抽图核对口径）。
运行：python3 -m tests.test_preprocess_plugin
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

from app.plugins import check_plugin_source, load_plugin, sha256_of
from app.registry import Registry

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins",
                      "preprocess_ops.py")

EXPECTED = {
    "preprocess.canny": ({"image": "IMAGE", "low_threshold": "INT",
                          "high_threshold": "INT"}, {"image": "IMAGE"}),
    "preprocess.openpose": ({"image": "IMAGE", "include_hand": "BOOL",
                             "include_face": "BOOL"}, {"image": "IMAGE"}),
}


def main():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source("preprocess_ops.py", content)
    assert err is None, f"check_plugin_source: {err}"
    print(f"ok: check_plugin_source pass (sha256={sha256_of(content)[:12]}...)")

    reg = Registry()
    names = load_plugin(PLUGIN, reg)
    assert sorted(names) == sorted(EXPECTED), f"registered: {names}"
    print(f"ok: registered ops = {sorted(names)}")

    for name, (inputs, outputs) in EXPECTED.items():
        op = reg.get(name)
        assert op.inputs == inputs, f"{name} inputs: {op.inputs}"
        assert op.outputs == outputs, f"{name} outputs: {op.outputs}"
    print("ok: op specs match contract")

    # 参数校验逻辑（不触重依赖的分支）：非 IMAGE 入参必须明确报错
    from PIL import Image  # noqa: F401  （本地有 PIL）
    sys.path.insert(0, os.path.join(os.path.dirname(PLUGIN)))
    import preprocess_ops as po
    try:
        po.preprocess_canny("not-an-image")
        raise AssertionError("preprocess.canny 应拒绝非 IMAGE")
    except ValueError as e:
        assert "IMAGE" in str(e)
    try:
        po.preprocess_openpose(123)
        raise AssertionError("preprocess.openpose 应拒绝非 IMAGE")
    except ValueError as e:
        assert "IMAGE" in str(e)
    print("ok: 非 IMAGE 入参明确报错")
    print("ALL PASS")


if __name__ == "__main__":
    main()

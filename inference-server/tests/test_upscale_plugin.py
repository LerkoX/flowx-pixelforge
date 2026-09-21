"""upscale 插件算子契约测试：插件文件可加载、注册契约正确。

本机无 torch/numpy/spandrel：插件模块级零重依赖（懒加载纪律），桩掉
torch/diffusers 后走 app.plugins.load_plugin 真实加载链路；数值正确性
由真机验收覆盖（抽图核对口径）。
运行：python3 -m tests.test_upscale_plugin
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

from app.plugins import check_plugin_source, load_plugin
from app.registry import Registry

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins",
                      "upscale_ops.py")

EXPECTED = {
    "upscale_model.load": ({"name": "STRING"}, {"upscale_model": "UPSCALE_MODEL"}),
    "image.upscale_with_model": ({"image": "IMAGE",
                                  "upscale_model": "UPSCALE_MODEL",
                                  "tile": "INT"}, {"image": "IMAGE"}),
}


def main():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source("upscale_ops.py", content)
    assert err is None, err

    reg = Registry()
    claimed = load_plugin(PLUGIN, reg)
    assert sorted(claimed) == sorted(EXPECTED), claimed

    for name, (inputs, outputs) in EXPECTED.items():
        op = reg.get(name)
        assert op.inputs == inputs, (name, op.inputs)
        assert op.outputs == outputs, (name, op.outputs)

    # 非 IMAGE 输入拒绝（fn 体内 isinstance 检查在懒加载之前，本地可测）
    fn = reg.get("image.upscale_with_model").fn
    try:
        fn(image="not-a-pil", upscale_model=object(), tile=0)
        raise AssertionError("should reject non-IMAGE")
    except ValueError as e:
        assert "IMAGE" in str(e), e

    # 非句柄 upscale_model 拒绝（真机教训：裸 descriptor 不透传 parameters()）
    from PIL import Image as _PILImage
    try:
        fn(image=_PILImage.new("RGB", (8, 8)), upscale_model=object(), tile=0)
        raise AssertionError("should reject non-handle")
    except ValueError as e:
        assert "upscale_model.load" in str(e), e

    # 模型不存在 → FileNotFoundError（_resolve 先于重依赖 import，本地可测）
    fn_load = reg.get("upscale_model.load").fn
    try:
        fn_load("definitely_not_exist_model_12345")
        raise AssertionError("should raise FileNotFoundError")
    except FileNotFoundError as e:
        assert "not found" in str(e), e

    # 路径穿越拒绝
    try:
        fn_load("../etc/passwd")
        raise AssertionError("should reject path traversal")
    except ValueError as e:
        assert "纯文件名" in str(e), e

    print("PASS test_upscale_plugin")


if __name__ == "__main__":
    main()

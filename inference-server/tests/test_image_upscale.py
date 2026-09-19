"""image.upscale 算子单元测试（纯 PIL：无 GPU、无 torch 依赖，桩掉重依赖后导入 app.ops）。
运行：python3 -m tests.test_image_upscale  或  python3 tests/test_image_upscale.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ops.py 顶部 import torch / .samplers（依赖 diffusers），image_upscale 本身只用 PIL。
# 本地无 GPU 环境桩掉这两个重依赖即可测试纯图像逻辑。
sys.modules.setdefault("torch", types.ModuleType("torch"))
_fake_samplers = types.ModuleType("app.samplers")
_fake_samplers.SAMPLERS = {}
_fake_samplers.DEFAULT_SAMPLER = "euler"
_fake_samplers.make_scheduler = lambda *a, **k: None
sys.modules.setdefault("app.samplers", _fake_samplers)

from PIL import Image

from app.ops import image_upscale


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"ok: {name}")


def main():
    img = Image.new("RGB", (512, 512), (128, 64, 200))

    # 默认 2x 放大
    out = image_upscale(img)
    check("default scale=2 -> 1024x1024", out["image"].size == (1024, 1024))

    # 显式倍率
    out = image_upscale(img, scale=1.5)
    check("scale=1.5 -> 768x768", out["image"].size == (768, 768))

    # 目标尺寸优先于倍率
    out = image_upscale(img, scale=3.0, width=640, height=480)
    check("width/height override scale", out["image"].size == (640, 480))

    # 非 8 倍数向下对齐（VAE 编码要求）
    out = image_upscale(img, width=513, height=515)
    check("align down to multiple of 8", out["image"].size == (512, 512))

    # method 校验
    out = image_upscale(img, scale=2.0, method="bicubic")
    check("method=bicubic", out["image"].size == (1024, 1024))
    try:
        image_upscale(img, method="magic")
        raise SystemExit("FAIL: unknown method should raise")
    except ValueError as e:
        check("unknown method raises ValueError", "magic" in str(e))

    # 非法 scale
    try:
        image_upscale(img, scale=0)
        raise SystemExit("FAIL: scale=0 should raise")
    except ValueError:
        check("scale<=0 raises ValueError", True)

    # batch 列表逐张处理
    out = image_upscale([img, img], scale=2.0)
    check("batch list -> list of 2 @1024",
          isinstance(out["image"], list) and len(out["image"]) == 2
          and all(i.size == (1024, 1024) for i in out["image"]))

    # 原图不被修改（resize 返回新对象）
    check("input image unchanged", img.size == (512, 512))

    print("all image_upscale tests passed")


if __name__ == "__main__":
    main()

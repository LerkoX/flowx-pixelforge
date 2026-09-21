"""ipadapter 插件算子契约测试：注册契约 + 参数校验 + 目录清单分类。

本机无 torch/transformers：插件模块级仅 import torch（桩得住）；
transformers/safetensors 全部函数内懒加载，校验路径测试不触碰。
运行：python3 -m tests.test_ipadapter_plugin
"""
import os
import sys
import tempfile
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
from app.sniff import list_model_files

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins",
                      "ipadapter_ops.py")

EXPECTED = {
    "clip_vision.load": ({"name": "STRING"}, {"clip_vision": "CLIP_VISION"}),
    "ipadapter.load": ({"name": "STRING"}, {"ipadapter": "IPADAPTER"}),
    "ipadapter.apply": ({"model": "MODEL", "ipadapter": "IPADAPTER",
                         "clip_vision": "CLIP_VISION", "image": "IMAGE",
                         "weight": "FLOAT", "start_percent": "FLOAT",
                         "end_percent": "FLOAT"}, {"model": "MODEL"}),
}


class _StubPipe:
    """最小管道桩：只有 unet 属性（apply 的准入检查）。"""
    unet = object()


def main():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source("ipadapter_ops.py", content)
    assert err is None, f"check_plugin_source: {err}"
    print(f"ok: check_plugin_source pass (sha256={sha256_of(content)[:12]}...)")

    reg = Registry()
    names = load_plugin(PLUGIN, reg)
    assert sorted(names) == sorted(EXPECTED), f"registered: {names}"
    for name, (inputs, outputs) in EXPECTED.items():
        op = reg.get(name)
        assert op.inputs == inputs, f"{name} inputs: {op.inputs}"
        assert op.outputs == outputs, f"{name} outputs: {op.outputs}"
        assert op.description, f"{name} missing description"
        print(f"ok: {name} spec {list(inputs)} -> {list(outputs)}")

    apply_fn = reg.get("ipadapter.apply").fn
    load_fn = reg.get("ipadapter.load").fn
    img = Image.new("RGB", (8, 8))

    # apply 校验路径（编码前拦截，无需真实编码器/张量）
    try:
        apply_fn(model=object(), ipadapter={"state_dict": {}},
                 clip_vision=None, image=img)
        raise SystemExit("FAIL: 无 unet 的 model 应被拒")
    except ValueError as e:
        assert "unet" in str(e), e
        print(f"ok: apply 拒绝非 SD 管道: {str(e)[:36]}...")

    try:
        apply_fn(model=_StubPipe(), ipadapter={"state_dict": {}},
                 clip_vision=None, image="not-an-image")
        raise SystemExit("FAIL: 非 PIL image 应被拒")
    except ValueError as e:
        assert "PIL" in str(e), e
        print(f"ok: apply 拒绝非 PIL 图像: {str(e)[:36]}...")

    try:
        apply_fn(model=_StubPipe(), ipadapter=None, clip_vision=None, image=img)
        raise SystemExit("FAIL: 非法 ipadapter 应被拒")
    except ValueError as e:
        assert "ipadapter.load" in str(e), e
        print(f"ok: apply 拒绝非法 ipadapter: {str(e)[:36]}...")

    try:
        apply_fn(model=_StubPipe(), ipadapter={"state_dict": {}},
                 clip_vision=None, image=img,
                 start_percent=0.8, end_percent=0.2)
        raise SystemExit("FAIL: start>=end 应被拒")
    except ValueError as e:
        assert "start" in str(e), e
        print(f"ok: apply 拒绝非法步窗口: {str(e)[:36]}...")

    # ipadapter.load 校验路径（env 覆盖到临时目录）
    with tempfile.TemporaryDirectory() as d:
        os.environ["IPADAPTER_DIR"] = d
        try:
            load_fn("no-such-adapter")
            raise SystemExit("FAIL: 缺失文件应报 FileNotFoundError")
        except FileNotFoundError as e:
            assert "no-such-adapter" in str(e), e
            print(f"ok: load 缺失文件报错: {str(e)[:40]}...")
        try:
            load_fn("../escape")
            raise SystemExit("FAIL: 路径穿越应被拒")
        except ValueError as e:
            print(f"ok: load 拒绝路径穿越: {str(e)[:40]}...")
        # 顶层键校验：monkeypatch _load_state_dict 避开真实权重解析
        import importlib
        mod = importlib.import_module("flowx_plugin_ipadapter_ops")
        open(os.path.join(d, "bad.safetensors"), "wb").write(b"x")
        orig = mod._load_state_dict
        mod._load_state_dict = lambda p: {"foo": {}}
        try:
            load_fn("bad")
            raise SystemExit("FAIL: 缺顶层键应被拒")
        except ValueError as e:
            assert "image_proj" in str(e), e
            print(f"ok: load 校验顶层键: {str(e)[:40]}...")
        mod._load_state_dict = lambda p: {"image_proj": {"latents": 1},
                                          "ip_adapter": {}}
        got = load_fn("bad")
        assert got["ipadapter"]["plus"] is True
        mod._load_state_dict = lambda p: {"image_proj": {"weight": 1},
                                          "ip_adapter": {}}
        got = load_fn("bad")
        assert got["ipadapter"]["plus"] is False
        mod._load_state_dict = orig
        print("ok: plus/standard 按 image_proj.latents 键判别")
        del os.environ["IPADAPTER_DIR"]

    # clip_vision.load 校验路径
    cv_fn = reg.get("clip_vision.load").fn
    with tempfile.TemporaryDirectory() as d:
        os.environ["CLIP_VISION_DIR"] = d
        try:
            cv_fn("no-such-encoder")
            raise SystemExit("FAIL: 缺失目录应报 FileNotFoundError")
        except FileNotFoundError as e:
            print(f"ok: clip_vision.load 缺失目录报错: {str(e)[:40]}...")
        try:
            cv_fn("a/b")
            raise SystemExit("FAIL: 路径穿越应被拒")
        except ValueError as e:
            print(f"ok: clip_vision.load 拒绝路径穿越: {str(e)[:36]}...")
        del os.environ["CLIP_VISION_DIR"]

    # 清单分类：clip_vision/ipadapter 子目录 → 新 kind
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "clip_vision", "CLIP-ViT-H-14"))
        os.makedirs(os.path.join(d, "ipadapter"))
        open(os.path.join(d, "ipadapter", "ip-adapter_sd15.bin"),
             "wb").write(b"x")
        open(os.path.join(d, "ipadapter", "readme.txt"), "w").write("x")
        kinds = {(f["name"], f["kind"]) for f in list_model_files(d)}
        assert ("CLIP-ViT-H-14", "clip_vision") in kinds, kinds
        assert ("ip-adapter_sd15", "ipadapter") in kinds, kinds
        assert not any(k == "ipadapter" and n == "readme" for n, k in kinds)
        # 子目录本身不得作为顶层条目出现
        assert not any(n in ("clip_vision", "ipadapter") for n, _ in kinds)
        print("ok: list_model_files clip_vision/ipadapter 分类 + 顶层跳过")

    print("all ipadapter plugin tests passed")


if __name__ == "__main__":
    main()

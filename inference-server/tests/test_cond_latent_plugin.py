"""cond/latent 插件算子契约测试：插件文件可加载、注册契约正确。

本机无 torch：桩掉 torch 后走 app.plugins.load_plugin 真实加载链路
（数值正确性由真机验收覆盖——抽图核对口径）。
运行：python3 -m tests.test_cond_latent_plugin
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
                      "cond_latent_ops.py")

EXPECTED = {
    "cond.combine": ({"cond_a": "COND", "cond_b": "COND"}, {"cond": "COND"}),
    "cond.average": ({"cond_a": "COND", "cond_b": "COND", "weight": "FLOAT"},
                     {"cond": "COND"}),
    "latent.upscale": ({"latent": "LATENT", "scale": "FLOAT", "width": "INT",
                        "height": "INT", "method": "STRING"},
                       {"latent": "LATENT"}),
    "latent.composite": ({"dst": "LATENT", "src": "LATENT", "x": "INT",
                          "y": "INT", "feather": "INT"}, {"latent": "LATENT"}),
}


def main():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    # 上传前置校验契约（与 /admin/plugins 同源检查）
    err = check_plugin_source("cond_latent_ops.py", content)
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
        assert op.description, f"{name} missing description"
        print(f"ok: {name} spec {list(inputs)} -> {list(outputs)}")

    # SD3 dict COND 拒绝路径（无需真 torch 张量：dict 在 is_tensor 前拦截）
    try:
        reg.get("cond.combine").fn({"embeds": 1, "pooled": 2}, {"embeds": 3})
        raise SystemExit("FAIL: cond.combine should reject dict COND")
    except ValueError as e:
        assert "SD3" in str(e), e
        print(f"ok: cond.combine 拒绝 SD3 dict COND: {str(e)[:40]}...")

    print("all cond/latent plugin tests passed")


if __name__ == "__main__":
    main()

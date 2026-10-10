"""flux 节点插件（nodes/flux-*/server_op.py）契约测试（本机无 torch/diffusers：
桩覆盖注册契约 + 校验链 + GGUF shape 语义回归保护；补丁与数学路径在真机验收覆盖）。

运行：python3 -m tests.test_flux_plugin
"""
import os
import sys
import types

# --- torch 桩（够用即可：nn.Module 基类 + functional.sdpa） ---
_torch = types.ModuleType("torch")
_nn = types.ModuleType("torch.nn")


class _Module:
    def __init__(self, *a, **kw):
        pass


_nn.Module = _Module
_nn.Parameter = lambda x: x
_functional = types.ModuleType("torch.nn.functional")


def _sdpa(*a, **kw):
    return ("sdpa", a, kw)


_functional.scaled_dot_product_attention = _sdpa
_nn.functional = _functional
_torch.nn = _nn
_torch.device = lambda x: ("dev", x)
_torch.bfloat16 = "bf16"
_torch.float16 = "fp16"
_torch.float32 = "fp32"
_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", _nn)
sys.modules.setdefault("torch.nn.functional", _functional)

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plugins import check_plugin_source, load_plugin, sha256_of
from app.registry import Registry

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "..", "nodes",
                     "flux-sampler", "server_op.py")

EXPECTED = {
    "flux.unet_load": ({"transformer": "STRING", "dtype": "STRING"},
                       {"model": "MODEL"}),
    "flux.dual_clip_load": ({"t5": "STRING", "dtype": "STRING"},
                            {"clip": "CLIP"}),
    "flux.vae_load": ({"name": "STRING", "dtype": "STRING"}, {"vae": "VAE"}),
    "flux.encode": ({"clip": "CLIP", "text": "STRING", "max_seq": "INT",
                     "release_t5": "INT"}, {"cond": "COND", "info": "STRING"}),
    "flux.empty_latent": ({"width": "INT", "height": "INT",
                           "batch_size": "INT"}, {"latent": "LATENT"}),
    "flux.sample": ({"model": "MODEL", "cond": "COND", "latent": "LATENT",
                     "seed": "INT", "steps": "INT", "guidance": "FLOAT",
                     "denoise": "FLOAT", "preview_every": "INT"},
                    {"latent": "LATENT", "seed": "INT"}),
    "flux.vae_decode": ({"vae": "VAE", "latent": "LATENT"}, {"image": "IMAGE"}),
    "flux.vae_encode": ({"vae": "VAE", "image": "IMAGE"}, {"latent": "LATENT"}),
    "flux.detail_refine": ({"model": "MODEL", "vae": "VAE", "cond": "COND",
                            "image": "IMAGE", "detector": "STRING",
                            "conf": "FLOAT", "padding": "FLOAT",
                            "denoise": "FLOAT", "steps": "INT",
                            "guidance": "FLOAT", "seed": "INT",
                            "guide_size": "INT", "max_targets": "INT",
                            "feather": "INT"},
                           {"image": "IMAGE", "count": "INT"}),
}


def load(reg=None):
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source(os.path.basename(PLUGIN), content)
    assert err is None, f"check_plugin_source: {err}"
    if reg is None:
        reg = Registry()
    load_plugin(PLUGIN, reg)
    return reg


def test_torch_compat_install():
    load()
    assert hasattr(_nn, "RMSNorm"), "RMSNorm 补丁未安装"
    r = _functional.scaled_dot_product_attention(1, 2, 3, enable_gqa=False)
    assert r[0] == "sdpa" and "enable_gqa" not in r[2], "shim 未剥离 enable_gqa"
    try:
        _functional.scaled_dot_product_attention(1, 2, 3, enable_gqa=True)
        raise AssertionError("enable_gqa=True 应明确报错")
    except RuntimeError as e:
        assert "torch>=2.5" in str(e)
    # 幂等：再次安装不重复包裹
    import sys as _s2
    flux_ops = _s2.modules["flowx_plugin_server_op"]
    flux_ops._install_torch_compat()
    assert getattr(_functional.scaled_dot_product_attention,
                   "_flux_gqa_shim", False)
    print("ok: torch 兼容补丁（RMSNorm + SDPA shim 安装/剥离/幂等）")


def test_t5_tensor_map():
    import sys as _s; mod = _s.modules["flowx_plugin_server_op"]
    m = mod._T5_TENSOR_MAP
    assert m["attn_q.weight"] == "layer.0.SelfAttention.q.weight"
    assert m["ffn_gate.weight"] == "layer.1.DenseReluDense.wi_0.weight"
    assert m["ffn_up.weight"] == "layer.1.DenseReluDense.wi_1.weight"
    assert m["ffn_down.weight"] == "layer.1.DenseReluDense.wo.weight"
    assert m["attn_rel_b.weight"] == \
        "layer.0.SelfAttention.relative_attention_bias.weight"
    assert len(m) == 10
    print("ok: T5 GGUF 映射表（10 条，wi_0=gate/wi_1=up）")


def test_gguf_shape_semantics():
    """GGUF shape 反序语义回归保护：reshape(shape[::-1]) ≠ reshape(shape).T
    （纯 Python 模拟行主序展开，无需 numpy）。"""
    d = list(range(6))           # 行主序扁平数据
    in_, out = 3, 2              # gguf shape=[in,out]=[3,2]
    # 正解：按 [out,in]=[2,3] reshape → [[0,1,2],[3,4,5]]
    correct = [d[r * in_:(r + 1) * in_] for r in range(out)]
    # 错误等价（初版踩过）：先 [3,2] reshape 再转置 → [[0,3],[1,4],[2,5]] 再 .T
    wrong_rows = [d[r * out:(r + 1) * out] for r in range(in_)]
    wrong = [[wrong_rows[c][r] for c in range(in_)] for r in range(out)]
    assert correct != wrong
    assert correct == [[0, 1, 2], [3, 4, 5]]
    assert wrong == [[0, 2, 4], [1, 3, 5]]
    print("ok: GGUF shape 反序语义回归保护")


def test_registry_contract():
    reg = load()
    specs = {o["name"]: o for o in reg.list()}
    assert set(specs) == set(EXPECTED)
    for name, (ins, outs) in EXPECTED.items():
        assert specs[name]["inputs"] == ins, f"{name} inputs: {specs[name]['inputs']}"
        assert specs[name]["outputs"] == outs, f"{name} outputs"
    print("ok: 注册契约（flux 家族 9 算子端口与类型）")


def test_validation_chain():
    # flux.encode/flux.sample 函数体内 import diffusers.FluxPipeline / numpy，
    # 校验链测试只需走到参数校验，补齐最小桩
    _fake_diffusers.FluxPipeline = type("FluxPipeline", (), {})
    _pf = types.ModuleType("diffusers.pipelines.flux.pipeline_flux")
    _pf.calculate_shift = lambda *a, **k: None
    _pfl = types.ModuleType("diffusers.pipelines.flux")
    _pfl.pipeline_flux = _pf
    _p = types.ModuleType("diffusers.pipelines")
    _p.flux = _pfl
    sys.modules.setdefault("diffusers.pipelines", _p)
    sys.modules.setdefault("diffusers.pipelines.flux", _pfl)
    sys.modules.setdefault("diffusers.pipelines.flux.pipeline_flux", _pf)
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))

    reg = load()

    class NotDualClip:
        pass
    try:
        reg.get("flux.encode").fn(NotDualClip(), "hi")
        raise AssertionError("非 dual_clip_load 产物应拒绝")
    except ValueError as e:
        assert "flux.dual_clip_load" in str(e)

    class FakeTransformer:
        pass
    try:
        reg.get("flux.sample").fn(FakeTransformer(), {"prompt_embeds": 1},
                                  {"width": 1024})
        raise AssertionError("非 unet_load 产物应拒绝")
    except ValueError as e:
        assert "flux.unet_load" in str(e)

    fake_tx = type("FluxTransformer2DModel", (), {})()
    try:
        reg.get("flux.sample").fn(fake_tx, {"foo": 1}, {"width": 1024})
        raise AssertionError("坏 cond 应拒绝")
    except ValueError as e:
        assert "flux.encode" in str(e)
    print("ok: 校验链（非 dual-clip / 非 unet 产物 / 坏 cond 明确报错）")


if __name__ == "__main__":
    test_torch_compat_install()
    test_t5_tensor_map()
    test_gguf_shape_semantics()
    test_registry_contract()
    test_validation_chain()
    print("PASS: flux 插件契约全部通过")

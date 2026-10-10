"""加载旋钮全量化（dev-plan §43.1）测试。

覆盖：
- model_manager.load_motion：dtype/offload 旋钮的缓存键语义（auto 继承进程级
  默认 = 历史键名逐字节不变；显式指定 → 键扩展，同模型不同旋钮独立常驻）
- ops.motion_load 透传
- ipadapter 插件 clip_vision.load：dtype 旋钮 + (name, dtype) 缓存键
- flux 插件三个 loader：compute dtype 旋钮 + 缓存键扩展 + 默认值解析

无需 GPU/真 torch/diffusers：stub 注入后直接测（模式同 test_vram_governance /
test_flux_plugin）。

运行：python3 tests/test_load_knobs.py
"""
import os
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_vram_governance import install_stubs  # noqa: E402  (torch stub 先行)

torch, STATE = install_stubs()

import diffusers  # noqa: E402  (install_stubs 放的 stub)

import app.model_manager as mm  # noqa: E402
from app.model_manager import ModelManager  # noqa: E402
from app import ops as core_ops  # noqa: E402
from app.plugins import load_plugin  # noqa: E402
from app.registry import Registry  # noqa: E402

NODES = os.path.join(os.path.dirname(__file__), "..", "..", "nodes")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeVAE:
    def __init__(self):
        self.dtype = None

    def to(self, dtype):
        self.dtype = dtype
        return self

    def decode(self, z, *a, **k):
        return ("decoded", z)


class _FakeScheduler:
    def __init__(self):
        self.config = {"num_train_timesteps": 1000}


class _FakeBasePipe:
    """_instantiate 产物替身（CPU 实例，组件移交组合管）。"""

    def __init__(self, dtype):
        self.dtype = dtype
        self.vae = _FakeVAE()
        self.text_encoder = object()
        self.tokenizer = object()
        self.unet = object()
        self.scheduler = _FakeScheduler()
        self.feature_extractor = None


class _FakeAnimateDiffPipeline:
    instances = []

    def __init__(self, vae=None, text_encoder=None, tokenizer=None, unet=None,
                 motion_adapter=None, scheduler=None, feature_extractor=None,
                 image_encoder=None):
        self.vae = vae
        self.scheduler = scheduler
        self.motion_adapter = motion_adapter
        self.set_pb = None
        _FakeAnimateDiffPipeline.instances.append(self)

    def set_progress_bar_config(self, disable=True):
        self.set_pb = disable


class _FakeMotionAdapter:
    calls = []

    @classmethod
    def from_pretrained(cls, path, variant=None, torch_dtype=None):
        cls.calls.append(("pretrained", path, variant, torch_dtype))
        return cls()

    @classmethod
    def from_single_file(cls, path, torch_dtype=None):
        cls.calls.append(("single_file", path, torch_dtype))
        return cls()


class _FakeDDIM:
    @classmethod
    def from_config(cls, cfg):
        o = cls()
        o.config = dict(cfg)
        return o


def _install_diffusers_fakes():
    diffusers.AnimateDiffPipeline = _FakeAnimateDiffPipeline
    diffusers.MotionAdapter = _FakeMotionAdapter
    diffusers.DDIMScheduler = _FakeDDIM


def _motion_mgr():
    """构造可跑 load_motion 的管理器：重活全部替换为 fake。"""
    mgr = ModelManager()
    mgr.resolve = lambda name: ("/fake/" + name, name)
    mgr.resolve_motion = lambda m: "/fake/motion/" + m + ".ckpt"
    mgr.estimate = lambda *a, **k: 100
    mgr._evict_if_needed = lambda *a, **k: None
    mgr._instantiate = lambda path, loader, cls_name, dtype=torch.float16, \
        use_t5=False: _FakeBasePipe(dtype)
    applied = []

    def _apply(pipe, mode=None):
        applied.append(mode)
        return pipe

    mgr._apply_offload = _apply
    mgr._applied = applied
    old_sniff = mm.sniff.sniff_arch
    mm.sniff.sniff_arch = lambda p: ("single_file", "StableDiffusionPipeline")
    return mgr, old_sniff


# ---------------------------------------------------------------------------
# 1. load_motion 缓存键语义
# ---------------------------------------------------------------------------

def test_motion_key_default_unchanged():
    """auto/auto（历史调用形态）→ 缓存键 '{base}+motion:{motion}' 逐字节不变，
    幂等命中；offload 用进程级 OFFLOAD_MODE，dtype 用 fp16。"""
    _install_diffusers_fakes()
    mgr, old_sniff = _motion_mgr()
    try:
        key1, new1 = mgr.load_motion("base", "mm_sd_v15_v2")
        assert key1 == "base+motion:mm_sd_v15_v2", key1
        assert new1 is True
        key2, new2 = mgr.load_motion("base", "mm_sd_v15_v2")
        assert key2 == key1 and new2 is False, (key2, new2)
        assert mgr._applied[-1] == mm.OFFLOAD_MODE, mgr._applied
        # 底模与运动模块都按 fp16（auto 默认）
        call = _FakeMotionAdapter.calls[-1]
        assert call[0] == "single_file" and call[-1] == torch.float16, call
    finally:
        mm.sniff.sniff_arch = old_sniff
    print("[ok] motion.load 默认键名不变 + auto 继承进程级默认")


def test_motion_key_knob_extends():
    """显式 dtype/offload → 键扩展 '|dtype=..|offload=..'；同模型不同旋钮
    独立常驻条目（同一 LRU）。"""
    _install_diffusers_fakes()
    mgr, old_sniff = _motion_mgr()
    try:
        k_auto, _ = mgr.load_motion("base", "mm")
        k_bf16, new = mgr.load_motion("base", "mm", dtype="bf16")
        exp = f"base+motion:mm|dtype=bf16|offload={mm.OFFLOAD_MODE}"
        assert k_bf16 == exp and new is True, k_bf16
        k_seq, new = mgr.load_motion("base", "mm", offload="none")
        exp2 = f"base+motion:mm|dtype=fp16|offload=none"
        assert k_seq == exp2 and new is True, k_seq
        # 三个条目同模型不同旋钮 → 独立常驻
        assert len(mgr._pipes) == 3, mgr.resident()
        # 幂等：同旋钮组合命中缓存
        again, new = mgr.load_motion("base", "mm", dtype="bf16")
        assert again == k_bf16 and new is False
        # offload 旋钮真正传到 _apply_offload
        assert mgr._applied[-1] is None or True  # fake 记录见下
        assert "none" in mgr._applied, mgr._applied
        # 显式 dtype 传到运动模块加载
        kinds = [c for c in _FakeMotionAdapter.calls if c[-1] == "bf16"]
        assert kinds, _FakeMotionAdapter.calls
    finally:
        mm.sniff.sniff_arch = old_sniff
    print("[ok] motion.load 显式旋钮 → 缓存键扩展 + 独立常驻 + 幂等")


def test_motion_knob_validation():
    mgr, old_sniff = _motion_mgr()
    try:
        for bad in ("int8", "fp64"):
            try:
                mgr.load_motion("base", "mm", dtype=bad)
                raise AssertionError(f"dtype={bad} 应报错")
            except ValueError as e:
                assert "unknown dtype" in str(e)
        try:
            mgr.load_motion("base", "mm", offload="layer")
            raise AssertionError("offload=layer 应报错")
        except ValueError as e:
            assert "unknown offload mode" in str(e)
    finally:
        mm.sniff.sniff_arch = old_sniff
    print("[ok] motion.load 非法旋钮值明确报错")


def test_ops_motion_load_passthrough():
    """ops.motion_load 把 dtype/offload 透传进 models.load_motion。"""
    calls = []

    class _Models:
        def load_motion(self, ckpt, motion, dtype="auto", offload="auto"):
            calls.append((ckpt, motion, dtype, offload))
            return "K", True

        def get(self, key):
            return f"pipe:{key}"

    out = core_ops.motion_load(_Models(), "b", "m",
                               dtype="bf16", offload="none")
    assert calls == [("b", "m", "bf16", "none")], calls
    assert out == {"model": "pipe:K", "clip": "pipe:K", "vae": "pipe:K"}
    print("[ok] ops.motion_load 旋钮透传")


# ---------------------------------------------------------------------------
# 2. ipadapter 插件 clip_vision.load dtype 旋钮
# ---------------------------------------------------------------------------

def _load_ipadapter_plugin():
    reg = Registry()
    plugin = os.path.join(NODES, "_common", "server_ops", "ipadapter.py")
    load_plugin(plugin, reg)
    return reg, sys.modules["flowx_plugin_ipadapter"]


def test_clip_vision_dtype_knob():
    reg, mod = _load_ipadapter_plugin()
    # 默认值解析
    assert mod._resolve_vision_dtype(None) == "fp16"
    assert mod._resolve_vision_dtype("") == "fp16"
    assert mod._resolve_vision_dtype("auto") == "fp16"
    assert mod._resolve_vision_dtype("fp32") == "fp32"
    try:
        mod._resolve_vision_dtype("int8")
        raise AssertionError("int8 应报错")
    except ValueError as e:
        assert "unknown dtype" in str(e)

    # 缓存键 (name, dtype)：stub transformers
    calls = []

    class _Enc:
        @classmethod
        def from_pretrained(cls, path, torch_dtype=None):
            calls.append(("enc", path, torch_dtype))
            o = cls()
            o.dtype = torch_dtype
            return o

        def eval(self):
            return self

    class _Proc:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

    tr = types.ModuleType("transformers")
    tr.CLIPVisionModelWithProjection = _Enc
    tr.CLIPImageProcessor = _Proc
    old_tr = sys.modules.get("transformers")
    sys.modules["transformers"] = tr

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "enc"))
    old_dir = os.environ.get("CLIP_VISION_DIR")
    os.environ["CLIP_VISION_DIR"] = tmp
    try:
        fn = reg.get("clip_vision.load").fn
        mod._VISION_CACHE.clear()
        r1 = fn("enc")                       # auto → fp16
        assert calls[-1] == ("enc", os.path.join(tmp, "enc"), torch.float16)
        n = len(calls)
        fn("enc")                            # 命中缓存
        assert len(calls) == n, "auto 二次调用应命中 (name, fp16) 缓存"
        r3 = fn("enc", dtype="fp32")         # 显式 → 独立缓存条目
        assert calls[-1][-1] == torch.float32
        assert set(mod._VISION_CACHE) == {("enc", "fp16"), ("enc", "fp32")}, \
            list(mod._VISION_CACHE)
        assert r1["clip_vision"] is not r3["clip_vision"]
    finally:
        if old_dir is None:
            os.environ.pop("CLIP_VISION_DIR", None)
        else:
            os.environ["CLIP_VISION_DIR"] = old_dir
        if old_tr is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = old_tr
    print("[ok] clip_vision.load dtype 旋钮 + (name,dtype) 缓存键 + 幂等")


# ---------------------------------------------------------------------------
# 3. flux 插件三个 loader 的 compute dtype 旋钮
# ---------------------------------------------------------------------------

def _load_flux_plugin(flux_dir):
    os.environ["FLUX_DIR"] = flux_dir
    # flux 插件导入期安装 torch 兼容补丁，需要 nn/functional 最小面
    if not hasattr(torch, "nn"):
        nn = types.ModuleType("torch.nn")
        nn.Module = type("Module", (), {"__init__": lambda self, *a, **k: None})
        nn.Parameter = lambda x: x
        fn = types.ModuleType("torch.nn.functional")
        fn.scaled_dot_product_attention = lambda *a, **k: None
        nn.functional = fn
        torch.nn = nn
        sys.modules["torch.nn"] = nn
        sys.modules["torch.nn.functional"] = fn
    reg = Registry()
    plugin = os.path.join(NODES, "_common", "server_ops", "flux.py")
    load_plugin(plugin, reg)
    return reg, sys.modules["flowx_plugin_flux"]


class _FakeModels:
    """flux 插件 _register 所需的最小管理器面。"""

    def __init__(self):
        self._pipes = {}
        self._archs = {}
        self._sizes = {}
        self._last_used = {}
        self._lock = threading.Lock()

    def _evict_if_needed(self, *a, **k):
        pass

    def get(self, key):
        self._last_used[key] = time.time()
        return self._pipes[key]


def _fake_app_main():
    mod = types.ModuleType("app.main")
    mod.models = _FakeModels()
    sys.modules["app.main"] = mod
    return mod


def test_flux_dtype_helpers():
    tmp = tempfile.mkdtemp()
    reg, mod = _load_flux_plugin(tmp)
    r = mod._resolve_compute_dtype
    assert r(None, "fp16") == "fp16" and r("", "bf16") == "bf16"
    assert r("auto", "bf16") == "bf16" and r("fp32", "fp16") == "fp32"
    try:
        r("int8", "fp16")
        raise AssertionError("int8 应报错")
    except ValueError as e:
        assert "unknown dtype" in str(e)
    s = mod._dtype_suffix
    assert s(None) == "" and s("") == "" and s("auto") == ""
    assert s("bf16") == "|dtype=bf16"
    os.environ.pop("FLUX_DIR", None)
    print("[ok] flux dtype 解析器/键后缀（auto→默认，显式→扩展）")


def test_flux_unet_load_dtype_knob():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "f.gguf"), "wb") as f:
        f.write(b"\0" * 16)
    os.makedirs(os.path.join(tmp, "transformer"))
    with open(os.path.join(tmp, "transformer", "config.json"), "w") as f:
        f.write("{}")

    quant_calls, tr_calls = [], []

    class _Quant:
        def __init__(self, compute_dtype=None):
            quant_calls.append(compute_dtype)
            self.compute_dtype = compute_dtype

    class _TR:
        @classmethod
        def from_single_file(cls, path, config=None,
                             quantization_config=None, torch_dtype=None):
            tr_calls.append((path, torch_dtype,
                             quantization_config.compute_dtype))
            o = cls()
            o.dtype = torch_dtype
            return o

    old_q = getattr(diffusers, "GGUFQuantizationConfig", None)
    old_t = getattr(diffusers, "FluxTransformer2DModel", None)
    diffusers.GGUFQuantizationConfig = _Quant
    diffusers.FluxTransformer2DModel = _TR

    reg, mod = _load_flux_plugin(tmp)
    fake_main = _fake_app_main()
    try:
        fn = reg.get("flux.unet_load").fn
        fn("f.gguf")                       # auto → 历史键名 + fp16
        assert "flux-unet:f.gguf" in fake_main.models._pipes, \
            list(fake_main.models._pipes)
        assert quant_calls[-1] == torch.float16
        assert tr_calls[-1][1] == torch.float16
        fn("f.gguf")                       # 幂等命中
        assert len(tr_calls) == 1
        fn("f.gguf", dtype="bf16")         # 显式 → 键扩展 + bf16 compute
        assert "flux-unet:f.gguf|dtype=bf16" in fake_main.models._pipes
        assert quant_calls[-1] == "bf16"
        assert len(fake_main.models._pipes) == 2
    finally:
        if old_q is None:
            del diffusers.GGUFQuantizationConfig
        else:
            diffusers.GGUFQuantizationConfig = old_q
        if old_t is None:
            del diffusers.FluxTransformer2DModel
        else:
            diffusers.FluxTransformer2DModel = old_t
        sys.modules.pop("app.main", None)
        os.environ.pop("FLUX_DIR", None)
    print("[ok] flux.unet_load dtype 旋钮：默认键名不变，显式独立常驻")


def test_flux_dual_clip_and_vae_dtype_knob():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "t5.gguf"), "wb") as f:
        f.write(b"\0" * 8)
    for d in ("text_encoder_2", "text_encoder", "tokenizer",
              "tokenizer_2", "vae"):
        os.makedirs(os.path.join(tmp, d))
    with open(os.path.join(tmp, "text_encoder_2", "config.json"), "w") as f:
        f.write("{}")

    clip_dtypes, vae_dtypes = [], []

    class _CL:
        @classmethod
        def from_pretrained(cls, path, torch_dtype=None):
            clip_dtypes.append(torch_dtype)
            return cls()

    class _Tok:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

    tr = types.ModuleType("transformers")
    tr.CLIPTextModel = _CL
    tr.CLIPTokenizer = _Tok
    tr.T5TokenizerFast = _Tok
    old_tr = sys.modules.get("transformers")
    sys.modules["transformers"] = tr

    class _VAE:
        @classmethod
        def from_pretrained(cls, path, torch_dtype=None):
            vae_dtypes.append((path, torch_dtype))
            return cls()

    old_vae = getattr(diffusers, "AutoencoderKL", None)
    diffusers.AutoencoderKL = _VAE

    reg, mod = _load_flux_plugin(tmp)
    fake_main = _fake_app_main()
    try:
        cfn = reg.get("flux.dual_clip_load").fn
        cfn("t5.gguf")                     # auto → 历史键名 + bf16
        assert "flux-clip:t5=t5.gguf" in fake_main.models._pipes
        assert clip_dtypes[-1] == "bf16"
        cfn("t5.gguf", dtype="fp16")
        assert "flux-clip:t5=t5.gguf|dtype=fp16" in fake_main.models._pipes
        assert clip_dtypes[-1] == "fp16"

        vfn = reg.get("flux.vae_load").fn
        vfn("vae")                         # auto → fp16
        assert "flux-vae:vae" in fake_main.models._pipes
        assert vae_dtypes[-1][1] == torch.float16
        vfn("vae", dtype="bf16")
        assert "flux-vae:vae|dtype=bf16" in fake_main.models._pipes
        assert vae_dtypes[-1][1] == "bf16"
    finally:
        if old_tr is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = old_tr
        if old_vae is None:
            del diffusers.AutoencoderKL
        else:
            diffusers.AutoencoderKL = old_vae
        sys.modules.pop("app.main", None)
        os.environ.pop("FLUX_DIR", None)
    print("[ok] flux.dual_clip_load / flux.vae_load dtype 旋钮 + 键扩展")


def main():
    test_motion_key_default_unchanged()
    test_motion_key_knob_extends()
    test_motion_knob_validation()
    test_ops_motion_load_passthrough()
    test_clip_vision_dtype_knob()
    test_flux_dtype_helpers()
    test_flux_unet_load_dtype_knob()
    test_flux_dual_clip_and_vae_dtype_knob()
    print("\n✓ 加载旋钮全量化测试全部通过")


if __name__ == "__main__":
    main()

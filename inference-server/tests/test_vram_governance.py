"""显存治理护栏测试（dev-plan §21.3 任务 1/2/3）。

无需 GPU、无需真 torch：注入最小 torch/diffusers stub 后直接测 ModelManager 与
引擎护栏逻辑（淘汰决策、pin 保护、显式卸载、OOM 自愈、引擎 pin 生命周期）。

运行：python3 tests/test_vram_governance.py
"""
import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GB = 1024 ** 3


class OOM(RuntimeError):
    """torch.cuda.OutOfMemoryError 的替身。"""


class _FakeCuda:
    def __init__(self, state):
        self._state = state

    def is_available(self):
        return self._state["available"]

    def mem_get_info(self):
        return self._state["free"], self._state["total"]

    def empty_cache(self):
        self._state["empty_cache_calls"] += 1

    def memory_reserved(self, device=0):
        return self._state["reserved"]

    def memory_allocated(self, device=0):
        return self._state["allocated"]


def install_stubs(free=8 * GB, available=True):
    """装 torch/diffusers stub；返回 (torch, state) 以便测试改余量。"""
    state = {"free": free, "total": 8 * GB, "available": available,
             "empty_cache_calls": 0, "reserved": 0, "allocated": 0}
    torch = types.ModuleType("torch")
    cuda = _FakeCuda(state)
    cuda.OutOfMemoryError = OOM
    torch.cuda = cuda
    torch.OutOfMemoryError = OOM
    torch.float16, torch.bfloat16, torch.float32 = "fp16", "bf16", "fp32"
    torch.no_grad = lambda: contextlib.nullcontext()
    torch.zeros = lambda *a, **k: object()
    sys.modules["torch"] = torch
    if "diffusers" not in sys.modules:
        # app.samplers 在模块级 import 具体 scheduler 类，给同名占位即可
        diffusers = types.ModuleType("diffusers")
        for name in ("DDIMScheduler", "DPMSolverMultistepScheduler",
                     "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
                     "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
            setattr(diffusers, name, type(name, (), {}))
        sys.modules["diffusers"] = diffusers
    return torch, state


torch, STATE = install_stubs()

from app import engine  # noqa: E402  (必须在 stub 之后导入)
from app.model_manager import ModelManager, VramGuard, oom_help  # noqa: E402
from app.ops import model_unload  # noqa: E402


class FakePipe:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<FakePipe {self.name}>"


def make_manager(entries, free=8 * GB):
    """entries: [(key, size_bytes, last_used)] —— 直接摆进内部字典（跳过真实加载）。"""
    STATE["free"] = free
    m = ModelManager()
    for key, size, ts in entries:
        m._pipes[key] = FakePipe(key)
        m._sizes[key] = size
        m._last_used[key] = ts
        m._archs[key] = "FakePipeline"
    m._evicted = []
    m._orig_do_evict = m._do_evict

    def _track(victim):
        m._evicted.append(victim)
        return m._orig_do_evict(victim)

    m._do_evict = _track
    return m


# ---------- 任务 1：预算式淘汰 ----------

def test_count_cap_still_works():
    """条目上限兜底：MAX_RESIDENT=2、显存充足 ⇒ 只淘汰 LRU 一个。"""
    import app.model_manager as mm
    old = mm.MAX_RESIDENT
    try:
        mm.MAX_RESIDENT = 2
        m = make_manager([("a", GB, 1.0), ("b", GB, 2.0)], free=8 * GB)
        m._evict_if_needed(GB)
        assert m._evicted == ["a"], m._evicted
    finally:
        mm.MAX_RESIDENT = old


def test_memory_budget_evicts_until_enough():
    """显存预算：上限放宽到 8，但余量只有 1GB、要装 2GB+reserve ⇒ 按 LRU 连淘汰两个。"""
    import app.model_manager as mm
    old = mm.MAX_RESIDENT
    try:
        mm.MAX_RESIDENT = 8
        m = make_manager([("a", GB, 1.0), ("b", GB, 2.0), ("c", GB, 3.0)],
                         free=1 * GB)
        m._evict_if_needed(2 * GB)  # reserve 1GB + factor 1.25 ⇒ 需 3.5GB
        assert m._evicted == ["a", "b", "c"], m._evicted
        assert m.resident() == []
    finally:
        mm.MAX_RESIDENT = old


def test_memory_budget_keeps_when_roomy():
    import app.model_manager as mm
    old = mm.MAX_RESIDENT
    try:
        mm.MAX_RESIDENT = 8
        m = make_manager([("a", GB, 1.0)], free=7 * GB)
        m._evict_if_needed(2 * GB)
        assert m._evicted == []
        assert m.resident() == ["a"]
    finally:
        mm.MAX_RESIDENT = old


def test_free_bytes_counts_reclaimable_cache():
    """采样后驱动余量 0 但缓存池有几 GB ⇒ 可用量要把可回收部分算进来。

    实测教训：wf58 跑完 mem_get_info 报 free=0（PyTorch 缓存分配器留着复用），
    若只看驱动余量就会把刚加载的 majicmix 白白踢掉（重载 ~2 分钟）。
    """
    STATE.update(free=0, reserved=6 * GB, allocated=2 * GB)
    m = make_manager([], free=0)
    assert m._free_bytes() == 4 * GB          # 0 + (6GB-2GB)
    STATE.update(free=1 * GB, reserved=0, allocated=0)
    assert m._free_bytes() == 1 * GB          # 无缓存时就是驱动余量
    STATE.update(available=False)
    assert m._free_bytes() is None
    STATE.update(available=True, free=8 * GB)


def test_resident_weights_report():
    m = make_manager([("a", 2 * GB, 1.0), ("b", GB / 2, 2.0)])
    assert m.resident_weights() == 2 * GB + GB / 2


def test_pin_protects_in_use_model():
    """pin 住的（执行中在用）不淘汰：宁可放行，也不抽走手上正用的模型。"""
    import app.model_manager as mm
    old = mm.MAX_RESIDENT
    try:
        mm.MAX_RESIDENT = 1
        m = make_manager([("in-use", GB, 1.0), ("idle", GB, 2.0)], free=0)
        m.pin("in-use")
        m._evict_if_needed(2 * GB)
        assert m._evicted == ["idle"], m._evicted
    finally:
        mm.MAX_RESIDENT = old


# ---------- 任务 2：显式卸载 ----------

def test_unload_matching_and_all():
    m = make_manager([("majicmixRealistic_v7|dtype=fp16", GB, 1.0),
                      ("vae:vae-ft-mse|dtype=fp32", GB / 3, 2.0),
                      ("cn:control_v11p_sd15_canny|dtype=fp16", GB / 2, 3.0)])
    assert m.find_keys("majicmixRealistic_v7") == ["majicmixRealistic_v7|dtype=fp16"]
    assert m.find_keys("vae-ft-mse") == ["vae:vae-ft-mse|dtype=fp32"]
    assert m.find_keys("control_v11p_sd15_canny") == ["cn:control_v11p_sd15_canny|dtype=fp16"]
    assert m.unload("majicmixRealistic_v7") == ["majicmixRealistic_v7|dtype=fp16"]
    assert len(m.resident()) == 2
    # all / 空 / * 都是"全部"
    assert len(m.unload("all")) == 2
    assert m.resident() == []
    # 没有匹配 ⇒ 空列表，不报错
    assert m.unload("nope") == []
    assert m.unload("") == []


def test_unload_skips_pinned():
    m = make_manager([("busy", GB, 1.0), ("idle", GB, 2.0)])
    m.pin("busy")
    assert m.unload("all") == ["idle"]
    assert m.resident() == ["busy"]


def test_evict_idle_oom_helper():
    m = make_manager([("a", GB, 1.0), ("b", GB, 2.0), ("c", GB, 3.0)])
    m.pin("a")
    assert m.evict_idle(1) == 1
    assert "b" not in m.resident() and "a" in m.resident()
    assert m.evict_idle(5) == 1  # 只剩 c 可淘汰（a 被 pin）
    assert m.resident() == ["a"]
    assert m.evict_idle(1) == 0  # 全在用 ⇒ 0（触发上层给指引）


def test_details_report():
    m = make_manager([("a", 2 * GB, 1.0)])
    m.pin("a")
    got = m.details()[0]
    assert got["key"] == "a" and got["est_vram_mb"] == 2048 and got["pinned"] is True
    assert got["arch"] == "FakePipeline" and got["idle_seconds"] >= 0


def test_gc_does_not_unload_models():
    """/gc 清缓存不卸模型（历史坑）：显式卸载必须走 unload。"""
    m = make_manager([("a", GB, 1.0)])
    m.unload("all")
    assert m.resident() == []


# ---------- 任务 3：OOM 自愈（引擎层） ----------

class FakeGuard:
    """假护栏：记录 pin/unpin 调用，evict_for_oom 按预设返回可淘汰个数。"""

    def __init__(self, can_evict=1):
        self.can_evict = can_evict
        self.pinned = []
        self.unpinned = []
        self.evict_calls = 0

    def pin_objects(self, objects):
        for obj in objects:
            self.pinned.append(getattr(obj, "name", obj))
        return list(self.pinned)

    def unpin(self, keys):
        self.unpinned.extend(keys)

    def evict_for_oom(self, limit=1):
        self.evict_calls += 1
        return self.can_evict


class FakeOp:
    def __init__(self, fn, name="fake.op"):
        self.fn = fn
        self.name = name


def test_engine_oom_retry_recovers():
    calls = {"n": 0}

    def fn(model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OOM("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"ok": True}

    guard = FakeGuard(can_evict=1)
    engine.set_vram_guard(guard)
    try:
        out = engine._run_op(FakeOp(fn), {"model": FakePipe("m1")})
        assert out == {"ok": True}
        assert calls["n"] == 2                      # 重试了一次
        assert guard.evict_calls == 1               # 且触发了淘汰
        assert engine.oom_state()["recovered"] >= 1
        assert guard.pinned == ["m1"] and guard.unpinned == ["m1"]
    finally:
        engine.set_vram_guard(None)


def test_engine_oom_without_evictable_raises_helpful_error():
    def fn(model=None):
        raise OOM("CUDA out of memory")

    guard = FakeGuard(can_evict=0)
    engine.set_vram_guard(guard)
    before = engine.oom_state()["failed"]
    try:
        engine._run_op(FakeOp(fn, "sample"), {"model": FakePipe("m1")})
        raise AssertionError("应当抛出可读错误")
    except RuntimeError as e:
        assert "sample" in str(e)
        assert "offload=model|sequential" in str(e)   # 带自救指引
        assert "model-unload" in str(e) or "/model/unload" in str(e)
    finally:
        engine.set_vram_guard(None)
    assert engine.oom_state()["failed"] == before + 1
    assert guard.unpinned == ["m1"]              # 异常路径也必须解 pin


def test_engine_unpins_even_on_non_oom_error():
    def fn(model=None):
        raise ValueError("bad input")

    guard = FakeGuard()
    engine.set_vram_guard(guard)
    try:
        try:
            engine._run_op(FakeOp(fn), {"model": FakePipe("m2")})
            raise AssertionError("应当抛错")
        except ValueError:
            pass
        assert guard.pinned == ["m2"] and guard.unpinned == ["m2"]
    finally:
        engine.set_vram_guard(None)


def test_vram_guard_pins_by_object_identity():
    m = make_manager([("m1", GB, 1.0)])
    guard = VramGuard(m)
    keys = guard.pin_objects([m._pipes["m1"]])
    assert keys == ["m1"] and "m1" in m._pins
    guard.unpin(keys)
    assert m._pins == set()
    # 不认识的普通对象（如 LATENT 张量）被忽略，不报错
    assert guard.pin_objects([object(), None]) == []


# ---------- 算子层：model.unload ----------

def test_model_unload_op_returns_strings():
    m = make_manager([("a", GB, 1.0)])
    out = model_unload(m, "a")
    assert out == {"unloaded": "a", "resident": "(empty)"}
    out = model_unload(m, "")
    assert out == {"unloaded": "(none)", "resident": "(empty)"}


def test_oom_help_mentions_actions():
    text = oom_help("sample")
    for kw in ("offload=model", "VRAM_RESERVE_MB", "model-unload", "VRAM_BUDGET"):
        assert kw in text


def main():
    cases = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in cases:
        fn()
        print(f"{fn.__name__} OK")
    print(f"ALL VRAM GOVERNANCE TESTS PASSED ({len(cases)} cases)")


if __name__ == "__main__":
    main()

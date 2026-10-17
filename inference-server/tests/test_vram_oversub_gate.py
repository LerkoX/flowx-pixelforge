"""B1/B4 超订闸门与总量仲裁测试（exec 795 事故，2026-10-16）。

无需 GPU、无需真 torch：沿用 tests/test_vram_governance.py 的最小 torch stub，
直接测 ModelManager 的物理口径占用快照、offload 分账、降档判定与租约总量仲裁。

事故现场（真机数据）：
- GTX 1080 8GB / WSL2：租约 794（FLUX ~7.4GB）+ 795（SDXL+CN ~9.0GB）共 16.4GB
- `/health` 全程 `status=ok`、`vram_low=false`、`degraded.count=0`（OOM 自愈没触发）
- 唯一线索：`vram_process_reserved_mb=9276 > vram_total_mb=8192`、`driver_free=0`
- 后果：采样 170~210s/步（干净态 40s/步），PCIe 双向 1.5/2.5-3.0 GB/s 搬了 100 分钟

运行：python3 tests/test_vram_oversub_gate.py
"""
import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MB = 1024 ** 2
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


def install_stubs():
    """装 torch/diffusers stub；返回 state 以便测试改物理状态。"""
    state = {"free": 8 * GB, "total": 8 * GB, "available": True,
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
        diffusers = types.ModuleType("diffusers")
        for name in ("DDIMScheduler", "DPMSolverMultistepScheduler",
                     "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
                     "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
            setattr(diffusers, name, type(name, (), {}))
        sys.modules["diffusers"] = diffusers
    return state


STATE = install_stubs()

from app import lease  # noqa: E402
from app.model_manager import KNOBS, ModelManager  # noqa: E402


class FakePipe:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<FakePipe {self.name}>"


def make_manager(entries=(), free=8 * GB, reserved=0, allocated=0, total=8 * GB):
    """entries: [(key, size_bytes)]（直接摆进内部字典，跳过真实加载）。"""
    STATE.update({"free": free, "reserved": reserved, "allocated": allocated,
                  "total": total})
    m = ModelManager()
    for i, (key, size) in enumerate(entries):
        m._pipes[key] = FakePipe(key)
        m._sizes[key] = size
        m._last_used[key] = float(i)
        m._archs[key] = "FakePipeline"
    return m


# ---------- B1：物理口径占用快照 ----------

def test_occupancy_detects_oversubscription():
    """实测现场：reserved 9276MB > 物理 8192MB ⇒ 已在借主机内存。"""
    m = make_manager(free=0, reserved=9276 * MB, allocated=8000 * MB)
    occ = m.occupancy()
    assert occ["available"] is True
    assert occ["total_bytes"] == 8 * GB
    assert occ["reserved_bytes"] == 9276 * MB
    assert occ["reclaimable_bytes"] == 1276 * MB
    assert occ["oversubscribed"] is True          # ← 旧 /health 完全看不出来
    assert occ["saturated"] is True
    assert occ["headroom_bytes"] == 0             # 物理真余量 0
    assert occ["oversub_ratio"] > 1.0
    # 对照：旧的乐观口径 vram_low = (free + reclaimable) < reserve(1024MB)
    # → 0 + 1276MB ≥ 1024MB ⇒ vram_low=false（把人骗了的地方，见 /health 注释）


def test_occupancy_healthy_cached_is_not_oversubscribed():
    """健康态：分配器把整卡缓存下来（reserved==total, driver_free=0）不算超订。"""
    m = make_manager(free=0, reserved=8 * GB, allocated=5 * GB)
    occ = m.occupancy()
    assert occ["saturated"] is True               # 物理上没余量（可能是缓存）
    assert occ["oversubscribed"] is False         # 但没有借主机内存 ⇒ 不误报
    assert occ["headroom_bytes"] == 0


def test_occupancy_without_cuda():
    STATE["available"] = False
    try:
        m = make_manager()
        occ = m.occupancy()
        assert occ["available"] is False and occ["oversubscribed"] is False
    finally:
        STATE["available"] = True


# ---------- B1：offload 分账（预算别再按权重全量算） ----------

def test_offload_budget_accounting():
    """offload=model/sequential 的条目只按 floor 占显存预算。

    修复前：SDXL 条目按权重全量记 6616MB×1.25 = 8270MB > 整卡 8GB ⇒
    预算条件 free+freed ≥ need+reserve 永远无法满足 ⇒ 每次加载都把别的条目全清掉
    （实测 flux 四条 6470/477/381/123MB 被逐个顶掉），清完自己也装不下。
    """
    floor = ModelManager._offload_floor()
    assert floor > 0
    assert ModelManager.vram_budget_of(6616 * MB, "none") == 6616 * MB
    assert ModelManager.vram_budget_of(6616 * MB, "model") == floor
    assert ModelManager.vram_budget_of(6616 * MB, "sequential") == floor
    assert ModelManager.vram_budget_of(0, "model") == floor


def test_vram_size_uses_recorded_offload_mode():
    m = make_manager([("legacy", 100 * MB)])
    # 兼容：只填了 _sizes 的老条目/测试仍返回权重体积
    assert m._vram_size("legacy") == 100 * MB
    m._vram_sizes["pony"] = m.vram_budget_of(6616 * MB, "model")
    m._sizes["pony"] = 6616 * MB
    m._pipes["pony"] = FakePipe("pony")
    # 诊断量保持权重体积，预算是 floor
    assert m._sizes["pony"] == 6616 * MB
    assert m._vram_size("pony") == ModelManager._offload_floor()
    assert m.vram_budget_bytes() == 100 * MB + ModelManager._offload_floor()


def test_offload_entry_credit_is_honest_in_eviction_plan():
    """淘汰计划不再把 offload 条目的**CPU 权重**当成“释放了显存”。

    修复前：淘汰一个 offload=model 的大管道会被记账为释放 6.6GB 显存，
    于是计划只淘汰它一个就“达标”——而实际 VRAM 只掉了一个 floor。
    分账后：按 floor 计（1GB），该腾多少就得腾多少（宁可多淘汰几个）。
    """
    import app.model_manager as _mm
    old = _mm.MAX_RESIDENT
    _mm.MAX_RESIDENT = 8          # 关掉条目数兜底，只验显存预算
    try:
        m = make_manager([("pony|offload=model", 6616 * MB), ("idle", 2 * GB)],
                         free=512 * MB)
        m._vram_sizes["pony|offload=model"] = ModelManager.vram_budget_of(
            6616 * MB, "model")
        assert m._vram_size("pony|offload=model") == ModelManager._offload_floor()
        evicted = []
        orig = m._do_evict
        m._do_evict = lambda k: (evicted.append(k), orig(k))[1]
        # 需要 2GB + reserve 1GB，而物理只剩 512MB ⇒ 必须腾出 ≥ 2.5GB；
        # 诚实账目：floor(1GB) + idle(2GB) = 3GB ⇒ 两个都要淘汰；
        # 旧账目会把 pony 记成 6.6GB，只淘汰它一个。
        m._evict_if_needed(2 * GB)
        assert sorted(evicted) == ["idle", "pony|offload=model"], evicted
    finally:
        _mm.MAX_RESIDENT = old


# ---------- B1：降档判定 ----------

def test_needs_downgrade_on_oversubscription():
    m = make_manager(free=0, reserved=9276 * MB, allocated=8000 * MB)
    need, why = m._needs_downgrade(1024 * MB)
    assert need is True and "兜底" in why


def test_needs_downgrade_when_weights_never_fit():
    """8GB 卡塞 6.6GB 权重（×1.25 = 8.25GB > 8GB − reserve）：静态不可能。"""
    m = make_manager(free=8 * GB, reserved=0)
    need, why = m._needs_downgrade(6616 * MB)
    assert need is True and "不可行" in why
    # 小模型正常放行
    assert m._needs_downgrade(64 * MB)[0] is False


def test_needs_downgrade_without_cuda():
    STATE["available"] = False
    try:
        assert make_manager()._needs_downgrade(6616 * MB)[0] is False
    finally:
        STATE["available"] = True


# ---------- B4：租约总量仲裁 ----------

def test_total_budget_flags_other_lease_oversub():
    """实测场景：别的租约占 7.4GB，本次还要 6GB ⇒ 8GB 卡装不下，要说话。"""
    m = make_manager([("flux-unet", int(7.4 * GB))])
    m.leases.touch("794", ["flux-unet"])
    lease.bind("795")
    try:
        assert m._other_leases_vram() == int(7.4 * GB)
        assert m._check_total_budget(6 * GB) is True        # 警告路径（默认）
    finally:
        lease.unbind()


def test_total_budget_strict_raises_with_guidance():
    m = make_manager([("flux-unet", int(7.4 * GB))])
    m.leases.touch("794", ["flux-unet"])
    old = KNOBS.get("vram_strict")
    KNOBS["vram_strict"] = 1
    lease.bind("795")
    try:
        try:
            m._check_total_budget(6 * GB)
            raise AssertionError("strict 模式应拒绝加载")
        except RuntimeError as e:
            msg = str(e)
            assert "超订" in msg
            for hint in ("model/unload", "offload=model|sequential", "重启"):
                assert hint in msg, msg
    finally:
        KNOBS["vram_strict"] = old
        lease.unbind()


def test_total_budget_quiet_when_no_other_lease():
    m = make_manager([("mine", int(6 * GB))])
    lease.bind("795")
    try:
        assert m._check_total_budget(6 * GB) is False       # 自己的条目不算“他人”
    finally:
        lease.unbind()


def test_total_budget_counts_offload_floor():
    """分账同样作用于仲裁：别人的 offload 大条目只按 floor 计。"""
    m = make_manager([("other|offload=model", 7 * GB)])
    m._vram_sizes["other|offload=model"] = ModelManager.vram_budget_of(
        7 * GB, "model")
    m.leases.touch("A", ["other|offload=model"])
    lease.bind("B")
    try:
        assert m._other_leases_vram() == ModelManager._offload_floor()
        assert m._check_total_budget(6 * GB) is False
    finally:
        lease.unbind()


def main():
    checks = [
        test_occupancy_detects_oversubscription,
        test_occupancy_healthy_cached_is_not_oversubscribed,
        test_occupancy_without_cuda,
        test_offload_budget_accounting,
        test_vram_size_uses_recorded_offload_mode,
        test_offload_entry_credit_is_honest_in_eviction_plan,
        test_needs_downgrade_on_oversubscription,
        test_needs_downgrade_when_weights_never_fit,
        test_needs_downgrade_without_cuda,
        test_total_budget_flags_other_lease_oversub,
        test_total_budget_strict_raises_with_guidance,
        test_total_budget_quiet_when_no_other_lease,
        test_total_budget_counts_offload_floor,
    ]
    for fn in checks:
        fn()
        print(f"{fn.__name__} OK")
    print("ALL OVERSUB GATE TESTS PASSED "
          f"({len(checks)} cases)")


if __name__ == "__main__":
    main()

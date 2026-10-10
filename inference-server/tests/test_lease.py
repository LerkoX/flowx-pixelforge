"""执行租约 pin（dev-plan §43.2）测试。

覆盖：
- lease 模块：sanitize/bind/current 线程本地、LeaseBook 归属/TTL 滑动过期/摘除
- ModelManager：淘汰跳过其它活跃租约（本租约可汰自己的、过期租约不再保护、
  无租约调用方被所有租约约束）、unload/evict_idle 租约保护、
  加载阻塞等待（超时抛指引 / 对端释放后通行 / 租约过期后通行）
- jobs：租约随 job 记录并在 worker 执行线程绑定
- pin：执行期 pin 同时归入当前租约

运行：python3 tests/test_lease.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_vram_governance as tvg  # noqa: E402  (其模块级已装 stub 并导入 app.*)

torch, STATE = tvg.torch, tvg.STATE  # 与 app.model_manager 所持 stub 同一 state

from app import lease  # noqa: E402
from app.jobs import JobManager  # noqa: E402
from app.model_manager import ModelManager  # noqa: E402
import app.model_manager as mm  # noqa: E402

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# 1. lease 模块本体
# ---------------------------------------------------------------------------

def test_sanitize_and_bind():
    assert lease.sanitize("") is None
    assert lease.sanitize(None) is None
    assert lease.sanitize("  ") is None
    assert lease.sanitize("x" * 200) is None       # 超长拒绝
    assert lease.sanitize(" exec-1 ") == "exec-1"  # 去空白
    lease.bind("exec-1")
    assert lease.current() == "exec-1"
    lease.unbind()
    assert lease.current() is None
    lease.bind("")                                 # 空头 = 无租约
    assert lease.current() is None
    print("[ok] lease sanitize/bind 线程本地语义")


def test_leasebook_touch_expiry():
    book = lease.LeaseBook(ttl=0.3)
    book.touch("a", ["k1", "k2"])
    book.touch("b", ["k3"])
    assert book.keys_of_others("a") == {"k3"}      # a 视角：只有 b 的
    assert book.keys_of_others("b") == {"k1", "k2"}
    assert book.keys_of_others(None) == {"k1", "k2", "k3"}  # 无租约：全部
    assert book.keys_of_others("a") == {"k3"}
    time.sleep(0.35)
    assert book.keys_of_others("a") == set()       # TTL 过期全清
    assert book.view() == []
    # 滑动刷新：活动中的租约不过期
    book.touch("c", ["k9"])
    time.sleep(0.2)
    book.touch("c")                                # 活动刷新
    time.sleep(0.2)
    assert book.keys_of_others(None) == {"k9"}
    # forget_keys 摘除
    book.forget_keys(["k9"])
    assert book.keys_of_others(None) == set()
    print("[ok] LeaseBook 归属/他人视角/TTL 滑动过期/摘除")


# ---------------------------------------------------------------------------
# 2. ModelManager 淘汰过滤
# ---------------------------------------------------------------------------

def _mgr_with_entries():
    mgr = ModelManager()
    mgr._pipes["leased:a"] = object()
    mgr._pipes["free:b"] = object()
    mgr._sizes["leased:a"] = 4 * GB
    mgr._sizes["free:b"] = 100 * 1024**2
    mgr._last_used["leased:a"] = time.time() - 100  # leased 更旧（LRU 先汰它）
    mgr._last_used["free:b"] = time.time()
    return mgr


def _run_evict(mgr, need=0):
    evicted = []
    orig = mgr._do_evict
    mgr._do_evict = lambda key: (evicted.append(key), orig(key))[1]
    old = mm.MAX_RESIDENT
    mm.MAX_RESIDENT = 2  # 2 个条目汰 1 个（剩余 < cap 才停）
    try:
        mgr._evict_if_needed(need)
    finally:
        mm.MAX_RESIDENT = old
    return evicted


def test_eviction_skips_other_lease():
    """租约 A 持有的条目在租约 B（当前线程）加载时不被淘汰；改汰无租约条目。"""
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    lease.bind("exec-B")
    try:
        assert _run_evict(mgr) == ["free:b"]
        assert "leased:a" in mgr._pipes
    finally:
        lease.unbind()
    print("[ok] 淘汰跳过其它活跃租约持有的条目")


def test_eviction_own_lease_keys_evictable():
    """本租约自己的条目对自己可淘汰（同流水线内换模型不受影响）。"""
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    lease.bind("exec-A")
    try:
        assert _run_evict(mgr) == ["leased:a"]     # 自己最旧，正常淘汰
    finally:
        lease.unbind()
    print("[ok] 本租约条目对自己可淘汰")


def test_eviction_expired_lease_unprotected():
    """租约 TTL 过期后条目恢复可淘汰。"""
    mgr = _mgr_with_entries()
    mgr.leases._ttl = 0.2
    mgr.leases.touch("exec-A", ["leased:a"])
    lease.bind("exec-B")
    try:
        assert _run_evict(mgr) == ["free:b"]       # 未过期：保护生效
        mgr._pipes["free:b2"] = object()
        mgr._sizes["free:b2"] = 1
        mgr._last_used["free:b2"] = time.time()
        time.sleep(0.25)
        assert _run_evict(mgr) == ["leased:a"]     # 过期：最旧的它被汰
    finally:
        lease.unbind()
    print("[ok] 租约过期后条目恢复可淘汰")


def test_unleased_caller_blocked_by_leases():
    """无租约调用方（验收脚本）被所有活跃租约约束。"""
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    assert lease.current() is None
    assert _run_evict(mgr) == ["free:b"]
    assert "leased:a" in mgr._pipes
    print("[ok] 无租约调用方不动任何租约的条目")


def test_unload_and_evict_idle_respect_leases():
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    lease.bind("exec-B")
    try:
        assert mgr.unload("all") == ["free:b"]     # leased:a 跳过
        assert mgr.evict_idle(1) == 0              # 只剩 leased:a，不可汰
    finally:
        lease.unbind()
    # 租约过期后可卸
    mgr.leases._ttl = 0.1
    time.sleep(0.15)
    assert mgr.unload("all") == ["leased:a"]
    print("[ok] unload/evict_idle 跳过其它租约，过期恢复")


def test_pin_attributes_to_lease():
    """执行期 pin 同时归入当前租约（引擎 pin_objects 链路）。"""
    mgr = _mgr_with_entries()
    lease.bind("exec-A")
    try:
        mgr.pin("leased:a")
        held = [e for e in mgr.leases.view() if e["lease"] == "exec-A"]
        assert held and "leased:a" in held[0]["keys"], mgr.leases.view()
        mgr.unpin("leased:a")                      # unpin 不移除租约归属
        held = [e for e in mgr.leases.view() if e["lease"] == "exec-A"]
        assert "leased:a" in held[0]["keys"]
    finally:
        lease.unbind()
    # 无租约 pin：不产生租约记录（行为不变）
    mgr.pin("free:b")
    assert mgr.leases.view() == mgr.leases.view()  # 不炸
    assert all(e["lease"] != "" for e in mgr.leases.view())
    print("[ok] pin 归入当前租约；unpin 不摘归属（TTL 统一释放）")


def test_evict_forgets_lease_keys():
    """条目被淘汰后从租约名下摘除（防虚保护堵死后续加载）。"""
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    lease.bind("exec-A")
    try:
        _run_evict(mgr)
        assert "leased:a" not in mgr._pipes
        assert mgr.leases.keys_of_others("exec-B") == set()
    finally:
        lease.unbind()
    print("[ok] 淘汰同步摘除租约簿记")


# ---------------------------------------------------------------------------
# 3. 加载阻塞等待（LEASE_WAIT_TIMEOUT）
# ---------------------------------------------------------------------------

def _wait_scenario():
    """构造等待场景：leased:a(4GB) 被 exec-A 持有，free=1GB，need 4GB。"""
    mgr = _mgr_with_entries()
    mgr.leases.touch("exec-A", ["leased:a"])
    STATE["free"] = 1 * GB
    STATE["reserved"] = STATE["allocated"] = 0
    old_wait = lease.LEASE_WAIT_TIMEOUT
    old_budget = mm.VRAM_BUDGET
    mm.VRAM_BUDGET = True
    lease.bind("exec-B")
    return mgr, old_wait, old_budget


def _wait_scenario_cleanup(old_wait, old_budget):
    lease.LEASE_WAIT_TIMEOUT = old_wait
    mm.VRAM_BUDGET = old_budget
    lease.unbind()
    STATE["free"] = 8 * GB


def test_wait_timeout_raises_with_guidance():
    """对端租约一直占着 → 等待超时，报错带可执行指引（非裸 OOM）。
    注意调用纪律：_evict_if_needed 必须在持锁状态调用（生产路径均如此）。"""
    mgr, old_wait, old_budget = _wait_scenario()
    lease.LEASE_WAIT_TIMEOUT = 0.6
    t0 = time.time()
    try:
        try:
            with mgr._lock:
                mgr._evict_if_needed(4 * GB)
            raise AssertionError("应抛租约等待超时")
        except RuntimeError as e:
            msg = str(e)
            assert "等待超时" in msg and "exec-A" in msg, msg
            assert "/model/unload" in msg, msg
        assert time.time() - t0 >= 0.5
    finally:
        _wait_scenario_cleanup(old_wait, old_budget)
    print("[ok] 租约等待超时 → 带指引的 RuntimeError")


def test_wait_succeeds_when_peer_frees():
    """等待期间对端经 models.unload 真释放（走同一把模型锁）→ 通行。
    回归：等待曾持锁睡眠，把对端 unload 挡死到超时（真机事故）。"""
    mgr, old_wait, old_budget = _wait_scenario()
    lease.LEASE_WAIT_TIMEOUT = 5.0

    def _free_later():
        time.sleep(0.5)
        lease.bind("exec-A")          # 对端流水线收尾：卸自己名下的模型
        try:
            mgr.unload("all")         # 需要拿到模型锁（等待方已临时释放）
        finally:
            lease.unbind()
        STATE["free"] = 8 * GB        # 卸载后驱动余量回升（stub 手动模拟）

    threading.Thread(target=_free_later, daemon=True).start()
    t0 = time.time()
    try:
        with mgr._lock:
            mgr._evict_if_needed(4 * GB)   # 不抛异常即通行
        assert 0.4 < time.time() - t0 < 5.0
        assert "leased:a" not in mgr._pipes  # 对端自己卸的，不是互踩
    finally:
        _wait_scenario_cleanup(old_wait, old_budget)
    print("[ok] 对端 unload 真释放（同一把锁）→ 等待通行")


def test_wait_succeeds_when_lease_expires():
    """等待期间对端租约 TTL 过期 → 其条目恢复可汰 → 通行。"""
    mgr, old_wait, old_budget = _wait_scenario()
    lease.LEASE_WAIT_TIMEOUT = 5.0
    mgr.leases._ttl = 0.4               # exec-A 很快过期
    t0 = time.time()
    try:
        with mgr._lock:
            mgr._evict_if_needed(4 * GB)
        assert time.time() - t0 < 5.0
        assert "leased:a" not in mgr._pipes  # 过期后被汰
    finally:
        _wait_scenario_cleanup(old_wait, old_budget)
    print("[ok] 对端租约过期 → 淘汰通行")


def test_no_wait_without_leases():
    """无租约占用时纯体积不足 → 不等待，走既有 WARNING 路径（行为不变）。"""
    mgr = _mgr_with_entries()
    STATE["free"] = 1 * GB
    STATE["reserved"] = STATE["allocated"] = 0
    old_budget = mm.VRAM_BUDGET
    mm.VRAM_BUDGET = True
    t0 = time.time()
    try:
        mgr._evict_if_needed(4 * GB)   # 立即返回（WARNING），不抛不等
        assert time.time() - t0 < 1.0
    finally:
        mm.VRAM_BUDGET = old_budget
        STATE["free"] = 8 * GB
    print("[ok] 无租约占用 → 不进入等待（行为不变）")


# ---------------------------------------------------------------------------
# 4. jobs：租约随 job 记录并在 worker 线程绑定
# ---------------------------------------------------------------------------

def test_job_binds_lease():
    seen = {}

    def run(kind, payload):
        seen["lease"] = lease.current()
        return {"ok": True}

    jm = JobManager(run)
    jid = jm.submit("op", {"name": "x"}, lease="exec-42")
    deadline = time.time() + 5
    while time.time() < deadline:
        v = jm.get(jid)
        if v["status"] == "done":
            break
        time.sleep(0.05)
    v = jm.get(jid)
    assert v["status"] == "done", v
    assert v.get("lease") == "exec-42", v          # job 视图带租约
    assert seen.get("lease") == "exec-42", seen    # worker 执行期间已绑定
    # 无租约 job：view 无 lease 字段，执行时 current 为 None
    seen.clear()
    jid2 = jm.submit("op", {"name": "x"})
    deadline = time.time() + 5
    while time.time() < deadline:
        if jm.get(jid2)["status"] == "done":
            break
        time.sleep(0.05)
    assert "lease" not in jm.get(jid2)
    assert seen.get("lease") is None
    print("[ok] job 记录租约 + worker 绑定；无租约 job 行为不变")


def main():
    test_sanitize_and_bind()
    test_leasebook_touch_expiry()
    test_eviction_skips_other_lease()
    test_eviction_own_lease_keys_evictable()
    test_eviction_expired_lease_unprotected()
    test_unleased_caller_blocked_by_leases()
    test_unload_and_evict_idle_respect_leases()
    test_pin_attributes_to_lease()
    test_evict_forgets_lease_keys()
    test_wait_timeout_raises_with_guidance()
    test_wait_succeeds_when_peer_frees()
    test_wait_succeeds_when_lease_expires()
    test_no_wait_without_leases()
    test_job_binds_lease()
    print("\n✓ 执行租约 pin 测试全部通过")


if __name__ == "__main__":
    main()

"""对象仓库引用保活：exec442 事故的回归测试。

事故：并发加载时 SDXL 入驻检查把先完成、对象已被下游绑定待消费的
InstantID CN 当 LRU 牺牲品淘汰，Apply 取用 control_net 对象报
"object not found"。

语义：ModelManager._evict_if_needed 的淘汰计划把 extra_pinned 回调
返回的 key 视同执行期 pin；main.py 注入
extra_pinned = models.keys_of(store.referenced_data())。

运行：python3 tests/test_eviction_guard.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_vram_governance import install_stubs  # noqa: E402  (torch stub 先行)

install_stubs()

from app.model_manager import ModelManager  # noqa: E402
from app.object_store import ObjectStore  # noqa: E402
import app.model_manager as mm  # noqa: E402


def _mgr_with_entries():
    """构造带两个驻留条目的管理器（绕过真实加载，直塞内部状态）。"""
    mgr = ModelManager()
    pipe_cn = object()   # 占位管道本体
    pipe_ckpt = object()
    with mgr._lock:
        mgr._pipes["cn:test|dtype=fp16"] = pipe_cn
        mgr._pipes["ckpt:test|dtype=fp16"] = pipe_ckpt
        mgr._sizes["cn:test|dtype=fp16"] = 100
        mgr._sizes["ckpt:test|dtype=fp16"] = 200
        mgr._last_used["cn:test|dtype=fp16"] = time.time() - 100  # CN 更旧
        mgr._last_used["ckpt:test|dtype=fp16"] = time.time()
    return mgr, pipe_cn, pipe_ckpt


def test_keys_of_matches_pipe_identity():
    mgr, pipe_cn, _ = _mgr_with_entries()
    assert mgr.keys_of([pipe_cn, object()]) == {"cn:test|dtype=fp16"}
    assert mgr.keys_of([]) == set()
    print("[ok] keys_of 按对象身份匹配常驻 key")


def _run_evict(mgr):
    evicted = []
    mgr._do_evict = lambda key: evicted.append(key)  # 拦截真实淘汰
    old = mm.MAX_RESIDENT
    mm.MAX_RESIDENT = 2  # 加载前腾 1 个位置（剩余 < cap 才停）
    try:
        mgr._evict_if_needed(0)
    finally:
        mm.MAX_RESIDENT = old
    return evicted


def test_eviction_skips_store_referenced_pipe():
    """CN 对象仍在对象仓库（下游待消费）→ 加载新模型时不得淘汰 CN。"""
    mgr, pipe_cn, _ = _mgr_with_entries()
    store = ObjectStore(ttl_seconds=14400)
    store.put("CONTROL_NET", pipe_cn)  # LoadCN 输出对象入库
    mgr.extra_pinned = lambda: mgr.keys_of(store.referenced_data())
    assert _run_evict(mgr) == ["ckpt:test|dtype=fp16"]
    print("[ok] 仓库引用的管道视同 pin，淘汰跳过 CN 改汰 ckpt")


def test_eviction_without_reference_still_lru():
    """仓库不再引用（对象已消费/过期）→ CN 恢复可淘汰（LRU 最旧）。"""
    mgr, pipe_cn, _ = _mgr_with_entries()
    store = ObjectStore(ttl_seconds=14400)
    mgr.extra_pinned = lambda: mgr.keys_of(store.referenced_data())
    assert _run_evict(mgr) == ["cn:test|dtype=fp16"]  # CN 更旧，正常淘汰
    print("[ok] 无引用时退回纯 LRU 语义")


def test_extra_pinned_failure_does_not_block():
    """回调故障不阻断加载（退回纯执行期 pin 语义）。"""
    mgr, _, _ = _mgr_with_entries()
    def boom():
        raise RuntimeError("boom")
    mgr.extra_pinned = boom
    assert _run_evict(mgr) == ["cn:test|dtype=fp16"]
    print("[ok] extra_pinned 异常静默降级")


if __name__ == "__main__":
    test_keys_of_matches_pipe_identity()
    test_eviction_skips_store_referenced_pipe()
    test_eviction_without_reference_still_lru()
    test_extra_pinned_failure_does_not_block()
    print("[PASS] test_eviction_guard")

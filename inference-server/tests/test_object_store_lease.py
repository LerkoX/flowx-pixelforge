"""B2：对象仓库按租约收敛 + 孤儿回收测试（exec 795 事故，2026-10-16）。

问题现场：仓库条目**不区分租约**，`extra_pinned` 把上一轮/另一条流水线留下的
model/clip/vae 视图也当“不可淘汰”，配上实测环境 `OBJECT_TTL_SECONDS=86400`（24h）
—— flux 的四个条目（unet 6470MB + clip 477 + vae 381 + redux 123）就这样赖在
8GB 卡上一整天，与 SDXL+ControlNet 一起把显存顶爆（16.4GB 申报 / 8GB 物理），
最后靠人工 `model.unload` 才松开（13:36 的 evicting 日志）。

修法（本文件所测）：
1. `put()` 记下所属租约；`referenced_data(lease_id)` 只返回**本租约**的条目
   （`lease_id=None` 保持旧的全量语义，脚本/验收调用方不受影响）；
2. 孤儿回收：条目所属租约已不活跃且龄期超过 `orphan_ttl_seconds` → 回收；
   未注入 `lease_alive` 时**关闭**（兼容旧行为）；
3. 长任务靠 jobs 进度心跳续租，不会被当孤儿（exec 795 采样 5425s ≫ 租约 TTL 1800s）。

运行：python3 tests/test_object_store_lease.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import lease                      # noqa: E402
from app.jobs import Job                   # noqa: E402
from app.lease import LeaseBook            # noqa: E402
from app.object_store import ObjectStore   # noqa: E402


def _store(**kw):
    kw.setdefault("ttl_seconds", 10 ** 9)   # 常规 TTL 放到无穷远，只验孤儿规则
    return ObjectStore(**kw)


def test_put_stamps_lease_and_referenced_data_is_scoped():
    st = _store()
    lease.bind("795")
    st.put("MODEL", {"obj": 1})
    lease.unbind()
    lease.bind("794")
    st.put("MODEL", {"obj": 2})
    lease.unbind()
    st.put("MODEL", {"obj": 3})             # 无租约调用方（脚本/验收）
    # 旧语义保持：不传租约 = 全量
    assert len(st.referenced_data()) == 3
    # 新语义：只看本租约 —— flux（794）的条目不再钉住 SDXL（795）的淘汰
    assert st.referenced_data("795") == [{"obj": 1}]
    assert st.referenced_data("794") == [{"obj": 2}]
    assert st.referenced_data("796") == []


def test_orphan_sweep_drops_items_of_finished_lease():
    st = _store(orphan_ttl_seconds=0,
                lease_alive=lambda lid: lid == "live")
    lease.bind("live")
    st.put("MODEL", {"k": "live"})
    lease.unbind()
    lease.bind("dead")
    st.put("MODEL", {"k": "dead"})
    lease.unbind()
    st.put("MODEL", {"k": "no-lease"})      # 无租约条目不适用孤儿规则
    time.sleep(0.01)
    st._sweep()
    kinds = st.stats()
    assert kinds["count"] == 2, kinds        # dead 被回收，live / 无租约保留


def test_orphan_sweep_disabled_without_lease_alive():
    st = _store(orphan_ttl_seconds=0)        # 未注入 lease_alive
    lease.bind("dead")
    st.put("MODEL", {"k": "dead"})
    lease.unbind()
    time.sleep(0.01)
    st._sweep()
    assert st.stats()["count"] == 1          # 机制关闭 ⇒ 兼容旧行为


def test_orphan_sweep_with_real_leasebook():
    """端到端：租约活跃 → 保留；租约释放（执行结束）→ 回收。"""
    book = LeaseBook(ttl=60)
    st = _store(orphan_ttl_seconds=0, lease_alive=book.is_active)
    lease.bind("795")
    st.put("VAE", {"v": 1})
    lease.unbind()
    book.touch("795")                        # = jobs 进度心跳续租的效果
    time.sleep(0.01)
    st._sweep()
    assert st.stats()["count"] == 1          # 执行还在跑，别动它的对象
    book.release("795")                      # 执行结束
    time.sleep(0.01)
    st._sweep()
    assert st.stats()["count"] == 0          # 回收孤儿（其钉住的模型本体随之可淘汰）


def test_leasebook_is_active():
    book = LeaseBook(ttl=60)
    assert book.is_active("A") is False
    book.touch("A")
    assert book.is_active("A") is True
    assert book.is_active("B") is False
    assert book.is_active(None) is False
    assert book.release("A") is True
    assert book.is_active("A") is False
    # TTL 过期后不再活跃（滑动过期）
    short = LeaseBook(ttl=0.01)
    short.touch("C")
    assert short.is_active("C") is True
    time.sleep(0.02)
    assert short.is_active("C") is False


def test_job_progress_renews_lease():
    """长任务（90 分钟采样）每次进度心跳续租，避免被当“执行已结束”。"""
    touched = []
    job = Job("op", {}, lease="795", lease_touch=lambda lid: touched.append(lid))
    job.set_progress(3, 30)
    assert touched == ["795"]
    assert job.progress == {"current": 3, "total": 30}
    # 兼容：无租约 / 未注入回调（老构造方式）不得报错
    Job("op", {}).set_progress(1, 2)
    Job("op", {}, lease="795").set_progress(1, 2)
    # 续租回调抛错不影响任务
    bad = Job("op", {}, lease="795",
              lease_touch=lambda lid: (_ for _ in ()).throw(RuntimeError("x")))
    bad.set_progress(2, 3)
    assert bad.progress["current"] == 2


def main():
    checks = [
        test_put_stamps_lease_and_referenced_data_is_scoped,
        test_orphan_sweep_drops_items_of_finished_lease,
        test_orphan_sweep_disabled_without_lease_alive,
        test_orphan_sweep_with_real_leasebook,
        test_leasebook_is_active,
        test_job_progress_renews_lease,
    ]
    for fn in checks:
        fn()
        print(f"{fn.__name__} OK")
    print(f"ALL OBJECT STORE LEASE TESTS PASSED ({len(checks)} cases)")


if __name__ == "__main__":
    main()

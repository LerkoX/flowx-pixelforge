"""异步任务体系集成测试（无需 GPU/torch：假算子 + 真实 JobManager/engine 接线）。

覆盖：graph/op 任务完成、失败、running 中断（节点间检查点）、pending 中断、
广播中断、进度视图、结果淘汰。
运行：python3 -m tests.test_jobs  或  python3 tests/test_jobs.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import engine
from app.jobs import JobManager
from app.object_store import ObjectStore
from app.registry import Registry


def build_registry(node_sleep=0.0):
    reg = Registry()

    @reg.register("make", inputs={"value": "INT"}, outputs={"obj": "LATENT"})
    def _(value):
        time.sleep(node_sleep)
        return {"obj": value}

    @reg.register("passthrough", inputs={"x": "LATENT"}, outputs={"y": "LATENT"})
    def _(x):
        time.sleep(node_sleep)
        return {"y": x}

    @reg.register("boom", inputs={}, outputs={"obj": "LATENT"})
    def _():
        raise ValueError("intentional failure")

    return reg


def chain_graph(n):
    """n 个节点串成一条链：0=make，其余 passthrough。"""
    nodes = {"0": {"op": "make", "inputs": {"value": 1}}}
    for i in range(1, n):
        nodes[str(i)] = {"op": "passthrough",
                         "inputs": {"x": [str(i - 1), "obj" if i == 1 else "y"]}}
    return {"nodes": nodes}


def make_manager(reg, retained=200):
    store = ObjectStore()

    def run(kind, payload):
        if kind == "graph":
            return engine.run_graph(store, reg, payload, cache={})
        return engine.run_op(store, reg, payload["name"], payload.get("inputs"))

    return JobManager(run, retained=retained)


def wait(jobman, jid, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = jobman.get(jid)
        if v["status"] in ("done", "failed", "cancelled"):
            return v
        time.sleep(0.01)
    raise AssertionError(f"job {jid} did not finish in {timeout}s: {jobman.get(jid)}")


def wait_status(jobman, jid, status, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if jobman.get(jid)["status"] == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {jid} never reached '{status}': {jobman.get(jid)}")


def main():
    # --- graph 任务完成 ---
    reg = build_registry()
    jm = make_manager(reg)
    jid = jm.submit("graph", chain_graph(3))
    v = wait(jm, jid)
    assert v["status"] == "done", v
    assert v["result"]["nodes"]["2"]["y"]["type"] == "LATENT", v["result"]
    assert v["started_at"] and v["finished_at"], v
    print("graph job done OK")

    # --- op 任务完成 ---
    jid = jm.submit("op", {"name": "make", "inputs": {"value": 42}})
    v = wait(jm, jid)
    assert v["status"] == "done" and v["result"]["outputs"]["obj"]["type"] == "LATENT", v
    print("op job done OK")

    # --- 失败任务带 error ---
    jid = jm.submit("op", {"name": "boom", "inputs": {}})
    v = wait(jm, jid)
    assert v["status"] == "failed" and "intentional failure" in v["error"], v
    print("failed job OK:", v["error"])

    # --- running 中断：节点间检查点生效，后续节点不执行 ---
    reg_slow = build_registry(node_sleep=0.05)
    jm2 = make_manager(reg_slow)
    jid = jm2.submit("graph", chain_graph(50))  # ~2.5s
    wait_status(jm2, jid, "running")
    r = jm2.interrupt(jid)
    assert r["interrupted"] == [jid], r
    v = wait(jm2, jid)
    assert v["status"] == "cancelled", v
    assert v["error"] == "cancelled", v
    print("running interrupt OK (cancelled at node boundary)")

    # --- pending 中断：排队任务不启动 ---
    slow = jm2.submit("graph", chain_graph(50))   # 占住 worker
    queued = jm2.submit("graph", chain_graph(3))
    wait_status(jm2, slow, "running")
    assert jm2.get(queued)["status"] == "pending"
    r = jm2.interrupt(queued)
    assert r["interrupted"] == [queued]
    assert jm2.get(queued)["status"] == "cancelled"
    jm2.interrupt(slow)  # 收尾，释放 worker
    wait(jm2, slow)
    v = wait(jm2, queued)
    assert v["error"] == "cancelled before start", v
    print("pending interrupt OK")

    # --- 广播中断：running + 全部 pending ---
    a = jm2.submit("graph", chain_graph(50))
    b = jm2.submit("graph", chain_graph(3))
    wait_status(jm2, a, "running")
    r = jm2.interrupt()
    assert set(r["interrupted"]) == {a, b}, r
    assert wait(jm2, a)["status"] == "cancelled"
    assert wait(jm2, b)["status"] == "cancelled"
    print("broadcast interrupt OK")

    # --- 中断不存在的任务 ---
    try:
        jm2.interrupt("no-such-job")
        raise AssertionError("KeyError not raised")
    except KeyError as e:
        print("unknown job interrupt OK:", e)

    # --- 进度视图 ---
    jm3 = make_manager(build_registry())
    jid = jm3.submit("op", {"name": "make", "inputs": {"value": 1}})
    v = wait(jm3, jid)
    assert v["progress"] == {"current": 0, "total": 0, "percent": None}, v["progress"]
    job = jm3._jobs[jid]
    job.set_progress(7, 20)
    assert jm3.get(jid)["progress"]["percent"] == 35.0
    print("progress view OK")

    # --- 结果淘汰：retained=2，最旧完成任务被逐出 ---
    jm4 = make_manager(build_registry(), retained=2)
    ids = []
    for i in range(4):
        jid = jm4.submit("op", {"name": "make", "inputs": {"value": i}})
        wait(jm4, jid)
        ids.append(jid)
    for old in ids[:2]:
        try:
            jm4.get(old)
            raise AssertionError(f"job {old} should have been evicted")
        except KeyError:
            pass
    assert jm4.get(ids[-1])["status"] == "done"
    print("retention eviction OK")

    # --- 同步 run_graph 上下文：check_cancelled 为空操作（无 job 绑定） ---
    from app.execution import check_cancelled, current
    assert current() is None
    check_cancelled()  # 不抛
    print("sync context no-op OK")

    print("\nALL JOB TESTS PASSED")


if __name__ == "__main__":
    main()

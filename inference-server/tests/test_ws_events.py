"""WS 事件体系测试（无 GPU/fastapi 可跑，项目测试惯例：stub 重依赖、纯 stdlib）。

覆盖：
1. app/events 广播器：订阅/发布/退订/满队列丢弃/订阅数观测；
2. jobs.py 事件发布点：提交→running→progress→stage→done 全序列到达订阅者，
   failed / cancelled / pending-中断 同样发布（用真实 JobManager + 假 run_fn 驱动）；
3. 静态约束（fastapi 在本机不可导入，不起服务）：
   - main.py 注册 /ws 且鉴权失败关 4401、finally unsubscribe、30s 应用层 ping；
   - monitor.html 有 WS 驱动刷新且保留轮询兜底。

运行：python3 tests/test_ws_events.py
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import events
from app.jobs import JobManager

FAILED = []


def check(label, cond):
    print(("✓" if cond else "✗"), label)
    if not cond:
        FAILED.append(label)


def drain(q, n, timeout=3.0):
    out = []
    t0 = time.time()
    while len(out) < n and time.time() - t0 < timeout:
        try:
            out.append(q.get(timeout=0.2))
        except Exception:
            pass
    return out


# ---------- 1. 广播器 ----------
def test_broadcaster():
    q1, q2 = events.subscribe(), events.subscribe()
    n0 = events.subscriber_count()
    events.publish("progress", "job1", current=3, total=10)
    e1 = q1.get(timeout=2)
    e2 = q2.get(timeout=2)
    check("广播到达全部订阅者", e1["current"] == 3 and e2["current"] == 3)
    check("事件带 type/job_id/ts", e1["type"] == "progress"
          and e1["job_id"] == "job1" and e1["ts"] > 0)
    events.unsubscribe(q1)
    events.publish("progress", "job1", current=4, total=10)
    check("退订后不再收到", q1.empty() and q2.get(timeout=2)["current"] == 4)
    check("subscriber_count 退订后回落", events.subscriber_count() == n0 - 1)
    events.unsubscribe(q2)
    # 满队列丢弃而不阻塞（推送方绝不许影响任务执行）
    q3 = events.subscribe()
    for i in range(1200):
        events.publish("progress", "j", current=i, total=1200)
    check("满队列丢弃不阻塞（收到 <= 容量上限）", q3.qsize() <= 1000)
    events.unsubscribe(q3)


# ---------- 2. jobs.py 发布点（真实 JobManager + 假 run_fn） ----------
def make_manager(run):
    return JobManager(run, retained=10)


def test_job_event_sequence():
    def run(kind, payload):
        from app.execution import current
        job = current()
        job.set_stage(1, 2, "N1", "op.a")
        job.set_progress(1, 3)
        job.set_stage(2, 2, "N2", "op.b")
        job.set_progress(3, 3)
        return {"image": "obj-1"}
    jm = make_manager(run)
    q = events.subscribe()
    jid = jm.submit("graph", {"nodes": {"N1": {"op": "op.a"}, "N2": {"op": "op.b"}}})
    evs = drain(q, 7)
    events.unsubscribe(q)
    types = [e["type"] for e in evs]
    check("完整事件序列 submitted→running→stage→progress→…→done",
          types == ["submitted", "running", "stage", "progress",
                    "stage", "progress", "done"])
    check("事件归属正确 job", all(e.get("job_id") == jid for e in evs))
    done = [e for e in evs if e["type"] == "done"]
    check("done 事件带 result", done and done[0].get("result") == {"image": "obj-1"})
    prog = [e for e in evs if e["type"] == "progress"]
    check("progress 带 percent", prog and prog[-1].get("percent") == 100.0)


def test_failed_and_cancelled_events():
    def boom(kind, payload):
        raise ValueError("炸了")
    jm = make_manager(boom)
    q = events.subscribe()
    jid = jm.submit("op", {"name": "op.x", "inputs": {}})
    evs = drain(q, 3)
    types = [e["type"] for e in evs]
    check("failed 事件发布且带 error",
          types[-1] == "failed" and "炸了" in (evs[-1].get("error") or ""))

    # pending 中断：占住 worker，让目标任务排在队列里再取消
    import threading
    gate = threading.Event()
    def slow(kind, payload):
        gate.wait(5)
        return {}
    jm2 = make_manager(slow)
    jm2.submit("op", {"name": "op.slow", "inputs": {}})
    jid2 = jm2.submit("op", {"name": "op.x", "inputs": {}})
    time.sleep(0.3)  # 等 slow 占住 worker
    before = events.subscriber_count()
    q2 = events.subscribe()
    jm2.interrupt(jid2)
    evs2 = drain(q2, 1)
    events.unsubscribe(q2)
    gate.set()
    check("pending 取消立即发 cancelled 事件",
          evs2 and evs2[0]["type"] == "cancelled" and evs2[0]["job_id"] == jid2)
    check("订阅计数无泄漏", events.subscriber_count() == before)


# ---------- 3. 静态约束（main.py / monitor.html） ----------
MAIN = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
MONITOR = os.path.join(os.path.dirname(__file__), "..", "app", "static",
                       "monitor.html")


def test_static_constraints():
    src = open(MAIN, encoding="utf-8").read()
    check("main.py 注册 /ws", '@app.websocket("/ws")' in src)
    check("/ws 鉴权失败关 4401", "4401" in src)
    check("/ws finally 退订（防队列泄漏）",
          re.search(r"finally:[\s\S]{0,160}?events\.unsubscribe", src) is not None)
    check("/ws 30s 应用层 ping（泵线程超时生成）", '{"type": "ping"}' in src)
    check("/ws 单消费者泵（防 wait_for 取消泄漏吞事件，exec 824 事故）",
          "call_soon_threadsafe" in src
          and "wait_for(asyncio.to_thread" not in src)
    check("/ws 带 job_id 时先发 snapshot", '"type": "snapshot"' in src)
    check("job_id 过滤（无关事件不推）",
          'ev.get("job_id") not in (None, job_id)' in src)

    html = open(MONITOR, encoding="utf-8").read()
    check("监控页建立 WS 连接", "new WebSocket(" in html and "/ws?token=" in html)
    check("监控页保留轮询兜底（setInterval 仍在）", "setInterval(poll, POLL_MS)" in html)
    check("监控页 WS 事件节流刷新", "wsLastPush" in html)
    check("监控页断线重连（指数退避）", "scheduleWsRetry" in html)
    check("保存 token 后重连 WS", "reconnectWs()" in html)


def main():
    test_broadcaster()
    test_job_event_sequence()
    test_failed_and_cancelled_events()
    test_static_constraints()
    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()

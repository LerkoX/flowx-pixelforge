"""WS 事件广播（/ws 的推送源）：job 状态/进度/阶段事件的进程内 pub-sub。

为什么独立成模块：事件产生在 job worker 线程（jobs.py / engine 上报），
消费在 asyncio 事件循环（main.py 的 /ws 端点）——队列是两者之间唯一
安全的桥（queue.Queue 天然跨线程；asyncio.Queue 不是）。

纪律：
- 推送方绝不允许阻塞/抛错影响任务执行：put_nowait + 满队列丢事件
  （客户端有轮询兜底，丢事件只损失实时性，不损失正确性）。
- 订阅者持引用期间才会收到事件；/ws 断开必须 unsubscribe（finally），
  否则队列泄漏。
"""
import queue
import threading
import time

_MAXSIZE = 1000  # 单订阅者积压上限（30s 心跳兜底，正常远用不到）

_lock = threading.Lock()
_subs = []  # list[queue.Queue]


def publish(event_type, job_id=None, **fields):
    """向全部订阅者广播一个事件。任何线程可调；永抛不出异常。"""
    ev = {"type": event_type, "ts": round(time.time(), 3)}
    if job_id:
        ev["job_id"] = job_id
    ev.update(fields)
    with _lock:
        subs = list(_subs)
    for q in subs:
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass  # 慢订阅者丢事件：客户端轮询兜底（见模块 docstring）
        except Exception:
            pass


def subscribe():
    """注册一个订阅者，返回其事件队列。"""
    q = queue.Queue(maxsize=_MAXSIZE)
    with _lock:
        _subs.append(q)
    return q


def unsubscribe(q):
    """摘除订阅者（/ws 断开在 finally 里调用）。"""
    with _lock:
        if q in _subs:
            _subs.remove(q)


def subscriber_count():
    """观测用（测试/排障）：当前活跃订阅数。"""
    with _lock:
        return len(_subs)

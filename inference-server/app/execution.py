"""执行上下文：当前线程正在服务的异步任务（若有）。

job worker 线程执行前 bind(job)，引擎在节点间、采样循环在每步调用
check_cancelled() 协作式响应 /interrupt（对齐 ComfyUI 的取消语义：
只有执行检查点能感知取消，不强制杀线程）。

同步 /graph、/op 上下文无 job 绑定，check_cancelled() 为空操作，行为不变。
"""
import threading


class JobCancelled(Exception):
    """任务被取消：由取消检查点抛出，冒泡到 job worker 标记 cancelled。"""


_LOCAL = threading.local()


def bind(job):
    _LOCAL.job = job


def unbind():
    _LOCAL.job = None


def current():
    """当前线程绑定的 job（同步调用上下文为 None）。"""
    return getattr(_LOCAL, "job", None)


def check_cancelled():
    """取消检查点：当前 job 被取消则抛 JobCancelled。无 job 时为空操作。"""
    job = current()
    if job is not None and job.cancel_event.is_set():
        raise JobCancelled(f"job '{job.id}' cancelled")

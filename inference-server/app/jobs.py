"""异步任务体系：POST /jobs 提交 → job_id 轮询进度 → /interrupt 取消。

GPU 是串行资源：所有任务进 FIFO 队列，单个 worker 线程逐个执行（与同步
/graph、/op 经 engine.EXEC_LOCK 互斥）。取消通过 threading.Event +
执行检查点（engine 节点间 / sample 每步）协作式生效，不杀线程。

任务结果保留在内存，完成任务超过 retained 上限时按完成时间淘汰最旧。
"""
import queue
import threading
import time
import uuid

from .execution import JobCancelled, bind, unbind

MAX_RETAINED = 200  # 已完成任务保留上限


class Job:
    def __init__(self, kind, payload):
        self.id = uuid.uuid4().hex
        self.kind = kind            # "graph" | "op"
        self.payload = payload
        self.status = "pending"     # pending/running/done/failed/cancelled
        self.cancel_event = threading.Event()
        self.progress = {"current": 0, "total": 0}
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None

    def set_progress(self, current, total):
        self.progress = {"current": current, "total": total}

    def view(self):
        cur, tot = self.progress["current"], self.progress["total"]
        v = {"id": self.id, "kind": self.kind, "status": self.status,
             "progress": {**self.progress,
                          "percent": round(cur / tot * 100, 1) if tot else None},
             "created_at": self.created_at,
             "started_at": self.started_at,
             "finished_at": self.finished_at}
        if self.status == "done":
            v["result"] = self.result
        if self.error is not None:
            v["error"] = self.error
        return v


class JobManager:
    """run_fn(kind, payload) -> result：在 worker 线程内调用（由 main.py 注入，
    分派到 engine.run_graph / engine.run_op），本类不感知引擎细节。"""

    def __init__(self, run_fn, retained=MAX_RETAINED):
        self._run = run_fn
        self._retained = retained
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True,
                         name="job-worker").start()

    def submit(self, kind, payload) -> str:
        job = Job(kind, payload)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job)
        return job.id

    def get(self, job_id) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"job '{job_id}' not found")
        return job.view()

    def list(self):
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at)
        return [j.view() for j in jobs]

    def interrupt(self, job_id=None):
        """取消：指定 job_id 只取消它；否则取消当前 running + 全部 pending。
        running 任务置取消标志（检查点生效）；pending 任务直接标记 cancelled。"""
        with self._lock:
            if job_id is not None:
                job = self._jobs.get(job_id)
                if job is None:
                    raise KeyError(f"job '{job_id}' not found")
                targets = [job]
            else:
                targets = [j for j in self._jobs.values()
                           if j.status in ("pending", "running")]
        for job in targets:
            job.cancel_event.set()
            if job.status == "pending":
                job.status = "cancelled"
                job.error = "cancelled before start"
                job.finished_at = time.time()
        return {"interrupted": [j.id for j in targets]}

    def _worker(self):
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue  # 排队期间被取消，直接跳过
            job.status = "running"
            job.started_at = time.time()
            bind(job)
            try:
                job.result = self._run(job.kind, job.payload)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
                job.error = "cancelled"
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
            finally:
                unbind()
                job.finished_at = time.time()
                self._evict()

    def _evict(self):
        with self._lock:
            finished = [j for j in self._jobs.values()
                        if j.status in ("done", "failed", "cancelled")]
            if len(finished) <= self._retained:
                return
            finished.sort(key=lambda j: j.finished_at or 0)
            for j in finished[:len(finished) - self._retained]:
                del self._jobs[j.id]

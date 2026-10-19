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

from . import events
from .execution import JobCancelled, bind, unbind
from . import lease as _lease

MAX_RETAINED = 200  # 已完成任务保留上限


def _summarize(kind, payload):
    """任务内容摘要（监控页「任务队列」展示用，零成本：只读该 job 自己的 payload）。

    op 任务 title = 算子名；graph 任务 title = "graph · N 节点"，并给出去重算子
    清单（ops）。以前监控页只能看到 kind="graph"/"op"，无法分辨是哪个模型
    在跑——排障时要另开 /jobs/{id} 手查 result 才知道。
    """
    payload = payload or {}
    if kind == "op":
        name = payload.get("name") or "op"
        return {"title": name, "ops": [name], "nodes": 1}
    nodes = payload.get("nodes") or {}
    ops = []
    for spec in nodes.values():
        name = (spec or {}).get("op")
        if name and name not in ops:
            ops.append(name)
    return {"title": f"graph · {len(nodes)} 节点", "ops": ops, "nodes": len(nodes)}


class Job:
    def __init__(self, kind, payload, lease=None, lease_touch=None):
        self.id = uuid.uuid4().hex
        self.kind = kind            # "graph" | "op"
        self.payload = payload
        self.summary = _summarize(kind, payload)  # 展示用摘要（title/ops/nodes）
        self.lease = lease          # 执行租约（§43.2，X-FlowX-Lease 头；可空）
        # 续租回调（B2）：由 JobManager 注入（实际是 ModelManager.leases.touch，
        # 因为租约登记簿归 ModelManager 所有，本模块不持簿）
        self._lease_touch = lease_touch
        self.status = "pending"     # pending/running/done/failed/cancelled
        self.cancel_event = threading.Event()
        self.progress = {"current": 0, "total": 0}
        # 节点级阶段（engine.run_graph 每节点上报）：{"index","total","node","op"}。
        # 与 progress 互补——stage 说明「在哪一个算子」，progress 说明「走到第几步」：
        # 加载阶段 progress 恒为 0/0（无百分比），光看进度条会一直空着。
        self.stage = None
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None

    def set_stage(self, index, total, node_id=None, op_name=None):
        """节点级进度上报（engine 调用；同步 /graph 上下文无 job 时不会走到这里）。
        同时续租：节点间切换也是长任务的活性心跳。"""
        self.stage = {"index": index, "total": total,
                      "node": node_id, "op": op_name}
        events.publish("stage", self.id, index=index, total=total,
                       node=node_id, op=op_name)
        self._touch_lease()

    def _touch_lease(self):
        """续租（B2）：租约 TTL 默认 30min，而 30 步采样可能跑 90 分钟。"""
        if self.lease and self._lease_touch is not None:
            try:
                self._lease_touch(self.lease)
            except Exception:  # noqa: BLE001 - 续租失败不阻断任务
                pass

    def set_progress(self, current, total):
        self.progress = {"current": current, "total": total}
        events.publish("progress", self.id, current=current, total=total,
                       percent=round(current / total * 100, 1) if total else None)
        # 进度心跳兼作**续租**（B2）：租约 TTL 默认 30min，而 30 步采样可能跑
        # 90 分钟（exec 795 实测 5425s）。不续租的后果有两个：① 本执行的常驻
        # 条目失去租约保护被别的执行淘汰；② 对象仓库把它创建的对象当孤儿回收
        # （对象仓里的 vae/latent 视图一没，后续 VAEDecode 报 object not found）。
        self._touch_lease()

    def view(self):
        cur, tot = self.progress["current"], self.progress["total"]
        v = {"id": self.id, "kind": self.kind, "status": self.status,
             "title": self.summary["title"],
             "ops": self.summary["ops"][:6],
             "nodes": self.summary["nodes"],
             "progress": {**self.progress,
                          "percent": round(cur / tot * 100, 1) if tot else None},
             "stage": self.stage,
             # 排队时长（提交 → 开始执行）：判断是不是 GPU 被前一个任务占着
             "wait_seconds": (round(self.started_at - self.created_at, 1)
                              if self.started_at else None),
             "created_at": self.created_at,
             "started_at": self.started_at,
             "finished_at": self.finished_at}
        if self.lease:
            v["lease"] = self.lease
        if self.status == "done":
            v["result"] = self.result
        if self.error is not None:
            v["error"] = self.error
        return v


class JobManager:
    """run_fn(kind, payload) -> result：在 worker 线程内调用（由 main.py 注入，
    分派到 engine.run_graph / engine.run_op），本类不感知引擎细节。"""

    def __init__(self, run_fn, retained=MAX_RETAINED, lease_touch=None):
        self._run = run_fn
        self._retained = retained
        # 续租回调（B2）：main.py 注入 models.leases.touch，长任务进度心跳时续租
        self._lease_touch = lease_touch
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True,
                         name="job-worker").start()

    def submit(self, kind, payload, lease=None) -> str:
        job = Job(kind, payload, lease=lease, lease_touch=self._lease_touch)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job)
        events.publish("submitted", job.id, kind=kind,
                       title=job.summary["title"], nodes=job.summary["nodes"])
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
            views = [j.view() for j in jobs]
        # 排队位次（1 起）：pending 按提交顺序排名（running 占住 worker，也算一位）。
        # 监控页据此显示「排队 #2，前面还有 1 个在跑」——否则 pending 任务只能
        # 看到一个光秃秃的 pending 状态，看不出卡在哪里。
        pos = 0
        for v in views:
            if v["status"] in ("pending", "running"):
                pos += 1
                if v["status"] == "pending":
                    v["queue_pos"] = pos
        return views

    def queue_state(self):
        """队列整体状态（监控页头部与 ETA 展示）：worker 是否空闲、队头等了多久。"""
        now = time.time()
        with self._lock:
            jobs = list(self._jobs.values())
        running = [j for j in jobs if j.status == "running"]
        pending = sorted([j for j in jobs if j.status == "pending"],
                         key=lambda j: j.created_at)
        out = {"running": len(running), "pending": len(pending),
               "worker_busy": bool(running)}
        if running:
            j = running[0]
            out["running_id"] = j.id
            out["running_title"] = j.summary["title"]
            out["running_seconds"] = (round(now - j.started_at, 1)
                                      if j.started_at else None)
        if pending:
            out["head_wait_seconds"] = round(now - pending[0].created_at, 1)
        return out

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
                events.publish("cancelled", job.id,
                               error="cancelled before start")
        return {"interrupted": [j.id for j in targets]}

    def _worker(self):
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue  # 排队期间被取消，直接跳过
            job.status = "running"
            job.started_at = time.time()
            events.publish("running", job.id,
                           wait_seconds=round(job.started_at - job.created_at, 1))
            bind(job)
            _lease.bind(job.lease)
            try:
                job.result = self._run(job.kind, job.payload)
                job.status = "done"
                events.publish("done", job.id, result=job.result)
            except JobCancelled:
                job.status = "cancelled"
                job.error = "cancelled"
                events.publish("cancelled", job.id, error="cancelled")
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                events.publish("failed", job.id, error=job.error)
            finally:
                _lease.unbind()
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

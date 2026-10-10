"""执行租约 pin（dev-plan §43.2）：并发执行"排队而非互踩"。

问题（§21.8 事故）：两条 GPU 流水线并发时，B 加载新模型的显存淘汰会把 A
正在用/待消费的模型顶掉，A 的下一个算子报 object not found。

设计（方案经逐条确认，决策记录见 §43.3）：
- 标识：节点经 HTTP 头 `X-FlowX-Lease: <execution_id>` 携带执行标识
  （FLOWX_EXECUTION_ID 由 Studio 注入节点环境，flowx_client.py 统一加头）；
  不带头的调用方（验收脚本等）无租约，行为与租约机制引入前逐字节一致。
- 归属：租约名下加载的模型 + 算子执行中用到的模型（复用引擎 pin_objects
  识别）归入该租约；算子级 pin（ModelManager._pins）保留——防同租约内
  OOM 自愈误杀正在前向的模型。
- 保护：淘汰决策（含显式 unload）跳过**其它活跃租约**持有的条目；
  本租约自己的条目对自己可淘汰（同流水线内换模型不受影响）。
- 释放：TTL 滑动过期唯一途径（LEASE_TTL_SECONDS，默认 30min，每次租约
  活动刷新）——不依赖调用方收尾，节点崩溃/进程被杀/隧道断均自愈；
  显式 release 端点不做（无真实调用方，见决策 1）。
- 排队：加载需腾地方但可淘汰条目都被其它租约占着时，阻塞等待
  （LEASE_WAIT_TIMEOUT，默认 300s）周期重查；超时失败带 oom_help 风格
  指引。等待期间持模型锁＝后来的加载自然排队。
"""
import os
import threading
import time

LEASE_TTL = float(os.environ.get("LEASE_TTL_SECONDS", "1800"))
LEASE_WAIT_TIMEOUT = float(os.environ.get("LEASE_WAIT_TIMEOUT", "300"))
# 租约标识合法性上限（防畸形头撑爆内存；execution_id 是 uuid/短串）
MAX_LEASE_ID_LEN = 128

_local = threading.local()


def sanitize(lease_id):
    """请求头值 → 合法租约 id（非法/超长一律视为无租约）。"""
    if not lease_id:
        return None
    lid = str(lease_id).strip()
    if not lid or len(lid) > MAX_LEASE_ID_LEN:
        return None
    return lid


def bind(lease_id):
    """把当前线程归入租约（同步端点在请求线程、job worker 在执行线程调用）。"""
    _local.lease = sanitize(lease_id)


def unbind():
    _local.lease = None


def current():
    return getattr(_local, "lease", None)


class LeaseBook:
    """租约登记簿：lease_id -> {"keys": set[model_key], "seen": 最后活跃时间}。

    线程安全（自带锁）；登记/查询路径都不持 ModelManager._lock，
    等待循环里也能被其它线程正常刷新/过期。
    """

    def __init__(self, ttl=LEASE_TTL):
        self._ttl = float(ttl)
        self._leases = {}
        self._lock = threading.Lock()

    def touch(self, lease_id, keys=()):
        """租约活动：刷新活跃时间（滑动过期），并把 keys 归入该租约名下。"""
        lid = sanitize(lease_id)
        if lid is None:
            return
        with self._lock:
            e = self._leases.setdefault(lid, {"keys": set(), "seen": 0.0})
            e["seen"] = time.time()
            if keys:
                e["keys"].update(k for k in keys if k)

    def forget_keys(self, keys):
        """条目被淘汰/卸载后从所有租约名下摘除（簿记卫生，防虚保护）。"""
        if not keys:
            return
        with self._lock:
            for e in self._leases.values():
                e["keys"].difference_update(keys)

    def _sweep_locked(self, now):
        expired = [l for l, e in self._leases.items()
                   if now - e["seen"] > self._ttl]
        for l in expired:
            del self._leases[l]
        return expired

    def sweep(self):
        """清理过期租约（等待循环周期调用）；返回过期的 lease_id 列表。"""
        with self._lock:
            return self._sweep_locked(time.time())

    def keys_of_others(self, lease_id):
        """其它活跃租约持有的 key 集合（淘汰跳过）；顺带清扫过期租约。"""
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            out = set()
            for l, e in self._leases.items():
                if l != lease_id:
                    out |= e["keys"]
            return out

    def leases_holding(self, keys, exclude=None):
        """持有 keys 中任一条目的活跃租约 id 列表（等待超时指引用）。"""
        now = time.time()
        ks = set(keys)
        with self._lock:
            self._sweep_locked(now)
            return sorted(l for l, e in self._leases.items()
                          if l != exclude and e["keys"] & ks)

    def view(self):
        """观测视图（GET /leases）：活跃租约及其持有条目/空闲时长。"""
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            return [{"lease": l,
                     "keys": sorted(e["keys"]),
                     "idle_seconds": round(now - e["seen"], 1)}
                    for l, e in sorted(self._leases.items())]


def wait_help(others, waited):
    """租约等待超时的可执行指引（oom_help 风格）。"""
    return (f"加载等待超时（{waited:.0f}s）：显存被其它执行租约持有的模型占满，"
            f"并发执行排队未轮到。占用租约：{sorted(others)}。"
            f"建议：① 稍后重试（对端流水线结束后租约 TTL 到期自动释放）；"
            f"② 让对端流水线收尾加 model-unload 节点主动腾显存；"
            f"③ 调大 LEASE_WAIT_TIMEOUT 或错开两条流水线的启动时间；"
            f"④ 确认无并发后 POST /model/unload 清空常驻再跑")

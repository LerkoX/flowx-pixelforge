"""系统资源探测（psutil 可选依赖）：CPU/内存/swap/进程 RSS/磁盘占用。

torch 的显存账本由 /health 覆盖、GPU 硬件指标由 gpu_probe 经 NVML 覆盖；
本模块补它们都没有的「系统侧」视角：RAM 与 swap（model offload 的权重全躺在
进程内存里，RAM 吃紧 → swap → 采样速度雪崩）、推理进程自身 RSS、模型盘余量。

psutil 是可选依赖：缺失或读取失败时 snapshot() 返回 available=False + 全 None
字段，监控页对应区块降级显示「—」，核心推理路径不受影响（本模块只在指标
采样时被调用）。

容器化口径注意：Docker Desktop（WSL2）里 psutil 读到的是 WSL2 虚拟机的
内存/CPU，不是 Windows 物理机——监控页已标注「WSL2 VM 口径」。
cpu_percent 用 interval=None 非阻塞模式：首次调用返回无意义值，由常驻采样
线程（metrics_history，每 2s 一帧）持续调用后即正常。
"""
import os
import threading

_lock = threading.Lock()
_psutil = None    # 初始化成功后缓存的 psutil 模块
_state = None     # None=未探测, True=可用, False=不可用（只探测一次）
_reason = ""      # 不可用原因（透出给监控页诊断）
_proc = None      # 本进程句柄缓存（RSS 采样用）


def _reset():
    """清初始化缓存（测试用；生产路径只探测一次）。"""
    global _psutil, _state, _reason, _proc
    with _lock:
        _psutil, _state, _reason, _proc = None, None, "", None


def _init():
    """惰性初始化 psutil；返回 (psutil_module|None, ok)。"""
    global _psutil, _state, _reason, _proc
    if _state is not None:
        return _psutil, _state
    with _lock:
        if _state is not None:
            return _psutil, _state
        try:
            import psutil
            _safe(psutil.cpu_percent, None)  # 预热：丢掉首次的无意义读数
            _proc = psutil.Process(os.getpid())
            _psutil, _state = psutil, True
        except Exception as e:
            _state, _reason = False, f"{type(e).__name__}: {e}"
    return _psutil, _state


def _safe(fn, *args):
    """单项指标采集容错：任一项读不到返回 None。"""
    try:
        return fn(*args)
    except Exception:
        return None


def snapshot(disk_paths=()):
    """一帧系统快照（metrics_history 采样器 / 监控页的数据源）。

    disk_paths: [(label, path), ...]——对每个路径取 disk_usage（如模型盘）；
    任一字段读不到为 None，不拖垮整帧。bytes→MB 取整。
    """
    ps, ok = _init()
    if not ok:
        return {"available": False, "reason": _reason}
    vm = _safe(ps.virtual_memory)
    sw = _safe(ps.swap_memory)
    out = {
        "available": True,
        "reason": "",
        "cpu_percent": _safe(ps.cpu_percent, None),  # interval=None 非阻塞
        "ram_used_mb": round(vm.used / 1024**2) if vm else None,
        "ram_total_mb": round(vm.total / 1024**2) if vm else None,
        "ram_percent": getattr(vm, "percent", None) if vm else None,
        "swap_used_mb": round(sw.used / 1024**2) if sw else None,
        "swap_total_mb": round(sw.total / 1024**2) if sw else None,
        "swap_percent": getattr(sw, "percent", None) if sw else None,
        # 本进程 RSS：model offload 的权重宿主，offload 压力的第一手指标
        "proc_rss_mb": _safe(lambda: round(_proc.memory_info().rss / 1024**2)),
        "disks": [],
    }
    for label, path in disk_paths or ():
        du = _safe(ps.disk_usage, path)
        out["disks"].append({
            "label": label, "path": path,
            "used_mb": round(du.used / 1024**2) if du else None,
            "free_mb": round(du.free / 1024**2) if du else None,
            "total_mb": round(du.total / 1024**2) if du else None,
            "percent": getattr(du, "percent", None) if du else None,
        })
    return out

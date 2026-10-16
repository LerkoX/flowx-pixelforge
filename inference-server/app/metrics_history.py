"""指标历史：服务端后台采样 + 环形缓存（监控页图表的数据源）。

之前监控页的历史曲线由前端内存自攒（刷新页面即清零，且页面没开就没人采）。
本模块把采样挪到服务端：常驻线程每 INTERVAL 秒采一帧（GPU/NVML + 系统/psutil
+ torch 显存账本 + 任务队列 + 对象仓库），追加到环形缓冲（默认 6h@2s=10800 帧，
每帧数百字节，内存可忽略），GET /metrics/history?minutes=N 回放窗口。
进程重启历史清零（不落盘——监控数据不值得持久化）。

容错纪律：单个采集器（NVML/psutil/torch）失败只让对应字段为 None，
采样线程任何异常都不退出；torch 在函数内惰性 import（本模块保持纯 stdlib
可导入，测试环境无 torch 也能跑）。

sec_per_step：对 running 任务的 progress 做相邻帧差分（EMA 平滑），
「每步多少秒」直接可见（75s/步 这种 offload 瓶颈数字不用掐表算）。
"""
import threading
import time
from collections import deque

from . import gpu_probe, sys_probe

INTERVAL = 2.0              # 采样间隔（秒）
RETAIN_SECONDS = 6 * 3600   # 保留窗口（默认 6 小时）

_frames = deque(maxlen=int(RETAIN_SECONDS / INTERVAL))
_lock = threading.Lock()
_thread = None
_started = False

# 由 start() 注入的依赖（main.py 装配）；测试可直接替换
_store = None
_jobman = None
_disk_paths = ()

# sec_per_step 差分状态：上一帧 running job 的 (job_id, current, t) 与 EMA
_last_prog = None
_step_ema = None

_EMA_ALPHA = 0.4  # 瞬时 sec/step 的 EMA 权重（约 2~3 帧收敛）


def _torch_vram():
    """torch 显存账本快照；torch 不可用/CUDA 不可用时返回空 dict（字段为 None）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        return {
            "vram_alloc_mb": round(allocated / 1024**2),
            # reclaimable = 分配器缓存保留（可回收），与 /health 口径一致
            "vram_cache_mb": round(max(0, reserved - allocated) / 1024**2),
            "vram_driver_free_mb": round(free / 1024**2),
            "vram_torch_total_mb": round(total / 1024**2),
        }
    except Exception:
        return {}


def _jobs_view():
    """任务队列快照：running/pending 计数 + 当前 running job（取最早开始的）。"""
    if _jobman is None:
        return {}
    try:
        jobs = _jobman.list()
        running = [j for j in jobs if j.get("status") == "running"]
        pending = [j for j in jobs if j.get("status") == "pending"]
        out = {"jobs_running": len(running), "jobs_pending": len(pending)}
        if running:
            j = running[0]
            p = j.get("progress") or {}
            out.update({
                "run_job": j.get("id"),
                "run_current": p.get("current"),
                "run_total": p.get("total"),
                "run_percent": p.get("percent"),
            })
        return out
    except Exception:
        return {}


def _sec_per_step(frame, now):
    """running job 进度差分 → 每步秒数（EMA 平滑）；无进度变化时沿用上一值。"""
    global _last_prog, _step_ema
    jid, cur = frame.get("run_job"), frame.get("run_current")
    if jid is None or cur is None:
        _last_prog, _step_ema = None, None
        return None
    if _last_prog and _last_prog[0] == jid and cur > _last_prog[1]:
        inst = (now - _last_prog[2]) / (cur - _last_prog[1])
        _step_ema = inst if _step_ema is None else \
            (1 - _EMA_ALPHA) * _step_ema + _EMA_ALPHA * inst
    if _last_prog is None or _last_prog[0] != jid or cur != _last_prog[1]:
        _last_prog = (jid, cur, now)
    return round(_step_ema, 2) if _step_ema is not None else None


def _sample(now=None):
    """采一帧追加到环形缓冲（独立可测：采集器全部容错）。"""
    now = now if now is not None else time.time()
    frame = {"t": now}

    try:
        g = gpu_probe.snapshot()
        gpu0 = g["gpus"][0] if g.get("available") and g.get("gpus") else None
        if gpu0:
            frame.update({
                "gpu_util": gpu0.get("utilization_percent"),
                "gpu_memctrl": gpu0.get("mem_controller_percent"),
                "gpu_temp": gpu0.get("temperature_c"),
                "gpu_power_w": gpu0.get("power_w"),
                "gpu_fan": gpu0.get("fan_percent"),
                "gpu_sm_clock": gpu0.get("sm_clock_mhz"),
                "gpu_mem_clock": gpu0.get("mem_clock_mhz"),
                "pcie_tx_kbps": gpu0.get("pcie_tx_kbps"),
                "pcie_rx_kbps": gpu0.get("pcie_rx_kbps"),
                "throttle": gpu0.get("throttle_reasons"),
                "vram_used_mb": gpu0.get("vram_used_mb"),
                "vram_total_mb": gpu0.get("vram_total_mb"),
            })
    except Exception:
        pass

    try:
        s = sys_probe.snapshot(_disk_paths)
        if s.get("available"):
            frame.update({
                "cpu_pct": s.get("cpu_percent"),
                "ram_pct": s.get("ram_percent"),
                "ram_used_mb": s.get("ram_used_mb"),
                "ram_total_mb": s.get("ram_total_mb"),
                "swap_pct": s.get("swap_percent"),
                "swap_used_mb": s.get("swap_used_mb"),
                "proc_rss_mb": s.get("proc_rss_mb"),
            })
            if s.get("disks"):
                d = s["disks"][0]
                frame.update({"disk_label": d.get("label"),
                              "disk_free_mb": d.get("free_mb"),
                              "disk_pct": d.get("percent")})
    except Exception:
        pass

    frame.update(_torch_vram())
    frame.update(_jobs_view())

    if _store is not None:
        try:
            frame["store_objects"] = _store.stats().get("count")
        except Exception:
            pass

    frame["sec_per_step"] = _sec_per_step(frame, now)

    with _lock:
        _frames.append(frame)
    return frame


def _loop():
    while True:
        try:
            _sample()
        except Exception:
            pass  # 采样线程永不退出：单帧失败只丢一帧
        time.sleep(INTERVAL)


def start(store=None, jobman=None, disk_paths=()):
    """装配依赖并启动采样线程（幂等，重复调用只更新依赖不重复开线程）。"""
    global _store, _jobman, _disk_paths, _thread, _started
    _store, _jobman, _disk_paths = store, jobman, tuple(disk_paths or ())
    if _started:
        return
    _started = True
    _thread = threading.Thread(target=_loop, daemon=True, name="metrics-sampler")
    _thread.start()


def stop():
    """测试用：重置线程标志（daemon 线程随进程退出；本函数只清状态）。"""
    global _started, _thread, _last_prog, _step_ema
    _started, _thread = False, None
    _last_prog, _step_ema = None, None


def reset_frames():
    """测试用：清空历史缓冲。"""
    with _lock:
        _frames.clear()


def history(minutes=60):
    """回放窗口：返回最近 minutes 分钟的帧（升序）。

    minutes=None 或 <=0 返回全部。响应带采样间隔，前端按它对齐时间轴。
    """
    with _lock:
        frames = list(_frames)
    if minutes and minutes > 0:
        cutoff = time.time() - minutes * 60
        frames = [f for f in frames if f["t"] >= cutoff]
    return {"interval": INTERVAL, "retain_seconds": RETAIN_SECONDS,
            "frames": frames}

"""GPU 硬件指标探测（pynvml 可选依赖）：利用率/温度/功耗/风扇。

torch.cuda 只能给显存账本（/health 已覆盖）；SM 利用率、显存控制器利用率、
温度、功耗、风扇转速要经 NVML。pynvml 是可选依赖：缺失或驱动无 NVML 时
snapshot() 返回 available=False + 空列表，监控页降级为只画显存区，核心推理
路径完全不受影响（本模块只在 GET /gpu 被调用时才初始化）。

指标按卡返回列表（gpus[]），多卡机型直接铺开，单卡取 gpus[0]。
每次调用即时采样不做缓存（NVML 查询本身微秒级）；单个传感器读不到时
该字段为 None，不拖垮整帧（无风扇卡/老驱动常见）。
"""
import threading

_lock = threading.Lock()
_nvml = None    # 初始化成功后缓存的 pynvml 模块
_state = None   # None=未探测, True=可用, False=不可用（只探测一次，避免每请求重试报错）
_reason = ""    # 不可用原因（透出给监控页诊断）


def _reset():
    """清初始化缓存（测试用；生产路径只探测一次）。"""
    global _nvml, _state, _reason
    with _lock:
        _nvml, _state, _reason = None, None, ""


def _init():
    """惰性初始化 NVML；返回 (nvml_module|None, ok)。"""
    global _nvml, _state, _reason
    if _state is not None:
        return _nvml, _state
    with _lock:
        if _state is not None:
            return _nvml, _state
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml, _state = pynvml, True
        except Exception as e:
            _state, _reason = False, f"{type(e).__name__}: {e}"
    return _nvml, _state


def _safe(fn, *args):
    """单项指标采集容错：任一传感器读不到返回 None。"""
    try:
        return fn(*args)
    except Exception:
        return None


def _text(v):
    """pynvml<12 的 nvmlDeviceGetName 返回 bytes，12+ 返回 str，统一成 str。"""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def snapshot():
    """一帧 GPU 硬件快照（GET /gpu 的数据源）。

    返回 {"available": bool, "reason": str, "gpus": [...]}；
    功率 mW→W 保留 1 位小数，显存 bytes→MB 取整，指标读不到为 None。
    """
    nvml, ok = _init()
    if not ok:
        return {"available": False, "reason": _reason, "gpus": []}
    try:
        count = nvml.nvmlDeviceGetCount()
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}",
                "gpus": []}
    gpus = []
    for i in range(count):
        h = _safe(nvml.nvmlDeviceGetHandleByIndex, i)
        if h is None:
            continue
        util = _safe(nvml.nvmlDeviceGetUtilizationRates, h)
        mem = _safe(nvml.nvmlDeviceGetMemoryInfo, h)
        power = _safe(nvml.nvmlDeviceGetPowerUsage, h)            # mW
        limit = _safe(nvml.nvmlDeviceGetPowerManagementLimit, h)  # mW
        gpus.append({
            "index": i,
            "name": _text(_safe(nvml.nvmlDeviceGetName, h)),
            "utilization_percent": getattr(util, "gpu", None),
            "mem_controller_percent": getattr(util, "memory", None),
            "temperature_c": _safe(nvml.nvmlDeviceGetTemperature, h, 0),  # 0=GPU 核心
            "power_w": round(power / 1000, 1) if power is not None else None,
            "power_limit_w": round(limit / 1000, 1) if limit is not None else None,
            # 无风扇/水冷卡 nvmlDeviceGetFanSpeed 抛错 → None（_safe 归一）
            "fan_percent": _safe(nvml.nvmlDeviceGetFanSpeed, h),
            # NVML 口径显存（整卡，含其它进程）；torch 口径账本看 /health
            "vram_used_mb": round(mem.used / 1024**2) if mem is not None else None,
            "vram_total_mb": round(mem.total / 1024**2) if mem is not None else None,
        })
    return {"available": True, "reason": "", "gpus": gpus}

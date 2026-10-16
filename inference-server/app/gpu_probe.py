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


def _missing(*_a, **_kw):
    raise RuntimeError("pynvml 缺该接口")


def _fn(nvml, name):
    """按名取 NVML 接口，老版本缺接口时给一个必抛错的替身（经 _safe 归一为 None）。
    必须经此层中转：直接把 nvml.xxx 当参数求值会在 _safe 之外抛 AttributeError。"""
    return getattr(nvml, name, _missing)


def _text(v):
    """pynvml<12 的 nvmlDeviceGetName 返回 bytes，12+ 返回 str，统一成 str。"""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


# 降频原因位掩码（NVML 常量在不同 pynvml 版本命名不一，按位值硬解码，
# 与 nvmlClocksThrottleReason* 官方位定义一致）；读出 bitmask 后翻译成名称列表
_THROTTLE_BITS = [
    (0x01, "gpu_idle"),            # 空闲降频（正常，不算异常）
    (0x02, "app_clocks"),          # 应用时钟设定
    (0x04, "sw_power_cap"),        # 撞功耗墙
    (0x08, "hw_slowdown"),         # 硬件减速（过热/供电）
    (0x10, "sync_boost"),
    (0x20, "sw_thermal"),          # 软件热降频
    (0x40, "hw_thermal"),          # 硬件热降频
    (0x80, "hw_power_brake"),      # 供电刹车
]


def _throttle_names(mask):
    """降频位掩码 → 原因名列表；None 透传（读不到时）。"""
    if mask is None:
        return None
    return [name for bit, name in _THROTTLE_BITS if mask & bit]


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
        # PCIe 吞吐量（KB/s）：model offload 每步经 PCIe 搬数 GB 权重，
        # 「SM 100% 但功耗低」的搬运瓶颈用这个指标直接实锤
        pcie_tx = _safe(_fn(nvml, "nvmlDeviceGetPcieThroughput"), h,
                        getattr(nvml, "NVML_PCIE_UTIL_TX_BYTES", 1))
        pcie_rx = _safe(_fn(nvml, "nvmlDeviceGetPcieThroughput"), h,
                        getattr(nvml, "NVML_PCIE_UTIL_RX_BYTES", 0))
        # 时钟频率（MHz）+ 降频原因：区分「算力跑满」还是「被功耗/温度墙限频」
        sm_clock = _safe(_fn(nvml, "nvmlDeviceGetClockInfo"), h,
                         getattr(nvml, "NVML_CLOCK_SM", 1))
        mem_clock = _safe(_fn(nvml, "nvmlDeviceGetClockInfo"), h,
                          getattr(nvml, "NVML_CLOCK_MEM", 2))
        sm_clock_max = _safe(_fn(nvml, "nvmlDeviceGetMaxClockInfo"), h,
                             getattr(nvml, "NVML_CLOCK_SM", 1))
        mem_clock_max = _safe(_fn(nvml, "nvmlDeviceGetMaxClockInfo"), h,
                              getattr(nvml, "NVML_CLOCK_MEM", 2))
        throttle = _safe(_fn(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons"), h)
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
            "pcie_tx_kbps": pcie_tx,
            "pcie_rx_kbps": pcie_rx,
            "sm_clock_mhz": sm_clock,
            "mem_clock_mhz": mem_clock,
            "sm_clock_max_mhz": sm_clock_max,
            "mem_clock_max_mhz": mem_clock_max,
            "throttle_reasons": _throttle_names(throttle),
            # NVML 口径显存（整卡，含其它进程）；torch 口径账本看 /health
            "vram_used_mb": round(mem.used / 1024**2) if mem is not None else None,
            "vram_total_mb": round(mem.total / 1024**2) if mem is not None else None,
        })
    return {"available": True, "reason": "", "gpus": gpus}

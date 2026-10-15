"""GPU 硬件指标探测（app/gpu_probe.py）测试。

无需 GPU、无需真 pynvml：把假 pynvml 模块塞进 sys.modules 验证正常帧、
单位换算（mW→W / bytes→MB）、pynvml<12 的 bytes 设备名解码、单传感器
容错（读不到归 None 不拖垮整帧）、依赖缺失/驱动异常时 available=False 降级。

运行：python3 tests/test_gpu_probe.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import gpu_probe  # noqa: E402

GB = 1024 ** 3


class _Mem:
    def __init__(self, used, total):
        self.used, self.total = used, total


class _Util:
    def __init__(self, gpu, memory):
        self.gpu, self.memory = gpu, memory


def make_fake_nvml(name=b"Fake RTX 4090", fan=55, fail_sensors=()):
    """构造假 pynvml 模块；fail_sensors 列出要抛错的传感器函数名。"""
    m = types.SimpleNamespace()
    m.inits = 0

    def nvmlInit():
        m.inits += 1
    m.nvmlInit = nvmlInit
    m.nvmlDeviceGetCount = lambda: 1
    m.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"
    m.nvmlDeviceGetName = lambda h: name
    m.nvmlDeviceGetUtilizationRates = lambda h: _Util(gpu=87, memory=42)
    m.nvmlDeviceGetMemoryInfo = lambda h: _Mem(used=18 * GB, total=24 * GB)
    m.nvmlDeviceGetTemperature = lambda h, sensor: 66
    m.nvmlDeviceGetPowerUsage = lambda h: 305_500          # mW
    m.nvmlDeviceGetPowerManagementLimit = lambda h: 450_000  # mW
    m.nvmlDeviceGetFanSpeed = lambda h: fan
    for fn in fail_sensors:
        def boom(*a, _fn=fn):
            raise RuntimeError(f"{_fn} not supported")
        setattr(m, fn, boom)
    return m


def setup(fake=None):
    """注入假 pynvml（None=模拟未安装）并复位探测缓存。"""
    if fake is None:
        sys.modules.pop("pynvml", None)
        sys.modules["pynvml"] = None  # import 机制遇到 None 视为不可用
    else:
        sys.modules["pynvml"] = fake
    gpu_probe._reset()


def teardown():
    sys.modules.pop("pynvml", None)
    gpu_probe._reset()


def test_snapshot_full_frame():
    fake = make_fake_nvml()
    setup(fake)
    s = gpu_probe.snapshot()
    assert s["available"] is True and s["reason"] == ""
    g = s["gpus"][0]
    assert g["index"] == 0
    assert g["name"] == "Fake RTX 4090"          # bytes 设备名已解码
    assert g["utilization_percent"] == 87
    assert g["mem_controller_percent"] == 42
    assert g["temperature_c"] == 66
    assert g["power_w"] == 305.5                  # mW → W
    assert g["power_limit_w"] == 450.0
    assert g["fan_percent"] == 55
    assert g["vram_used_mb"] == 18 * 1024         # bytes → MB
    assert g["vram_total_mb"] == 24 * 1024


def test_init_runs_once():
    fake = make_fake_nvml()
    setup(fake)
    gpu_probe.snapshot()
    gpu_probe.snapshot()
    assert fake.inits == 1  # 探测缓存：失败/成功都只初始化一次


def test_sensor_failure_degrades_to_none():
    # 无风扇卡：风扇读不到归 None，其余字段正常
    setup(make_fake_nvml(fail_sensors=("nvmlDeviceGetFanSpeed",)))
    g = gpu_probe.snapshot()["gpus"][0]
    assert g["fan_percent"] is None
    assert g["utilization_percent"] == 87
    assert g["power_w"] == 305.5


def test_power_failure_degrades_to_none():
    setup(make_fake_nvml(fail_sensors=("nvmlDeviceGetPowerUsage",
                                       "nvmlDeviceGetPowerManagementLimit")))
    g = gpu_probe.snapshot()["gpus"][0]
    assert g["power_w"] is None and g["power_limit_w"] is None
    assert g["temperature_c"] == 66


def test_pynvml_missing():
    setup(None)
    s = gpu_probe.snapshot()
    assert s["available"] is False and s["gpus"] == []
    assert s["reason"]  # 透出不可用原因供监控页诊断


def test_nvml_init_failure():
    fake = make_fake_nvml()

    def boom():
        raise RuntimeError("NVML driver not loaded")
    fake.nvmlInit = boom
    setup(fake)
    s = gpu_probe.snapshot()
    assert s["available"] is False and "NVML driver not loaded" in s["reason"]
    # 失败后同样只探测一次（避免每请求重试刷日志）
    assert gpu_probe.snapshot()["available"] is False


def test_str_name_passthrough():
    # pynvml>=12 的 nvmlDeviceGetName 返回 str，不再解码
    setup(make_fake_nvml(name="RTX 4090"))
    assert gpu_probe.snapshot()["gpus"][0]["name"] == "RTX 4090"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
            print(f"PASS {t.__name__}")
    finally:
        teardown()
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()

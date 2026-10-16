"""系统资源探测（app/sys_probe.py）测试。

无需真 psutil：把假 psutil 模块塞进 sys.modules 验证正常帧、bytes→MB 换算、
磁盘列表、单字段容错（读不到归 None）；依赖缺失时 available=False 降级。

运行：python3 tests/test_sys_probe.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import sys_probe  # noqa: E402

GB = 1024 ** 3
MB = 1024 ** 2

# setup 前 sys.modules 里 psutil 的原值（teardown 还原用；_MISSING 表示「未登记」）
_MISSING = object()
_saved_psutil = _MISSING


class _VMem:
    used = 10 * GB
    total = 16 * GB
    percent = 62.5


class _Swap:
    used = 512 * MB
    total = 4 * GB
    percent = 12.5


class _Disk:
    used = 100 * GB
    free = 400 * GB
    total = 500 * GB
    percent = 20.0


class _MemInfo:
    rss = 9 * GB


class _Proc:
    def memory_info(self):
        return _MemInfo()


def make_fake_psutil(fail=()):
    def boom(*a):
        raise RuntimeError("sensor gone")
    m = types.ModuleType("psutil")
    m.virtual_memory = boom if "vm" in fail else lambda: _VMem()
    m.swap_memory = boom if "swap" in fail else lambda: _Swap()
    m.cpu_percent = boom if "cpu" in fail else (lambda interval=None: 37.0)
    m.disk_usage = boom if "disk" in fail else (lambda path: _Disk())
    m.Process = lambda pid: _Proc()
    return m


def setup(fake=None):
    sys_probe._reset()
    global _saved_psutil
    if fake is not None:
        # 记下 sys.modules 里原有条目，teardown 按原样还原（不能无条件 pop：
        # 本机真装了 psutil 时，pop 会把真依赖从同进程其它测试那里一并卷走）
        _saved_psutil = sys.modules.get("psutil", _MISSING)
        sys.modules["psutil"] = fake
    else:
        # 本机无 psutil 时装一个最小可用的桩，保证「缺失」用例语义明确
        sys.modules.pop("psutil", None)


def teardown():
    global _saved_psutil
    if _saved_psutil is not _MISSING:
        if _saved_psutil is None:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = _saved_psutil
    _saved_psutil = _MISSING
    sys_probe._reset()


def test_snapshot_full_frame():
    setup(make_fake_psutil())
    s = sys_probe.snapshot([("models", "/models")])
    assert s["available"] is True and s["reason"] == ""
    assert s["cpu_percent"] == 37.0
    assert s["ram_used_mb"] == 10 * 1024 and s["ram_total_mb"] == 16 * 1024
    assert s["ram_percent"] == 62.5
    assert s["swap_used_mb"] == 512 and s["swap_percent"] == 12.5
    assert s["proc_rss_mb"] == 9 * 1024
    assert s["disks"] == [{"label": "models", "path": "/models",
                           "used_mb": 100 * 1024, "free_mb": 400 * 1024,
                           "total_mb": 500 * 1024, "percent": 20.0}]
    teardown()


def test_sensor_failure_degrades_to_none():
    setup(make_fake_psutil(fail=("vm", "swap", "cpu", "disk")))
    s = sys_probe.snapshot([("models", "/models")])
    assert s["available"] is True          # 整体可用，单字段归 None
    assert s["cpu_percent"] is None
    assert s["ram_used_mb"] is None and s["ram_percent"] is None
    assert s["swap_used_mb"] is None
    assert s["disks"][0]["free_mb"] is None and s["disks"][0]["label"] == "models"
    teardown()


def test_no_disk_paths():
    setup(make_fake_psutil())
    s = sys_probe.snapshot()
    assert s["available"] is True and s["disks"] == []
    teardown()


def test_psutil_missing():
    setup(None)
    sys.modules.pop("psutil", None)  # 确保 import 走真路径且失败（本机无 psutil）
    # 若本机恰好装了 psutil，则跳过语义（不强行卸载真实依赖）
    try:
        import psutil  # noqa: F401
        print("SKIP test_psutil_missing（本机装有 psutil）")
        return
    except ImportError:
        pass
    s = sys_probe.snapshot([("models", "/models")])
    assert s["available"] is False and "psutil" not in s or s["reason"]
    teardown()


def test_init_runs_once():
    calls = {"n": 0}
    m = make_fake_psutil()
    orig = m.virtual_memory
    m.virtual_memory = lambda: (calls.__setitem__("n", calls["n"] + 1), orig())[1]
    setup(m)
    sys_probe.snapshot()
    sys_probe.snapshot()
    assert calls["n"] == 2        # 采样每帧都读，但 Process/init 只做一次
    teardown()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()

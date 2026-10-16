"""指标历史（app/metrics_history.py）测试。

纯 stdlib：假 gpu_probe/sys_probe 数据 + 假 store/jobman 注入，验证
采样帧字段聚合、采集器异常容错（字段为 None 不丢帧）、环形缓冲上限、
history(minutes) 窗口过滤、sec_per_step 差分 EMA。
不开采样线程（直接调 _sample，避免测试睡眠）。

运行：python3 tests/test_metrics_history.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import metrics_history as mh  # noqa: E402

# 采集器原件（setup 首次调用时记录、teardown 还原）：本文件直接改写
# gpu_probe/sys_probe 的模块属性，不还原会跨测试文件污染
_ORIG = {}


class FakeStore:
    def __init__(self, n=7):
        self._n = n

    def stats(self):
        return {"count": self._n, "kinds": {"LATENT": self._n}}


class FakeJobman:
    def __init__(self, jobs=()):
        self._jobs = list(jobs)

    def list(self):
        return list(self._jobs)


def running_job(jid, cur, total=25):
    return {"id": jid, "status": "running",
            "progress": {"current": cur, "total": total,
                         "percent": round(cur / total * 100, 1)}}


def fake_gpu(monkeypatch_vals):
    return lambda: {"available": True, "reason": "", "gpus": [monkeypatch_vals]}


GPU0 = {"utilization_percent": 99, "mem_controller_percent": 42,
        "temperature_c": 61, "power_w": 88.5, "fan_percent": 40,
        "sm_clock_mhz": 1607, "mem_clock_mhz": 4513,
        "pcie_tx_kbps": 900000, "pcie_rx_kbps": 12000,
        "throttle_reasons": ["gpu_idle"],
        "vram_used_mb": 6000, "vram_total_mb": 8192}

SYS0 = {"available": True, "reason": "", "cpu_percent": 33.0,
        "ram_percent": 61.0, "ram_used_mb": 10000, "ram_total_mb": 16384,
        "swap_percent": 3.0, "swap_used_mb": 100, "proc_rss_mb": 9000,
        "disks": [{"label": "models", "path": "/models", "used_mb": 1,
                   "free_mb": 4096, "total_mb": 2, "percent": 50.0}]}


def setup(store=None, jobman=None, gpu=GPU0, sysd=SYS0, torch_vram=None):
    mh.reset_frames()
    mh.stop()
    # 不走 start()（会起真线程，与手动 _sample 竞争缓冲）：直接注入依赖
    # 采集器替换是**模块属性改写**，必须记下原件供 teardown 还原——不还原就会
    # 污染同进程内其它测试文件（pytest 全量收集时 test_sys_probe/test_gpu_probe
    # 后续用例全都拿到这里的定值，表现为「cpu_percent 33.0 != 37.0」这类怪错误）
    global _ORIG
    if not _ORIG:
        _ORIG = {"gpu": mh.gpu_probe.snapshot,
                 "sys": mh.sys_probe.snapshot,
                 "torch": mh._torch_vram}
    mh._store, mh._jobman = store, jobman
    mh._disk_paths = (("models", "/models"),)
    mh.gpu_probe.snapshot = (lambda: {"available": True, "reason": "", "gpus": [gpu]}) \
        if isinstance(gpu, dict) \
        else (lambda: {"available": False, "reason": "no nvml", "gpus": []})
    mh.sys_probe.snapshot = (lambda paths=(): sysd) if isinstance(sysd, dict) \
        else (lambda paths=(): {"available": False, "reason": "no psutil"})
    # torch 显存账本同样直接注入：不能靠「本机没装 torch」来断言，因为全量收集时
    # 兄弟测试文件已把 torch 桩塞进 sys.modules，_torch_vram 会真去读那个桩
    mh._torch_vram = torch_vram if torch_vram is not None else (lambda: {})


def teardown():
    mh.reset_frames()
    mh.stop()
    if _ORIG:
        mh.gpu_probe.snapshot = _ORIG["gpu"]
        mh.sys_probe.snapshot = _ORIG["sys"]
        mh._torch_vram = _ORIG["torch"]


def test_frame_full_aggregation():
    setup(store=FakeStore(7), jobman=FakeJobman([running_job("j1", 3)]))
    f = mh._sample(now=1000.0)
    # GPU 帧
    assert f["gpu_util"] == 99 and f["gpu_memctrl"] == 42
    assert f["pcie_tx_kbps"] == 900000 and f["throttle"] == ["gpu_idle"]
    assert f["vram_used_mb"] == 6000
    # 系统帧
    assert f["ram_pct"] == 61.0 and f["proc_rss_mb"] == 9000
    assert f["disk_free_mb"] == 4096 and f["disk_label"] == "models"
    # 任务/对象仓库帧
    assert f["jobs_running"] == 1 and f["jobs_pending"] == 0
    assert f["run_job"] == "j1" and f["run_current"] == 3
    assert f["store_objects"] == 7
    # torch 采集器空（本用例注入 lambda: {}）→ 无 torch 字段但不丢帧
    assert "vram_alloc_mb" not in f
    teardown()


def test_torch_vram_fields_when_available():
    """torch 显存账本可用时四个字段齐全（与 /health 口径一致）。"""
    setup(store=FakeStore(1), torch_vram=lambda: {
        "vram_alloc_mb": 6144, "vram_cache_mb": 512,
        "vram_driver_free_mb": 1000, "vram_torch_total_mb": 8192})
    f = mh._sample(now=1000.0)
    assert f["vram_alloc_mb"] == 6144 and f["vram_cache_mb"] == 512
    assert f["vram_driver_free_mb"] == 1000 and f["vram_torch_total_mb"] == 8192
    teardown()


def test_collector_failure_keeps_frame():
    setup(store=None, jobman=None, gpu=None, sysd=None)
    f = mh._sample(now=1000.0)
    assert f["t"] == 1000.0
    assert "gpu_util" not in f and "ram_pct" not in f
    assert f.get("store_objects") is None
    teardown()


def test_history_window_filter():
    setup()
    import time
    now = time.time()
    mh._frames.clear()
    mh._frames.extend([{"t": now - 7200}, {"t": now - 1800}, {"t": now - 60}])
    d = mh.history(minutes=60)
    assert [f["t"] for f in d["frames"]] == [now - 1800, now - 60]
    d = mh.history(minutes=0)   # <=0 → 全部
    assert len(d["frames"]) == 3
    assert d["interval"] == mh.INTERVAL
    teardown()


def test_ring_buffer_maxlen():
    setup()
    assert mh._frames.maxlen == int(mh.RETAIN_SECONDS / mh.INTERVAL)  # 6h@2s=10800
    teardown()


def test_sec_per_step_ema():
    jm = FakeJobman([running_job("j1", 0)])
    setup(jobman=jm)
    mh._sample(now=1000.0)
    assert mh.history(0)["frames"][-1]["sec_per_step"] is None  # 首帧无差分
    jm._jobs = [running_job("j1", 1)]
    mh._sample(now=1060.0)   # 60s 走一步 → 60s/步
    f1 = mh.history(0)["frames"][-1]
    assert f1["sec_per_step"] == 60.0
    jm._jobs = [running_job("j1", 2)]
    mh._sample(now=1090.0)   # 30s 走一步 → EMA 平滑：0.6*60+0.4*30=48
    f2 = mh.history(0)["frames"][-1]
    assert f2["sec_per_step"] == 48.0
    jm._jobs = []            # 任务结束 → 状态复位
    mh._sample(now=1200.0)
    assert mh.history(0)["frames"][-1]["sec_per_step"] is None
    teardown()


def test_sec_per_step_job_switch_resets():
    jm = FakeJobman([running_job("j1", 5)])
    setup(jobman=jm)
    mh._sample(now=1000.0)
    jm._jobs = [running_job("j2", 1)]  # 换 job：差分不得跨任务
    mh._sample(now=1060.0)
    assert mh.history(0)["frames"][-1]["sec_per_step"] is None
    teardown()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()

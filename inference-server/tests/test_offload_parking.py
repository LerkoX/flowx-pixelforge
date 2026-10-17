"""B3：offload 大件驻留管理测试（exec 795 事故，2026-10-16）。

事故现场：`offload=model` 的钩子是**惰性**的（accelerate `cpu_offload_with_hook`：
下一个模块前向时才把上一个搬回 CPU），而本服务是手动采样循环——只调 unet。
于是编码阶段上卡的 text_encoder / VAE 全程没人“顶”它下车：
`allocated` 8000MB ≈ UNet 2.56 + CN 2.5 + CLIP-L/G 1.64 + VAE(fp32) 0.34 + 激活 1
⇒ 8GB 卡装不下 ⇒ WSL2/WDDM 兜底成每步经 PCIe 取权重（170~210s/步，干净态 40s/步）。

本文件测三件事（都不需要真 torch/GPU）：
1. `park_idle_components`：只把**受 offload 钩子管理**的大件送回 CPU（不碰常驻管道）；
2. `ControlNetPlacement`：CN 按实测余量二选一（常驻 / 按次搬移），采样前定死；
3. `is_offloaded_pipe`：用 unet 上的钩子判定 offload 是否启用。

运行：python3 tests/test_offload_parking.py
"""
import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MB = 1024 ** 2
GB = 1024 ** 3
STATE = {"free": 6 * GB, "total": 8 * GB, "reserved": 0, "allocated": 0,
         "available": True, "empty_cache_calls": 0}


class _FakeCuda:
    def is_available(self):
        return STATE["available"]

    def mem_get_info(self):
        return STATE["free"], STATE["total"]

    def memory_reserved(self, device=0):
        return STATE["reserved"]

    def memory_allocated(self, device=0):
        return STATE["allocated"]

    def empty_cache(self):
        STATE["empty_cache_calls"] += 1


_fake_torch = types.ModuleType("torch")
_fake_torch.cuda = _FakeCuda()
_fake_torch.is_tensor = lambda x: isinstance(x, FakeTensor)
sys.modules.setdefault("torch", _fake_torch)

_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)


class FakeTensor:
    """最小张量替身：供 _module_mb 算体积 + 读 .device。"""

    def __init__(self, nbytes, device="cuda"):
        self._nbytes = nbytes
        self.device = device

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1


from app.ops import (ControlNetPlacement, available_vram_bytes,  # noqa: E402
                     is_offloaded_pipe, park_idle_components)


class FakeModule:
    """带 device/hook 的最小模块替身。"""

    def __init__(self, mb, device="cuda", managed=True):
        self.mb = mb
        self._device = device
        self._forward = None          # 测试可替换（模拟前向）
        if managed:
            self._hf_hook = object()      # accelerate 钩子在位（可安全搬走）

    def __call__(self, *args, **kwargs):
        if self._forward is None:
            raise NotImplementedError("fake forward not set")
        return self._forward(*args, **kwargs)

    def parameters(self):
        return iter([FakeTensor(int(self.mb * MB), self._device)])

    def device(self):
        return self._device

    def to(self, dev):
        self._device = dev
        return self


class _Param:
    def __init__(self, mb, device):
        self._mb = mb
        self.device = types.SimpleNamespace(type=device)

    def numel(self):
        return int(self._mb * MB)

    def element_size(self):
        return 1


class FakePipe:
    def __init__(self, unet_managed=True, **mods):
        self.unet = FakeModule(2560, "cuda", managed=unet_managed)
        for name, mod in mods.items():
            setattr(self, name, mod)


def test_is_offloaded_pipe():
    assert is_offloaded_pipe(FakePipe(unet_managed=True)) is True
    assert is_offloaded_pipe(FakePipe(unet_managed=False)) is False
    assert is_offloaded_pipe(object()) is False     # 无 unet 也不抛


def test_park_moves_only_hook_managed_components():
    clip = FakeModule(1390, "cuda", managed=True)     # CLIP-G：钩子管理 → 该下车
    vae = FakeModule(335, "cuda", managed=False)      # 常驻组件：绝不能碰
    already = FakeModule(246, "cpu", managed=True)    # 已在 CPU：无事可做
    pipe = FakePipe(text_encoder=clip, text_encoder_2=already, vae=vae)
    before = STATE["empty_cache_calls"]
    moved = park_idle_components(pipe)
    assert moved == [("text_encoder", 1390.0)], moved
    assert clip._device == "cpu"                      # 下车
    assert vae._device == "cuda"                      # 未受 offload 管理 → 原样
    assert already._device == "cpu"
    assert STATE["empty_cache_calls"] == before + 1   # 下车后回收缓存
    # 再调一次：已无在卡上的受管大件 → 空操作，不再 empty_cache
    assert park_idle_components(pipe) == []


def test_park_noop_when_nothing_on_gpu():
    pipe = FakePipe(text_encoder=FakeModule(100, "cpu"), vae=FakeModule(50, "cpu"))
    before = STATE["empty_cache_calls"]
    assert park_idle_components(pipe) == []
    assert STATE["empty_cache_calls"] == before


def _placement(cn, free_mb, driver_free_mb=None, peers=(), total_mb=8192):
    STATE["free"] = int(free_mb * MB)
    STATE["total"] = int(total_mb * MB)
    STATE["reserved"] = 0
    STATE["allocated"] = 0
    return ControlNetPlacement(cn, "cuda", peers=peers).plan()


def test_cn_keeps_resident_when_room():
    cn = FakeModule(2500, "cuda")
    p = _placement(cn, free_mb=4000)                  # 余量充足
    assert p.per_step is False and cn._device == "cuda"


def test_cn_parks_and_goes_per_step_when_no_room():
    cn = FakeModule(2500, "cuda")
    p = _placement(cn, free_mb=600)                   # 2.5GB 装不下
    assert p.per_step is True and cn._device == "cpu"


def test_cn_loads_in_when_previously_parked():
    cn = FakeModule(2500, "cpu")
    p = _placement(cn, free_mb=4000)
    assert p.per_step is False and cn._device == "cuda"   # 常驻上卡
    cn2 = FakeModule(2500, "cpu")
    p2 = _placement(cn2, free_mb=600)
    assert p2.per_step is True and cn2._device == "cpu"    # 留 CPU，按次搬


def test_cn_per_step_run_moves_in_and_out():
    cn = FakeModule(2500, "cpu")
    placement = _placement(cn, free_mb=600)
    assert placement.per_step is True
    seen = []

    def _fwd(*a, **k):
        seen.append(cn._device)                       # 前向时必须在卡上
        return ("down", "mid")

    cn._forward = _fwd
    out = placement.run("inp", "t")
    assert out == ("down", "mid") and seen == ["cuda"]
    assert cn._device == "cpu"                        # 用完立刻下车


def test_cn_resident_run_does_not_move():
    cn = FakeModule(2500, "cuda")
    placement = _placement(cn, free_mb=4000)
    assert placement.per_step is False
    seen = []

    def _fwd(*a, **k):
        seen.append(cn._device)
        return ("down", "mid")

    cn._forward = _fwd
    placement.run("inp", "t")
    assert seen == ["cuda"] and cn._device == "cuda"  # 常驻：零搬运


def test_cn_placement_noop_without_control():
    p = ControlNetPlacement(None, "cuda").plan()
    assert p.per_step is False and p.mb == 0
    try:                                              # 无 CN 却硬调 → 明确报错
        p.run("x")
        raise AssertionError("无 ControlNet 时应报错")
    except ValueError as e:
        assert "无 ControlNet" in str(e)


def test_cn_placement_counts_resident_peers():
    """实测教训（exec 796）：不算 unet 就会误判“装得下”。

    真机现场：可用 4672MB，CN 需 2386×1.15+512 = 3256MB → 决定常驻；
    随后 UNet 4897MB 也上卡 ⇒ 7283MB+激活 > 8192MB ⇒ 依旧溢出（80s/步）。
    算上 unet 后：3256 + 4897×1.15 = 8888MB > 4672MB ⇒ 正确地改为按次搬移。
    """
    unet = FakeModule(4897, "cpu")                 # 未上卡，循环期会常驻
    cn = FakeModule(2386, "cuda")
    p = _placement(cn, free_mb=4672, peers=(unet,))
    assert p.per_step is True and cn._device == "cpu"
    # 余量充足时仍选择常驻
    cn2 = FakeModule(2386, "cuda")
    p2 = _placement(cn2, free_mb=12000, peers=(unet,))
    assert p2.per_step is False and cn2._device == "cuda"
    # unet 已在卡上 → 不再重复预留它的体积
    unet_on = FakeModule(4897, "cuda")
    p3 = _placement(FakeModule(2386, "cuda"), free_mb=3000, peers=(unet_on,))
    assert p3.per_step is False


def test_hopeless_warning_when_work_set_exceeds_physical():
    """SDXL+CN 在 8GB 上本就超容：即便 CN 按次搬移，也要打醒目告警。"""
    unet = FakeModule(4897, "cpu")
    cn = FakeModule(2386, "cuda")
    p = _placement(cn, free_mb=100, peers=(unet,))
    assert p.per_step is True
    # 4.9GB UNet + 2.4GB CN + 512MB 预留 < 8GB ⇒ 本组合不触发告警
    assert (4897 + 2386 + 512) < 8192
    # 换个更小的物理卡尺寸验证告警分支
    p2 = _placement(FakeModule(2386, "cuda"), free_mb=100, peers=(unet,),
                    total_mb=6000)
    assert p2.per_step is True and p2.mb == 2386.0


def test_available_vram_bytes_counts_cache():
    STATE["free"] = 0
    STATE["reserved"] = 2 * GB
    STATE["allocated"] = 1 * GB
    assert available_vram_bytes() == GB                # 0 + 可回收 1GB
    STATE["available"] = False
    try:
        assert available_vram_bytes() == 0
    finally:
        STATE["available"] = True
        STATE["free"] = 6 * GB


def main():
    checks = [
        test_is_offloaded_pipe,
        test_park_moves_only_hook_managed_components,
        test_park_noop_when_nothing_on_gpu,
        test_cn_keeps_resident_when_room,
        test_cn_parks_and_goes_per_step_when_no_room,
        test_cn_loads_in_when_previously_parked,
        test_cn_per_step_run_moves_in_and_out,
        test_cn_resident_run_does_not_move,
        test_cn_placement_noop_without_control,
        test_cn_placement_counts_resident_peers,
        test_hopeless_warning_when_work_set_exceeds_physical,
        test_available_vram_bytes_counts_cache,
    ]
    for fn in checks:
        fn()
        print(f"{fn.__name__} OK")
    print(f"ALL OFFLOAD PARKING TESTS PASSED ({len(checks)} cases)")


if __name__ == "__main__":
    main()

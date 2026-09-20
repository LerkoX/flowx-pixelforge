"""samplers.py 单元测试：sampler × scheduler 解耦（M2）。

本机无 diffusers：桩掉 diffusers 模块（假 Scheduler 类，__init__ 签名
复刻 v0.35.2 真实支持面），再导入 app.samplers 测纯映射逻辑。
运行：python3 -m tests.test_samplers  或  python3 tests/test_samplers.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeBase:
    """记录 from_config 收到的 kwargs 供断言。"""
    last_kwargs = None

    @classmethod
    def from_config(cls, config, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()


class _FakeEuler(_FakeBase):
    def __init__(self, num_train_timesteps=1000, timestep_spacing="leading",
                 use_karras_sigmas=False, use_exponential_sigmas=False,
                 use_beta_sigmas=False):
        pass


class _FakeEulerA(_FakeBase):
    def __init__(self, num_train_timesteps=1000, timestep_spacing="leading"):
        pass  # 真实类无 sigma 开关


class _FakeDDIM(_FakeBase):
    def __init__(self, num_train_timesteps=1000, timestep_spacing="leading"):
        pass  # 真实类无 sigma 开关


class _FakeLMS(_FakeEuler):
    pass


class _FakeDPMMulti(_FakeBase):
    def __init__(self, num_train_timesteps=1000, algorithm_type="dpmsolver++",
                 solver_type="midpoint", timestep_spacing="leading",
                 use_karras_sigmas=False, use_exponential_sigmas=False,
                 use_beta_sigmas=False, use_lu_lambdas=False):
        pass


class _FakeUniPC(_FakeEuler):
    pass


_fake_diffusers = types.ModuleType("diffusers")
_fake_diffusers.DDIMScheduler = _FakeDDIM
_fake_diffusers.DPMSolverMultistepScheduler = _FakeDPMMulti
_fake_diffusers.EulerAncestralDiscreteScheduler = _FakeEulerA
_fake_diffusers.EulerDiscreteScheduler = _FakeEuler
_fake_diffusers.LMSDiscreteScheduler = _FakeLMS
_fake_diffusers.UniPCMultistepScheduler = _FakeUniPC
sys.modules.setdefault("diffusers", _fake_diffusers)

from app import samplers
from app.samplers import (SAMPLERS, SCHEDULES, make_scheduler, resolve_sampler,
                          supported_schedules)


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"ok: {name}")


def expect_raise(name, fn, *needle):
    try:
        fn()
    except ValueError as e:
        check(f"{name} raises ValueError",
              all(n in str(e) for n in needle))
        return
    raise SystemExit(f"FAIL: {name} should raise ValueError")


def main():
    # 基础解析
    check("euler 默认 normal",
          resolve_sampler("euler") == ("euler", "normal"))
    check("euler+karras", resolve_sampler("euler", "karras") == ("euler", "karras"))
    check("scheduler=None 回退 normal",
          resolve_sampler("euler", None) == ("euler", "normal"))

    # 旧名兼容
    check("旧名 dpmpp_2m_karras -> (dpmpp_2m, karras)",
          resolve_sampler("dpmpp_2m_karras") == ("dpmpp_2m", "karras"))
    check("旧名 + 显式 normal 仍映射 karras",
          resolve_sampler("dpmpp_2m_karras", "normal") == ("dpmpp_2m", "karras"))
    check("旧名 + 显式 beta：显式优先",
          resolve_sampler("dpmpp_2m_karras", "beta") == ("dpmpp_2m", "beta"))

    # 支持矩阵（按假类签名动态判定）
    check("euler 支持全部 4 种",
          supported_schedules("euler") == ["normal", "karras", "exponential", "beta"])
    check("ddim 仅 normal", supported_schedules("ddim") == ["normal"])
    check("euler_a 仅 normal", supported_schedules("euler_a") == ["normal"])
    check("uni_pc 支持全部 4 种",
          supported_schedules("uni_pc") == ["normal", "karras", "exponential", "beta"])

    # 非法输入报错
    expect_raise("unknown sampler",
                 lambda: resolve_sampler("nope"), "nope")
    expect_raise("unknown scheduler", lambda: resolve_sampler("euler", "nope"),
                 "scheduler")
    expect_raise("euler_a+karras 不支持",
                 lambda: resolve_sampler("euler_a", "karras"),
                 "euler_a", "karras", "normal")
    expect_raise("ddim+beta 不支持",
                 lambda: resolve_sampler("ddim", "beta"), "ddim", "beta")

    # make_scheduler 注入的 kwargs
    make_scheduler("dpmpp_2m", "karras", {})
    check("dpmpp_2m+karras kwargs",
          _FakeDPMMulti.last_kwargs == {"algorithm_type": "dpmsolver++",
                                        "use_karras_sigmas": True})
    make_scheduler("dpmpp_2m_sde", "exponential", {})
    check("dpmpp_2m_sde+exponential kwargs",
          _FakeDPMMulti.last_kwargs == {"algorithm_type": "dpmsolver++",
                                        "solver_type": "midpoint",
                                        "use_exponential_sigmas": True})
    make_scheduler("euler", "normal", {})
    check("euler+normal 无多余 kwargs", _FakeEuler.last_kwargs == {})
    make_scheduler("dpmpp_2m_karras", "normal", {})  # 旧名走 make_scheduler
    check("旧名经 make_scheduler 仍注入 karras",
          _FakeDPMMulti.last_kwargs == {"algorithm_type": "dpmsolver++",
                                        "use_karras_sigmas": True})

    # 全集自检：每个 SAMPLERS 键都解析成功
    for name in SAMPLERS:
        resolve_sampler(name)
    check("所有 SAMPLERS 键可解析", True)
    check("SCHEDULES 含 normal", "normal" in SCHEDULES)

    print("all samplers tests passed")


if __name__ == "__main__":
    main()

"""采样器（更新公式）× sigma 排布（scheduler）解耦（M2，对标 ComfyUI 双参数）。

SAMPLERS 只决定用哪个 diffusers Scheduler 类 + 固定 kwargs（更新公式）；
SCHEDULES 只管 sigma 曲线开关（karras/exponential/beta）。
两者自由组合；支持性按类 __init__ 签名动态判定（不写死矩阵，
diffusers 升级后支持面变化零维护）。

已核实的支持矩阵（diffusers v0.35.2 源码）：
- euler / lms / dpmpp_2m / dpmpp_2m_sde / uni_pc：karras + exponential + beta
- euler_a / ddim：仅 normal（类构造无 sigma 开关）

旧一体名兼容：dpmpp_2m_karras → (dpmpp_2m, karras)，
仅当 scheduler 缺省/normal 时生效；显式给了非 normal scheduler 时显式值优先。
"""
import inspect

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

SAMPLERS = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "ddim": (DDIMScheduler, {}),
    "lms": (LMSDiscreteScheduler, {}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
    "dpmpp_2m_sde": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "dpmsolver++", "solver_type": "midpoint"},
    ),
    "uni_pc": (UniPCMultistepScheduler, {}),
}

SCHEDULES = {
    "normal": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}

# 旧一体名 → (sampler, scheduler)；仅 scheduler 缺省/normal 时映射生效
LEGACY_SAMPLERS = {
    "dpmpp_2m_karras": ("dpmpp_2m", "karras"),
}

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "normal"


def supported_schedules(sampler_name):
    """该 sampler 类支持的 sigma 排布列表（按 __init__ 签名动态判定）。"""
    cls, _ = SAMPLERS[sampler_name]
    params = inspect.signature(cls.__init__).parameters
    return [name for name in SCHEDULES
            if name == "normal" or all(k in params for k in SCHEDULES[name])]


def resolve_sampler(sampler_name, scheduler=DEFAULT_SCHEDULER):
    """(sampler_name, scheduler) → (基础 sampler 名, 生效 scheduler)；含旧名兼容。

    校验顺序：旧名映射 → sampler 存在性 → scheduler 存在性 → 组合支持性，
    非法输入一律 ValueError（错误信息列出可选项）。"""
    if sampler_name in LEGACY_SAMPLERS:
        base, legacy_sched = LEGACY_SAMPLERS[sampler_name]
        if not scheduler or scheduler == DEFAULT_SCHEDULER:
            scheduler = legacy_sched
        else:
            print(f"[samplers] 旧名 '{sampler_name}' 与显式 scheduler "
                  f"'{scheduler}' 同现：scheduler 优先（等效 "
                  f"'{base}' + '{scheduler}'）", flush=True)
        sampler_name = base
    if sampler_name not in SAMPLERS:
        raise ValueError(
            f"unknown sampler '{sampler_name}', available: {sorted(SAMPLERS)}"
            f"（兼容旧名: {sorted(LEGACY_SAMPLERS)}）")
    scheduler = scheduler or DEFAULT_SCHEDULER
    if scheduler not in SCHEDULES:
        raise ValueError(
            f"unknown scheduler '{scheduler}', available: {sorted(SCHEDULES)}")
    supported = supported_schedules(sampler_name)
    if scheduler not in supported:
        raise ValueError(
            f"sampler '{sampler_name}' 不支持 scheduler '{scheduler}'，"
            f"支持: {supported}")
    return sampler_name, scheduler


def make_scheduler(sampler_name, scheduler, config):
    """基于当前 workflow 的 scheduler 配置构造新 scheduler（公式 × sigma 曲线组合）。"""
    base, sched = resolve_sampler(sampler_name, scheduler)
    cls, kwargs = SAMPLERS[base]
    return cls.from_config(config, **kwargs, **SCHEDULES[sched])

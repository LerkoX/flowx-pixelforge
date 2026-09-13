"""ComfyUI 采样器名 → diffusers Scheduler 映射。"""
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
    "dpmpp_2m_karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpmpp_2m_sde": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "dpmsolver++", "solver_type": "midpoint"},
    ),
    "uni_pc": (UniPCMultistepScheduler, {}),
}

DEFAULT_SAMPLER = "euler"


def make_scheduler(name: str, config):
    """基于当前 workflow 的 scheduler 配置构造新 scheduler。"""
    if name not in SAMPLERS:
        raise ValueError(
            f"unknown sampler '{name}', available: {', '.join(sorted(SAMPLERS))}"
        )
    cls, kwargs = SAMPLERS[name]
    return cls.from_config(config, **kwargs)

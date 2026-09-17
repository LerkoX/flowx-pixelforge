"""模型架构嗅探：按文件内容/目录结构决定加载路径与管道类名（纯 stdlib，可单测）。

两条加载路径（ROADMAP §1 纪律，不自己实现 state_dict key 映射）：
- diffusers 目录：读 model_index.json 的 _class_name → from_pretrained
- safetensors 单文件：读头部 JSON 的 state_dict key 前缀嗅探 → from_single_file
- .ckpt（pickle）无法廉价嗅探，回退 SD1.x（历史行为兼容）

新架构（SD3/Flux/视频）在此加一条嗅探规则即可，引擎与算子零改动。
"""
import json
import os
import struct

# 单文件嗅探结果 → diffusers 管道类名（getattr(diffusers, ...) 解析）
_SD15 = "StableDiffusionPipeline"
_SDXL = "StableDiffusionXLPipeline"

# 图像管道类名集合：M3 VAE fp32 防黑图只作用于图像线；
# 视频管道（SVD/Wan）的 VAE 解码发生在 pipeline 内部，dtype 混用会崩，不纳入。
IMAGE_ARCHS = frozenset({_SD15, _SDXL})


def _safetensors_keys(path):
    """读 safetensors 头部 JSON（8 字节 LE 长度 + JSON），不加载权重。"""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def sniff_arch(path):
    """返回 (loader, class_name)：loader = 'pretrained'（diffusers 目录）
    或 'single_file'（safetensors/ckpt 单文件）。识别不出则抛 ValueError。"""
    if os.path.isdir(path):
        index = os.path.join(path, "model_index.json")
        if not os.path.isfile(index):
            raise ValueError(
                f"directory '{path}' has no model_index.json "
                f"(not a diffusers model directory)")
        with open(index) as f:
            cls = json.load(f).get("_class_name")
        if not cls:
            raise ValueError(f"'{index}' missing '_class_name' field")
        return "pretrained", cls

    if path.endswith(".safetensors"):
        keys = _safetensors_keys(path)

        def has(*prefixes):
            return any(k.startswith(prefixes) for k in keys)

        # ComfyUI 格式单文件
        if has("conditioner.embedders."):   # SDXL：双 text encoder 打包
            return "single_file", _SDXL
        if has("cond_stage_model.", "model.diffusion_model.", "first_stage_model."):
            return "single_file", _SD15
        # diffusers 格式单文件（组件级 key 前缀）
        if has("text_encoder_2."):
            return "single_file", _SDXL
        if has("unet.", "vae.", "text_encoder."):
            return "single_file", _SD15
        raise ValueError(
            f"cannot sniff architecture of '{path}' "
            f"(unknown state_dict key layout); 请改用 diffusers 目录格式存放")

    if path.endswith(".ckpt"):
        # pickle 容器无法廉价嗅探，按 SD1.x 处理（历史行为兼容）
        return "single_file", _SD15

    raise ValueError(f"unsupported model file '{path}' "
                     f"(expect .safetensors / .ckpt / diffusers directory)")

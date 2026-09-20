"""模型架构嗅探：按文件内容/目录结构决定加载路径与管道类名（纯 stdlib，可单测）。

两条加载路径（ROADMAP §1 纪律，不自己实现 state_dict key 映射）：
- diffusers 目录：读 model_index.json 的 _class_name → from_pretrained
- safetensors 单文件：读头部 JSON 的 state_dict key 前缀嗅探 → from_single_file
- .ckpt（pickle）无法廉价嗅探，回退 SD1.x（历史行为兼容）

组件级嗅探（sniff_component）：识别独立的 VAE / ControlNet 组件
（单文件 / diffusers 组件目录），与整管嗅探互斥（完整 checkpoint 会被明确拒绝）。

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


# ---------- 组件级嗅探（Load VAE / ControlNet Loader：独立组件单文件/目录） ----------

_VAE_CLS = "AutoencoderKL"
_CN_CLS = "ControlNetModel"
_COMPONENT_CLSES = (_VAE_CLS, _CN_CLS)


def sniff_component(path):
    """组件级嗅探。返回 (kind, loader)：
    kind='vae' / 'controlnet'；loader='pretrained'（diffusers 组件目录，有 config.json
    无 model_index.json）/ 'single_file'（safetensors 单文件）。
    完整 checkpoint（含 unet/text_encoder key）与整管目录会被明确拒绝。"""
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "model_index.json")):
            raise ValueError(
                f"'{path}' 是整管 diffusers 目录（含 model_index.json），"
                f"请用 checkpoint.load；组件加载只接受独立组件目录")
        cfg = os.path.join(path, "config.json")
        if not os.path.isfile(cfg):
            raise ValueError(
                f"directory '{path}' has neither model_index.json nor "
                f"config.json (not a diffusers component directory)")
        with open(cfg) as f:
            cls = json.load(f).get("_class_name")
        if cls == _VAE_CLS:
            return "vae", "pretrained"
        if cls == _CN_CLS:
            return "controlnet", "pretrained"
        raise ValueError(
            f"unsupported component '{cls}' in '{cfg}' "
            f"(当前只支持 {_COMPONENT_CLSES})")

    if path.endswith(".safetensors"):
        keys = _safetensors_keys(path)

        def has(*prefixes):
            return any(k.startswith(prefixes) for k in keys)

        # 完整 checkpoint 明确拒绝（防止误传整模被当成组件加载）
        if has("model.diffusion_model.", "unet.", "cond_stage_model.",
               "text_encoder.", "conditioner.embedders."):
            raise ValueError(
                f"'{path}' 是完整 checkpoint（含 unet/text_encoder key），"
                f"不是独立组件；请用 checkpoint.load")
        # ControlNet 单文件：diffusers 格式（controlnet_down_blocks.*）、
        # lllyasviel 原版（input_hint_block.*）或 ComfyUI 重打包格式
        # （control_model.* 统一前缀，from_single_file 均可自动转换）
        if has("controlnet_down_blocks.", "controlnet_mid_block.") \
                or has("input_hint_block.") or has("control_model."):
            return "controlnet", "single_file"
        # ComfyUI 格式 VAE 单文件 / diffusers 格式 VAE 单文件
        if has("first_stage_model.encoder.", "first_stage_model.decoder.") \
                or has("encoder.conv_in.", "decoder.conv_in."):
            return "vae", "single_file"
        raise ValueError(
            f"cannot sniff component of '{path}' (unknown state_dict key "
            f"layout)；组件加载支持 ComfyUI/diffusers 格式 VAE 单文件与 "
            f"ControlNet 单文件（control_v11* 等）")

    raise ValueError(f"unsupported component file '{path}' "
                     f"(expect .safetensors / diffusers component directory)")

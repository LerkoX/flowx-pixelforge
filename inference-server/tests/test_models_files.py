"""模型文件清单（GET /models/files 数据源）测试：

- 分类：checkpoint 目录/单文件、vae/controlnet 组件、upscale（.pth 直判 +
  .safetensors 名字线索兜底）、embedding、lora、motion
- 跳过功能目录（preprocessors/detectors/plugins.d/embeddings/motion 顶层不直列）
- name 去扩展名（resolve 可省略扩展名）；按 kind/name 排序

本机无 torch：构造假目录结构 + 最小 safetensors 头部（sniff 只读 JSON header）。
运行：python3 -m tests.test_models_files
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.sniff import list_model_files


def _st(path, keys):
    """写最小 safetensors 文件（仅 header，keys 决定嗅探结果）。"""
    header = json.dumps({k: {"dtype": "F16", "shape": [1],
                             "data_offsets": [0, 2]} for k in keys}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + b"\x00" * 2)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def main():
    tmp = tempfile.mkdtemp()
    models = os.path.join(tmp, "models")
    loras = os.path.join(tmp, "loras")
    os.makedirs(models)
    os.makedirs(loras)

    # checkpoint：diffusers 整管目录 + 完整 checkpoint 单文件
    os.makedirs(os.path.join(models, "v1-5-pruned-emaonly-fp16"))
    _touch(os.path.join(models, "v1-5-pruned-emaonly-fp16", "model_index.json"))
    # 目录名含点（sd3.5-medium）：不能按扩展名截断
    os.makedirs(os.path.join(models, "sd3.5-medium"))
    _touch(os.path.join(models, "sd3.5-medium", "model_index.json"))
    _st(os.path.join(models, "majicmixRealistic_v7.safetensors"),
        ["model.diffusion_model.input_blocks.0.0.weight"])
    # vae / controlnet 单文件（嗅探 key 布局）
    _st(os.path.join(models, "vae-ft-mse-840000-ema-pruned.safetensors"),
        ["first_stage_model.encoder.conv_in.weight"])
    _st(os.path.join(models, "control_v11p_sd15_canny_fp16.safetensors"),
        ["input_hint_block.0.weight"])
    # controlnet diffusers 组件目录
    cn_dir = os.path.join(models, "control_v11p_sd15_openpose")
    os.makedirs(cn_dir)
    with open(os.path.join(cn_dir, "config.json"), "w") as f:
        json.dump({"_class_name": "ControlNetModel"}, f)
    # upscale：.pth 直判 + .safetensors 名字线索兜底（未知布局）
    _touch(os.path.join(models, "RealESRGAN_x4plus.pth"))
    _st(os.path.join(models, "4x-UltraSharp.safetensors"),
        ["body.0.weight"])  # spandrel 系布局，sniff_component 不认
    # 功能目录：跳过顶层，embeddings/motion 单列
    for d in ("preprocessors", "detectors", "plugins.d"):
        os.makedirs(os.path.join(models, d))
    _touch(os.path.join(models, "embeddings", "easynegative.safetensors"))
    _touch(os.path.join(models, "motion", "mm_sd_v15_v2.ckpt"))
    # lora
    _touch(os.path.join(loras, "add-detail.safetensors"))
    # 杂项：非模型文件忽略
    _touch(os.path.join(models, "README.txt"))

    files = list_model_files(models, loras)
    by_name = {f["name"]: f["kind"] for f in files}

    expect = {
        "v1-5-pruned-emaonly-fp16": "checkpoint",
        "sd3.5-medium": "checkpoint",
        "majicmixRealistic_v7": "checkpoint",
        "vae-ft-mse-840000-ema-pruned": "vae",
        "control_v11p_sd15_canny_fp16": "controlnet",
        "control_v11p_sd15_openpose": "controlnet",
        "RealESRGAN_x4plus": "upscale",
        "4x-UltraSharp": "upscale",
        "easynegative": "embedding",
        "mm_sd_v15_v2": "motion",
        "add-detail": "lora",
    }
    for name, kind in expect.items():
        got = by_name.get(name)
        assert got == kind, f"{name}: 期望 {kind}，实际 {got}"
    assert "README" not in by_name, "非模型文件不应入清单"
    assert "sd3" not in by_name, "带点目录名被误截扩展名"
    for skip in ("preprocessors", "detectors", "plugins.d", "embeddings", "motion"):
        assert skip not in by_name, f"功能目录 {skip} 不应顶层入清单"
    # 排序：kind 分组有序
    kinds = [f["kind"] for f in files]
    assert kinds == sorted(kinds), "应按 kind 分组排序"

    print("ok: 10 类条目分类正确，功能目录跳过，排序稳定")


if __name__ == "__main__":
    main()

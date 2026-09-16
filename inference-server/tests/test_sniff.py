"""模型架构嗅探单元测试（纯 stdlib：构造假 safetensors 头部 / 假 diffusers 目录）。
运行：python3 -m tests.test_sniff  或  python3 tests/test_sniff.py
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.sniff import sniff_arch


def fake_safetensors(path, keys):
    """写一个仅含头部 JSON 的假 safetensors 文件（权重区为空，嗅探不读）。"""
    header = json.dumps({k: {} for k in keys}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def expect(path, loader, cls):
    got = sniff_arch(path)
    assert got == (loader, cls), f"{path}: expect ({loader}, {cls}), got {got}"
    print(f"OK: {os.path.basename(path)} -> {got}")


def expect_error(path, needle=""):
    try:
        got = sniff_arch(path)
        raise AssertionError(f"{path}: expect ValueError, got {got}")
    except ValueError as e:
        if needle:
            assert needle in str(e), f"{path}: '{needle}' not in {e}"
        print(f"OK: {os.path.basename(path)} -> ValueError: {str(e)[:60]}...")


def main():
    with tempfile.TemporaryDirectory() as d:
        # --- safetensors 单文件：ComfyUI 格式 key ---
        p = os.path.join(d, "v1-5.safetensors")
        fake_safetensors(p, ["cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                             "model.diffusion_model.input_blocks.0.0.weight",
                             "first_stage_model.encoder.conv_in.weight"])
        expect(p, "single_file", "StableDiffusionPipeline")

        p = os.path.join(d, "sdxl-base.safetensors")
        fake_safetensors(p, ["conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                             "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                             "model.diffusion_model.input_blocks.0.0.weight",
                             "first_stage_model.encoder.conv_in.weight"])
        expect(p, "single_file", "StableDiffusionXLPipeline")

        # --- safetensors 单文件：diffusers 组件级 key ---
        p = os.path.join(d, "sd15-diffusers.safetensors")
        fake_safetensors(p, ["unet.conv_in.weight", "vae.encoder.conv_in.weight",
                             "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight"])
        expect(p, "single_file", "StableDiffusionPipeline")

        p = os.path.join(d, "sdxl-diffusers.safetensors")
        fake_safetensors(p, ["unet.conv_in.weight", "vae.encoder.conv_in.weight",
                             "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight",
                             "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.weight"])
        expect(p, "single_file", "StableDiffusionXLPipeline")

        # --- 无法识别的 key 布局 ---
        p = os.path.join(d, "mystery.safetensors")
        fake_safetensors(p, ["some.transformer.block.weight"])
        expect_error(p, "diffusers 目录格式")

        # --- .ckpt 回退 SD1.x ---
        p = os.path.join(d, "old-model.ckpt")
        open(p, "wb").write(b"\x80\x04")  # pickle 魔数，嗅探不读内容
        expect(p, "single_file", "StableDiffusionPipeline")

        # --- 不支持的扩展名 ---
        p = os.path.join(d, "model.gguf")
        open(p, "wb").write(b"GGUF")
        expect_error(p, "unsupported model file")

        # --- diffusers 目录 ---
        wan = os.path.join(d, "Wan2.2-TI2V-5B-Diffusers")
        os.makedirs(wan)
        with open(os.path.join(wan, "model_index.json"), "w") as f:
            json.dump({"_class_name": "WanPipeline"}, f)
        expect(wan, "pretrained", "WanPipeline")

        # --- 目录缺 model_index.json ---
        bad = os.path.join(d, "not-a-model")
        os.makedirs(bad)
        expect_error(bad, "model_index.json")

        # --- model_index.json 缺 _class_name ---
        bad2 = os.path.join(d, "broken-model")
        os.makedirs(bad2)
        with open(os.path.join(bad2, "model_index.json"), "w") as f:
            json.dump({"_name_or_path": "x"}, f)
        expect_error(bad2, "_class_name")

    print("\nALL SNIFF TESTS PASSED")


if __name__ == "__main__":
    main()

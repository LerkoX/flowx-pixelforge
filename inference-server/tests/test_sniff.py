"""模型架构嗅探单元测试（纯 stdlib：构造假 safetensors 头部 / 假 diffusers 目录）。
运行：python3 -m tests.test_sniff  或  python3 tests/test_sniff.py
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.sniff import sniff_arch, sniff_component


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


def expect_comp(path, kind, loader):
    got = sniff_component(path)
    assert got == (kind, loader), f"{path}: expect ({kind}, {loader}), got {got}"
    print(f"OK: {os.path.basename(path)} -> component {got}")


def expect_comp_error(path, needle=""):
    try:
        got = sniff_component(path)
        raise AssertionError(f"{path}: expect ValueError, got {got}")
    except ValueError as e:
        if needle:
            assert needle in str(e), f"{path}: '{needle}' not in {e}"
        print(f"OK: {os.path.basename(path)} -> ValueError: {str(e)[:60]}...")


def main_component():
    """组件级嗅探（Load VAE）用例。"""
    with tempfile.TemporaryDirectory() as d:
        # --- ComfyUI 格式 VAE 单文件 ---
        p = os.path.join(d, "vae-ft-mse-840000.safetensors")
        fake_safetensors(p, ["first_stage_model.encoder.conv_in.weight",
                             "first_stage_model.decoder.conv_in.weight",
                             "first_stage_model.quant_conv.weight"])
        expect_comp(p, "vae", "single_file")

        # --- diffusers 格式 VAE 单文件（裸组件 key） ---
        p = os.path.join(d, "vae-diffusers.safetensors")
        fake_safetensors(p, ["encoder.conv_in.weight", "decoder.conv_in.weight",
                             "quant_conv.weight"])
        expect_comp(p, "vae", "single_file")

        # --- 完整 checkpoint 明确拒绝（ComfyUI / diffusers 两种布局） ---
        p = os.path.join(d, "full-comfy.safetensors")
        fake_safetensors(p, ["first_stage_model.encoder.conv_in.weight",
                             "model.diffusion_model.input_blocks.0.0.weight",
                             "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"])
        expect_comp_error(p, "完整 checkpoint")

        p = os.path.join(d, "full-diffusers.safetensors")
        fake_safetensors(p, ["vae.encoder.conv_in.weight", "unet.conv_in.weight",
                             "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight"])
        expect_comp_error(p, "完整 checkpoint")

        # --- 无法识别的组件 key ---
        p = os.path.join(d, "mystery-comp.safetensors")
        fake_safetensors(p, ["some.random.weight"])
        expect_comp_error(p, "unknown state_dict key layout")

        # --- diffusers 组件目录（config.json 无 model_index.json） ---
        vdir = os.path.join(d, "vae-dir")
        os.makedirs(vdir)
        with open(os.path.join(vdir, "config.json"), "w") as f:
            json.dump({"_class_name": "AutoencoderKL"}, f)
        expect_comp(vdir, "vae", "pretrained")

        # --- 整管目录拒绝 ---
        pdir = os.path.join(d, "pipe-dir")
        os.makedirs(pdir)
        with open(os.path.join(pdir, "model_index.json"), "w") as f:
            json.dump({"_class_name": "StableDiffusionPipeline"}, f)
        expect_comp_error(pdir, "checkpoint.load")

        # --- 两无目录 ---
        edir = os.path.join(d, "empty-dir")
        os.makedirs(edir)
        expect_comp_error(edir, "config.json")

        # --- 不支持的组件类 ---
        cdir = os.path.join(d, "clip-dir")
        os.makedirs(cdir)
        with open(os.path.join(cdir, "config.json"), "w") as f:
            json.dump({"_class_name": "CLIPTextModel"}, f)
        expect_comp_error(cdir, "unsupported component")

        # --- 不支持的扩展名 ---
        p = os.path.join(d, "vae.ckpt")
        open(p, "wb").write(b"\x80\x04")
        expect_comp_error(p, "unsupported component file")

    print("\nALL SNIFF COMPONENT TESTS PASSED")


if __name__ == "__main__":
    main()
    main_component()

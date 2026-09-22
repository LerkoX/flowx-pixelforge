"""显存预算计算测试（纯 stdlib，无需 torch/GPU）。

运行：python3 tests/test_vram.py     （或 python3 -m tests.test_vram）
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.vram import (DTYPE_BYTES, estimate_bytes, plan_eviction,
                      safetensors_bytes, target_bytes)


def _write_safetensors(path, tensors):
    """tensors: [(name, dtype, nbytes)] —— 按 safetensors 格式落一个假文件。"""
    header, offset = {}, 0
    for name, dtype, nbytes in tensors:
        header[name] = {"dtype": dtype, "shape": [nbytes],
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


def test_target_bytes():
    assert target_bytes("fp16") == 2 and target_bytes("bf16") == 2
    assert target_bytes("fp32") == 4 and target_bytes("auto") == 2


def test_safetensors_bytes(fp16_file, fp32_file):
    # fp16 存 + fp16 加载 = 原样；fp32 存 + fp16 加载 = 减半
    assert safetensors_bytes(fp16_file, "fp16") == 4096
    assert safetensors_bytes(fp32_file, "fp16") == 4096
    assert safetensors_bytes(fp32_file, "fp32") == 8192
    # fp16 存 + fp32 加载 = 翻倍
    assert safetensors_bytes(fp16_file, "fp32") == 8192


def test_safetensors_bytes_bad_file(fake_tmp):
    bad = os.path.join(fake_tmp, "bad.safetensors")
    with open(bad, "wb") as f:
        f.write(b"not-a-safetensors")
    assert safetensors_bytes(bad, "fp16") is None
    # 单文件兜底到文件大小（安全上界）
    assert estimate_bytes(bad, "fp16") == os.path.getsize(bad)


def test_estimate_dir(fake_tmp):
    d = os.path.join(fake_tmp, "diffusers-model")
    os.makedirs(os.path.join(d, "unet"))
    _write_safetensors(os.path.join(d, "unet", "diffusion_pytorch_model.safetensors"),
                       [("w", "F16", 2048)])
    _write_safetensors(os.path.join(d, "text_encoder.safetensors"),
                       [("w", "F32", 4096)])
    # 非权重文件不参与估算
    with open(os.path.join(d, "model_index.json"), "w") as f:
        f.write("{}")
    # fp16 加载：unet(F16,2048) 原样 + text_encoder(F32,4096) 减半
    assert estimate_bytes(d, "fp16") == 2048 + 2048
    # fp32 加载：unet(F16,2048) 翻倍 + text_encoder(F32,4096) 原样
    assert estimate_bytes(d, "fp32") == 4096 + 4096


def test_estimate_missing():
    assert estimate_bytes("/nonexistent/whatever", "fp16") is None


def test_plan_eviction_count_cap():
    """条目上限兜底：MAX_RESIDENT=2 时加载第 3 个要淘汰 1 个（LRU）。"""
    resident = [("a", 100, 1.0), ("b", 100, 2.0)]
    assert plan_eviction(resident, free_bytes=10**9, need_bytes=0,
                         reserve_bytes=0, max_resident=2) == ["a"]


def test_plan_eviction_memory_budget():
    """显存预算：余量 3GB、待加载 2GB + reserve 1GB ⇒ 必须再腾出 0 ⇒ 不淘汰。"""
    resident = [("a", 100, 1.0), ("b", 100, 2.0)]
    gb = 1024 ** 3
    assert plan_eviction(resident, free_bytes=3 * gb, need_bytes=2 * gb,
                         reserve_bytes=1 * gb, max_resident=0) == []
    # 余量只有 1GB ⇒ 需腾出 2GB ⇒ 两个都淘汰（LRU 顺序）
    assert plan_eviction(resident, free_bytes=1 * gb, need_bytes=2 * gb,
                         reserve_bytes=1 * gb, max_resident=0) == ["a", "b"]


def test_plan_eviction_pinned_skipped():
    """执行中在用（pin）的条目不动，宁可放行也不抽走手上的模型。"""
    resident = [("in-use", 4096, 1.0), ("idle", 4096, 2.0)]
    got = plan_eviction(resident, free_bytes=0, need_bytes=8192,
                        reserve_bytes=1024, max_resident=1, pinned={"in-use"})
    assert got == ["idle"]


def test_plan_eviction_unknown_size_and_empty():
    assert plan_eviction([], free_bytes=0, need_bytes=999,
                         reserve_bytes=0, max_resident=1) == []
    # 体积未知（0）：条目上限仍要淘汰，显存条件靠调用方兜底
    resident = [("x", None, 5.0), ("y", None, 1.0)]
    assert plan_eviction(resident, free_bytes=0, need_bytes=10 ** 9,
                         reserve_bytes=0, max_resident=2) == ["y", "x"]


def main():
    tmp = tempfile.mkdtemp(prefix="vram-test-")
    fp16 = os.path.join(tmp, "fp16.safetensors")
    fp32 = os.path.join(tmp, "fp32.safetensors")
    _write_safetensors(fp16, [("w", "F16", 4096)])
    _write_safetensors(fp32, [("w", "F32", 8192)])
    checks = [
        test_target_bytes,
        lambda: test_safetensors_bytes(fp16, fp32),
        lambda: test_safetensors_bytes_bad_file(tmp),
        lambda: test_estimate_dir(tmp),
        test_estimate_missing,
        test_plan_eviction_count_cap,
        test_plan_eviction_memory_budget,
        test_plan_eviction_pinned_skipped,
        test_plan_eviction_unknown_size_and_empty,
    ]
    for fn in checks:
        fn()
        print(f"{getattr(fn, '__name__', 'case')} OK")
    assert DTYPE_BYTES["F16"] == 2
    print("all vram budget tests OK")


if __name__ == "__main__":
    main()

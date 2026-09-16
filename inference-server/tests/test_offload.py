"""显存治理配置解析测试（纯 stdlib）。
运行：python3 -m tests.test_offload  或  python3 tests/test_offload.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.offload import resolve_offload_mode, resolve_quantization


def main():
    # --- 默认：无 offload ---
    assert resolve_offload_mode({}) == "none"
    print("default none OK")

    # --- 历史开关兼容：ENABLE_CPU_OFFLOAD=1 -> model ---
    assert resolve_offload_mode({"ENABLE_CPU_OFFLOAD": "1"}) == "model"
    assert resolve_offload_mode({"ENABLE_CPU_OFFLOAD": "0"}) == "none"
    print("legacy ENABLE_CPU_OFFLOAD mapping OK")

    # --- OFFLOAD_MODE 显式档位，且优先于历史开关 ---
    assert resolve_offload_mode({"OFFLOAD_MODE": "sequential"}) == "sequential"
    assert resolve_offload_mode({"OFFLOAD_MODE": "MODEL"}) == "model"  # 大小写不敏感
    assert resolve_offload_mode({"OFFLOAD_MODE": "none",
                                 "ENABLE_CPU_OFFLOAD": "1"}) == "none"
    print("OFFLOAD_MODE priority OK")

    # --- 非法档位 ---
    try:
        resolve_offload_mode({"OFFLOAD_MODE": "magic"})
        raise AssertionError("invalid mode accepted")
    except ValueError as e:
        print("invalid mode rejected OK:", e)

    # --- 量化配置：默认 none；fp8 识别通过（实现后置） ---
    assert resolve_quantization({}) == "none"
    assert resolve_quantization({"QUANTIZATION": "FP8"}) == "fp8"
    try:
        resolve_quantization({"QUANTIZATION": "int4"})
        raise AssertionError("invalid quantization accepted")
    except ValueError as e:
        print("quantization config OK;", e)

    print("\nALL OFFLOAD CONFIG TESTS PASSED")


if __name__ == "__main__":
    main()

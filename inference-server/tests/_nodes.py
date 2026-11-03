"""节点仓定位（拆分后：节点代码在独立仓库 flowx-pixelforge-nodes）。

2026-11 起 `nodes/` 从本仓拆出为独立仓库。本仓的算子契约测试（`test_*_plugin.py`
等）仍需要真实节点代码作为输入——它们是「服务端插件机制 × 真实算子」的集成测试，
不是纯算子单测（纯算子契约由节点仓的 `_tools/check-op-signatures.py` 纯 AST 静态
门禁覆盖，无需本仓参与）。

解析顺序：
  1. 环境变量 `FLOWX_NODES_DIR`（CI 或特殊布局用）
  2. 与本仓并列的同级目录 `../flowx-pixelforge-nodes`（本地推荐布局）
"""
import os
import pathlib

_TEST_DIR = pathlib.Path(__file__).resolve().parent

NODES = pathlib.Path(
    os.environ.get("FLOWX_NODES_DIR")
    or _TEST_DIR.parent.parent.parent / "flowx-pixelforge-nodes")


def require() -> pathlib.Path:
    """取节点仓根目录；不存在时给出可操作的报错（比 FileNotFoundError 好排查）。"""
    if not NODES.is_dir():
        raise RuntimeError(
            f"找不到节点仓：{NODES}\n"
            f"节点代码已拆到独立仓库 flowx-pixelforge-nodes。请二选一：\n"
            f"  1) 把两个仓库并列 clone（../flowx-pixelforge-nodes）；\n"
            f"  2) 设置 FLOWX_NODES_DIR=/path/to/flowx-pixelforge-nodes")
    return NODES

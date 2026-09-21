"""插件系统硬化测试（插件状态漂移事故根因修复）：

- load_plugin 如实上报全部声明算子名（含覆盖既有名），不做静默丢弃
- detect_conflicts：跨文件重名 / 遮蔽核心算子 / 同文件重传不冲突
- scan_plugins：跨文件重名 ERROR 日志 + 后加载者生效；reserved 核心遮蔽 ERROR

本机无 torch：插件样例全部为零依赖内联源码，直接走 app.plugins 真实链路。
运行：python3 -m tests.test_plugin_hardening
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plugins import detect_conflicts, load_plugin, scan_plugins
from app.registry import Registry

PLUGIN_A = '''
def register(registry):
    registry.register("shared.op", inputs={}, outputs={"v": "INT"},
                      description="A 版")(lambda: {"v": 1})
    registry.register("a.only", inputs={}, outputs={"v": "INT"},
                      description="A 独有")(lambda: {"v": 1})
'''

PLUGIN_B = '''
def register(registry):
    registry.register("shared.op", inputs={}, outputs={"v": "INT"},
                      description="B 版")(lambda: {"v": 2})
    registry.register("b.only", inputs={}, outputs={"v": "INT"},
                      description="B 独有")(lambda: {"v": 2})
'''

PLUGIN_CORE_SHADOW = '''
def register(registry):
    registry.register("core.op", inputs={}, outputs={"v": "INT"},
                      description="遮蔽核心")(lambda: {"v": 9})
'''


def _write(d, name, src):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(src)
    return p


def main():
    tmp = tempfile.mkdtemp()
    reg = Registry()
    reg.register("core.op", {}, {"v": "INT"}, "核心算子")(lambda: {"v": 0})

    # 1) load_plugin 如实上报全部声明名（含覆盖既有 core.op 的名）
    pa = _write(tmp, "a_plug.py", PLUGIN_A)
    claimed = load_plugin(pa, reg)
    assert claimed == ["a.only", "shared.op"], claimed
    assert reg.get("shared.op").description == "A 版"

    # 2) detect_conflicts：B 插件与 A 跨文件重名
    plugin_ops = {n: {"file": "a_plug.py", "sha256": "x"} for n in claimed}
    plugin_ops["core.op"] = None  # core.op 不在归属簿（核心算子）
    plugin_ops.pop("core.op")
    before = dict(reg._ops)
    pb = _write(tmp, "b_plug.py", PLUGIN_B)
    claimed_b = load_plugin(pb, reg)
    assert claimed_b == ["b.only", "shared.op"], claimed_b
    assert reg.get("shared.op").description == "B 版", "register 覆盖语义"
    conflicts = detect_conflicts(claimed_b, "b_plug.py", plugin_ops, before)
    assert conflicts == [{"op": "shared.op", "owner": "a_plug.py"}], conflicts

    # 3) 同文件重传：无冲突（幂等更新）
    conflicts_same = detect_conflicts(claimed, "a_plug.py", plugin_ops, before)
    assert conflicts_same == [], conflicts_same

    # 4) 遮蔽核心算子 → owner='core'
    pc = _write(tmp, "c_plug.py", PLUGIN_CORE_SHADOW)
    claimed_c = load_plugin(pc, reg)
    conflicts_core = detect_conflicts(claimed_c, "c_plug.py", plugin_ops, before)
    assert conflicts_core == [{"op": "core.op", "owner": "core"}], conflicts_core

    # 5) scan_plugins：跨文件重名 ERROR 日志 + 后加载者（b_plug.py）生效
    tmp2 = tempfile.mkdtemp()
    _write(tmp2, "a_plug.py", PLUGIN_A)
    _write(tmp2, "b_plug.py", PLUGIN_B)
    _write(tmp2, "c_plug.py", PLUGIN_CORE_SHADOW)
    reg2 = Registry()
    reg2.register("core.op", {}, {"v": "INT"}, "核心算子")(lambda: {"v": 0})
    buf = io.StringIO()
    with redirect_stdout(buf):
        loaded = scan_plugins(tmp2, reg2, reserved=set(reg2._ops))
    log = buf.getvalue()
    assert "ERROR" in log and "shared.op" in log, log
    assert "遮蔽核心算子 'core.op'" in log, log
    assert reg2.get("shared.op").description == "B 版", "后加载者生效"
    assert loaded["b_plug.py"] == ["b.only", "shared.op"], loaded

    print("PASS test_plugin_hardening")


if __name__ == "__main__":
    main()

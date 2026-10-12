"""插件归属簿（plugin_ops）对账单测 —— 无需 GPU/torch，直跑：
    python3 tests/test_plugin_registry.py

覆盖 dev-plan §52.1 的三个可证缺陷（A1 幽灵归属 / A2 陈旧算子泄漏 / A3 回滚不完整）：
- 幽灵归属（归属簿指向已删除的文件）不算冲突，可被新文件接管并清理
- 同文件重传后，"上次声明、这次不再声明"的算子必须从注册表与归属簿摘除
- 只摘"归属仍是本文件"的条目（force 接手过的别人的名字不许误摘）
- 幂等重传无副作用
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import plugins                      # noqa: E402
from app.registry import Registry            # noqa: E402

FAILED = 0
PASSED = 0


def check(label, cond, extra=""):
    global FAILED, PASSED
    if cond:
        PASSED += 1
        print(f"PASS {label}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  {extra}" if extra else ""))


def reg_op(reg, name):
    """注意：Registry.register 是**装饰器工厂**（返回 deco），必须套到函数上才真注册。"""
    @reg.register(name, {"image": "IMAGE"}, {"image": "IMAGE"}, "t")
    def _fn(image=None):
        return {"image": image}
    return _fn


def make_reg():
    reg = Registry()
    for n in ("colorize.load", "colorize.apply"):
        reg_op(reg, n)
    return reg


def main():
    with tempfile.TemporaryDirectory() as td:
        # ---------- A1 幽灵归属 ----------
        plugin_ops = {"colorize.load": {"file": "colorize_aaaa.py", "sha256": "x"},
                      "colorize.apply": {"file": "colorize_aaaa.py", "sha256": "x"}}
        # 盘上没有 colorize_aaaa.py（手工删除 + 未重启）⇒ 两个算子都是幽灵
        stale = plugins.stale_owners(["colorize.load"], "colorize_bbbb.py",
                                     plugin_ops, td)
        check("A1 归属指向不存在的文件 ⇒ 识别为幽灵", stale == ["colorize.load"], str(stale))
        # 盘上有文件时不算幽灵
        (pathlib.Path(td) / "colorize_aaaa.py").write_text("x")
        stale2 = plugins.stale_owners(["colorize.load"], "colorize_bbbb.py",
                                      plugin_ops, td)
        check("A1 归属文件确实存在 ⇒ 不算幽灵（正常 409）", stale2 == [], str(stale2))
        (pathlib.Path(td) / "colorize_aaaa.py").unlink()

        # 幽灵场景下的冲突判定：本地视图剔除幽灵后应无冲突
        owner_view = {k: v for k, v in plugin_ops.items() if k not in set(stale)}
        before = {"colorize.load": object(), "colorize.apply": object()}
        before_view = {k: v for k, v in before.items() if k not in set(stale)}
        conf = plugins.detect_conflicts(["colorize.load"], "colorize_bbbb.py",
                                        owner_view, before_view)
        check("A1 幽灵算子不产生 409（本地视图剔除后无冲突）", conf == [], str(conf))
        conf_raw = plugins.detect_conflicts(["colorize.load"], "colorize_bbbb.py",
                                            plugin_ops, before)
        check("A1 对照：不清幽灵时误报为「已删除文件拥有」（假 409 的真身）",
              conf_raw and conf_raw[0]["owner"] == "colorize_aaaa.py", str(conf_raw))

        # ---------- A2 陈旧算子泄漏 ----------
        reg = make_reg()
        plugin_ops = {"colorize.load": {"file": "f.py", "sha256": "old"},
                      "colorize.apply": {"file": "f.py", "sha256": "old"},
                      "colorize.legacy": {"file": "f.py", "sha256": "old"}}
        reg_op(reg, "colorize.legacy")
        rep = plugins.reconcile("f.py", ["colorize.load", "colorize.apply"],
                                reg, plugin_ops, sha256="new")
        check("A2 陈旧算子被告知摘除", rep["stale_removed"] == ["colorize.legacy"],
              str(rep))
        check("A2 陈旧算子已从注册表摘除", "colorize.legacy" not in reg._ops)
        check("A2 陈旧算子已从归属簿删除", "colorize.legacy" not in plugin_ops)
        check("A2 本次声明的归属已更新",
              plugin_ops["colorize.load"] == {"file": "f.py", "sha256": "new"},
              str(plugin_ops.get("colorize.load")))
        check("A2 仍在声明的算子没被误摘",
              "colorize.load" in reg._ops and "colorize.apply" in reg._ops)

        # ---------- 归属不是本文件的条目不许误摘（force 接管的场景）----------
        plugin_ops = {"other.op": {"file": "other.py", "sha256": "z"},
                      "mine.op": {"file": "f.py", "sha256": "z"}}
        reg_op(reg, "other.op")
        rep2 = plugins.reconcile("f.py", ["mine.op"], reg, plugin_ops, sha256="z2")
        check("越权保护：不摘别的文件的算子",
              rep2["stale_removed"] == [] and "other.op" in reg._ops
              and plugin_ops["other.op"]["file"] == "other.py", str(rep2))

        # ---------- 幂等重传无副作用 ----------
        reg3 = make_reg()
        po3 = {"colorize.load": {"file": "f.py", "sha256": "s"},
               "colorize.apply": {"file": "f.py", "sha256": "s"}}
        rep3 = plugins.reconcile("f.py", ["colorize.load", "colorize.apply"],
                                 reg3, po3, sha256="s")
        check("幂等重传：无陈旧摘除、归属不变",
              rep3["stale_removed"] == [] and po3["colorize.load"]["sha256"] == "s"
              and set(po3) == {"colorize.load", "colorize.apply"}, str(rep3))

        # ---------- 模块名口径（回滚路径依赖） ----------
        check("plugin_module_name 与 load_plugin 同口径",
              plugins.plugin_module_name("/models/plugins.d/seg_abc123.py")
              == "flowx_plugin_seg_abc123")

        # ---------- 真加载一次：声明的算子被如实上报 ----------
        # 与真实族文件同款：register(...)(fn)
        src = ("def _op(image=None):\n"
               "    return {}\n\n\n"
               "def register(registry):\n"
               "    registry.register('demo.op', {'image': 'IMAGE'},\n"
               "                      {'image': 'IMAGE'}, 'd')(_op)\n")
        f = pathlib.Path(td) / "demo_plugin.py"
        f.write_text(src)
        reg4 = Registry()
        names = plugins.load_plugin(str(f), reg4)
        check("真实加载：上报声明的算子名", names == ["demo.op"], str(names))
        check("真实加载：注册表可见", "demo.op" in reg4._ops)
        check("真实加载：模块已进 sys.modules 且带既有算子时 register 已还原",
              f"flowx_plugin_demo_plugin" in sys.modules
              and not hasattr(reg4.register, "_recording"))

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

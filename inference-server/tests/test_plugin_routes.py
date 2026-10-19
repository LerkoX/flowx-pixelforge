"""插件契约 v2 测试：register_routes 路由自注册（无 fastapi/torch 可跑）。

RouteRegistrar 的 depends/app 全部用桩注入（生产侧 main.py 注入 fastapi.Depends）。
覆盖：
1. 插件声明 register_routes → 路由挂上、/ext/ 前缀、auth 依赖自动附加；
2. 非 /ext/ 前缀 → 拒载，且**已加的路由整体回滚**（半路失败不残留）；
3. 老插件（无 register_routes）→ 行为与契约 v1 完全一致；
4. 声明了 register_routes 但加载方没给注册器 → 明确报错（不静默跳过）；
5. **负例**：摘掉前缀校验 ⇒ 裸路径被接受（证明校验真的在起作用）；
6. 多方法（POST）透传。

运行：python3 tests/test_plugin_routes.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import plugins
from app.registry import Registry

FAILED = []


def check(label, cond):
    print(("✓" if cond else "✗"), label)
    if not cond:
        FAILED.append(label)


class FakeRoute:
    def __init__(self, path):
        self.path = path


class FakeApp:
    """最小 FastAPI 桩：add_api_route 记录参数，router.routes 供回滚摘除。"""
    def __init__(self):
        self.calls = []
        class R:  # noqa: D106
            routes = []
        self.router = R

    def add_api_route(self, path, endpoint, methods=None, dependencies=None, **kw):
        self.calls.append({"path": path, "endpoint": endpoint,
                           "methods": methods, "dependencies": dependencies})
        self.router.routes.append(FakeRoute(path))


def fake_depends(dep):
    return ("auth-attached", dep)


def fake_auth():
    pass


def write_plugin(tmp, name, body):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write(body)
    return p


def make_registrar(app):
    return plugins.RouteRegistrar(app, fake_auth, fake_depends)


# ---------- 1. 正常注册 ----------
def test_routes_registered():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_plugin(tmp, "mesh_dl.py", """
def register(registry):
    registry.register("mesh.make", {"image": "IMAGE"}, {"mesh": "MESH"})

def register_routes(add_route):
    def dl():
        return {"ok": 1}
    add_route("/ext/meshes/{oid}", dl)
""")
        reg = Registry()
        app = FakeApp()
        rr = make_registrar(app)
        ops = plugins.load_plugin(path, reg, route_registrar=rr)
        check("算子照常注册", ops == ["mesh.make"])
        check("路由注册 1 条", len(app.calls) == 1)
        check("路径带 /ext/ 前缀", app.calls[0]["path"] == "/ext/meshes/{oid}")
        check("auth 依赖自动附加",
              app.calls[0]["dependencies"] == [("auth-attached", fake_auth)])
        check("默认 GET 方法", app.calls[0]["methods"] == ["GET"])
        check("注册器记录了路由（供回滚/归属簿）", len(rr.added) == 1)


# ---------- 2. 前缀违规 → 拒载 + 已加路由回滚 ----------
def test_prefix_enforced_and_rollback():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_plugin(tmp, "evil.py", """
def register(registry):
    registry.register("evil.op", {}, {})

def register_routes(add_route):
    add_route("/ext/ok", lambda: 1)   # 这条合法，先加上
    add_route("/health", lambda: 2)   # 这条违规，必须连带上一条一起滚
""")
        reg = Registry()
        app = FakeApp()
        rr = make_registrar(app)
        try:
            plugins.load_plugin(path, reg, route_registrar=rr)
            check("非 /ext/ 前缀必须拒载", False)
        except ValueError as e:
            check("非 /ext/ 前缀必须拒载", "/ext/" in str(e))
        check("半路失败：已加路由整体回滚（不残留 /ext/ok）",
              app.router.routes == [] and len(rr.added) == 0)


# ---------- 3. 老插件兼容（无 register_routes） ----------
def test_v1_plugin_compat():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_plugin(tmp, "old.py", """
def register(registry):
    registry.register("old.op", {}, {})
""")
        reg = Registry()
        app = FakeApp()
        ops = plugins.load_plugin(path, reg, route_registrar=make_registrar(app))
        check("v1 插件（无路由钩子）行为不变", ops == ["old.op"]
              and app.calls == [])


# ---------- 4. 声明了路由钩子但没给注册器 → 明确失败 ----------
def test_routes_hook_without_registrar_fails():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_plugin(tmp, "orphan.py", """
def register(registry):
    registry.register("orphan.op", {}, {})

def register_routes(add_route):
    add_route("/ext/x", lambda: 1)
""")
        reg = Registry()
        try:
            plugins.load_plugin(path, reg)  # 不给 route_registrar
            check("有钩子无注册器必须明确失败", False)
        except ValueError as e:
            check("有钩子无注册器必须明确失败", "register_routes" in str(e))


# ---------- 5. 负例：摘掉前缀校验 ⇒ 裸路径被接受 ----------
def test_negative_prefix_gate():
    app = FakeApp()
    rr = make_registrar(app)
    orig = plugins.RouteRegistrar.PREFIX
    try:
        plugins.RouteRegistrar.PREFIX = "/"   # 模拟"摘掉校验"
        rr("/health", lambda: 1)              # 裸路径必须能过（证明校验是闸门）
        check("负例：摘掉前缀校验后裸路径被接受", app.calls[0]["path"] == "/health")
    finally:
        plugins.RouteRegistrar.PREFIX = orig
    app2 = FakeApp()
    rr2 = make_registrar(app2)
    try:
        rr2("/health", lambda: 1)
        check("恢复校验后裸路径再次被拒", False)
    except ValueError:
        check("恢复校验后裸路径再次被拒", True)


# ---------- 6. 多方法透传 ----------
def test_methods_passthrough():
    app = FakeApp()
    rr = make_registrar(app)
    rr("/ext/upload", lambda: 1, methods=("POST",))
    check("POST 方法透传", app.calls[0]["methods"] == ["POST"])


def main():
    test_routes_registered()
    test_prefix_enforced_and_rollback()
    test_v1_plugin_compat()
    test_routes_hook_without_registrar_fails()
    test_negative_prefix_gate()
    test_methods_passthrough()
    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()

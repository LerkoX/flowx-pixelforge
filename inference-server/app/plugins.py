"""插件式算子：PLUGINS_DIR（默认 /models/plugins.d，bind-mount 持久化）下的 *.py
自动扫描注册 + 运行时热加载（/admin/plugins 上传接口配套）。

插件文件契约：模块级定义 `register(registry)` 函数；服务端 import 后调用之，
插件内部用 @registry.register(...) 声明一个或多个算子。插件可
`from app import ops` 复用核心原语（sample / vae_encode / resolve_pipe 等）。

安全约束（在 main.py 端点层强制）：INFERENCE_TOKEN 为空时 /admin/* 一律 404；
上传需 sha256 校验 + ast.parse 语法检查 + 原子落盘 + 加载失败回滚。
"""
import ast
import hashlib
import importlib.util
import os
import sys

MAX_PLUGIN_BYTES = 512 * 1024


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_plugin_source(filename: str, content: bytes):
    """上传前置校验：文件名合法 + 体积 + 语法。返回错误信息或 None。"""
    if os.path.basename(filename) != filename or not filename.endswith(".py") \
            or filename.startswith("_"):
        return f"bad filename {filename!r}: 需为纯文件名、.py 结尾、不以 _ 开头"
    if not content:
        return "empty content"
    if len(content) > MAX_PLUGIN_BYTES:
        return f"plugin too large: {len(content)} > {MAX_PLUGIN_BYTES} bytes"
    try:
        ast.parse(content)
    except SyntaxError as e:
        return f"syntax error: {e}"
    return None


def load_plugin(path: str, registry):
    """加载单个插件文件，返回本次新注册/覆盖的算子名列表。"""
    before = set(registry._ops)
    mod_name = "flowx_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # 插件间 import 可见
    spec.loader.exec_module(mod)
    reg = getattr(mod, "register", None)
    if not callable(reg):
        raise ValueError(
            f"plugin {os.path.basename(path)}: 缺少 register(registry) 函数")
    reg(registry)
    return sorted(set(registry._ops) - before)


def scan_plugins(plugins_dir: str, registry):
    """启动时扫描插件目录，逐个加载。单个失败只记日志，不拖垮服务。
    返回 {文件名: [算子名]}。"""
    loaded = {}
    if not os.path.isdir(plugins_dir):
        return loaded
    for fn in sorted(os.listdir(plugins_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(plugins_dir, fn)
        try:
            op_names = load_plugin(path, registry)
            loaded[fn] = op_names
            print(f"[plugins] {fn} -> {op_names}", flush=True)
        except Exception as e:
            print(f"[plugins] {fn} LOAD FAILED: {e}", flush=True)
    return loaded

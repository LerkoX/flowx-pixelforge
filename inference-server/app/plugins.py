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


def plugin_module_name(filename: str) -> str:
    """插件文件的模块名（`load_plugin` 与回滚路径共用同一口径）。"""
    return "flowx_plugin_" + os.path.splitext(os.path.basename(filename))[0]


def drop_bytecode_cache(path: str) -> int:
    """删掉该文件对应的字节码缓存，返回删除个数。

    ⚠️ 真机复现（dev-plan §52.7）：同文件名重传时，若**新内容与旧内容字节数相同**且写入
    **落在同一秒**，CPython 的 .pyc 有效性校验（mtime + size）会判定缓存仍有效 ⇒
    **执行旧代码**：插件热更新静默失效（实测两份 145 字节源码同秒写入，第二次 load_plugin
    仍返回第一次声明的算子名）。缓存文件名用的是**文件 stem**（`<stem>.cpython-310.pyc`），
    与模块名无关，所以按 stem 前缀匹配。
    """
    cache = os.path.join(os.path.dirname(os.path.abspath(path)), "__pycache__")
    if not os.path.isdir(cache):
        return 0
    stem = os.path.splitext(os.path.basename(path))[0]
    n = 0
    for fn in os.listdir(cache):
        if fn.startswith(stem + ".") and fn.endswith((".pyc", ".pyo")):
            try:
                os.remove(os.path.join(cache, fn))
                n += 1
            except OSError:
                pass
    return n


def load_plugin(path: str, registry):
    """加载单个插件文件，返回该插件声明注册的**全部**算子名（含覆盖既有名）。

    registry.register 是覆盖语义（后注册生效），因此"该文件声明了哪些算子"
    与"哪些是新增"是两回事——本函数用录制 shim 如实上报全部声明名，
    跨文件重名/核心算子遮蔽的拒绝策略在上层（main.py upload 端点 409 /
    scan_plugins ERROR 日志），本层不做静默丢弃。
    """
    recorded = []
    orig_register = registry.register

    def _recording(name, inputs, outputs, description=""):
        recorded.append(name)
        return orig_register(name, inputs, outputs, description)

    registry.register = _recording
    try:
        mod_name = plugin_module_name(path)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod  # 插件间 import 可见
        # 必须先失效字节码缓存：同文件名 + 等长内容 + 同秒重传会命中旧 .pyc（见上方注释）
        drop_bytecode_cache(path)
        spec.loader.exec_module(mod)
        reg = getattr(mod, "register", None)
        if not callable(reg):
            raise ValueError(
                f"plugin {os.path.basename(path)}: 缺少 register(registry) 函数")
        reg(registry)
    finally:
        del registry.register  # 摘除实例属性，恢复类方法
    return sorted(set(recorded))


def detect_conflicts(op_names, filename: str, plugin_ops: dict,
                     before_ops: dict):
    """上传冲突检测（纯函数，供端点在 force 判定前调用）。

    op_names: 插件本次声明的全部算子名；plugin_ops: 运行时归属簿
    {op: {"file": ...}}；before_ops: 加载前的 registry._ops 快照。
    冲突 = 算子已存在且归属不同主体：
    - 归属另一插件文件 → owner=该文件名
    - 已存在但不在归属簿 → 核心算子（不可遮蔽）→ owner='core'
    返回 [{"op": 名, "owner": 归属}]，按算子名排序。
    """
    conflicts = []
    for n in op_names:
        info = plugin_ops.get(n)
        if info is not None:
            if info["file"] != filename:
                conflicts.append({"op": n, "owner": info["file"]})
        elif n in before_ops:
            conflicts.append({"op": n, "owner": "core"})
    return sorted(conflicts, key=lambda c: c["op"])


def stale_owners(op_names, filename: str, plugin_ops: dict, plugins_dir: str):
    """找出"归属簿说属于**别的文件**、但那个文件已经不在盘上"的算子（幽灵归属）。

    真机来路：B1 清理遗留插件时手工 `rm plugins.d/xxx.py`（没重启服务）⇒ 归属簿仍写着
    `xxx.py` ⇒ ① 新文件声明同名算子被**假 409** 拦住（owner 是个已删除的文件）
    ② `DELETE /admin/plugins/xxx.py` 反而 404 ⇒ 幽灵条目既拦人又清不掉。
    本函数把它识别出来（返回 op 名列表），端点据此：不算冲突 + 告警 + 接管归属。
    """
    stale = []
    for n in op_names:
        info = plugin_ops.get(n)
        if info is None or info.get("file") == filename:
            continue
        other = info.get("file")
        if other and not os.path.isfile(os.path.join(plugins_dir, other)):
            stale.append(n)
    return sorted(stale)


def reconcile(claiming_file: str, claimed, registry, plugin_ops: dict,
              sha256: str = ""):
    """一次成功加载后的归属簿对账（纯逻辑，可单测）。

    做两件事：
    1. **摘除陈旧算子**：该文件上一次声明、这次不再声明的算子 —— 从 `registry._ops`
       摘掉并从归属簿删除（旧实现从不摘 ⇒ 改名后旧算子永久留在注册表，重启才自愈）；
       只摘"归属仍是本文件"的条目（force 接管过的名字不许误摘别人的）。
    2. **更新归属**：本次声明的算子全部记为 `claiming_file`。
    返回 {"stale_removed": [...], "claimed": [...]}。冲突/失败路径**不得**调用本函数。
    """
    claimed_set = set(claimed)
    stale = sorted(n for n, info in list(plugin_ops.items())
                   if info.get("file") == claiming_file and n not in claimed_set)
    for n in stale:
        registry._ops.pop(n, None)
        plugin_ops.pop(n, None)
    for n in claimed:
        plugin_ops[n] = {"file": claiming_file, "sha256": sha256}
    return {"stale_removed": stale, "claimed": sorted(claimed_set)}


def scan_plugins(plugins_dir: str, registry, reserved=()):
    """启动时扫描插件目录，逐个加载。单个失败只记日志，不拖垮服务。
    返回 {文件名: [算子名]}（算子名为该文件声明的全部名，含覆盖名）。

    reserved: 核心算子名集合——插件遮蔽核心算子是 ERROR（register 覆盖语义
    下仍会生效，但必须喊出来）。跨文件重名同样 ERROR（后加载者生效）。
    """
    loaded = {}
    owners = {}  # op -> 首个声明文件（仅用于日志）
    if not os.path.isdir(plugins_dir):
        return loaded
    for fn in sorted(os.listdir(plugins_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(plugins_dir, fn)
        try:
            op_names = load_plugin(path, registry)
            for n in op_names:
                if n in reserved:
                    print(f"[plugins] ERROR: {fn} 遮蔽核心算子 '{n}'（覆盖语义下"
                          f"插件生效）——请立即修正！", flush=True)
                elif n in owners:
                    print(f"[plugins] ERROR: 算子 '{n}' 跨文件重名：{owners[n]} 与 "
                          f"{fn}（后加载的 {fn} 生效）——请立即删除其一！",
                          flush=True)
                else:
                    owners[n] = fn
            loaded[fn] = op_names
            print(f"[plugins] {fn} -> {op_names}", flush=True)
        except Exception as e:
            print(f"[plugins] {fn} LOAD FAILED: {e}", flush=True)
    return loaded

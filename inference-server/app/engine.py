"""图执行引擎：拓扑排序 + 逐算子调度 + 跨调用结果缓存（对齐 ComfyUI 执行器语义）。

graph 格式（对齐 ComfyUI /prompt）：
{
  "nodes": {
    "1": {"op": "checkpoint.load", "inputs": {"ckpt": "v1-5"}},
    "2": {"op": "clip.encode", "inputs": {"clip": ["1", "clip"], "text": "a cat"}}
  }
}
对象端口引用：["节点ID", "输出端口"]；字面量直接写值。
"""
import datetime
import json
import threading

from . import execution
from .registry import OBJ_TYPES, meta_to_objects, outputs_to_store

_CACHE = {}  # 跨调用缓存：node_key -> {port: meta}
EXEC_LOCK = threading.Lock()  # GPU 串行：同步 /graph、/op 与异步 job worker 互斥

# ---------- 显存护栏（dev-plan §21.3 任务 1/3） ----------

# 注入点（main.py 装配 app.model_manager.VramGuard；测试可注入假护栏）。
# 引擎不依赖 ModelManager 本体：只要求 pin_objects / unpin / evict_for_oom 三个方法。
_VRAM_GUARD = None
# OOM 自愈状态：/health 的 degraded 字段、诊断用
_OOM_STATE = {"count": 0, "recovered": 0, "failed": 0,
              "last_at": None, "last_op": None}
_OOM_TYPES = None


def set_vram_guard(guard):
    global _VRAM_GUARD
    _VRAM_GUARD = guard


def oom_state():
    return dict(_OOM_STATE)


def _oom_exc_types():
    """torch 的 OOM 异常类型（延迟导入：纯 stdlib 环境下测试引擎不需要 torch）。"""
    global _OOM_TYPES
    if _OOM_TYPES is None:
        try:
            import torch
            types = [torch.cuda.OutOfMemoryError]
            if hasattr(torch, "OutOfMemoryError"):
                types.append(torch.OutOfMemoryError)  # 2.5+ 的通用别名
            _OOM_TYPES = tuple(types)
        except Exception:
            _OOM_TYPES = ()
    return _OOM_TYPES


def _note_oom(op_name):
    _OOM_STATE["count"] += 1
    _OOM_STATE["last_op"] = op_name
    _OOM_STATE["last_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")


def _call_op(op, kwargs):
    """执行算子：捕获 CUDA OOM → 淘汰非在用 LRU → 重试一次（自愈）。

    重试仍失败或无可淘汰条目 → 抛 RuntimeError，信息里带可执行指引
    （降档 offload / 卸载常驻模型 / 调小 reserve），不让调用方只看到裸 OOM。
    """
    try:
        return op.fn(**kwargs)
    except _oom_exc_types() as e:
        _note_oom(op.name)
        guard = _VRAM_GUARD
        freed = guard.evict_for_oom(1) if guard is not None else 0
        if freed <= 0:
            _OOM_STATE["failed"] += 1
            from .model_manager import oom_help
            raise RuntimeError(oom_help(op.name, f" 原始错误：{e}")) from e
        print(f"[engine] OOM in '{op.name}' → 已淘汰 {freed} 个非在用模型，重试一次",
              flush=True)
        try:
            out = op.fn(**kwargs)
        except _oom_exc_types() as e2:
            _OOM_STATE["failed"] += 1
            from .model_manager import oom_help
            raise RuntimeError(oom_help(op.name, f" 重试后仍失败：{e2}")) from e2
        _OOM_STATE["recovered"] += 1
        return out


def _run_op(op, kwargs):
    """带显存护栏的算子调用：执行期间 pin 在用模型（淘汰跳过），OOM 时自愈。"""
    guard = _VRAM_GUARD
    keys = guard.pin_objects(kwargs.values()) if guard is not None else []
    try:
        return _call_op(op, kwargs)
    finally:
        if guard is not None:
            guard.unpin(keys)


def _is_ref(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str)


def topo_sort(nodes: dict):
    """Kahn 拓扑排序；检测缺失引用与环。"""
    deps = {nid: set() for nid in nodes}
    for nid, spec in nodes.items():
        for v in (spec.get("inputs") or {}).values():
            if _is_ref(v):
                if v[0] not in nodes:
                    raise ValueError(f"node '{nid}' refs unknown node '{v[0]}'")
                if v[0] != nid:
                    deps[nid].add(v[0])
    order, ready = [], [n for n, d in deps.items() if not d]
    while ready:
        nid = ready.pop()
        order.append(nid)
        for other, d in deps.items():
            if nid in d:
                d.discard(nid)
                if not d and other not in order and other not in ready:
                    ready.append(other)
    if len(order) != len(nodes):
        leftover = sorted(set(nodes) - set(order))
        raise ValueError(f"graph has a cycle involving nodes: {leftover}")
    return order


def _node_key(nodes, nid, memo):
    """节点的内容寻址 key：算子名 + 递归展开的输入（引用用上游 key，字面量用值）。"""
    if nid in memo:
        return memo[nid]
    spec = nodes[nid]
    parts = []
    for k, v in sorted((spec.get("inputs") or {}).items()):
        if _is_ref(v):
            parts.append([k, "ref", _node_key(nodes, v[0], memo), v[1]])
        else:
            parts.append([k, "lit", v])
    key = json.dumps([spec["op"], parts], ensure_ascii=False)
    memo[nid] = key
    return key


def _cache_valid(store, meta: dict):
    return all("id" not in m or store.contains(m["id"]) for m in meta.values())


def run_graph(store, registry, graph: dict, cache=None):
    """执行整张图。返回 {"nodes": {nid: {port: meta}}, "cached": [命中缓存的节点]}。"""
    nodes = graph.get("nodes") or {}
    if not nodes:
        raise ValueError("empty graph: 'nodes' is required")
    cache = _CACHE if cache is None else cache
    order = topo_sort(nodes)

    results, out_meta, cached, memo = {}, {}, [], {}
    with EXEC_LOCK:
        for nid in order:
            execution.check_cancelled()  # 节点间取消检查点（异步 job 上下文生效）
            spec = nodes[nid]
            if "op" not in spec:
                raise ValueError(f"node '{nid}' missing 'op'")
            op = registry.get(spec["op"])
            raw_inputs = spec.get("inputs") or {}

            key = _node_key(nodes, nid, memo)
            hit = cache.get(key)
            if hit is not None and _cache_valid(store, hit):
                results[nid] = meta_to_objects(store, op, hit)
                out_meta[nid] = hit
                cached.append(nid)
                print(f"[engine] node '{nid}' ({op.name}) -> cache hit", flush=True)
                continue

            kwargs = registry.resolve_in_process(op, raw_inputs, results, out_meta)
            out = _run_op(op, kwargs)
            results[nid] = out
            out_meta[nid] = outputs_to_store(store, op, out)
            cache[key] = out_meta[nid]
            print(f"[engine] node '{nid}' ({op.name}) -> executed", flush=True)

    return {"nodes": out_meta, "cached": cached}


def run_op(store, registry, name: str, raw_inputs: dict):
    """单算子调用（/op）：对象端口用 {"$id": uuid} 引用。"""
    op = registry.get(name)
    with EXEC_LOCK:
        kwargs = registry.resolve_from_ids(store, op, raw_inputs or {})
        out = _run_op(op, kwargs)
        return {"outputs": outputs_to_store(store, op, out)}


def clear_cache():
    _CACHE.clear()

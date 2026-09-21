"""算子注册表：声明式类型规范 + 输入解析/类型校验（对齐 ComfyUI 的 INPUT_TYPES/RETURN_TYPES）。

对象类型：MODEL / CLIP / VAE / COND / LATENT / IMAGE / VIDEO / CONTROL_NET / CONTROL
/ UPSCALE_MODEL ——
以对象仓库 UUID 引用传递（VIDEO 落盘 mp4，仓库内为 {"path", "fps", "num_frames"} 元数据，
见 app/video.py；CONTROL_NET = ControlNetModel，CONTROL = controlnet.apply 的捆绑产物，见 app/ops.py；
UPSCALE_MODEL = spandrel 加载的像素放大模型句柄，见 plugins/upscale_ops.py）
字面量类型：INT / FLOAT / STRING / BOOL —— 直接传值
"""
import json

OBJ_TYPES = ("MODEL", "CLIP", "VAE", "COND", "LATENT", "IMAGE", "VIDEO",
             "CONTROL_NET", "CONTROL", "UPSCALE_MODEL")
LIT_TYPES = ("INT", "FLOAT", "STRING", "BOOL")


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"cannot cast {v!r} to BOOL")


_LIT_CAST = {"INT": int, "FLOAT": float, "STRING": str, "BOOL": _to_bool}


class Op:
    def __init__(self, name, fn, inputs, outputs, description=""):
        self.name = name
        self.fn = fn
        self.inputs = inputs    # {端口名: 类型}
        self.outputs = outputs  # {端口名: 类型}
        self.description = description
        for group in (inputs, outputs):
            for port, typ in group.items():
                if typ not in OBJ_TYPES + LIT_TYPES:
                    raise ValueError(f"op '{name}' port '{port}': unknown type '{typ}'")

    def cast_literal(self, typ, raw):
        try:
            return _LIT_CAST[typ](raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"op '{self.name}': cannot cast {raw!r} to {typ}: {e}")

    def spec(self):
        return {"name": self.name, "inputs": self.inputs,
                "outputs": self.outputs, "description": self.description}


class Registry:
    def __init__(self):
        self._ops = {}

    def register(self, name, inputs, outputs, description=""):
        def deco(fn):
            self._ops[name] = Op(name, fn, inputs, outputs, description)
            return fn
        return deco

    def get(self, name) -> Op:
        op = self._ops.get(name)
        if op is None:
            raise KeyError(
                f"unknown op '{name}', available: {sorted(self._ops)}")
        return op

    def list(self):
        return [op.spec() for op in self._ops.values()]

    def resolve_from_ids(self, store, op: Op, raw_inputs: dict):
        """/op 单算子调用：对象端口用 {"$id": uuid} 引用，从对象仓库解析。"""
        kwargs = {}
        for name, typ in op.inputs.items():
            if name not in raw_inputs:
                continue  # 未提供的输入落到算子函数的 Python 默认值
            raw = raw_inputs[name]
            if typ in OBJ_TYPES:
                oid = raw.get("id") or raw.get("$id") if isinstance(raw, dict) else None
                if not oid:
                    raise ValueError(
                        f"op '{op.name}' input '{name}' expects object ref {{\"$id\": ...}} ({typ})")
                kwargs[name] = store.get(oid, typ)["data"]
            else:
                if isinstance(raw, (dict, list)):
                    raise ValueError(
                        f"op '{op.name}' input '{name}' expects literal ({typ})")
                kwargs[name] = op.cast_literal(typ, raw)
        return kwargs

    def resolve_in_process(self, op: Op, raw_inputs: dict, results: dict, meta: dict):
        """/graph 图执行：对象端口用 ["node_id", "port"] 引用，直接取进程内对象。"""
        kwargs = {}
        for name, typ in op.inputs.items():
            if name not in raw_inputs:
                continue  # 未提供的输入落到算子函数的 Python 默认值
            raw = raw_inputs[name]
            if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[0], str):
                src, port = raw
                if src not in results or port not in results[src]:
                    raise ValueError(
                        f"op '{op.name}' input '{name}': bad ref [{src}, {port}]")
                src_type = meta[src][port]["type"]
                if src_type != typ:
                    raise TypeError(
                        f"op '{op.name}' input '{name}' expects {typ}, "
                        f"got {src_type} from [{src}, {port}]")
                kwargs[name] = results[src][port]
            else:
                if typ in OBJ_TYPES:
                    raise ValueError(
                        f"op '{op.name}' input '{name}' expects ref [node, port] ({typ})")
                kwargs[name] = op.cast_literal(typ, raw)
        return kwargs


def outputs_to_store(store, op: Op, out: dict):
    """算子返回值落对象仓库，生成 {port: {"id"/"value", "type"}} 元数据。"""
    meta = {}
    for port, typ in op.outputs.items():
        if port not in out:
            raise ValueError(f"op '{op.name}' did not return output '{port}'")
        v = out[port]
        if typ in OBJ_TYPES:
            oid = store.put(typ, v)
            meta[port] = {"id": oid, "type": typ}
        else:
            meta[port] = {"value": v, "type": typ}
    return meta


def meta_to_objects(store, op: Op, meta: dict):
    """元数据还原为对象字典（缓存命中时使用）。"""
    out = {}
    for port, typ in op.outputs.items():
        m = meta[port]
        out[port] = store.get(m["id"], typ)["data"] if typ in OBJ_TYPES else m["value"]
    return out

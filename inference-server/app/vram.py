"""显存预算计算（纯 stdlib，可单测；dev-plan §21.3 任务 1）。

把"这批模型装不下"这件事**算清楚**，且不依赖 torch/GPU：
- `estimate_bytes()`：待加载模型的实际体积（读 safetensors header 精确计算，
  而不是文件大小 —— 同样的文件按 fp16/fp32 加载显存占用差一倍）；
- `plan_eviction()`：给定当前余量与已常驻条目，决定淘汰谁（LRU + pin）。

model_manager 只负责取运行时数值（`torch.cuda.mem_get_info()`、已常驻条目的
体积/最近使用时间）并应用结果；决策逻辑放在这里，便于在没有 GPU 的环境单测。

不假设卡型（dev-plan §9.6 纪律 #1）：只看运行时余量与 reserve 配置。
"""
import json
import os
import struct

# safetensors header 里出现的 dtype → 每元素字节数
DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
               "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
               "BOOL": 1, "I1": 1}
# 加载精度 → 每元素字节数（模型加载后的权重占用按目标 dtype 计）
_TARGET_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
_HEADER_LIMIT = 64 * 1024 * 1024  # header 长度上限（防损坏文件把内存吃光）
_WEIGHT_EXT = (".safetensors", ".bin", ".ckpt", ".pt")


def target_bytes(dtype):
    """目标 dtype 的每元素字节数；未知按 fp16（服务默认精度）。"""
    return _TARGET_BYTES.get(dtype, 2)


def safetensors_bytes(path, dtype="fp16"):
    """safetensors 权重按目标 dtype 加载后的字节数（读 header 精确计算）。

    返回 None = 无法解析（损坏/非 safetensors），调用方回退到文件大小。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                return None
            n = struct.unpack("<Q", head)[0]
            if n <= 0 or n > _HEADER_LIMIT:
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None
    tgt = target_bytes(dtype)
    total = 0.0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offs = meta.get("data_offsets")
        if not (isinstance(offs, (list, tuple)) and len(offs) == 2):
            continue
        try:
            nbytes = max(0, int(offs[1]) - int(offs[0]))
        except (TypeError, ValueError):
            continue
        src = DTYPE_BYTES.get(str(meta.get("dtype") or "").upper())
        if not src:  # 未知 dtype：保守按目标精度算（不放大）
            src = tgt
        total += nbytes * tgt / src
    return int(total)


def estimate_bytes(path, dtype="fp16"):
    """待加载体积估算（字节）：目录 = 递归累加权重文件，单文件按格式算。

    None = 估不出来（调用方按 0 处理 ⇒ 只靠条目上限兜底，并打日志提示）。
    """
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            got = safetensors_bytes(path, dtype)
            if got is not None:
                return got
        try:
            # ckpt/pt 等 pickle 格式：文件大小是安全上界（fp32 存 → fp16 加载 ≤ 文件大小）
            return os.path.getsize(path)
        except OSError:
            return None
    if os.path.isdir(path):
        total = 0
        found = False
        for root, _dirs, files in os.walk(path):
            for fn in files:
                if not fn.endswith(_WEIGHT_EXT):
                    continue
                p = os.path.join(root, fn)
                got = None
                if fn.endswith(".safetensors"):
                    got = safetensors_bytes(p, dtype)
                if got is None:
                    try:
                        got = os.path.getsize(p)
                    except OSError:
                        continue
                total += got
                found = True
        return total if found else None
    return None


def plan_eviction(resident, free_bytes, need_bytes, reserve_bytes,
                  max_resident, pinned=()):
    """决定淘汰哪些常驻条目，返回按淘汰顺序排列的 key 列表。

    resident: [(key, bytes, last_used)]（体积未知写 0）；`pinned` 内的条目不动
    （执行中在用，淘汰它等于把手上正用的模型抽走）。
    两条停止条件（**都**满足才停）：
      1) 条目数：剩余 < max_resident（max_resident<=0 = 不限条目数，只看显存）；
      2) 显存：free + 已淘汰体积 >= need_bytes + reserve_bytes（need_bytes<=0 时跳过）。
    候选耗尽仍未满足 → 返回当前列表，由调用方决定放行（告警）还是报错。
    """
    pinned = set(pinned or ())
    candidates = sorted((r for r in resident if r[0] not in pinned),
                        key=lambda r: r[2])
    cap = max_resident if (max_resident and max_resident > 0) else 0
    evicted, freed = [], 0
    for key, size, _ts in candidates:
        try:
            size = int(size or 0)
        except (TypeError, ValueError):
            size = 0
        count_ok = (cap == 0) or (len(resident) - len(evicted) < cap)
        mem_ok = (need_bytes <= 0) or (free_bytes + freed >= need_bytes + reserve_bytes)
        if count_ok and mem_ok:
            break
        evicted.append(key)
        freed += size
    return evicted

"""FlowX 推理服务 = 远程能力运行时（ComfyUI 语义的公共能力层）。

接口：
- 系统：/health、/images/{id}（PNG 下载 / thumb 缩略图）、/videos/{id}（mp4 下载）、
  /gc（清对象仓库+缓存）、/model/unload（显式卸载常驻模型，等价 model.unload 算子）
- 能力运行时（同步）：/ops（列出算子）、/op（单算子）、/graph（整图执行，带缓存）
- 能力运行时（异步）：POST /jobs 提交 → GET /jobs/{id} 轮询状态/进度 →
  POST /interrupt 取消；视频等分钟级任务走此通道（同步 HTTP 会超时）

新增能力不再动本文件：算子全部由节点包携带 server_op.py 经 POST /admin/plugins
自注册（见 nodes/*/server_op.py）；本服务只保留引擎/对象仓库/任务体系/模型管理
与插件机制本身。
"""
import gc as _gc  # 别名：与下方 /gc 端点函数名冲突
import io
import os

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from . import engine, ops, plugins, preview, sniff
from . import model_manager as _mm
from .jobs import JobManager
from .model_manager import OFFLOAD_MODE, QUANTIZATION, ModelManager
from .object_store import ObjectStore
from .registry import Registry

TOKEN = os.environ.get("INFERENCE_TOKEN", "")
if not TOKEN:
    # 硬依赖（算子全面插件化起）：核心不再内置任何算子，全部经节点包
    # server_op.py → POST /admin/plugins 自注册；无 token 时 /admin/* 关闭，
    # 服务启动后注册表为空、任何节点都跑不动，等价不可用——宁可启动即失败。
    raise RuntimeError(
        "INFERENCE_TOKEN 未配置：算子已全面插件化（节点包 server_op.py 经 "
        "/admin/plugins 自注册），无 token 时 /admin/* 关闭、服务无任何算子可用。"
        "请设置 INFERENCE_TOKEN 环境变量后重启，并在流水线节点的 service_token "
        "参数填入同一值。")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "/videos")
# 插件算子目录：默认放 MODELS_DIR 下（bind-mount 持久化，重建容器不丢）
PLUGINS_DIR = os.environ.get("PLUGINS_DIR", "/models/plugins.d")
# 插件算子名 -> {"file": 文件名, "sha256": 内容哈希}；/ops 透出供客户端版本比对
plugin_ops = {}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "32")) * 1024 * 1024

app = FastAPI(title="flowx-inference-server")
models = ModelManager()
store = ObjectStore(ttl_seconds=int(os.environ.get("OBJECT_TTL_SECONDS", "3600")),
                    video_dir=VIDEO_DIR)
# 模型淘汰联动：LRU 顶掉的管道仍被对象仓库的 model/clip/vae 视图钉住，
# 必须在 evict 时按对象身份断开引用（先断引用后 GC，见 ModelManager._evict_if_needed）
models.on_evict = lambda pipe: store.discard_where(lambda d: d is pipe)
# 反向保护：对象仓库仍引用的常驻条目视同执行中 pin，淘汰计划跳过
# （并发加载竞态：后完成的大模型入驻检查不得淘汰先完成、对象已被下游
# 绑定待消费的模型——exec442 的 InstantID CN 误杀事故）
models.extra_pinned = lambda: models.keys_of(store.referenced_data())
# 显存护栏注入引擎：算子执行期间 pin 在用模型（淘汰跳过），OOM 时淘汰非在用条目后重试
engine.set_vram_guard(_mm.VramGuard(models))
registry = Registry()
os.makedirs(VIDEO_DIR, exist_ok=True)


def auth(authorization: str = Header(default="")):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid token")


def admin_auth(authorization: str = Header(default="")):
    """管理接口闸门：TOKEN 未配置时整个 /admin/* 不可用（404 而非 401，
    避免暴露能力存在性）；配置后与 auth 同规则。代码上传=远程执行能力，
    公网隧道场景必须配 token 才允许开启。"""
    if not TOKEN:
        raise HTTPException(status_code=404, detail="admin API disabled "
                            "(set INFERENCE_TOKEN to enable)")
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid token")


# ---------- 插件算子（启动扫描 + 运行时上传热加载） ----------

# 启动扫描：plugins.d 下既有插件全量注册（核心不再内置算子，reserved 恒为空集，
# 保留参数位防未来重新引入核心算子时失去防遮蔽告警）
_core_ops = set(registry._ops)
for _fn, _op_names in plugins.scan_plugins(PLUGINS_DIR, registry,
                                           reserved=_core_ops).items():
    _p = os.path.join(PLUGINS_DIR, _fn)
    with open(_p, "rb") as _f:
        _h = plugins.sha256_of(_f.read())
    for _n in _op_names:
        plugin_ops[_n] = {"file": _fn, "sha256": _h}


class PluginUpload(BaseModel):
    filename: str
    content: str
    sha256: str
    force: bool = False  # 显式接管跨文件重名算子（默认拒绝 409）


@app.post("/admin/plugins", dependencies=[Depends(admin_auth)])
def upload_plugin(p: PluginUpload):
    """上传插件（节点自注册通道）：hash 校验 → 语法检查 → 原子落盘 → 热加载。
    加载失败回滚文件；同文件重传为幂等更新；
    跨文件重名/遮蔽核心算子默认 409（全量回滚），force=true 显式接管
    （归属簿转移；旧文件仍在盘上，重启扫描顺序可能改变归属，建议尽快删除旧文件）。"""
    content = p.content.encode("utf-8")
    if plugins.sha256_of(content) != p.sha256:
        raise HTTPException(status_code=400, detail="sha256 mismatch")
    err = plugins.check_plugin_source(p.filename, content)
    if err:
        raise HTTPException(status_code=400, detail=err)
    os.makedirs(PLUGINS_DIR, exist_ok=True)
    path = os.path.join(PLUGINS_DIR, p.filename)
    backup = None
    if os.path.isfile(path):
        with open(path, "rb") as f:
            backup = f.read()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, path)
    before = dict(registry._ops)  # 注册表快照（Op 对象不可变，浅拷贝足够）
    try:
        claimed = plugins.load_plugin(path, registry)
    except Exception as e:
        # 回滚：恢复旧文件或删除；注册表全量恢复快照
        registry._ops.clear()
        registry._ops.update(before)
        if backup is not None:
            with open(path, "wb") as f:
                f.write(backup)
        else:
            os.unlink(path)
        raise HTTPException(status_code=400,
                            detail=f"plugin load failed (rolled back): {e}")
    conflicts = plugins.detect_conflicts(claimed, p.filename, plugin_ops, before)
    if conflicts and not p.force:
        # 冲突拒绝：注册表 + 文件全量回滚，409 报明归属
        registry._ops.clear()
        registry._ops.update(before)
        if backup is not None:
            with open(path, "wb") as f:
                f.write(backup)
        else:
            os.unlink(path)
        raise HTTPException(status_code=409, detail={
            "error": "op conflicts (rolled back)",
            "conflicts": conflicts,
            "hint": "同文件重传为幂等更新；确需接管其他文件的算子请显式传 "
                    "force=true（接管后建议删除旧文件，避免重启扫描顺序漂移）"})
    h = plugins.sha256_of(content)
    took_over = [c["op"] for c in conflicts]  # force 路径才有
    if took_over:
        print(f"[plugins] WARN: {p.filename} force 接管算子 {took_over}；"
              f"旧文件仍在盘上，重启后归属按扫描顺序可能漂移，建议删除旧文件",
              flush=True)
    for n in claimed:
        plugin_ops[n] = {"file": p.filename, "sha256": h}
    print(f"[plugins] uploaded {p.filename} -> {claimed}", flush=True)
    return {"plugin": p.filename, "ops": claimed, "sha256": h,
            "took_over": took_over}


@app.get("/admin/plugins", dependencies=[Depends(admin_auth)])
def list_plugins():
    files = {}
    for name, info in plugin_ops.items():
        files.setdefault(info["file"], {"sha256": info["sha256"],
                                          "ops": []})["ops"].append(name)
    return {"plugins_dir": PLUGINS_DIR, "plugins": files}


@app.delete("/admin/plugins/{filename}", dependencies=[Depends(admin_auth)])
def delete_plugin(filename: str):
    """卸载插件：删文件 + 从注册表摘除其算子（被引用中的执行不受影响）。"""
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="bad filename")
    path = os.path.join(PLUGINS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="plugin not found")
    removed = [n for n, info in list(plugin_ops.items())
               if info["file"] == filename]
    for n in removed:
        registry._ops.pop(n, None)
        plugin_ops.pop(n, None)
    os.unlink(path)
    return {"deleted": filename, "unregistered": removed}


# ---------- schemas ----------

class OpReq(BaseModel):
    name: str
    inputs: dict = Field(default_factory=dict)

class GraphReq(BaseModel):
    nodes: dict

class JobReq(BaseModel):
    """异步任务提交：nodes 非空 = graph 任务；否则 name = op 任务。"""
    nodes: dict | None = None
    name: str | None = None
    inputs: dict = Field(default_factory=dict)

class InterruptReq(BaseModel):
    job_id: str | None = None


# ---------- 系统端点 ----------

@app.get("/health")
def health():
    """健康检查 + 显存治理诊断（dev-plan §21.3 任务 1/3/4 的数据源）。

    vram_free_mb / resident_models 供 inference-ensure 的显存闸门（min_vram_mb）判断；
    resident_details 给每个常驻条目的体积估算与在用状态；degraded 记录 OOM 自愈史。
    """
    info = {"status": "ok", "cuda_available": torch.cuda.is_available(),
            "resident_models": models.resident(),
            "resident_details": models.details(),
            "offload_mode": OFFLOAD_MODE, "quantization": QUANTIZATION,
            "max_resident_models": _mm.MAX_RESIDENT,
            "vram_budget_eviction": _mm.VRAM_BUDGET,
            "vram_reserve_mb": _mm.VRAM_RESERVE_MB,
            "degraded": engine.oom_state()}
    try:  # 版本可观测性（验收/排障时要确认跑的是哪套 diffusers/torch）
        import diffusers as _diffusers
        info["diffusers_version"] = _diffusers.__version__
    except Exception:  # pragma: no cover
        pass
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        reserved = allocated = 0
        try:
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
        except Exception:  # pragma: no cover - 老驱动/异常时忽略
            pass
        reclaimable = max(0, reserved - allocated)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            # vram_free_mb = 驱动余量 + 本进程可回收缓存（"下一个模型能用多少"）。
            # 采样后驱动余量常为 0（缓存分配器留着复用），只看它会把正常态误报成退化。
            "vram_free_mb": round((free + reclaimable) / 1024**2),
            "vram_total_mb": round(total / 1024**2),
            "vram_driver_free_mb": round(free / 1024**2),
            "vram_reclaimable_mb": round(reclaimable / 1024**2),
            "vram_process_reserved_mb": round(reserved / 1024**2),
            "vram_process_allocated_mb": round(allocated / 1024**2),
            "resident_weights_mb": round(models.resident_weights() / 1024**2),
        })
        # 可用量低于 reserve 视为退化态（采样会走 host memory 兜底 → 分钟级/步）
        info["vram_low"] = (free + reclaimable) < _mm.VRAM_RESERVE_MB * 1024**2
        if info["vram_low"]:
            info["status"] = "degraded"
    return info


@app.get("/models", dependencies=[Depends(auth)])
def list_models():
    return {"resident_models": models.resident()}


@app.get("/models/files", dependencies=[Depends(auth)])
def list_model_files_endpoint():
    """磁盘模型文件清单（节点 widget 模型名下拉数据源）。
    按 checkpoint/vae/controlnet/upscale/embedding/lora/motion 分类。"""
    return {"files": sniff.list_model_files(_mm.MODELS_DIR,
                                            _mm.LORAS_DIR)}


@app.post("/images", dependencies=[Depends(auth)])
async def upload_image(request: Request):
    """上传图像（原始字节，Content-Type: image/png|jpeg），登记为 IMAGE 对象返回 id。
    供 FlowX 的 load-image 节点把客户端本地图片送入对象仓库。"""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"image too large: {len(body)} > {MAX_UPLOAD_BYTES} bytes")
    try:
        pil = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}")
    oid = store.put("IMAGE", pil, meta={"source": "upload"})
    return {"id": oid, "width": pil.width, "height": pil.height}


@app.get("/images/{image_id}", dependencies=[Depends(auth)])
def get_image(image_id: str, index: int = 0, thumb: int = 0):
    """下载图像 PNG；thumb>0 时返回最长边为该像素的 JPEG 缩略图（供节点画布预览）。"""
    try:
        data = store.get(image_id, "IMAGE")["data"]
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    pil = data if not isinstance(data, list) else data[index]
    if thumb > 0:
        pil = pil.copy()
        pil.thumbnail((thumb, thumb))
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=80)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/preview/{key}", dependencies=[Depends(auth)])
def get_preview(key: str):
    """采样实时预览帧：key 为异步 job_id（uuid 不可猜）。返回最新一帧 JPEG
    （每 job 只留一帧），X-Preview-Progress 头带进度；无帧 404。
    Studio 从节点 stdout 的 FLOWX_PREVIEW 标记拿到本地址后中转拉取给画布。"""
    frame = preview.BUFFER.get(key)
    if frame is None:
        raise HTTPException(status_code=404, detail="preview frame not found")
    jpeg, progress = frame
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache",
                             "X-Preview-Progress": f"{progress:.4f}"})


@app.get("/videos/{video_id}", dependencies=[Depends(auth)])
def get_video(video_id: str):
    """下载视频 mp4（对标 /images/{id}）：VIDEO 对象在 put 时已编码落盘。"""
    try:
        data = store.get(video_id, "VIDEO")["data"]
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    path = data.get("path") if isinstance(data, dict) else None
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404,
                            detail="video file missing (expired or not persisted)")
    return FileResponse(path, media_type="video/mp4",
                        filename=os.path.basename(path))


class UnloadReq(BaseModel):
    """target 空 = 全部（便于 curl -X POST .../model/unload -d '{}'）。"""
    target: str = ""


@app.post("/model/unload", dependencies=[Depends(auth)])
def model_unload(req: UnloadReq | None = None):
    """显式卸载常驻模型（等价 model.unload 算子；运维/自愈用）。
    body {"target": "majicmixRealistic_v7"} 或 {}（全部）。"""
    target = (req.target if req is not None else "") or ""
    return ops.model_unload(models, target)


@app.post("/gc", dependencies=[Depends(auth)])
def gc():
    # 注意：/gc 只清对象仓库与图缓存，**不卸常驻模型**（历史坑：清了缓存模型照样占显存）；
    # 要腾显存用 /model/unload（或 model.unload 算子）。
    engine.clear_cache()
    n = store.clear()
    # 清了引用还不够：管道组件/offload 钩子互相引用形成循环，
    # 必须 gc.collect 才真正回收（名字里叫 gc 就要做全套）。
    _gc.collect()
    torch.cuda.empty_cache()
    return {"cleared": n}


# ---------- 能力运行时端点（同步） ----------

@app.get("/ops", dependencies=[Depends(auth)])
def list_ops():
    specs = registry.list()
    for s in specs:
        info = plugin_ops.get(s["name"])
        if info:
            s["plugin_hash"] = info["sha256"]
            s["plugin_file"] = info["file"]
    return {"ops": specs}


@app.post("/op", dependencies=[Depends(auth)])
def run_op(req: OpReq):
    """单算子调用；对象端口用 {"$id": uuid} 引用，字面量直接传值。"""
    try:
        return engine.run_op(store, registry, req.name, req.inputs)
    except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
        # RuntimeError 含显存自愈失败的可执行指引（dev-plan §21.3 任务 3）：
        # 必须作为 detail 回到调用方节点，否则只剩一句 "HTTP 500"
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph", dependencies=[Depends(auth)])
def run_graph(req: GraphReq):
    """整图执行：拓扑调度 + 跨调用缓存；对象端口用 ["node_id", "port"] 引用。"""
    try:
        return engine.run_graph(store, registry, {"nodes": req.nodes})
    except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- 能力运行时端点（异步任务） ----------

def _execute_job(kind, payload):
    """job worker 的执行入口：分派到引擎（与同步端点共用缓存与执行锁）。"""
    if kind == "graph":
        return engine.run_graph(store, registry, {"nodes": payload["nodes"]})
    return engine.run_op(store, registry, payload["name"], payload.get("inputs") or {})


jobman = JobManager(_execute_job)


@app.post("/jobs", dependencies=[Depends(auth)])
def submit_job(req: JobReq):
    """提交异步任务（视频等分钟级任务走此通道）：返回 job_id，轮询 GET /jobs/{id}。
    body 二选一：{"nodes": {...}}（graph，同 /graph）或 {"name", "inputs"}（op，同 /op）。"""
    if req.nodes:
        return {"job_id": jobman.submit("graph", {"nodes": req.nodes}),
                "status": "pending"}
    if req.name:
        return {"job_id": jobman.submit(
            "op", {"name": req.name, "inputs": req.inputs}),
                "status": "pending"}
    raise HTTPException(status_code=400,
                        detail="job requires 'nodes' (graph) or 'name' (op)")


@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs():
    return {"jobs": jobman.list()}


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    """任务状态/进度轮询：done 时带 result（同 /graph 返回），failed/cancelled 时带 error。"""
    try:
        return jobman.get(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/interrupt", dependencies=[Depends(auth)])
def interrupt(req: InterruptReq | None = None):
    """取消任务：带 job_id 取消指定任务；空 body 取消当前 running + 全部 pending。
    running 任务在下一个检查点（节点间 / 采样每步）生效。"""
    try:
        return jobman.interrupt(None if req is None else req.job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

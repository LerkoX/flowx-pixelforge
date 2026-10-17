"""FlowX 推理服务 = 远程能力运行时（ComfyUI 语义的公共能力层）。

接口：
- 系统：/health、/gpu（NVML 硬件指标：利用率/温度/功耗/风扇/PCIe/时钟）、
  /metrics/history（服务端采样历史：监控页图表数据源）、/monitor（监控页）、
  /images/{id}（PNG 下载 / thumb 缩略图）、/videos/{id}（mp4 下载）、
  POST /images（图片上传）、POST /videos（视频上传，字节入站 + 返回服务端路径）、
  /gc（清对象仓库+缓存）、/model/unload（显式卸载常驻模型，等价 model.unload 算子）
  /model/policy（读/写显存淘汰策略旋钮，运维节点 vram-ops 用）
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
import re
import sys
import time

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from . import engine, gpu_probe, lease, metrics_history, ops, plugins, policy, preview, sniff, sys_probe
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
# 视频上传落盘目录（POST /videos）：load-video 节点把 FlowX 侧本地视频经此送入
# 服务端，随后由 video.load 算子解码成 VIDEO 对象；与 VIDEO_DIR（对象 mp4 落盘）
# 分开——上传物是「原片」，对象目录是「仓库产物」，两者生命周期不同。
VIDEO_INPUT_DIR = os.environ.get(
    "VIDEO_INPUT_DIR", os.path.join(os.environ.get("INPUT_DIR", "/input"), "videos"))
# 插件算子目录：默认放 MODELS_DIR 下（bind-mount 持久化，重建容器不丢）
PLUGINS_DIR = os.environ.get("PLUGINS_DIR", "/models/plugins.d")
# 监控页等静态资源（GET /monitor）；随 COPY app/ 进镜像，按本文件路径定位
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# 插件算子名 -> {"file": 文件名, "sha256": 内容哈希}；/ops 透出供客户端版本比对
plugin_ops = {}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "32")) * 1024 * 1024
# 视频上传另有体积上限（原片动辄数百 MB，不宜与图片共用 32MB 闸门）
MAX_VIDEO_UPLOAD_BYTES = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "512")) * 1024 * 1024

app = FastAPI(title="flowx-inference-server")
models = ModelManager()
store = ObjectStore(ttl_seconds=int(os.environ.get("OBJECT_TTL_SECONDS", "3600")),
                    video_dir=VIDEO_DIR,
                    # B2：孤儿对象回收（租约结束 + 龄期超限 → 回收）。
                    # 实测环境 OBJECT_TTL_SECONDS=86400（24h），没有这层回收的话，
                    # 上一轮流水线留下的 model/clip/vae 视图能把 7.4GB 常驻条目
                    # 钉一整天（exec 795：靠人工 model.unload 才松开）。
                    lease_alive=lambda lid: models.leases.is_active(lid),
                    orphan_ttl_seconds=int(os.environ.get(
                        "OBJECT_TTL_ORPHAN_SECONDS", "1800")))
# 模型淘汰联动：LRU 顶掉的管道仍被对象仓库的 model/clip/vae 视图钉住，
# 必须在 evict 时按对象身份断开引用（先断引用后 GC，见 ModelManager._evict_if_needed）
models.on_evict = lambda pipe: store.discard_where(lambda d: d is pipe)
# 反向保护：对象仓库仍引用的常驻条目视同执行中 pin，淘汰计划跳过
# （并发加载竞态：后完成的大模型入驻检查不得淘汰先完成、对象已被下游
# 绑定待消费的模型——exec442 的 InstantID CN 误杀事故）
# B2 修正（exec 795）：只保护**本租约**创建的对象。原先不看租约，把上一轮/另一条
# 流水线遗留的 model/clip/vae 视图也当“不可淘汰”（配上 24h 对象 TTL ⇒ flux 的
# 7.4GB 常驻条目钉了一整天，与 SDXL+CN 一起把 8GB 顶爆）。跨租约的保护由
# lease.keys_of_others 负责，不靠这一层。
models.extra_pinned = lambda: models.keys_of(
    store.referenced_data(lease.current()))
# 显存护栏注入引擎：算子执行期间 pin 在用模型（淘汰跳过），OOM 时淘汰非在用条目后重试
engine.set_vram_guard(_mm.VramGuard(models))
registry = Registry()
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)


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


def _bind_lease(x_flowx_lease: str):
    """X-FlowX-Lease 头 → 当前线程租约绑定 + 活动刷新（TTL 滑动）。
    返回净化后的租约 id（无头/非法 → None，调用方行为与租约机制引入前一致）。"""
    lid = lease.sanitize(x_flowx_lease)
    lease.bind(lid)
    if lid:
        models.leases.touch(lid)
    return lid


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
        sys.modules.pop(plugins.plugin_module_name(p.filename), None)
        if backup is not None:
            with open(path, "wb") as f:
                f.write(backup)
        else:
            os.unlink(path)
        raise HTTPException(status_code=400,
                            detail=f"plugin load failed (rolled back): {e}")
    # 幽灵归属（A1）：归属簿写的文件已不在盘上（手工 rm 后未重启）⇒ 不算冲突，
    # 否则新文件被"已删除的文件"假 409 拦住、且 DELETE 反 404 清不掉。这里只在
    # **本地视图**里剔除幽灵（不动全局 plugin_ops，失败路径必须零副作用）。
    stale = plugins.stale_owners(claimed, p.filename, plugin_ops, PLUGINS_DIR)
    stale_set = set(stale)
    if stale:
        print(f"[plugins] WARN: {p.filename} 发现幽灵归属 {stale}"
              f"（归属簿指向已不存在的文件）——本次视为空闲，接管后清理", flush=True)
    owner_view = {k: v for k, v in plugin_ops.items() if k not in stale_set}
    before_view = {k: v for k, v in before.items() if k not in stale_set}
    conflicts = plugins.detect_conflicts(claimed, p.filename, owner_view, before_view)
    if conflicts and not p.force:
        # 冲突拒绝：注册表 + 文件 + 模块缓存全量回滚，409 报明归属
        registry._ops.clear()
        registry._ops.update(before)
        sys.modules.pop(plugins.plugin_module_name(p.filename), None)
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
    # 归属簿对账（A2）：摘除本文件"上次声明、这次不再声明"的陈旧算子（旧实现从不摘，
    # 改名后旧算子永久留在注册表），再登记本次声明的归属。
    rep = plugins.reconcile(p.filename, claimed, registry, plugin_ops, sha256=h)
    if rep["stale_removed"]:
        print(f"[plugins] {p.filename} 摘除陈旧算子 {rep['stale_removed']}", flush=True)
    # 幽灵条目：文件已不在盘上；本次没接着声明的名字，注册表里也要摘掉（避免无主算子）
    adopted = []
    for n in stale:
        if n in set(claimed):
            adopted.append(n)
        else:
            registry._ops.pop(n, None)
            plugin_ops.pop(n, None)
    print(f"[plugins] uploaded {p.filename} -> {claimed}", flush=True)
    return {"plugin": p.filename, "ops": claimed, "sha256": h,
            "took_over": took_over, "stale_removed": rep["stale_removed"],
            "adopted_stale": sorted(adopted)}


@app.get("/admin/plugins", dependencies=[Depends(admin_auth)])
def list_plugins():
    files = {}
    for name, info in plugin_ops.items():
        ent = files.setdefault(info["file"], {"sha256": info["sha256"], "ops": [],
                                              "missing_file": True})
        ent["ops"].append(name)
        if os.path.isfile(os.path.join(PLUGINS_DIR, info["file"])):
            ent["missing_file"] = False
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
def health(modules: int = 0):
    """健康检查 + 显存治理诊断（dev-plan §21.3 任务 1/3/4 的数据源）。

    vram_free_mb / resident_models 供 inference-ensure 的显存闸门（min_vram_mb）判断；
    resident_details 给每个常驻条目的体积估算与在用状态；degraded 记录 OOM 自愈史。

    B1 新增（exec 795 事故）：`vram_oversubscribed` / `vram_oversub_ratio` /
    `vram_headroom_mb`（物理口径真余量）/ `vram_budget_mb`（按 offload 分账后的
    显存预算）。为什么必须加：WSL2/WDDM 下 CUDA 超订**不报 OOM**，旧的
    `vram_low`（乐观口径 driver_free + reclaimable）在兜底态下被缓存掩盖成
    false、status 恒 ok —— 跑了 100 分钟的 PCIe 抖动没有任何接口能看出来。
    `?modules=1` 额外返回 `module_map`（每个常驻条目的顶层组件在哪个设备上，
    用于实锤 offload 到底有没有把 text_encoder/vae 搬下车）。
    """
    info = {"status": "ok", "cuda_available": torch.cuda.is_available(),
            "resident_models": models.resident(),
            "resident_details": models.details(),
            "offload_mode": OFFLOAD_MODE, "quantization": QUANTIZATION,
            "max_resident_models": _mm.MAX_RESIDENT,
            "vram_budget_eviction": _mm.VRAM_BUDGET,
            "vram_reserve_mb": _mm.VRAM_RESERVE_MB,
            "vram_strict": _mm.KNOBS.get("vram_strict", 0),
            "offload_floor_mb": _mm.KNOBS.get("offload_floor_mb"),
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
        # B1：物理口径（兜底态检测）——与上面的乐观口径互补，不替代
        occ = models.occupancy()
        info.update({
            "vram_oversubscribed": occ["oversubscribed"],
            "vram_saturated": occ["saturated"],
            "vram_oversub_ratio": occ["oversub_ratio"],
            "vram_headroom_mb": round(occ["headroom_bytes"] / 1024**2),
            "vram_budget_mb": round(models.vram_budget_bytes() / 1024**2),
        })
        if occ["oversubscribed"]:
            # 超订态：采样正在经 PCIe 从主机内存取权重，必须显式报出行来
            # （否则前端与闸门都把这种“静默退化”当健康，exec 795 教训）
            info["status"] = "degraded"
        if modules:
            info["module_map"] = {k: models.module_map(k)
                                  for k in models.resident()}
    return info


@app.get("/gpu", dependencies=[Depends(auth)])
def gpu_stats():
    """GPU 硬件指标（监控页数据源）：利用率/温度/功耗/风扇 + NVML 口径显存。

    pynvml 为可选依赖：不可用时 available=False，监控页降级只画 /health 的显存区。
    """
    return gpu_probe.snapshot()


@app.get("/monitor")
def monitor_page():
    """监控页面（单文件 HTML，轮询 /health /gpu /leases /jobs 自绘图表）。

    页面本身不带数据所以不设鉴权；数据端点各有 Bearer 闸门，token 在页面内
    输入后存浏览器 localStorage，不经服务端。
    """
    return FileResponse(os.path.join(STATIC_DIR, "monitor.html"),
                        media_type="text/html")


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


@app.post("/videos", dependencies=[Depends(auth)])
async def upload_video(request: Request, filename: str = ""):
    """上传视频（原始字节，Content-Type: video/mp4|octet-stream），落盘到
    VIDEO_INPUT_DIR 并返回**服务端路径**——供 load-video 节点把 FlowX 侧本地视频
    送入服务端（对称于 POST /images）。

    本端点只做字节入站：解码成 VIDEO 对象由 `video.load` 算子完成（算子全面
    插件化后 main.py 不再耦合算子实现）。filename 只用来保留扩展名与可读性，
    落盘名带时间戳前缀，不覆盖既有文件。
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    if len(body) > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"video too large: {len(body)} > {MAX_VIDEO_UPLOAD_BYTES} bytes "
                   f"(raise MAX_VIDEO_UPLOAD_MB to allow bigger uploads)")
    base = os.path.basename(filename or "upload.mp4") or "upload.mp4"
    stem, ext = os.path.splitext(base)
    ext = (ext or ".mp4").lower()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:80] or "upload"
    name = f"{int(time.time() * 1000)}_{safe}{ext}"
    path = os.path.join(VIDEO_INPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(body)
    print(f"[upload] video {name} ({len(body)} bytes) -> {path}", flush=True)
    return {"path": path, "filename": name, "bytes": len(body)}


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


@app.get("/model/policy", dependencies=[Depends(auth)])
def model_policy_get():
    """读显存/淘汰策略旋钮（可写项 + 只读观测项）。

    定位：单流水线串行时的策略实验工具（dev-plan §P1）。并发执行下动旋钮会互相干扰。
    """
    _attach_policy()
    snap = policy.snapshot()
    snap["lease_ttl_seconds"] = lease.LEASE_TTL
    snap["lease_wait_timeout_seconds"] = lease.LEASE_WAIT_TIMEOUT
    snap["offload_mode"] = OFFLOAD_MODE
    snap["quantization"] = QUANTIZATION
    return snap


class PolicyReq(BaseModel):
    """可写旋钮（只传要改的字段）；其它字段视为非法。

    B1/B2/B4 新增（exec 795）：vram_strict / offload_floor_mb /
    object_ttl_orphan_seconds —— 容器是手工 docker run 的，env 改不了，
    这三个旋钮必须能热改（沿用现有的 /model/policy 通道）。
    """
    max_resident_models: int | None = None
    vram_reserve_mb: int | None = None
    vram_strict: int | None = None
    offload_floor_mb: int | None = None
    object_ttl_orphan_seconds: int | None = None


@app.post("/model/policy", dependencies=[Depends(auth)])
def model_policy_set(req: PolicyReq):
    """写策略旋钮（写后立即生效：淘汰逻辑每次调用都读模块级常量）。

    body 例：{"vram_reserve_mb": 768} / {"vram_strict": 1}
    （vram_strict=1：物理装不下时拒绝加载而非掉进 host memory 兜底）。
    越界或不可写键 ⇒ 400。参考 GET /model/policy 的当前值。
    """
    _attach_policy()
    patch = {k: v for k, v in (req.model_dump().items() if hasattr(req, "model_dump")
                               else req.dict().items()) if v is not None}
    if not patch:
        writable = sorted(policy.WRITERS) or ["max_resident_models",
                                              "vram_reserve_mb"]
        raise HTTPException(status_code=400, detail="policy: 需至少给一个可写字段"
                            f"（{' / '.join(writable)}）")
    try:
        return policy.update(patch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _attach_policy():
    """把 model_manager 的模块级旋钮注入 policy（每次调用幂等重挂，防测试替换后失效）。"""
    policy.attach(
        readers={"max_resident_models": lambda: _mm.MAX_RESIDENT,
                 "vram_reserve_mb": lambda: _mm.VRAM_RESERVE_MB,
                 # B1/B2/B4 运行时旋钮（env 给初值；这里可热改，不必重建容器）
                 "vram_strict": lambda: int(_mm.KNOBS.get("vram_strict", 0)),
                 "offload_floor_mb": lambda: int(_mm.KNOBS.get("offload_floor_mb", 0)),
                 "object_ttl_orphan_seconds": lambda: int(
                     store.orphan_ttl_seconds or 0)},
        writers={"max_resident_models": lambda v: setattr(_mm, "MAX_RESIDENT", v),
                 "vram_reserve_mb": lambda v: setattr(_mm, "VRAM_RESERVE_MB", v),
                 "vram_strict": lambda v: _mm.KNOBS.__setitem__(
                     "vram_strict", 1 if int(v) else 0),
                 "offload_floor_mb": lambda v: _mm.KNOBS.__setitem__(
                     "offload_floor_mb", int(v)),
                 "object_ttl_orphan_seconds": lambda v: setattr(
                     store, "orphan_ttl_seconds", int(v))},
    )


@app.post("/model/unload", dependencies=[Depends(auth)])
def model_unload(req: UnloadReq | None = None,
                 x_flowx_lease: str = Header(default="")):
    """显式卸载常驻模型（等价 model.unload 算子；运维/自愈用）。
    body {"target": "majicmixRealistic_v7"} 或 {}（全部）。
    其它活跃租约持有的条目跳过（X-FlowX-Lease 头标识本租约，§43.2）——
    有跳过时响应带 skipped / skipped_reasons / skipped_detail 三键说明原因。"""
    _bind_lease(x_flowx_lease)
    try:
        target = (req.target if req is not None else "") or ""
        return ops.model_unload(models, target)
    finally:
        lease.unbind()


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
def run_op(req: OpReq, x_flowx_lease: str = Header(default="")):
    """单算子调用；对象端口用 {"$id": uuid} 引用，字面量直接传值。"""
    _bind_lease(x_flowx_lease)
    try:
        try:
            return engine.run_op(store, registry, req.name, req.inputs)
        except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
            # RuntimeError 含显存自愈失败的可执行指引（dev-plan §21.3 任务 3）：
            # 必须作为 detail 回到调用方节点，否则只剩一句 "HTTP 500"
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        lease.unbind()


@app.post("/graph", dependencies=[Depends(auth)])
def run_graph(req: GraphReq, x_flowx_lease: str = Header(default="")):
    """整图执行：拓扑调度 + 跨调用缓存；对象端口用 ["node_id", "port"] 引用。"""
    _bind_lease(x_flowx_lease)
    try:
        try:
            return engine.run_graph(store, registry, {"nodes": req.nodes})
        except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        lease.unbind()


# ---------- 能力运行时端点（异步任务） ----------

def _execute_job(kind, payload):
    """job worker 的执行入口：分派到引擎（与同步端点共用缓存与执行锁）。"""
    if kind == "graph":
        return engine.run_graph(store, registry, {"nodes": payload["nodes"]})
    return engine.run_op(store, registry, payload["name"], payload.get("inputs") or {})


# 续租回调（B2）：长任务（如 90 分钟的采样）每次进度心跳刷新租约活跃时间 ——
# 否则租约 TTL 一到就“看似执行结束”，导致本执行的模型失去并发保护、
# 对象仓库把它的对象当孤儿回收（exec 795 采样跑了 5425s，TTL 只有 1800s）。
jobman = JobManager(_execute_job,
                    lease_touch=lambda lid: models.leases.touch(lid))

# 指标历史采样：服务端后台线程（2s/帧，环形缓冲 6h），监控页图表数据源；
# 依赖在此装配（对象仓库统计 + 任务队列 + 模型盘占用）
metrics_history.start(store=store, jobman=jobman,
                      disk_paths=[("models", _mm.MODELS_DIR)])


@app.get("/metrics/history", dependencies=[Depends(auth)])
def metrics_history_feed(minutes: int = 60):
    """指标历史回放（监控页图表数据源）：最近 minutes 分钟的采样帧（升序）。

    服务端后台线程常驻采样，页面刷新/未打开期间历史不丢；进程重启清零。
    """
    return metrics_history.history(minutes)


@app.post("/jobs", dependencies=[Depends(auth)])
def submit_job(req: JobReq, x_flowx_lease: str = Header(default="")):
    """提交异步任务（视频等分钟级任务走此通道）：返回 job_id，轮询 GET /jobs/{id}。
    body 二选一：{"nodes": {...}}（graph，同 /graph）或 {"name", "inputs"}（op，同 /op）。
    X-FlowX-Lease 头携带的执行租约随 job 记录，worker 执行期间生效（§43.2）。"""
    lid = _bind_lease(x_flowx_lease)
    try:
        if req.nodes:
            return {"job_id": jobman.submit("graph", {"nodes": req.nodes},
                                            lease=lid),
                    "status": "pending"}
        if req.name:
            return {"job_id": jobman.submit(
                "op", {"name": req.name, "inputs": req.inputs},
                lease=lid),
                    "status": "pending"}
        raise HTTPException(status_code=400,
                            detail="job requires 'nodes' (graph) or 'name' (op)")
    finally:
        lease.unbind()


@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs():
    """任务队列总览（监控页）：jobs 列表 + queue 汇总（worker 忙闲 / 队头等待）。

    每条 job 带 title（op 名或 graph·N 节点）/ ops / stage（当前节点）/ queue_pos
    （pending 排队位次）/ wait_seconds（排队时长）——旧版只有 kind="graph|op"，
    监控页无法分辨是哪个算子、卡在哪个节点。
    """
    return {"jobs": jobman.list(), "queue": jobman.queue_state()}


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str, x_flowx_lease: str = Header(default="")):
    """任务状态/进度轮询：done 时带 result（同 /graph 返回），failed/cancelled 时带 error。
    轮询也算租约活动（节点每 2s 轮询 = 长任务期间租约 TTL 持续滑动刷新）。"""
    lid = lease.sanitize(x_flowx_lease)
    if lid:
        models.leases.touch(lid)
    try:
        return jobman.get(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/leases", dependencies=[Depends(auth)])
def list_leases():
    """执行租约观测（dev-plan §43.2）：活跃租约及其持有的常驻条目/空闲时长。"""
    return {"leases": models.leases.view(),
            "lease_ttl_seconds": lease.LEASE_TTL,
            "lease_wait_timeout_seconds": lease.LEASE_WAIT_TIMEOUT}


@app.delete("/leases/{lease_id}", dependencies=[Depends(auth)])
def release_lease(lease_id: str):
    """显式释放执行租约（监控页清理僵尸租约）：执行已结束但 TTL 未到期的
    残留租约会挡住 /model/unload（全部条目被视为「其它租约持有」而跳过，
    实机 exec772 事故）。只摘除租约登记、不卸模型——正在执行的算子仍受
    pin 保护；释放后其名下常驻条目失去并发保护，调用前确认对应执行已结束。"""
    if not models.leases.release(lease_id):
        raise HTTPException(status_code=404,
                            detail=f"lease '{lease_id}' 不存在或已过期")
    print(f"[lease] released '{lease_id}'（显式释放，监控页）", flush=True)
    return {"released": lease_id}


@app.post("/interrupt", dependencies=[Depends(auth)])
def interrupt(req: InterruptReq | None = None):
    """取消任务：带 job_id 取消指定任务；空 body 取消当前 running + 全部 pending。
    running 任务在下一个检查点（节点间 / 采样每步）生效。"""
    try:
        return jobman.interrupt(None if req is None else req.job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

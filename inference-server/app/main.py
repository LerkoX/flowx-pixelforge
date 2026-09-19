"""FlowX 推理服务 = 远程能力运行时（ComfyUI 语义的公共能力层）。

接口：
- 系统：/health、/images/{id}（PNG 下载 / thumb 缩略图）、/videos/{id}（mp4 下载）、
  /gc（清对象仓库+缓存）
- 能力运行时（同步）：/ops（列出算子）、/op（单算子）、/graph（整图执行，带缓存）
- 能力运行时（异步）：POST /jobs 提交 → GET /jobs/{id} 轮询状态/进度 →
  POST /interrupt 取消；视频等分钟级任务走此通道（同步 HTTP 会超时）

新增能力只需在下方注册一个算子函数，无需新增端点。
"""
import gc as _gc  # 别名：与下方 /gc 端点函数名冲突
import io
import os

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from . import engine, execution, ops, preview
from .jobs import JobManager
from .model_manager import OFFLOAD_MODE, QUANTIZATION, ModelManager
from .object_store import ObjectStore
from .registry import Registry

TOKEN = os.environ.get("INFERENCE_TOKEN", "")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "/videos")
EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "/models/embeddings")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "32")) * 1024 * 1024

app = FastAPI(title="flowx-inference-server")
models = ModelManager()
store = ObjectStore(ttl_seconds=int(os.environ.get("OBJECT_TTL_SECONDS", "3600")),
                    video_dir=VIDEO_DIR)
# 模型淘汰联动：LRU 顶掉的管道仍被对象仓库的 model/clip/vae 视图钉住，
# 必须在 evict 时按对象身份断开引用（先断引用后 GC，见 ModelManager._evict_if_needed）
models.on_evict = lambda pipe: store.discard_where(lambda d: d is pipe)
registry = Registry()
os.makedirs(VIDEO_DIR, exist_ok=True)


def auth(authorization: str = Header(default="")):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid token")


# ---------- 算子注册（新能力在这里加一行） ----------

@registry.register(
    "checkpoint.load",
    inputs={"ckpt": "STRING"},
    outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
    description="加载 checkpoint 到显存（幂等），输出 model/clip/vae 三个对象")
def _op_checkpoint_load(ckpt):
    return ops.checkpoint_load(models, ckpt)


@registry.register(
    "clip.encode",
    inputs={"clip": "CLIP", "text": "STRING"},
    outputs={"cond": "COND"},
    description="CLIP Text Encode：文本 → conditioning")
def _op_clip_encode(clip, text):
    return ops.clip_encode(clip, text)


@registry.register(
    "latent.empty",
    inputs={"width": "INT", "height": "INT", "batch_size": "INT"},
    outputs={"latent": "LATENT"},
    description="Empty Latent Image：按宽高/batch 创建零 latent（均可省略，默认 512x512x1）")
def _op_latent_empty(width=512, height=512, batch_size=1):
    return ops.latent_empty(width, height, batch_size)


@registry.register(
    "sample",
    inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "latent": "LATENT",
            "seed": "INT", "steps": "INT", "cfg": "FLOAT",
            "sampler_name": "STRING", "denoise": "FLOAT",
            "preview_every": "INT"},
    outputs={"latent": "LATENT", "seed": "INT"},
    description="KSampler：seed/steps/cfg/sampler_name/denoise 均有默认值；"
                "异步 job 执行且 preview_every>0 时，逐步把 latent 预览帧（JPEG）"
                "留在 GET /preview/{job_id}（只留最新一帧），供 Studio 中转拉取")
def _op_sample(model, pos, neg, latent, seed=-1, steps=20, cfg=7.0,
               sampler_name="euler", denoise=1.0, preview_every=1):
    hint = "sd15"
    if preview_every > 0:
        pipe, _ = ops.resolve_pipe(model)
        hint = preview.model_hint_of(pipe)

    def on_step(latents, i, total):
        job = execution.current()
        if job is None:
            return  # 同步 /op 调用无 job 上下文：不录预览（预览走异步 job 通道）
        job.set_progress(i + 1, total)  # 异步 job 进度（无需预览也上报）
        if preview_every > 0:
            rec = preview.recorder_for(job.id, preview_every, hint)
            if rec.want(i, total):
                rec.push(latents, (i + 1) / total)

    return ops.sample(model, pos, neg, latent, seed, steps, cfg,
                      sampler_name, denoise, preview_cb=on_step,
                      interrupt_check=execution.check_cancelled)


@registry.register(
    "motion.load",
    inputs={"ckpt": "STRING", "motion": "STRING"},
    outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
    description="Motion 加载：SD1.x checkpoint + MotionAdapter → AnimateDiffPipeline（文生视频）。"
                "ckpt 为 MODELS_DIR 下底模名（同 checkpoint.load）；motion 为 MODELS_DIR/motion/ 下的"
                " diffusers 目录名或 safetensors 文件名（可省略扩展名）。直接收底模名而非 MODEL 引用："
                "组合需全新实例化底模，预先 checkpoint.load 会白占一份内存")
def _op_motion_load(ckpt, motion):
    return ops.motion_load(models, ckpt, motion)


@registry.register(
    "lora.apply",
    inputs={"model": "MODEL", "lora": "STRING", "strength": "FLOAT"},
    outputs={"model": "MODEL", "clip": "CLIP"},
    description="LoRA 加载：给 MODEL 挂增量补丁（可多个串联叠加），strength 默认 1.0；lora 为 LORAS_DIR 下文件名（可省略扩展名）")
def _op_lora_apply(model, lora, strength=1.0):
    return ops.lora_apply(models, model, lora, strength)


@registry.register(
    "vae.decode",
    inputs={"vae": "VAE", "latent": "LATENT"},
    outputs={"image": "IMAGE"},
    description="VAE Decode：latent → PIL 图像")
def _op_vae_decode(vae, latent):
    return ops.vae_decode(vae, latent)


@registry.register(
    "vae.encode",
    inputs={"vae": "VAE", "image": "IMAGE"},
    outputs={"latent": "LATENT"},
    description="VAE Encode：PIL 图像 → latent（图生图入口，配 sample 的 denoise<1 使用）")
def _op_vae_encode(vae, image):
    return ops.vae_encode(vae, image)


def _video_on_step(preview_every):
    """视频采样逐步回调：job 进度上报 + 进度卡片预览（3D latent 无法廉价投影）。"""
    def on_step(latents, i, total):
        job = execution.current()
        if job is None:
            return  # 同步 /op 调用无 job 上下文：不录预览
        job.set_progress(i + 1, total)  # 异步 job 进度（轮询通道）
        if preview_every > 0:
            rec = preview.recorder_for(job.id, preview_every)
            if rec.want(i, total):
                # 视频 3D latent 无法廉价投影成图，录纯渲染的进度卡片帧
                rec.push_pil(preview.progress_card(i + 1, total), (i + 1) / total)
    return on_step


@registry.register(
    "video.sample",
    inputs={"model": "MODEL", "prompt": "STRING", "neg_prompt": "STRING",
            "image": "IMAGE", "width": "INT", "height": "INT",
            "num_frames": "INT", "fps": "INT", "steps": "INT", "cfg": "FLOAT",
            "seed": "INT", "decode_chunk_size": "INT",
            "preview_every": "INT"},
    outputs={"video": "VIDEO", "seed": "INT"},
    description="Video Sample：文/图生视频，按管道签名自适应（Wan TI2V 文本+可选首帧 / "
                "SVD 纯图生视频，cfg 映射 min/max_guidance_scale，fps 进采样条件）。"
                "分钟级任务，请经 POST /jobs 异步执行；进度经 job 轮询上报，"
                "preview_every>0 时进度卡片帧留在 GET /preview/{job_id}；"
                "/interrupt 可取消")
def _op_video_sample(model, prompt="", neg_prompt="", image=None,
                     width=832, height=480, num_frames=121, fps=24,
                     steps=50, cfg=5.0, seed=-1, decode_chunk_size=0,
                     preview_every=0):
    return ops.video_sample(model, prompt, neg_prompt, image, width, height,
                            num_frames, fps, steps, cfg, seed, decode_chunk_size,
                            preview_cb=_video_on_step(preview_every),
                            interrupt_check=execution.check_cancelled)


@registry.register(
    "video.sample_latent",
    inputs={"model": "MODEL", "prompt": "STRING", "neg_prompt": "STRING",
            "image": "IMAGE", "width": "INT", "height": "INT",
            "num_frames": "INT", "fps": "INT", "steps": "INT", "cfg": "FLOAT",
            "seed": "INT", "preview_every": "INT"},
    outputs={"latent": "LATENT", "seed": "INT"},
    description="Video Sample（latent 模式）：只采样不解码，输出 3D latent 供 "
                "vae.decode_video 接力——decode 精度（fp32）与分块成为流水线可调参数。"
                "分钟级任务，请经 POST /jobs 异步执行")
def _op_video_sample_latent(model, prompt="", neg_prompt="", image=None,
                            width=832, height=480, num_frames=121, fps=24,
                            steps=50, cfg=5.0, seed=-1, preview_every=0):
    return ops.video_sample(model, prompt, neg_prompt, image, width, height,
                            num_frames, fps, steps, cfg, seed,
                            output_type="latent",
                            preview_cb=_video_on_step(preview_every),
                            interrupt_check=execution.check_cancelled)


@registry.register(
    "vae.decode_video",
    inputs={"vae": "VAE", "latents": "LATENT", "num_frames": "INT",
            "decode_chunk_size": "INT", "force_fp32": "BOOL", "fps": "INT"},
    outputs={"video": "VIDEO"},
    description="VAE Decode（视频）：3D latent → VIDEO。force_fp32 防 Pascal fp16 "
                "解码过曝/亮度漂移；decode_chunk_size 分块控制显存峰值（SVD 有效）")
def _op_vae_decode_video(vae, latents, num_frames=0, decode_chunk_size=14,
                         force_fp32=True, fps=24):
    return ops.vae_decode_video(vae, latents, num_frames, decode_chunk_size,
                                force_fp32, fps)


@registry.register(
    "embedding.load",
    inputs={"clip": "CLIP", "names": "STRING"},
    outputs={"clip": "CLIP"},
    description="Textual Inversion 加载：把 EMBEDDINGS_DIR 下的 embedding（逗号分隔，"
                "如 badhandv4,EasyNegative）载进 CLIP 文本编码器；之后正/反提示词里直接写"
                "该词生效（常用于负面压制缺陷）。幂等，已加载自动跳过")
def _op_embedding_load(clip, names):
    return ops.embedding_load(clip, EMBEDDINGS_DIR, names)


@registry.register(
    "image.upscale",
    inputs={"image": "IMAGE", "scale": "FLOAT", "width": "INT",
            "height": "INT", "method": "STRING"},
    outputs={"image": "IMAGE"},
    description="Image Upscale：按 scale 倍率（默认 2.0）或目标 width/height 放大图像，"
                "method 支持 lanczos/bicubic/bilinear/nearest；结果对齐 8 的倍数。"
                "hires.fix 前置：放大 → vae.encode → sample(denoise 0.3~0.5) 精修细节")
def _op_image_upscale(image, scale=2.0, width=0, height=0, method="lanczos"):
    return ops.image_upscale(image, scale, width, height, method)


@registry.register(
    "image.load",
    inputs={"name": "STRING"},
    outputs={"image": "IMAGE"},
    description="Load Image：读 INPUT_DIR 下的服务端本地图片（仅文件名）；客户端上传用 POST /images")
def _op_image_load(name):
    return ops.image_load(INPUT_DIR, name)


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
    info = {"status": "ok", "cuda_available": torch.cuda.is_available(),
            "resident_models": models.resident(),
            "offload_mode": OFFLOAD_MODE, "quantization": QUANTIZATION}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_free_mb": round(free / 1024**2),
            "vram_total_mb": round(total / 1024**2),
        })
    return info


@app.get("/models", dependencies=[Depends(auth)])
def list_models():
    return {"resident_models": models.resident()}


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


@app.post("/gc", dependencies=[Depends(auth)])
def gc():
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
    return {"ops": registry.list()}


@app.post("/op", dependencies=[Depends(auth)])
def run_op(req: OpReq):
    """单算子调用；对象端口用 {"$id": uuid} 引用，字面量直接传值。"""
    try:
        return engine.run_op(store, registry, req.name, req.inputs)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph", dependencies=[Depends(auth)])
def run_graph(req: GraphReq):
    """整图执行：拓扑调度 + 跨调用缓存；对象端口用 ["node_id", "port"] 引用。"""
    try:
        return engine.run_graph(store, registry, {"nodes": req.nodes})
    except (KeyError, TypeError, ValueError) as e:
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

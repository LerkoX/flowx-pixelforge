"""FlowX 推理服务 = 远程能力运行时（ComfyUI 语义的公共能力层）。

接口：
- 系统：/health、/images/{id}（PNG 下载 / thumb 缩略图）、/gc（清对象仓库+缓存）
- 能力运行时：/ops（列出算子）、/op（单算子）、/graph（整图执行，带缓存）

新增能力只需在下方注册一个算子函数，无需新增端点。
"""
import io
import os

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import engine, ops, preview
from .model_manager import ModelManager
from .object_store import ObjectStore
from .registry import Registry

TOKEN = os.environ.get("INFERENCE_TOKEN", "")

app = FastAPI(title="flowx-inference-server")
models = ModelManager()
store = ObjectStore(ttl_seconds=int(os.environ.get("OBJECT_TTL_SECONDS", "3600")))
registry = Registry()


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
            "preview_callback_url": "STRING", "preview_token": "STRING",
            "preview_every": "INT"},
    outputs={"latent": "LATENT", "seed": "INT"},
    description="KSampler：seed/steps/cfg/sampler_name/denoise 均有默认值；"
                "提供 preview_callback_url 时逐步 POST latent 预览帧（JPEG）到该地址")
def _op_sample(model, pos, neg, latent, seed=-1, steps=20, cfg=7.0,
               sampler_name="euler", denoise=1.0, preview_callback_url="",
               preview_token="", preview_every=1):
    pusher = None
    if preview_callback_url:
        pipe, _ = ops.resolve_pipe(model)
        pusher = preview.PreviewPusher(
            preview_callback_url, preview_token, every=preview_every,
            hint=preview.model_hint_of(pipe))

    def on_step(latents, i, total):
        if pusher is not None and pusher.want(i, total):
            pusher.push(latents, (i + 1) / total)

    return ops.sample(model, pos, neg, latent, seed, steps, cfg,
                      sampler_name, denoise, preview_cb=on_step)


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


# ---------- schemas ----------

class OpReq(BaseModel):
    name: str
    inputs: dict = Field(default_factory=dict)

class GraphReq(BaseModel):
    nodes: dict


# ---------- 系统端点 ----------

@app.get("/health")
def health():
    info = {"status": "ok", "cuda_available": torch.cuda.is_available(),
            "resident_models": models.resident()}
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


@app.post("/gc", dependencies=[Depends(auth)])
def gc():
    engine.clear_cache()
    return {"cleared": store.clear()}


# ---------- 能力运行时端点 ----------

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

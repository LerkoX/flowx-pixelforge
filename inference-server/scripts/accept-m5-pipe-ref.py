#!/usr/bin/env python3
"""M5 官方管道参考图生成（在推理容器内运行，output 写 INPUT_DIR）。

用途：给 accept-m5.py 的 t9 提供"官方 diffusers 管道"出图，做逐像素比对
（验证 FlowX 的手动采样循环 + SDXL added_cond_kwargs 与 diffusers
StableDiffusionXLPipeline.denoising loop 完全一致）。

为什么要在**另一个容器**里跑：8GB 卡装不下两个常驻 SDXL 进程
（UNet 5.1G + 双 TE 1.9G，即便 offload=model 也 >7G）⇒ 先让服务端
`model.unload`（accept-m5 t9 已做），再用同一镜像起一次性容器跑本脚本。

用法（在能连到推理 daemon 的机器上）：
  docker -H tcp://<解析后IP>:<port> run --rm --gpus all \
    --volumes-from flowx-inference-server \
    -e CKPT=<模型名> -e SIZE=1024 -e STEPS=25 -e CFG=6.0 -e SEED=5150 \
    -v <repo>/inference-server/scripts:/scripts \
    --entrypoint python3 flowx-inference-server:pascal \
    /scripts/accept-m5-pipe-ref.py

产出：INPUT_DIR/accept-m5-ref.png（INPUT_DIR 绑定挂载在宿主机 D:\\flowx-data\\input）
      之后 accept-m5.py 用 image.load 读取该图并与服务端出图逐像素比对。
环境变量：CKPT（必填）/SIZE/STEPS/CFG/PROMPT/NEG/SEED/OFFLOAD（默认 model）
          OUT_NAME（默认 accept-m5-ref.png）/INPUT_DIR（默认 /input）
"""
import os
import time

import torch

from app import ops
from app.model_manager import ModelManager

CKPT = os.environ.get("CKPT", "")
SIZE = int(os.environ.get("SIZE", "1024"))
STEPS = int(os.environ.get("STEPS", "25"))
CFG = float(os.environ.get("CFG", "6.0"))
SEED = int(os.environ.get("SEED", "5150"))
OFFLOAD = os.environ.get("OFFLOAD", "model")
PROMPT = os.environ.get("PROMPT", "a cute cat sitting on a windowsill, "
                                   "soft sunset light, masterpiece, "
                                   "best quality")
NEG = os.environ.get("NEG", "lowres, blurry, worst quality, watermark, text")
OUT_NAME = os.environ.get("OUT_NAME", "accept-m5-ref.png")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")


def main():
    if not CKPT:
        raise SystemExit("需指定 CKPT=<模型名>")
    models = ModelManager()
    key, _ = models.load(CKPT, dtype="fp16", offload=OFFLOAD)
    pipe = models.get(key)
    device = ops.exec_device_of(pipe)

    gen = torch.Generator(device=device).manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=PROMPT, negative_prompt=NEG, width=SIZE, height=SIZE,
                   num_inference_steps=STEPS, guidance_scale=CFG,
                   generator=gen, output_type="pil")
    dt = time.time() - t0
    pil = out.images[0]

    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, OUT_NAME)
    pil.save(path)
    import hashlib
    with open(path, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    print(f"[pipe-ref] {type(pipe).__name__} size={pil.size} steps={STEPS} "
          f"cfg={CFG} seed={SEED} {dt:.0f}s -> {path} sha1={sha[:12]}",
          flush=True)


if __name__ == "__main__":
    main()

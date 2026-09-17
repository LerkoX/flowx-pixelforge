"""M3 验收：VAE fp32 防黑图 —— 连续 N 次 vae.decode 无黑图。

用法：
  # 容器内（默认 http://127.0.0.1:8100）
  docker exec -i flowx-inference-server python /dev/stdin < scripts/accept-vae-fp32.py
  # 容器外（隧道 / 局域网）
  BASE=http://192.168.5.211:8100 python3 scripts/accept-vae-fp32.py

流程：checkpoint.load → clip.encode ×2 → 循环 N 次
（sample(steps=2, seed 递增, denoise=1) → vae.decode → GET /images/{id} 统计亮度）。
判定：任一解码结果均亮度 < BLACK_MEAN 视为黑图，失败退出码 1。
"""
import io
import json
import os
import sys
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8100").rstrip("/")
TOKEN = os.environ.get("TOKEN", "")
N = int(os.environ.get("N", "50"))
BLACK_MEAN = float(os.environ.get("BLACK_MEAN", "2.0"))  # 0~255，低于此值视为黑图


def call(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def op(name, **inputs):
    # 对象端口用 {"$id": ...} 引用，字面量直接传值（str 以字面量处理）
    wrapped = {k: ({"$id": v} if k in ("vae", "model", "clip", "cond", "latent",
                                      "pos", "neg", "image") else v)
               for k, v in inputs.items()}
    resp = call("/op", {"name": name, "inputs": wrapped})
    return {k: m.get("id", m.get("value")) for k, m in resp.get("outputs", {}).items()}


def image_stats(image_id):
    req = urllib.request.Request(f"{BASE}/images/{image_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(raw)).convert("L")
    st = ImageStat.Stat(im)
    return st.mean[0], st.stddev[0]


def main():
    ckpt = os.environ.get("CKPT", "v1-5-pruned-emaonly")
    print(f"[accept] base={BASE} ckpt={ckpt} rounds={N}", flush=True)

    out = op("checkpoint.load", ckpt=ckpt)
    vae, model, clip = out["vae"], out["model"], out["clip"]
    print(f"[accept] checkpoint loaded: vae={vae}", flush=True)

    pos = op("clip.encode", clip=clip,
             text="a colorful landscape photo, vibrant, high quality")["cond"]
    neg = op("clip.encode", clip=clip, text="")["cond"]

    latent0 = op("latent.empty", width=512, height=512, batch_size=1)["latent"]

    worst = (1e9, None)
    black = 0
    for i in range(N):
        lat = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
                 seed=1000 + i, steps=2, cfg=7.0,
                 sampler_name="euler", denoise=1.0)["latent"]
        img = op("vae.decode", vae=vae, latent=lat)["image"]
        mean, std = image_stats(img)
        if mean < worst[0]:
            worst = (mean, i)
        if mean < BLACK_MEAN:
            black += 1
            print(f"[accept] round {i}: BLACK mean={mean:.2f} std={std:.2f}",
                  flush=True)
        elif i % 10 == 0 or i == N - 1:
            print(f"[accept] round {i}: mean={mean:.2f} std={std:.2f}", flush=True)

    print(f"[accept] done: {N} rounds, black={black}, "
          f"worst_mean={worst[0]:.2f} (round {worst[1]})", flush=True)
    if black:
        print(f"[accept] FAIL: {black}/{N} 黑图", flush=True)
        sys.exit(1)
    print("[accept] PASS: 无黑图", flush=True)


if __name__ == "__main__":
    main()

"""第二档验收：KSampler Advanced 分段采样 + vae.load + cond/latent 插件算子。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-tier2.py
  # VAE_NAME 默认 vae-ft-mse-840000-ema-pruned（MODELS_DIR 下文件名，可省扩展名）

流程与判定（ckpt 默认 v1-5-pruned-emaonly-fp16，OUT_DIR 存抽验图）：
  t1 回归：默认参数 sample（euler）与 M2 基线 t4_euler_normal.png 逐字节一致
  t2 分段接力一致性：同 seed euler 20 步一次跑 vs 0→10 / 10→20 两段接力
     （第二段 add_noise=false）出图逐字节一致
  t3 提前停：end_at_step=10 的 latent 直接 decode 出半成品噪声图（存盘抽图，
     亮度区间放宽）
  t4 vae.load：外挂 VAE decode 同一 latent，出图正常且与内置 VAE 结果有差异
     （逐字节不同即证明真的走了外挂 VAE）；顺带验证 vae.encode 兼容 shim
  t5 cond.combine / cond.average：组合/加权提示词出图正常（存盘抽图）
  t6 latent.upscale：latent 2x 放大后 decode，图像尺寸 512→1024
  t7 latent.composite：两张不同 seed 图 latent 拼接（x=256px, feather=32px）
     decode 出图正常（存盘抽图核对位置/过渡）
全部通过退出码 0，否则 1。
"""
import hashlib
import io
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8100").rstrip("/")
TOKEN = os.environ.get("TOKEN", "")
CKPT = os.environ.get("CKPT", "v1-5-pruned-emaonly-fp16")
VAE_NAME = os.environ.get("VAE_NAME", "vae-ft-mse-840000-ema-pruned")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-tier2-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
STEPS = 20
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "cond", "cond_a", "cond_b",
             "latent", "image", "dst", "src"}

failures = []


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


def run_job(op_name, /, **inputs):
    wrapped = {k: ({"$id": v} if k in OBJ_PORTS else v)
               for k, v in inputs.items()}
    jid = call("/jobs", {"name": op_name, "inputs": wrapped})["job_id"]
    t0 = time.time()
    while True:
        st = call(f"/jobs/{jid}")
        if st["status"] == "done":
            return {k: m.get("id", m.get("value"))
                    for k, m in st["result"].get("outputs", {}).items()}, None
        if st["status"] in ("failed", "cancelled"):
            return None, st.get("error", "(no error)")
        if time.time() - t0 > TIMEOUT:
            call("/interrupt", {"job_id": jid})
            return None, "timeout"
        time.sleep(3)


def op(op_name, /, **inputs):
    out, err = run_job(op_name, **inputs)
    if err:
        raise RuntimeError(f"op {op_name} failed: {err}")
    return out


def fetch_image(image_id):
    req = urllib.request.Request(f"{BASE}/images/{image_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def save(tag, png):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{tag}.png"), "wb") as f:
        f.write(png)


def check_img(tag, png, lo=10.0, hi=245.0):
    """亮度均值区间检查 + 存盘；返回 (sha1, mean, size)。"""
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(png))
    mean = ImageStat.Stat(im.convert("L")).mean[0]
    sha = hashlib.sha1(png).hexdigest()
    print(f"[accept] {tag}: sha1={sha[:12]} mean={mean:.1f} size={im.size}",
          flush=True)
    save(tag, png)
    if not (lo < mean < hi):
        failures.append(f"{tag}: 亮度异常 mean={mean:.1f}")
    return sha, mean, im.size


def decode(vae, latent, tag, **kw):
    img_id = op("vae.decode", vae=vae, latent=latent)["image"]
    return check_img(tag, fetch_image(img_id), **kw)


def main():
    print(f"[accept] base={BASE} ckpt={CKPT} vae={VAE_NAME}", flush=True)

    out = op("checkpoint.load", ckpt=CKPT)
    model, clip, vae = out["model"], out["clip"], out["vae"]
    pos = op("clip.encode", clip=clip, text=PROMPT)["cond"]
    neg = op("clip.encode", clip=clip, text="")["cond"]
    latent0 = op("latent.empty", width=512, height=512, batch_size=1)["latent"]

    # t1 回归：默认参数 vs M2 基线逐字节一致
    lat_full = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
                  seed=SEED, steps=STEPS, cfg=7.0)["latent"]
    sha_full, _, _ = decode(vae, lat_full, "t1_default_euler")
    if os.path.isfile(BASELINE_T1):
        with open(BASELINE_T1, "rb") as f:
            base_sha = hashlib.sha1(f.read()).hexdigest()
        if sha_full == base_sha:
            print("[accept] t1 PASS: 默认参数与 M2 基线逐字节一致", flush=True)
        else:
            failures.append(f"t1: 默认回归不一致 {sha_full[:12]} != {base_sha[:12]}")
    else:
        print(f"[accept] t1 SKIP: 基线不存在 {BASELINE_T1}（仅记录出图）", flush=True)

    # t2 分段接力：0→10（end_at_step）+ 10→20（start_at_step + add_noise=false）
    lat_seg1 = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
                  seed=SEED, steps=STEPS, cfg=7.0, end_at_step=10)["latent"]
    lat_seg2 = op("sample", model=model, pos=pos, neg=neg, latent=lat_seg1,
                  seed=SEED, steps=STEPS, cfg=7.0, start_at_step=10,
                  add_noise=False)["latent"]
    sha_seg, _, _ = decode(vae, lat_seg2, "t2_segmented_euler")
    if sha_seg == sha_full:
        print("[accept] t2 PASS: 分段接力 ≡ 一次跑（逐字节一致）", flush=True)
    else:
        failures.append(f"t2: 分段接力不一致 {sha_seg[:12]} != {sha_full[:12]}")

    # t3 提前停：第一段 latent 直接 decode = 半成品噪声图（亮度区间放宽）
    decode(vae, lat_seg1, "t3_half_baked", lo=2.0, hi=253.0)

    # t4 vae.load：外挂 VAE decode 同一 latent
    vae_ext = op("vae.load", name=VAE_NAME)["vae"]
    sha_ext, _, _ = decode(vae_ext, lat_full, "t4_external_vae")
    if sha_ext != sha_full:
        print("[accept] t4 PASS: 外挂 VAE 生效（与内置结果不同）", flush=True)
    else:
        failures.append("t4: 外挂 VAE 与内置结果逐字节相同，疑似未生效")
    # vae.encode 兼容 shim（外挂 VAE 编码原图回 latent 再解码）
    img_id = op("vae.decode", vae=vae, latent=lat_full)["image"]
    lat_re = op("vae.encode", vae=vae_ext, image=img_id)["latent"]
    decode(vae_ext, lat_re, "t4b_ext_vae_roundtrip")

    # t5 cond.combine / cond.average
    ca = op("clip.encode", clip=clip, text="a cat")["cond"]
    cb = op("clip.encode", clip=clip,
            text="sitting on a windowsill, warm sunset light")["cond"]
    cond_comb = op("cond.combine", cond_a=ca, cond_b=cb)["cond"]
    lat = op("sample", model=model, pos=cond_comb, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0)["latent"]
    decode(vae, lat, "t5a_cond_combine")
    cond_avg = op("cond.average", cond_a=ca, cond_b=cb, weight=0.7)["cond"]
    lat = op("sample", model=model, pos=cond_avg, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0)["latent"]
    decode(vae, lat, "t5b_cond_average")

    # t6 latent.upscale：latent 2x → decode 尺寸翻倍
    lat_up = op("latent.upscale", latent=lat_full, scale=2.0,
                method="bicubic")["latent"]
    _, _, size = decode(vae, lat_up, "t6_latent_upscale", lo=2.0, hi=253.0)
    if size == (1024, 1024):
        print("[accept] t6 PASS: latent 2x upscale -> 1024x1024", flush=True)
    else:
        failures.append(f"t6: upscale 尺寸错误 {size} != (1024, 1024)")

    # t7 latent.composite：两个不同 seed 的 latent 左右拼接
    lat_b = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
               seed=777, steps=STEPS, cfg=7.0)["latent"]
    lat_comp = op("latent.composite", dst=lat_full, src=lat_b,
                  x=256, y=0, feather=32)["latent"]
    decode(vae, lat_comp, "t7_latent_composite")

    if failures:
        print(f"[accept] FAIL ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print(f"[accept] ALL PASS（抽验图已存 {OUT_DIR}，请人工看图确认内容）", flush=True)


if __name__ == "__main__":
    main()

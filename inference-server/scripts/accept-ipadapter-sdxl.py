"""SDXL IPAdapter（vit-h 变体）+ ip-adapter-plus_sd15 验收（功能拓展批次 C）。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-ipadapter-sdxl.py

流程与判定（OUT_DIR 存抽验图）：
  t1 SD1.5 回归：默认 sample 与 M2 基线逐字节一致（权重到盘未污染任何路径）
  t2 SDXL + ip-adapter_sdxl_vit-h（768/16 步 offload=model，同种子四链）：
     链 D 无 IPA 基线（prompt only）
     链 A weight=0    → 必须与链 D 逐字节一致（weight=0 零影响 + 零污染）
     链 B weight=0.8 全程 → 应向参考图（橘白卡通猫窗台）风格/构图漂移
     链 C weight=0.8 窗口[0,0.5) → 介于 A/B 之间（存图人工核对）
  t3 SD1.5 + ip-adapter-plus_sd15（Resampler 路径，512/20 步）：
     链 P0 weight=0 与 链 P0b 无 IPA 基线逐字节一致；
     链 P1 weight=0.8 生效（存图人工核对）
  t4 显存回收：model.unload all + gc
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
SD15 = os.environ.get("SD15_CKPT", "v1-5-pruned-emaonly-fp16")
SDXL = os.environ.get("SDXL_CKPT", "sd_xl_base_1.0")
IPA_SDXL = os.environ.get("IPA_SDXL", "ip-adapter_sdxl_vit-h")
IPA_PLUS = os.environ.get("IPA_PLUS", "ip-adapter-plus_sd15")
CLIP_VISION = os.environ.get("CLIP_VISION", "CLIP-ViT-H-14")
REF = os.environ.get("REF", "ipa_ref_cat.png")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-ipadapter-sdxl-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
T1_PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
PROMPT = "a dog sitting in a garden, photograph, high quality, sharp focus"
NEG = "lowres, blurry, worst quality, watermark, text"
SEED = 42
SDXL_SIZE = 768
SDXL_STEPS = 16
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "3600"))

# 对象端口（服务端对象仓句柄），调用时需包装 {"$id": ...}（同 tier2/3 脚本）
OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control", "clip_vision", "ipadapter"}

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
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(png))
    mean = ImageStat.Stat(im.convert("L")).mean[0]
    sha = hashlib.sha1(png).hexdigest()
    print(f"[accept] {tag}: sha1={sha[:12]} mean={mean:.1f} size={im.size}",
          flush=True)
    save(tag, png)
    if not (lo < mean < hi):
        failures.append(f"{tag}: 亮度异常 mean={mean:.1f}")
    return sha


def main():
    t0 = time.time()

    # ---------- t1 SD1.5 回归：与 M2 基线逐字节一致 ----------
    mcv = op("sd.checkpoint.load", ckpt=SD15)
    pos = op("sd.clip.encode", clip=mcv["clip"], text=T1_PROMPT)["cond"]
    neg = op("sd.clip.encode", clip=mcv["clip"], text="")["cond"]
    lat = op("sd.latent.empty", width=512, height=512, batch_size=1)["latent"]
    r = op("sd.sample", model=mcv["model"], pos=pos, neg=neg, latent=lat,
           seed=424242, steps=20, cfg=7.0, sampler_name="euler")
    png = fetch_image(op("sd.vae.decode", vae=mcv["vae"],
                         latent=r["latent"])["image"])
    save("t1_regression", png)
    sha = hashlib.sha1(png).hexdigest()
    with open(BASELINE_T1, "rb") as f:
        base_sha = hashlib.sha1(f.read()).hexdigest()
    if sha != base_sha:
        failures.append(f"t1: 回归不一致 {sha[:12]} != {base_sha[:12]}")
    print(f"[accept] t1 回归: {'一致' if sha == base_sha else '不一致!'}",
          flush=True)

    # ---------- t2 SDXL + ip-adapter_sdxl_vit-h ----------
    # TTL 兑底：条件/latent/模型视图在每链采样前重建（缓存命中秒回）
    def fresh_sdxl():
        m = op("sd.checkpoint.load", ckpt=SDXL, offload="model")
        p = op("sd.clip.encode", clip=m["clip"], text=PROMPT,
               width=SDXL_SIZE, height=SDXL_SIZE)["cond"]
        n = op("sd.clip.encode", clip=m["clip"], text=NEG,
               width=SDXL_SIZE, height=SDXL_SIZE)["cond"]
        l = op("sd.latent.empty", width=SDXL_SIZE, height=SDXL_SIZE,
               batch_size=1)["latent"]
        return m, p, n, l

    def sdxl_sample(tag, model_view, extra_prefix=""):
        m, p, n, l = fresh_sdxl()
        r = op("sd.sample", model=model_view, pos=p, neg=n, latent=l,
               seed=SEED, steps=SDXL_STEPS, cfg=7.0,
               sampler_name="euler", scheduler="normal", denoise=1.0)
        img = op("sd.vae.decode", vae=m["vae"], latent=r["latent"])["image"]
        return check_img(f"{extra_prefix}{tag}", fetch_image(img))

    cv = op("clip_vision.load", name=CLIP_VISION)["clip_vision"]
    ipa_sd = op("ipadapter.load", name=IPA_SDXL)["ipadapter"]
    ref = op("image.load", name=REF)["image"]

    m, p, n, l = fresh_sdxl()
    r = op("sd.sample", model=m["model"], pos=p, neg=n, latent=l,
           seed=SEED, steps=SDXL_STEPS, cfg=7.0,
           sampler_name="euler", scheduler="normal", denoise=1.0)
    sha_d = check_img("t2D_sdxl_baseline",
                      fetch_image(op("sd.vae.decode", vae=m["vae"],
                                     latent=r["latent"])["image"]))

    def apply_sdxl(weight, lo, hi):
        # ipadapter/clip_vision/ref 对象有 TTL，每次 apply 前刷新
        cv_ = op("clip_vision.load", name=CLIP_VISION)["clip_vision"]
        ipa_ = op("ipadapter.load", name=IPA_SDXL)["ipadapter"]
        ref_ = op("image.load", name=REF)["image"]
        m_, _, _, _ = fresh_sdxl()
        return op("ipadapter.apply", model=m_["model"], ipadapter=ipa_,
                  clip_vision=cv_, image=ref_, weight=weight,
                  start_percent=lo, end_percent=hi)["model"]

    sha_a = sdxl_sample("t2A_sdxl_ipa_w0", apply_sdxl(0.0, 0.0, 1.0))
    if sha_a != sha_d:
        failures.append(f"t2: weight=0 与无 IPA 基线不一致 "
                        f"{sha_a[:12]} != {sha_d[:12]}（零污染违反）")
    else:
        print("[accept] t2 weight=0 与基线逐字节一致（零污染 ✓）", flush=True)
    sha_b = sdxl_sample("t2B_sdxl_ipa_w08", apply_sdxl(0.8, 0.0, 1.0))
    if sha_b == sha_d:
        failures.append("t2: weight=0.8 全程与基线逐字节一致（IPA 疑似未生效）")
    sha_c = sdxl_sample("t2C_sdxl_ipa_w08_win", apply_sdxl(0.8, 0.0, 0.5))
    if sha_c == sha_d:
        failures.append("t2: 窗口链 C 与基线逐字节一致（IPA 疑似未生效）")

    # ---------- t3 SD1.5 + ip-adapter-plus_sd15（Resampler 路径） ----------
    def fresh_sd15():
        m = op("sd.checkpoint.load", ckpt=SD15)
        p = op("sd.clip.encode", clip=m["clip"], text=PROMPT)["cond"]
        n = op("sd.clip.encode", clip=m["clip"], text=NEG)["cond"]
        l = op("sd.latent.empty", width=512, height=512, batch_size=1)["latent"]
        return m, p, n, l

    def sd15_sample(model_view):
        m, p, n, l = fresh_sd15()
        r = op("sd.sample", model=model_view, pos=p, neg=n, latent=l,
               seed=SEED, steps=20, cfg=7.0, sampler_name="euler")
        return m, fetch_image(op("sd.vae.decode", vae=m["vae"],
                                 latent=r["latent"])["image"])

    def apply_sd15(weight):
        cv_ = op("clip_vision.load", name=CLIP_VISION)["clip_vision"]
        ipa_ = op("ipadapter.load", name=IPA_PLUS)["ipadapter"]
        ref_ = op("image.load", name=REF)["image"]
        m_, _, _, _ = fresh_sd15()
        return op("ipadapter.apply", model=m_["model"], ipadapter=ipa_,
                  clip_vision=cv_, image=ref_, weight=weight,
                  start_percent=0.0, end_percent=1.0)["model"]

    m, p, n, l = fresh_sd15()
    r = op("sd.sample", model=m["model"], pos=p, neg=n, latent=l,
           seed=SEED, steps=20, cfg=7.0, sampler_name="euler")
    sha_p0b = check_img("t3P0b_sd15_baseline",
                        fetch_image(op("sd.vae.decode", vae=m["vae"],
                                       latent=r["latent"])["image"]))

    m_, png = sd15_sample(apply_sd15(0.0))
    sha_p0 = check_img("t3P0_sd15_plus_w0", png)
    if sha_p0 != sha_p0b:
        failures.append(f"t3: plus weight=0 与基线不一致 "
                        f"{sha_p0[:12]} != {sha_p0b[:12]}")
    else:
        print("[accept] t3 plus weight=0 与基线逐字节一致（零污染 ✓）",
              flush=True)
    m_, png = sd15_sample(apply_sd15(0.8))
    sha_p1 = check_img("t3P1_sd15_plus_w08", png)
    if sha_p1 == sha_p0b:
        failures.append("t3: plus weight=0.8 与基线逐字节一致（未生效）")

    # ---------- t4 显存回收 ----------
    op("model.unload", target="all")
    call("/gc", {})
    h = call("/health")
    print(f"[accept] t4 resident={h.get('resident_models')} "
          f"vram_free={h.get('vram_free_mb')}MB", flush=True)

    dt = time.time() - t0
    print(f"\n[accept] 用时 {dt:.0f}s", flush=True)
    if failures:
        print("[accept] FAILED:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    print("[accept] ALL PASS（抽图请人工核对 t2B 风格漂移 / t2C 窗口 / t3P1）",
          flush=True)


if __name__ == "__main__":
    main()

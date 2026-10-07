"""FaceDetailer（face.mask + 局部精修编排）验收（功能拓展批次 2）。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-facedetailer.py

流程与判定（OUT_DIR 存抽验图）：
  t1 SD1.5 回归：默认 sample 与 M2 基线逐字节一致
  t2 face.mask：真人照片 → feathered 矩形 mask + crop 框
     （判定：白区占比合理、x/y/w/h 8 倍数且在图内、feather 产生中间灰阶）
  t3 端到端 FaceDetailer：txt2img 人像（512，人脸必糊）→ face.mask →
     本地 PIL crop（与 image-crop 节点同语义）→ vae.encode →
     sample(denoise=0.45) → vae.decode → 本地 PIL composite
     （与 image-composite 节点同语义）。判定：
     - mask 全黑区域：精修结果与原图**逐像素一致**（非面部区域不变式）
     - mask 白区中心：精修结果与原图存在差异（重绘生效）
     - 抽图人工核对面部清晰度提升
  t4 显存回收
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
PHOTO = os.environ.get("PHOTO", "test_person.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-facedetailer-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
T1_PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
PORTRAIT = ("portrait photo of a young woman, freckles, looking at camera, "
            "soft window light, highly detailed")
FIX_PROMPT = ("beautiful detailed face, sharp focus, detailed eyes, "
              "natural skin texture, soft window light")
NEG = "lowres, blurry, worst quality, watermark, text, deformed"
SEED = 1234
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control", "mask"}

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


def upload_image(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    req = urllib.request.Request(BASE + "/images", data=buf.getvalue(),
                                 method="POST")
    req.add_header("Content-Type", "image/png")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["id"]


def save(tag, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{tag}.png"), "wb") as f:
        f.write(data)


def main():
    from PIL import Image, ImageChops, ImageStat
    t0 = time.time()

    # ---------- t1 SD1.5 回归 ----------
    mcv = op("sd.checkpoint.load", ckpt=CKPT)
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

    # ---------- t2 face.mask 机械判定 ----------
    photo = op("image.load", name=PHOTO)["image"]
    photo_im = Image.open(io.BytesIO(fetch_image(photo)))
    fm = op("face.mask", image=photo, face_index=-1, det_thresh=0.2,
            expand=0.6, feather=16.0)
    mask_im = Image.open(io.BytesIO(fetch_image(fm["mask"]))).convert("L")
    fx, fy, fw, fh = fm["x"], fm["y"], fm["width"], fm["height"]
    W, H = photo_im.size
    print(f"[accept] t2 photo={W}x{H} crop=({fx},{fy},{fw}x{fh})", flush=True)
    if mask_im.size != (W, H):
        failures.append(f"t2: mask 尺寸 {mask_im.size} != 原图 {(W, H)}")
    for v, nm in ((fx, "x"), (fy, "y"), (fw, "width"), (fh, "height")):
        if v % 8 != 0:
            failures.append(f"t2: {nm}={v} 非 8 倍数")
    if not (0 <= fx and 0 <= fy and fx + fw <= W and fy + fh <= H):
        failures.append(f"t2: crop 框越界 ({fx},{fy},{fw},{fh}) vs {(W, H)}")
    area = fw * fh / (W * H)
    if not (0.01 < area < 0.8):
        failures.append(f"t2: crop 面积占比异常 {area:.3f}")
    hist = mask_im.histogram()
    white = sum(hist[200:]) / (W * H)
    mid = sum(hist[10:200]) / (W * H)
    print(f"[accept] t2 白区={white:.3f} 中间灰阶={mid:.3f} "
          f"面积占比={area:.3f}", flush=True)
    if mid <= 0.0005:
        failures.append(f"t2: feather 未生效（中间灰阶占比 {mid:.5f}）")
    save("t2_face_mask", fetch_image(fm["mask"]))

    # ---------- t3 端到端 FaceDetailer ----------
    # 3.1 生成人像（512 小图，人脸必糊——FaceDetailer 的典型场景）
    p3 = op("sd.clip.encode", clip=mcv["clip"], text=PORTRAIT)["cond"]
    n3 = op("sd.clip.encode", clip=mcv["clip"], text=NEG)["cond"]
    lat3 = op("sd.latent.empty", width=512, height=512, batch_size=1)["latent"]
    r3 = op("sd.sample", model=mcv["model"], pos=p3, neg=n3, latent=lat3,
            seed=SEED, steps=20, cfg=7.0, sampler_name="euler")
    orig_id = op("sd.vae.decode", vae=mcv["vae"], latent=r3["latent"])["image"]
    orig_im = Image.open(io.BytesIO(fetch_image(orig_id))).convert("RGB")
    save("t3a_portrait_orig", fetch_image(orig_id))

    # 3.2 face.mask（AI 生成图：det_thresh=0.2 语义生效的关键场景）
    try:
        fm3 = op("face.mask", image=orig_id, face_index=-1, det_thresh=0.2,
                 expand=0.6, feather=16.0)
    except RuntimeError as e:
        failures.append(f"t3: 生成图人脸检测失败（det_thresh=0.2 仍无脸）: {e}")
        print("[accept] FAILED:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    cx, cy, cw, ch = fm3["x"], fm3["y"], fm3["width"], fm3["height"]
    mask3_im = Image.open(io.BytesIO(fetch_image(fm3["mask"]))).convert("L")
    print(f"[accept] t3 crop=({cx},{cy},{cw}x{ch})", flush=True)
    save("t3b_face_mask", fetch_image(fm3["mask"]))

    # 3.3 crop（本地 PIL，与 image-crop 节点同语义）→ **放大到短边 512**
    # （FaceDetailer 关键一步：小 crop 直接重采样必崩，先放大再采样再缩回）
    # → 精修重采样（沿用原图 prompt 保风格，低 denoise）
    crop_im = orig_im.crop((cx, cy, cx + cw, cy + ch))
    crop_id = upload_image(crop_im)
    scale = 512.0 / min(cw, ch)
    up_w = round(cw * scale / 8) * 8
    up_h = round(ch * scale / 8) * 8
    up_id = op("image.upscale", image=crop_id, width=up_w, height=up_h,
               method="lanczos")["image"]
    print(f"[accept] t3 crop {cw}x{ch} -> upscale {up_w}x{up_h}", flush=True)
    lat_fix = op("sd.vae.encode", vae=mcv["vae"], image=up_id)["latent"]
    p_fix = op("sd.clip.encode", clip=mcv["clip"],
               text=PORTRAIT + ", " + FIX_PROMPT)["cond"]
    r_fix = op("sd.sample", model=mcv["model"], pos=p_fix, neg=n3,
               latent=lat_fix, seed=SEED + 1, steps=20, cfg=7.0,
               sampler_name="euler", denoise=0.35)
    fix_id = op("sd.vae.decode", vae=mcv["vae"], latent=r_fix["latent"])["image"]
    fix_up_im = Image.open(io.BytesIO(fetch_image(fix_id))).convert("RGB")
    save("t3c_face_fixed", fetch_image(fix_id))
    # 缩回 crop 原尺寸（LANCZOS，客户端 PIL 语义）
    fix_im = fix_up_im.resize((cw, ch), Image.LANCZOS)

    # 3.4 composite（本地 PIL，与 image-composite 节点同语义：alpha blend）
    comp_im = orig_im.copy()
    comp_im.paste(fix_im, (cx, cy), mask3_im.crop((cx, cy, cx + cw, cy + ch)))
    buf = io.BytesIO()
    comp_im.save(buf, format="PNG")
    save("t3d_composited", buf.getvalue())

    # 判定 1：mask 全黑区域逐像素一致（非面部区域不变式）
    diff = ImageChops.difference(comp_im, orig_im).convert("L")
    marr = list(mask3_im.getdata())
    darr = list(diff.getdata())
    outside_diff = max(d for d, m in zip(darr, marr) if m < 10)
    print(f"[accept] t3 mask 外最大差异={outside_diff}", flush=True)
    if outside_diff > 2:
        failures.append(f"t3: 非面部区域被改动（最大差异 {outside_diff}）")
    # 判定 2：mask 白区中心存在差异（重绘生效）
    inside = [d for d, m in zip(darr, marr) if m > 240]
    if not inside:
        failures.append("t3: mask 白区为空")
    else:
        inside_mean = sum(inside) / len(inside)
        print(f"[accept] t3 白区平均差异={inside_mean:.1f}", flush=True)
        if inside_mean < 3:
            failures.append(f"t3: 精修区域几乎无变化（平均差异 {inside_mean:.1f}）")

    # ---------- t4 回收 ----------
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
    print("[accept] ALL PASS（抽图请人工核对 t3a 原图 vs t3d 精修合成）",
          flush=True)


if __name__ == "__main__":
    main()

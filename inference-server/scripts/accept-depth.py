"""第三档批次 3 验收：深度预处理器（preprocess.depth）+ depth ControlNet 端到端。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-depth.py

流程与判定（ckpt 默认 v1-5-pruned-emaonly-fp16，OUT_DIR 存抽验图）：
  t1 回归：默认参数 sample 与 M2 基线逐字节一致（本批次改动未污染采样路径）
  t2 preprocess.depth：真人照片（INPUT_DIR/test_fullbody2.jpg）→ 灰度深度图
     （近亮远暗；判定：输出尺寸=输入尺寸、R=G=B 纯灰度、亮度合理区间非纯黑白）
  t3 端到端：深度图 + control_v11f1p_sd15_depth_fp16 控制采样，prompt 换成
     "雪中小屋"——构图/空间层次应跟随深度图（存图人工核对）
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
CN_DEPTH = os.environ.get("CN_DEPTH", "control_v11f1p_sd15_depth_fp16")
PHOTO = os.environ.get("PHOTO", "test_fullbody2.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-depth-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
STEPS = 20
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

# 对象端口（服务端对象仓句柄），调用时需包装 {"$id": ...}（同 tier2/3 脚本）
OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control"}

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
    return sha, mean, im.size


def main():
    from PIL import Image
    t0 = time.time()

    # ---------- t1 回归：默认路径逐字节一致 ----------
    mcv = op("sd.checkpoint.load", ckpt=CKPT)
    pos = op("sd.clip.encode", clip=mcv["clip"], text=PROMPT)["cond"]
    neg = op("sd.clip.encode", clip=mcv["clip"], text="")["cond"]
    lat = op("sd.latent.empty", width=512, height=512, batch_size=1)["latent"]
    r = op("sd.sample", model=mcv["model"], pos=pos, neg=neg, latent=lat,
           seed=SEED, steps=STEPS, cfg=7.0, sampler_name="euler")
    img_id = op("sd.vae.decode", vae=mcv["vae"], latent=r["latent"])["image"]
    png = fetch_image(img_id)
    save("t1_regression", png)
    sha = hashlib.sha1(png).hexdigest()
    with open(BASELINE_T1, "rb") as f:
        base_sha = hashlib.sha1(f.read()).hexdigest()
    if sha != base_sha:
        failures.append(f"t1: 回归不一致 {sha[:12]} != {base_sha[:12]}")
    print(f"[accept] t1 回归: {'一致' if sha == base_sha else '不一致!'}",
          flush=True)

    # ---------- t2 preprocess.depth：灰度深度图 ----------
    photo = op("image.load", name=PHOTO)["image"]
    photo_png = fetch_image(photo)
    photo_size = Image.open(io.BytesIO(photo_png)).size

    t2_id = op("preprocess.depth", image=photo, detect_resolution=512)["image"]
    t2_png = fetch_image(t2_id)
    _, _, t2_size = check_img("t2_person_depth", t2_png, lo=15.0, hi=230.0)
    # 输出尺寸必须=输入尺寸（算子契约）
    if t2_size != photo_size:
        failures.append(f"t2: 输出尺寸 {t2_size} != 输入 {photo_size}")
    # 深度图为纯灰度：R=G=B（通道极差≈0）；且应有灰度层次（非平坦）
    im2 = Image.open(io.BytesIO(t2_png)).convert("RGB")
    px2 = list(im2.resize((64, 64)).getdata())
    max_ch_diff = max(max(r, g, b) - min(r, g, b) for r, g, b in px2)
    lumas = [(r + g + b) / 3 for r, g, b in px2]
    spread = max(lumas) - min(lumas)
    print(f"[accept] t2 通道极差={max_ch_diff} 灰度跨度={spread:.0f}",
          flush=True)
    if max_ch_diff > 2:
        failures.append(f"t2: 非纯灰度图（通道极差 {max_ch_diff}）")
    if spread < 60:
        failures.append(f"t2: 灰度层次不足（跨度 {spread:.0f}，疑似平坦输出）")

    # ---------- t3 端到端：depth 控制采样 ----------
    cn = op("sd.controlnet.load", name=CN_DEPTH)["control_net"]
    ctrl = op("sd.controlnet.apply", control_net=cn, image=t2_id,
              strength=1.0)["control"]
    pos3 = op("sd.clip.encode", clip=mcv["clip"],
              text="a small wooden cabin in a snowy forest, night, "
                   "warm light in the window, highly detailed")["cond"]
    lat3 = op("sd.latent.empty", width=512, height=512, batch_size=1)["latent"]
    r3 = op("sd.sample", model=mcv["model"], pos=pos3, neg=neg, latent=lat3,
            seed=777, steps=STEPS, cfg=7.0, sampler_name="euler",
            control=ctrl)
    img3 = op("sd.vae.decode", vae=mcv["vae"], latent=r3["latent"])["image"]
    check_img("t3_cabin_depth", fetch_image(img3))

    # 回收：CN 常驻显存释放（8GB 卡纪律）
    try:
        call("/model/unload", {"target": f"cn:{CN_DEPTH}|dtype=fp16"})
        call("/gc", {})
    except Exception as e:
        print(f"[accept] unload cn 失败（不判负）: {e}", flush=True)

    dt = time.time() - t0
    print(f"\n[accept] 用时 {dt:.0f}s", flush=True)
    if failures:
        print("[accept] FAILED:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    print("[accept] ALL PASS（抽图请人工核对 t2 深度图/t3 构图跟随）",
          flush=True)


if __name__ == "__main__":
    main()

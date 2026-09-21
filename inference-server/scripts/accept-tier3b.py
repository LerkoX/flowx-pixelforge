"""第三档批次 2 验收：预处理器算子（preprocess.canny / preprocess.openpose）
+ openpose ControlNet 端到端（人物替换保动作）。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-tier3b.py

流程与判定（ckpt 默认 v1-5-pruned-emaonly-fp16，OUT_DIR 存抽验图）：
  t1 回归：默认参数 sample 与 M2 基线逐字节一致（依赖栈变动未污染采样路径）
  t2 preprocess.canny：真人照片（INPUT_DIR/test_person.jpg）→ 真 cv2 canny
     线稿（存图人工核对；边缘占比应在合理区间）
  t3 preprocess.openpose：真人照片 → 骨架图（存图人工核对骨架形态；
     黑底彩色，亮度低但非纯黑）
  t3b preprocess.openpose(include_hand=true)：手部精化路径连通（存图）
  t4 端到端：openpose 骨架 + control_v11p_sd15_openpose_fp16 控制采样，
     prompt 换"宇航员"——人物特征替换、姿势对齐骨架（存图人工核对，
     本批次业务目标）
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
CN_OPENPOSE = os.environ.get("CN_OPENPOSE", "control_v11p_sd15_openpose_fp16")
PHOTO = os.environ.get("PHOTO", "test_fullbody2.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-tier3b-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
STEPS = 20
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

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
    t0 = time.time()

    # ---------- t1 回归：默认路径逐字节一致 ----------
    mcv = op("checkpoint.load", ckpt=CKPT)
    pos = op("clip.encode", clip=mcv["clip"], text=PROMPT)["cond"]
    neg = op("clip.encode", clip=mcv["clip"], text="")["cond"]
    lat = op("latent.empty", width=512, height=512, batch_size=1)["latent"]
    r = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat,
           seed=SEED, steps=STEPS, cfg=7.0, sampler_name="euler")
    img_id = op("vae.decode", vae=mcv["vae"], latent=r["latent"])["image"]
    png = fetch_image(img_id)
    save("t1_regression", png)
    sha = hashlib.sha1(png).hexdigest()
    with open(BASELINE_T1, "rb") as f:
        base_sha = hashlib.sha1(f.read()).hexdigest()
    if sha != base_sha:
        failures.append(f"t1: 回归不一致 {sha[:12]} != {base_sha[:12]}")
    print(f"[accept] t1 回归: {'一致' if sha == base_sha else '不一致!'}", flush=True)

    # ---------- t2 真 canny ----------
    photo = op("image.load", name=PHOTO)["image"]
    t2_id = op("preprocess.canny", image=photo,
               low_threshold=100, high_threshold=200)["image"]
    t2_png = fetch_image(t2_id)
    check_img("t2_person_canny", t2_png, lo=2.0, hi=80.0)
    # 边缘占比机械判定：线稿应稀疏（黑底白线，1%~25% 白像素）
    from PIL import Image
    import statistics
    im = Image.open(io.BytesIO(t2_png)).convert("L")
    px = list(im.getdata())
    white = sum(1 for v in px if v > 127) / len(px)
    print(f"[accept] t2 边缘占比: {white:.3f}", flush=True)
    if not (0.005 < white < 0.25):
        failures.append(f"t2: 边缘占比异常 {white:.3f}")

    # ---------- t3 openpose 骨架 ----------
    t3_id = op("preprocess.openpose", image=photo,
               include_hand=False, include_face=False)["image"]
    t3_png = fetch_image(t3_id)
    check_img("t3_person_openpose", t3_png, lo=0.3, hi=60.0)
    # 骨架图为黑底彩色：非黑像素占比应稀疏，且存在彩色（通道间差异）像素
    im3 = Image.open(io.BytesIO(t3_png)).convert("RGB")
    px3 = list(im3.getdata())
    nonblack = sum(1 for r, g, b in px3 if max(r, g, b) > 60) / len(px3)
    colored = sum(1 for r, g, b in px3
                  if max(r, g, b) - min(r, g, b) > 60) / len(px3)
    print(f"[accept] t3 非黑占比={nonblack:.3f} 彩色占比={colored:.3f}", flush=True)
    if not (0.002 < nonblack < 0.4):
        failures.append(f"t3: 非黑占比异常 {nonblack:.3f}")
    if colored <= 0.001:
        failures.append(f"t3: 未检测到彩色骨架像素 {colored:.3f}")

    # ---------- t3b openpose include_hand 路径连通 ----------
    t3b_id = op("preprocess.openpose", image=photo,
                include_hand=True, include_face=False)["image"]
    t3b_png = fetch_image(t3b_id)
    check_img("t3b_person_openpose_hand", t3b_png, lo=0.3, hi=60.0)
    # 本测试照手部可见，include_hand=true 应产生更多骨架像素
    if t3b_png == t3_png:
        failures.append("t3b: include_hand=true 与 false 逐字节一致（手部未生效）")

    # ---------- t4 端到端：openpose 控制 + 人物替换 ----------
    cn = op("controlnet.load", name=CN_OPENPOSE)["control_net"]
    ctrl = op("controlnet.apply", control_net=cn, image=t3_id,
              strength=1.0)["control"]
    pos4 = op("clip.encode", clip=mcv["clip"],
              text="an astronaut woman in a white spacesuit without helmet, "
                   "head tilted back looking up at the sky, highly detailed")["cond"]
    lat4 = op("latent.empty", width=512, height=512, batch_size=1)["latent"]
    r4 = op("sample", model=mcv["model"], pos=pos4, neg=neg, latent=lat4,
            seed=777, steps=STEPS, cfg=7.0, sampler_name="euler",
            control=ctrl)
    img4 = op("vae.decode", vae=mcv["vae"], latent=r4["latent"])["image"]
    check_img("t4_astronaut_pose", fetch_image(img4))

    dt = time.time() - t0
    print(f"\n[accept] 用时 {dt:.0f}s", flush=True)
    if failures:
        print("[accept] FAILED:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    print("[accept] ALL PASS（抽图请人工核对 t2/t3/t3b/t4）", flush=True)


if __name__ == "__main__":
    main()

"""t3c 补票：preprocess.openpose include_face=true 精化路径验收。

背景：§十三遗留②——face 精化权重早已下载但验收从未覆盖 include_face=true。
用法：
  BASE=http://<地址> TOKEN=<token> python3 scripts/accept-tier3c-openpose-face.py

判定：
  1. 算子正常跑通出图（非黑、亮度在骨架图合理区间）
  2. 与 include_face=false 产物**非逐字节一致**（face 路径真实生效）
  3. 人脸区域出现关键点：用 PIL 对肖像上半部检测彩色（非灰）像素占比，
     与 include_face=false 版本对比应显著增加（face 关键点是彩色圆点+连线）
  4. 存 OUT_DIR 抽图人工核对
退出码 0=通过。
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
PHOTO = os.environ.get("PHOTO", "test_person.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-tier3c-out"))
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "600"))

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


def run_op(op_name, /, **inputs):
    wrapped = {k: ({"$id": v} if k == "image" else v)
               for k, v in inputs.items()}
    jid = call("/jobs", {"name": op_name, "inputs": wrapped})["job_id"]
    t0 = time.time()
    while True:
        st = call(f"/jobs/{jid}")
        if st["status"] == "done":
            return {k: m.get("id", m.get("value"))
                    for k, m in st["result"].get("outputs", {}).items()}
        if st["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"op {op_name} failed: {st.get('error')}")
        if time.time() - t0 > TIMEOUT:
            call("/interrupt", {"job_id": jid})
            raise RuntimeError(f"op {op_name} timeout")
        time.sleep(3)


def fetch_image(image_id):
    req = urllib.request.Request(f"{BASE}/images/{image_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def face_color_ratio(png):
    """肖像上半部（人脸所在）彩色像素占比：骨架黑底上彩色=非灰像素。"""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, h // 2))
    px = list(top.getdata())
    colored = sum(1 for r, g, b in px
                  if max(r, g, b) - min(r, g, b) > 40 and max(r, g, b) > 60)
    return colored / len(px)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    photo = run_op("image.load", name=PHOTO)["image"]
    print(f"[t3c] 照片加载: {photo}", flush=True)

    noface_id = run_op("preprocess.openpose", image=photo,
                       include_hand=False, include_face=False)["image"]
    face_id = run_op("preprocess.openpose", image=photo,
                     include_hand=False, include_face=True)["image"]
    nf_png, f_png = fetch_image(noface_id), fetch_image(face_id)

    from PIL import Image, ImageStat
    for tag, png in (("noface", nf_png), ("face", f_png)):
        im = Image.open(io.BytesIO(png))
        mean = ImageStat.Stat(im.convert("L")).mean[0]
        sha = hashlib.sha1(png).hexdigest()[:12]
        with open(os.path.join(OUT_DIR, f"t3c_{tag}.png"), "wb") as f:
            f.write(png)
        print(f"[t3c] {tag}: sha1={sha} mean={mean:.1f} size={im.size}",
              flush=True)
        if not (0.3 < mean < 60.0):
            failures.append(f"{tag}: 亮度异常 mean={mean:.1f}")

    if hashlib.sha1(nf_png).hexdigest() == hashlib.sha1(f_png).hexdigest():
        failures.append("include_face=true 与 false 逐字节一致——face 路径未生效")

    r_nf, r_f = face_color_ratio(nf_png), face_color_ratio(f_png)
    print(f"[t3c] 上半部彩色像素占比: noface={r_nf:.4f} face={r_f:.4f}",
          flush=True)
    if r_f <= r_nf * 1.2:
        failures.append(f"face 版本彩色像素未显著增加 ({r_f:.4f} vs {r_nf:.4f})")

    if failures:
        for f_ in failures:
            print(f"[FAIL] {f_}")
        sys.exit(1)
    print("[t3c] 全部通过（抽图存", OUT_DIR, "）")
    sys.exit(0)


if __name__ == "__main__":
    main()

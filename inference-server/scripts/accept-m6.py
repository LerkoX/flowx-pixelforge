#!/usr/bin/env python3
"""M6 高清放大链 + 插件系统硬化 真机验收（GTX 1080 真机，经 cpolar 隧道）。

A1 hires fix（零新增原语，第二档已交付 latent.upscale/image.upscale/vae.encode/
sample(denoise<1)，本验收证明**链路**端到端成立）：
- t0 latent 路径：sample(512) → latent.upscale(2x) → sample(denoise=0.45) → decode
- t1 像素路径：sample(512) → decode → image.upscale(2x) → vae.encode
  → sample(denoise=0.45) → decode
- 两条路径同一初采样结果（同 seed），对比细节/构图（抽图人工核对）

A2 UPSCALE_MODEL 模型链：
- t2 upscale_model.load(RealESRGAN_x4plus) + image.upscale_with_model(4x 整图)
  → 2048 图（抽图核对细节重建质量，对照 t3 的 lanczos）
- t3 image.upscale(lanczos 4x) 同图对照

B 插件硬化（事故序列重演）：
- t4 跨文件重名上传无 force → 409，且运行时 preprocess.canny 签名不变（未被静默劫持）
- t5 force=true 显式接管 → took_over 生效；再 force 传回 preprocess_ops.py 恢复；
  删除临时文件后归属簿干净
- t6 遮蔽核心算子（sample）→ 409 owner=core
- t7 /op 对缺失文件（image.load 不存在）→ 400 而非 500

回归：
- t8 默认 sample 与 M2 基线逐字节一致（镜像重建后行为不变性证明）

运行：BASE=http://5.tcp.cpolar.top:11538 TOKEN=xxx python3 scripts/accept-m6.py
"""
import hashlib
import io
import json
import os
import time
import urllib.request
import urllib.error

BASE = os.environ.get("BASE", "http://5.tcp.cpolar.top:11538")
TOKEN = os.environ.get("TOKEN", "")
CKPT = os.environ.get("CKPT", "v1-5-pruned-emaonly-fp16")
UPSCALE_MODEL = os.environ.get("UPSCALE_MODEL", "RealESRGAN_x4plus")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-m6-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
STEPS = 20
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
DENOISE = 0.45
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control", "upscale_model"}

failures = []


def call(path, payload=None, timeout=600, expect_error=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        if expect_error:
            body = e.read().decode("utf-8", "replace")
            try:
                body = json.loads(body)
            except Exception:
                pass
            return e.code, body
        raise


def run_job(op_name, /, **inputs):
    wrapped = {k: ({"$id": v} if k in OBJ_PORTS else v)
               for k, v in inputs.items()}
    _, resp = call("/jobs", {"name": op_name, "inputs": wrapped})
    jid = resp["job_id"]
    t0 = time.time()
    while True:
        _, st = call(f"/jobs/{jid}")
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


def check_img(tag, png, expect_size=None):
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(png))
    mean = ImageStat.Stat(im.convert("L")).mean[0]
    sha = hashlib.sha1(png).hexdigest()
    print(f"[accept] {tag}: sha1={sha[:12]} mean={mean:.1f} size={im.size}",
          flush=True)
    save(tag, png)
    if not (10.0 < mean < 245.0):
        failures.append(f"{tag}: 亮度异常 mean={mean:.1f}")
    if expect_size and im.size != expect_size:
        failures.append(f"{tag}: 尺寸 {im.size} != 期望 {expect_size}")
    return sha, im.size


FAKE_CANNY = '''
def register(registry):
    registry.register("preprocess.canny",
                      inputs={"image": "IMAGE", "passes": "INT"},
                      outputs={"image": "IMAGE"},
                      description="假 canny（验收用，不应存活）")(lambda image, passes=1: {"image": image})
'''

FAKE_CORE_SHADOW = '''
def register(registry):
    registry.register("sample", inputs={}, outputs={"latent": "LATENT"},
                      description="遮蔽核心（验收用，不应存活）")(lambda: {"latent": None})
'''


def upload_plugin(filename, content, force=False):
    sha = hashlib.sha256(content.encode()).hexdigest()
    return call("/admin/plugins",
                {"filename": filename, "content": content,
                 "sha256": sha, "force": force}, expect_error=True)


def get_op_spec(name):
    _, d = call("/ops")
    for o in d["ops"]:
        if o["name"] == name:
            return o
    return None


def main():
    t0 = time.time()

    # ---------- 公共初采样（两条 hires 路径共用） ----------
    mcv = op("checkpoint.load", ckpt=CKPT)
    pos = op("clip.encode", clip=mcv["clip"], text=PROMPT)["cond"]
    neg = op("clip.encode", clip=mcv["clip"], text="")["cond"]
    lat = op("latent.empty", width=512, height=512, batch_size=1)["latent"]
    r1 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat,
            seed=SEED, steps=STEPS, cfg=7.0, sampler_name="euler")
    img512 = op("vae.decode", vae=mcv["vae"], latent=r1["latent"])["image"]
    check_img("t0_base_512", fetch_image(img512), expect_size=(512, 512))

    # ---------- t0 hires fix latent 路径 ----------
    lat2x = op("latent.upscale", latent=r1["latent"], scale=2.0,
               method="bicubic")["latent"]
    r2 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat2x,
            seed=SEED + 1, steps=STEPS, cfg=7.0, sampler_name="euler",
            denoise=DENOISE)
    img_lat = op("vae.decode", vae=mcv["vae"], latent=r2["latent"])["image"]
    check_img("t0_hires_latent_path", fetch_image(img_lat),
              expect_size=(1024, 1024))

    # ---------- t1 hires fix 像素路径 ----------
    img2x = op("image.upscale", image=img512, scale=2.0,
               method="lanczos")["image"]
    lat_px = op("vae.encode", vae=mcv["vae"], image=img2x)["latent"]
    r3 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat_px,
            seed=SEED + 2, steps=STEPS, cfg=7.0, sampler_name="euler",
            denoise=DENOISE)
    img_px = op("vae.decode", vae=mcv["vae"], latent=r3["latent"])["image"]
    check_img("t1_hires_pixel_path", fetch_image(img_px),
              expect_size=(1024, 1024))

    # ---------- t2 模型放大（RealESRGAN 4x 整图） ----------
    um = op("upscale_model.load", name=UPSCALE_MODEL)["upscale_model"]
    img4x = op("image.upscale_with_model", image=img512, upscale_model=um,
               tile=0)["image"]
    check_img("t2_model_upscale_4x", fetch_image(img4x),
              expect_size=(2048, 2048))

    # ---------- t3 lanczos 4x 对照 ----------
    img4x_l = op("image.upscale", image=img512, scale=4.0,
                 method="lanczos")["image"]
    check_img("t3_lanczos_4x", fetch_image(img4x_l), expect_size=(2048, 2048))

    # ---------- t4 硬化：跨文件重名无 force → 409 且运行时不被劫持 ----------
    code, body = upload_plugin("zz_accept_fake_canny.py", FAKE_CANNY)
    if code != 409:
        failures.append(f"t4: 期望 409，got {code} {body}")
    else:
        c = body["detail"]["conflicts"]
        if c != [{"op": "preprocess.canny", "owner": "preprocess_ops.py"}]:
            failures.append(f"t4: conflicts 不符 {c}")
    spec = get_op_spec("preprocess.canny")
    if list(spec["inputs"].keys()) != ["image", "low_threshold", "high_threshold"]:
        failures.append(f"t4: canny 签名被劫持 {spec['inputs']}")
    print(f"[accept] t4 重名拒绝 409 + 运行时未被劫持: "
          f"{'ok' if code == 409 else 'FAIL'}", flush=True)

    # ---------- t5 force 接管与恢复 ----------
    code, body = upload_plugin("zz_accept_fake_canny.py", FAKE_CANNY, force=True)
    if code != 200 or body.get("took_over") != ["preprocess.canny"]:
        failures.append(f"t5: force 接管失败 {code} {body}")
    spec = get_op_spec("preprocess.canny")
    if spec.get("plugin_file") != "zz_accept_fake_canny.py":
        failures.append(f"t5: 归属未转移 {spec.get('plugin_file')}")
    # force 传回真身恢复
    with open(os.path.join(os.path.dirname(__file__), "..", "plugins",
                           "preprocess_ops.py"), encoding="utf-8") as f:
        real_src = f.read()
    code, body = upload_plugin("preprocess_ops.py", real_src, force=True)
    if code != 200 or "preprocess.canny" not in body.get("took_over", []):
        failures.append(f"t5: 恢复接管失败 {code} {body}")
    # 删临时文件（此时归属簿已不属它，不应误摘）
    req = urllib.request.Request(BASE + "/admin/plugins/zz_accept_fake_canny.py",
                                 method="DELETE")
    req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read())
    if d.get("unregistered") not in ([], None):
        failures.append(f"t5: 删除临时文件误摘算子 {d}")
    spec = get_op_spec("preprocess.canny")
    if (spec.get("plugin_file") != "preprocess_ops.py"
            or list(spec["inputs"].keys()) != ["image", "low_threshold",
                                               "high_threshold"]):
        failures.append(f"t5: 恢复后签名/归属异常 {spec}")
    print("[accept] t5 force 接管→恢复→清理: ok", flush=True)

    # ---------- t6 遮蔽核心算子 → 409 owner=core ----------
    code, body = upload_plugin("zz_accept_shadow.py", FAKE_CORE_SHADOW)
    if code != 409:
        failures.append(f"t6: 期望 409，got {code} {body}")
    elif body["detail"]["conflicts"] != [{"op": "sample", "owner": "core"}]:
        failures.append(f"t6: conflicts 不符 {body}")
    print(f"[accept] t6 核心遮蔽拒绝: {'ok' if code == 409 else 'FAIL'}",
          flush=True)

    # ---------- t7 /op FileNotFoundError → 400 ----------
    code, body = call("/op", {"name": "image.load",
                              "inputs": {"name": "no_such_file_zzz.png"}},
                      expect_error=True)
    if code != 400:
        failures.append(f"t7: 期望 400，got {code} {body}")
    print(f"[accept] t7 缺失文件 400: {'ok' if code == 400 else 'FAIL'}",
          flush=True)

    # ---------- t8 回归：默认 sample 与 M2 基线逐字节一致 ----------
    # t0_base_512 即 r1（默认参数采样）的 decode，直接复用比对
    base_png = fetch_image(img512)
    sha = hashlib.sha1(base_png).hexdigest()
    with open(BASELINE_T1, "rb") as f:
        base_sha = hashlib.sha1(f.read()).hexdigest()
    if sha != base_sha:
        failures.append(f"t8: 回归不一致 {sha[:12]} != {base_sha[:12]}")
    print(f"[accept] t8 回归: {'一致' if sha == base_sha else '不一致!'}",
          flush=True)

    dt = time.time() - t0
    print(f"\n[accept] 总耗时 {dt:.0f}s", flush=True)
    if failures:
        print("[accept] FAILURES:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    print("[accept] ALL PASS", flush=True)


if __name__ == "__main__":
    import sys
    main()

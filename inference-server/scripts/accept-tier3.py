"""第三档验收（M4）：ControlNet 加载/应用 + cond.set_area + latent.set_noise_mask。

用法：
  BASE=http://<隧道地址> TOKEN=<token> python3 scripts/accept-tier3.py
  # CN_NAME 默认 control_v11p_sd15_canny_fp16（MODELS_DIR 下文件名，可省扩展名）

流程与判定（ckpt 默认 v1-5-pruned-emaonly-fp16，OUT_DIR 存抽验图）：
  t1 回归：默认参数 sample 与 M2 基线逐字节一致（采样循环改造未污染默认路径）
  t2 controlnet 加载：controlnet.load 成功（CONTROL_NET 对象）
  t3 canny 控制：t1 出图经 PIL FIND_EDGES 生成线稿作 hint，换 prompt
     （cat→dog）出图——构图应对齐线稿（存图人工核对）
  t4 strength=0：同 seed 同 prompt + control(strength=0) ≡ t1 逐字节一致
     （注入路径零强度回归）
  t5 空步窗口：start_percent=0.999（窗口 round 后为空）≡ t1 逐字节一致；
     另跑 start=0.5/end=1.0 半程窗口出图正常（存图人工核对窗口语义）
  t6 cond.set_area：左半 "a cat" 右半 "a dog" 各 set_area 后 combine，
     出图左右分区构图（存图人工核对）
  t7 latent.set_noise_mask：t1 出图 vae.encode 后包右半白 mask，
     sample(pos="a dog...", denoise=0.75) ——右半重绘、左半保持原构图
     （存图人工核对）
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
CN_NAME = os.environ.get("CN_NAME", "control_v11p_sd15_canny_fp16")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-tier3-out"))
BASELINE_T1 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
STEPS = 20
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "cond", "cond_a", "cond_b",
             "latent", "image", "dst", "src", "control_net", "control", "mask"}

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


def upload_image(png_bytes):
    req = urllib.request.Request(f"{BASE}/images", data=png_bytes,
                                 method="POST")
    req.add_header("Content-Type", "image/png")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["id"]


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


def make_hint_png(base_png):
    """PIL FIND_EDGES 生成 canny 风格线稿（本批不引 controlnet_aux，
    验收用轻量边缘图代替；真 canny/openpose 预处理在第三档批次 2）。"""
    from PIL import Image, ImageFilter, ImageOps
    im = Image.open(io.BytesIO(base_png)).convert("L")
    edges = im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges, cutoff=1)
    buf = io.BytesIO()
    edges.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_right_half_mask_png(w=512, h=512):
    """右半白（重绘区）左半黑的灰度 mask。"""
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([w // 2, 0, w, h], fill=255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main():
    print(f"[accept] base={BASE} ckpt={CKPT} cn={CN_NAME}", flush=True)

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

    # t2 controlnet 加载
    t0 = time.time()
    control_net = op("controlnet.load", name=CN_NAME)["control_net"]
    print(f"[accept] t2 PASS: controlnet.load 成功 "
          f"（{time.time()-t0:.1f}s，含冷加载）", flush=True)

    # hint：t1 出图的边缘线稿（上传 /images 登记为 IMAGE 对象）
    base_png = fetch_image(op("vae.decode", vae=vae, latent=lat_full)["image"])
    hint_png = make_hint_png(base_png)
    save("t3_hint_edges", hint_png)
    hint_id = upload_image(hint_png)

    # t3 canny 控制：换 prompt（cat→dog）构图应对齐线稿
    pos_dog = op("clip.encode", clip=clip,
                 text="a dog sitting on a windowsill, warm sunset light, "
                      "high quality")["cond"]
    control = op("controlnet.apply", control_net=control_net, image=hint_id,
                 strength=1.0)["control"]
    lat = op("sample", model=model, pos=pos_dog, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0, control=control)["latent"]
    decode(vae, lat, "t3_canny_control_dog")

    # t4 strength=0 ≡ t1 逐字节一致（零强度注入回归）
    control0 = op("controlnet.apply", control_net=control_net, image=hint_id,
                  strength=0.0)["control"]
    lat = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0, control=control0)["latent"]
    sha0, _, _ = decode(vae, lat, "t4_strength_zero")
    if sha0 == sha_full:
        print("[accept] t4 PASS: strength=0 ≡ 无 control（逐字节一致）", flush=True)
    else:
        failures.append(f"t4: strength=0 与无 control 不一致 "
                        f"{sha0[:12]} != {sha_full[:12]}")

    # t5a 空步窗口（round 后 lo>=hi）≡ t1 逐字节一致
    control_win = op("controlnet.apply", control_net=control_net, image=hint_id,
                     strength=1.0, start_percent=0.999,
                     end_percent=1.0)["control"]
    lat = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0, control=control_win)["latent"]
    sha_win, _, _ = decode(vae, lat, "t5a_empty_window")
    if sha_win == sha_full:
        print("[accept] t5a PASS: 空步窗口 ≡ 无 control（逐字节一致）", flush=True)
    else:
        failures.append(f"t5a: 空步窗口与无 control 不一致 "
                        f"{sha_win[:12]} != {sha_full[:12]}")

    # t5b 半程窗口 [0.5,1.0)：出图正常（窗口语义存图人工核对）
    control_half = op("controlnet.apply", control_net=control_net,
                      image=hint_id, strength=1.0, start_percent=0.5,
                      end_percent=1.0)["control"]
    lat = op("sample", model=model, pos=pos_dog, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0, control=control_half)["latent"]
    decode(vae, lat, "t5b_half_window_dog")

    # t6 cond.set_area：左猫右狗分区构图
    ca = op("clip.encode", clip=clip, text="a cat")["cond"]
    cb = op("clip.encode", clip=clip, text="a dog")["cond"]
    seg_a = op("cond.set_area", cond=ca, x=0, y=0, width=256, height=512,
               strength=1.0)["cond"]
    seg_b = op("cond.set_area", cond=cb, x=256, y=0, width=256, height=512,
               strength=1.0)["cond"]
    pos_area = op("cond.combine", cond_a=seg_a, cond_b=seg_b)["cond"]
    lat = op("sample", model=model, pos=pos_area, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0)["latent"]
    decode(vae, lat, "t6_set_area_cat_dog")

    # t7 latent.set_noise_mask：原图右半重绘成狗（inpaint）
    lat_orig = op("vae.encode", vae=vae,
                      image=op("vae.decode", vae=vae,
                               latent=lat_full)["image"])["latent"]
    mask_id = upload_image(make_right_half_mask_png())
    lat_masked = op("latent.set_noise_mask", latent=lat_orig,
                    mask=mask_id)["latent"]
    lat = op("sample", model=model, pos=pos_dog, neg=neg, latent=lat_masked,
             seed=777, steps=STEPS, cfg=7.0, denoise=0.75)["latent"]
    decode(vae, lat, "t7_noise_mask_inpaint")

    if failures:
        print(f"[accept] FAIL ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print(f"[accept] ALL PASS（抽验图已存 {OUT_DIR}，请人工看图确认内容）", flush=True)


if __name__ == "__main__":
    main()

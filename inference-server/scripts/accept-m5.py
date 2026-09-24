#!/usr/bin/env python3
"""M5 SDXL 支持 真机验收（GTX 1080 8GB，经 cpolar 隧道；dev-plan 二十五）。

A 基础链路（M5 验收主体：SDXL 文生图 / 图生图 / hires）
- t0 预检：/health（diffusers 版本、显存口径）、/models/files（SDXL checkpoint
      存在，否则打印清单并退出）、/ops（clip.encode 已带 width/height/clip_skip）
- t1 SDXL txt2img 1024：checkpoint.load(offload=model) → clip.encode ×2（size=1024）
      → latent.empty(1024) → sample → vae.decode
      内容核对：尺寸 1024、非黑图、有结构方差（防"流程绿但出废片"，exec 191 教训）
- t2 SDXL i2i：t1 出图 → vae.encode → sample(denoise=0.6) → decode（构图保留）
- t3 SDXL hires：t1 latent → latent.upscale(1.5x) → sample(denoise=0.45) → decode
      （尺寸 1536，抽图核对细节）
- t4 SDXL ControlNet（需 SDXL ControlNet 权重；缺失则 SKIP 并打印下载指引）
      controlnet.load + controlnet.apply + sample(control=...)（验 added_cond_kwargs 注入）
- t5 offload 三档可行性（none/model/sequential）+ 每档耗时：验证 sequential 与
      SDXL encode_prompt 是否兼容（checkpoint-loader 里 SD1.x 标注为不兼容）
- t6 显存：SDXL 常驻 → model.unload → /health 余量回收（对比 resident_weights_mb）

B 回归 / 组合
- t7 SD1.5 txt2img 与 accept-m2 基线 t4_euler_normal.png sha1 逐字节一致（行为不变性）
- t8 SDXL + cond.set_area/cond.combine（区域构图）：两段区域提示词 + combine → 出图
      非黑图（验证 SDXLSegments 每段 pooled 注入路径）

C 可选：与官方 diffusers 管道逐像素一致（需 docker 直连 daemon）
- 设 PIPE_REF=1 且能 docker exec 到推理容器时执行 scripts/accept-m5-pipe-ref.py，
      比对两者 PNG 是否逐像素一致（最强口径；隧道不稳时跳过并人工核对）

运行：
  BASE=http://x.tcp.cpolar.top:PORT TOKEN=xxx SDXL_CKPT=<模型名> \
    python3 scripts/accept-m5.py
  # 只跑部分：ONLY=t1,t7 python3 scripts/accept-m5.py
"""
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8100")
TOKEN = os.environ.get("TOKEN", "")
SDXL_CKPT = os.environ.get("SDXL_CKPT", "")
SD15_CKPT = os.environ.get("SD15_CKPT", "v1-5-pruned-emaonly-fp16")
SDXL_CN = os.environ.get("SDXL_CN", "")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                "accept-m5-out"))
BASELINE_M2 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}

SDXL_SIZE = int(os.environ.get("SDXL_SIZE", "1024"))
SDXL_STEPS = int(os.environ.get("SDXL_STEPS", "25"))
SDXL_CFG = float(os.environ.get("SDXL_CFG", "6.0"))
SDXL_SEED = int(os.environ.get("SDXL_SEED", "5150"))
PROMPT = os.environ.get(
    "PROMPT", "a cute cat sitting on a windowsill, soft sunset light, "
              "masterpiece, best quality")
NEG = os.environ.get("NEG", "lowres, blurry, worst quality, watermark, text")

SD15_STEPS, SD15_CFG, SD15_SEED = 20, 7.0, 424242
SD15_PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"

TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "1800"))
OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control", "control_net_ref",
             "cond", "cond_a", "cond_b"}  # cond*：cond.set_area/combine/average 的对象入参

failures = []
skips = []


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


def run(op_name, /, **inputs):
    """不抛异常的 op：返回 (out, err)，供"某档不可用"类断言使用。"""
    return run_job(op_name, **inputs)


def fetch_image(image_id):
    req = urllib.request.Request(f"{BASE}/images/{image_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def save(tag, png):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{tag}.png")
    with open(path, "wb") as f:
        f.write(png)
    return path


def check_img(tag, png, expect_size=None, min_std=3.0):
    """内容级核对（抽图看图纪律的自动化部分）：尺寸 + 非黑图 + 有结构方差。"""
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(png))
    stat = ImageStat.Stat(im.convert("L"))
    mean, std = stat.mean[0], stat.stddev[0]
    ok = True
    if expect_size and im.size != expect_size:
        failures.append(f"{tag}: 尺寸 {im.size} != 期望 {expect_size}")
        ok = False
    if std < min_std:
        failures.append(f"{tag}: 方差过低 {std:.2f}（疑似废片/纯色）")
        ok = False
    if mean < 2.0 or mean > 253.0:
        failures.append(f"{tag}: 亮度异常 mean={mean:.1f}（疑似黑图/白图）")
        ok = False
    path = save(tag, png)
    sha = hashlib.sha1(png).hexdigest()
    print(f"[accept] {tag}: {'OK' if ok else 'FAIL'} size={im.size} "
          f"mean={mean:.1f} std={std:.2f} sha1={sha[:12]} -> {path}", flush=True)
    return sha


def skip(tag, reason):
    skips.append(f"{tag}: {reason}")
    print(f"[accept] {tag}: SKIP（{reason}）", flush=True)


def want(tag):
    return not ONLY or tag in ONLY


def main():
    t_start = time.time()

    # ---------------- t0 预检 ----------------
    if want("t0"):
        _, h = call("/health")
        print(f"[accept] t0 /health: status={h.get('status')} "
              f"gpu={h.get('gpu_name')} diffusers={h.get('diffusers_version')} "
              f"torch={h.get('torch_version')} "
              f"vram_free={h.get('vram_free_mb')}MB "
              f"resident={h.get('resident_models')}", flush=True)
        _, ops = call("/ops")
        names = {o["name"] for o in ops.get("ops", [])}
        for need in ("clip.encode", "sample", "checkpoint.load", "vae.decode",
                     "latent.upscale", "cond.set_area", "cond.combine"):
            if need not in names:
                failures.append(f"t0: 算子缺失 {need}")
        spec = next((o for o in ops["ops"] if o["name"] == "clip.encode"), {})
        for need in ("width", "height", "clip_skip"):
            if need not in (spec.get("inputs") or {}):
                failures.append(f"t0: clip.encode 缺入参 {need}"
                                f"（服务端未升级到 M5 版本？）")
        _, files = call("/models/files")
        ckpts = [f["name"] for f in files.get("files", [])
                 if f.get("kind") == "checkpoint"]
        print(f"[accept] t0 checkpoint 清单（{len(ckpts)}）: {ckpts}", flush=True)
        if not SDXL_CKPT:
            failures.append(
                "t0: 未指定 SDXL_CKPT（用 SDXL_CKPT=<模型名> 指定；"
                f"当前可用: {ckpts}）")
        elif SDXL_CKPT not in ckpts:
            failures.append(f"t0: SDXL_CKPT={SDXL_CKPT} 不在模型清单里"
                            f"（可用: {ckpts}）")
        else:
            print(f"[accept] t0 SDXL checkpoint = {SDXL_CKPT} ✓", flush=True)
        if SDXL_CN and SDXL_CN not in [f["name"] for f in files.get("files", [])]:
            failures.append(f"t0: SDXL_CN={SDXL_CN} 不在模型清单里")
        if failures:
            print("[accept] FAILURES:", *failures, sep="\n  - ", flush=True)
            sys.exit(1)

    if not SDXL_CKPT:
        print("[accept] 未指定 SDXL_CKPT：跳过 SDXL 用例（t1~t6/t8），"
              "仅跑 t7 回归等不依赖 SDXL 的用例", flush=True)

    mcv = pos = neg = lat0 = img1 = None
    if SDXL_CKPT:
        # SDXL 底模加载（offload=model：8GB 卡上 UNet 5.1G + 双 TE 1.9G 的默认选择）
        mcv = op("checkpoint.load", ckpt=SDXL_CKPT, offload="model")
        print(f"[accept] SDXL 加载完成 model={mcv['model']}", flush=True)

    def mkconds():
        """重建 pos/neg/lat0。对象仓库 TTL=3600s，长套件后段引用会过期，
        故在每个用到它们的测试前刷新（编码很快，缓存命中秒回）。"""
        nonlocal pos, neg, lat0
        pos = op("clip.encode", clip=mcv["clip"], text=PROMPT,
                 width=SDXL_SIZE, height=SDXL_SIZE)["cond"]
        neg = op("clip.encode", clip=mcv["clip"], text=NEG,
                 width=SDXL_SIZE, height=SDXL_SIZE)["cond"]
        lat0 = op("latent.empty", width=SDXL_SIZE, height=SDXL_SIZE,
                  batch_size=1)["latent"]

    def fresh_sdxl():
        """TTL 兑底：model/clip/vae 视图与条件一起重建（checkpoint.load 缓存命中）。"""
        nonlocal mcv
        mcv = op("checkpoint.load", ckpt=SDXL_CKPT, offload="model")
        mkconds()

    if SDXL_CKPT:
        mkconds()

    # ---------------- t1 txt2img ----------------
    r1 = None
    if want("t1") and SDXL_CKPT:
        t0 = time.time()
        r1 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat0,
                seed=SDXL_SEED, steps=SDXL_STEPS, cfg=SDXL_CFG,
                sampler_name="euler", scheduler="normal", denoise=1.0)
        img = op("vae.decode", vae=mcv["vae"], latent=r1["latent"])["image"]
        check_img("t1_sdxl_txt2img",
                  fetch_image(img), expect_size=(SDXL_SIZE, SDXL_SIZE))
        print(f"[accept] t1 采样+解码 {time.time() - t0:.0f}s "
              f"(seed={r1['seed']}, steps={SDXL_STEPS}, cfg={SDXL_CFG})",
              flush=True)

    # ---------------- t2 i2i ----------------
    if want("t2") and SDXL_CKPT:
        if r1 is None:
            r1 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat0,
                    seed=SDXL_SEED, steps=SDXL_STEPS, cfg=SDXL_CFG,
                    sampler_name="euler", scheduler="normal", denoise=1.0)
            img = op("vae.decode", vae=mcv["vae"], latent=r1["latent"])["image"]
            check_img("t1_sdxl_txt2img", fetch_image(img),
                      expect_size=(SDXL_SIZE, SDXL_SIZE))
        img1 = op("vae.decode", vae=mcv["vae"], latent=r1["latent"])["image"]
        lat_i2i = op("vae.encode", vae=mcv["vae"],
                     image=img1)["latent"]
        r2 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat_i2i,
                seed=SDXL_SEED + 1, steps=SDXL_STEPS, cfg=SDXL_CFG,
                sampler_name="euler", scheduler="normal", denoise=0.6)
        img2 = op("vae.decode", vae=mcv["vae"], latent=r2["latent"])["image"]
        check_img("t2_sdxl_i2i", fetch_image(img2),
                  expect_size=(SDXL_SIZE, SDXL_SIZE))

    # ---------------- t3 hires ----------------
    if want("t3") and SDXL_CKPT:
        if r1 is None:
            r1 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=lat0,
                    seed=SDXL_SEED, steps=SDXL_STEPS, cfg=SDXL_CFG,
                    sampler_name="euler", scheduler="normal", denoise=1.0)
        up = op("latent.upscale", latent=r1["latent"], scale=1.5)["latent"]
        r3 = op("sample", model=mcv["model"], pos=pos, neg=neg, latent=up,
                seed=SDXL_SEED + 2, steps=SDXL_STEPS, cfg=SDXL_CFG,
                sampler_name="euler", scheduler="normal", denoise=0.45)
        img3 = op("vae.decode", vae=mcv["vae"], latent=r3["latent"])["image"]
        check_img("t3_sdxl_hires_1536", fetch_image(img3),
                  expect_size=(int(SDXL_SIZE * 1.5), int(SDXL_SIZE * 1.5)))

    # ---------------- t4 SDXL ControlNet ----------------
    if want("t4") and SDXL_CKPT:
        if not SDXL_CN:
            skip("t4", "未指定 SDXL_CN（SDXL ControlNet 权重名）")
        else:
            mkconds()  # t4 前刷新（t1~t3 已耗时 ~20 分钟）
            cn = op("controlnet.load", name=SDXL_CN)["control_net"]
            # hint：从 img1（t2 产物）抽 Canny 线稿；或用 HINT_IMG 指定 INPUT_DIR 里的图
            hint = None
            if os.environ.get("HINT_IMG"):
                hint = op("image.load", name=os.environ["HINT_IMG"])["image"]
            elif img1 is not None:
                hint = op("preprocess.canny", image=img1)["image"]
            if hint is None:
                skip("t4", "无 hint 图（需先跑 t1/t2 产出 img1，或指定 HINT_IMG）")
            else:
                ctl = op("controlnet.apply", control_net=cn, image=hint,
                         strength=0.8)["control"]
                # 8GB 卡上 SDXL+CN 峰值 ~8.2GB 会換页抖动（~90s/步），
                # t4 验证链路机制即可，默认降到 10 步（SDXL_CN_STEPS 可调）
                cn_steps = int(os.environ.get("SDXL_CN_STEPS", "10"))
                r4 = op("sample", model=mcv["model"], pos=pos, neg=neg,
                        latent=lat0, seed=SDXL_SEED + 3,
                        steps=min(SDXL_STEPS, cn_steps),
                        cfg=SDXL_CFG, sampler_name="euler", scheduler="normal",
                        denoise=1.0, control=ctl)
                img4 = op("vae.decode", vae=mcv["vae"], latent=r4["latent"])["image"]
                check_img("t4_sdxl_controlnet", fetch_image(img4),
                          expect_size=(SDXL_SIZE, SDXL_SIZE))
                # 卸载 CN：避免其常驻 1.25GB 拖慢后续 t5~t8 采样（90s/步 抖动）
                call("/model/unload", {"target": f"cn:{SDXL_CN}|dtype=fp16"})

    # ---------------- t5 offload 三档 ----------------
    if want("t5") and SDXL_CKPT:
        mkconds()  # t5 三档重载耗时 ~25 分钟，pos/neg/lat0 先刷新
        for mode in ("none", "model", "sequential"):
            t0 = time.time()
            out, err = run("checkpoint.load", ckpt=SDXL_CKPT, offload=mode)
            if err:
                print(f"[accept] t5 offload={mode}: 加载/常驻失败 -> "
                      f"{str(err)[:160]}", flush=True)
                continue
            key = out["model"]
            out2, err2 = run("sample", model=key, pos=pos, neg=neg,
                             latent=lat0, seed=SDXL_SEED, steps=3, cfg=SDXL_CFG,
                             sampler_name="euler", scheduler="normal")
            dt = time.time() - t0
            if err2:
                print(f"[accept] t5 offload={mode}: 3 步采样失败 -> "
                      f"{str(err2)[:160]}", flush=True)
            else:
                print(f"[accept] t5 offload={mode}: 3 步采样 OK，"
                      f"加载+采样 {dt:.0f}s", flush=True)
            op("model.unload", target="all")

    # ---------------- t6 显存回收 ----------------
    if want("t6") and SDXL_CKPT:
        _, h0 = call("/health")
        op("model.unload", target="all")
        _, h1 = call("/health")
        print(f"[accept] t6 unload: resident_weights "
              f"{h0.get('resident_weights_mb')}MB -> "
              f"{h1.get('resident_weights_mb')}MB, vram_free "
              f"{h0.get('vram_free_mb')}MB -> {h1.get('vram_free_mb')}MB",
              flush=True)
        if h1.get("resident_models"):
            failures.append(f"t6: unload 后仍常驻 {h1['resident_models']}")

    # ---------------- t7 SD1.5 回归（逐字节） ----------------
    if want("t7"):
        m15 = op("checkpoint.load", ckpt=SD15_CKPT, offload="none")
        p15 = op("clip.encode", clip=m15["clip"], text=SD15_PROMPT)["cond"]
        n15 = op("clip.encode", clip=m15["clip"], text="")["cond"]
        l15 = op("latent.empty", width=512, height=512, batch_size=1)["latent"]
        r15 = op("sample", model=m15["model"], pos=p15, neg=n15, latent=l15,
                 seed=SD15_SEED, steps=SD15_STEPS, cfg=SD15_CFG,
                 sampler_name="euler", scheduler="normal", denoise=1.0)
        img15 = op("vae.decode", vae=m15["vae"], latent=r15["latent"])["image"]
        png15 = fetch_image(img15)
        sha15 = hashlib.sha1(png15).hexdigest()
        save("t7_sd15_regression", png15)
        if os.path.isfile(BASELINE_M2):
            with open(BASELINE_M2, "rb") as f:
                base = hashlib.sha1(f.read()).hexdigest()
            if sha15 != base:
                failures.append(f"t7: SD1.5 回归不一致 {sha15[:12]} != "
                                f"{base[:12]}（M5 改动破坏了 SD1.x 行为？）")
            print(f"[accept] t7 SD1.5 回归: "
                  f"{'逐字节一致' if sha15 == base else '不一致!'} "
                  f"sha1={sha15[:12]}", flush=True)
        else:
            skip("t7", f"缺 M2 基线 {BASELINE_M2}")

    # ---------------- t8 SDXL + 区域条件（SDXLSegments） ----------------
    if want("t8") and SDXL_CKPT:
        fresh_sdxl()  # 走到这里通常已 >50 分钟：model 视图与条件全部重建
        pos_a = op("cond.set_area", cond=pos, x=0, y=0,
                   width=SDXL_SIZE, height=SDXL_SIZE // 2,
                   strength=1.0)["cond"]
        pos_b = op("cond.set_area", cond=op(
            "clip.encode", clip=mcv["clip"],
            text="a mountain landscape, snow peak",
            width=SDXL_SIZE, height=SDXL_SIZE)["cond"],
            x=0, y=SDXL_SIZE // 2, width=SDXL_SIZE, height=SDXL_SIZE // 2,
            strength=1.0)["cond"]
        pos_c = op("cond.combine", cond_a=pos_a, cond_b=pos_b)["cond"]
        neg_c = op("cond.combine", cond_a=neg, cond_b=neg)["cond"]
        r8 = op("sample", model=mcv["model"], pos=pos_c, neg=neg_c,
                latent=lat0, seed=SDXL_SEED + 4, steps=SDXL_STEPS,
                cfg=SDXL_CFG, sampler_name="euler", scheduler="normal")
        img8 = op("vae.decode", vae=mcv["vae"], latent=r8["latent"])["image"]
        check_img("t8_sdxl_area_combine", fetch_image(img8),
                  expect_size=(SDXL_SIZE, SDXL_SIZE))

    # ---------------- t9（可选）与官方管道逐像素一致 ----------------
    # 参考图由容器内脚本生成（同一模型/种子/步数/cfg，官方 denoising loop，
    # 输出写 INPUT_DIR/accept-m5-ref.png）；本步骤做像素级比对。
    # 两条进程不能同时常驻 SDXL（8GB 不够），故先卸载本进程常驻模型，
    # 参考脚本在**另一个容器**里跑（命令见下方打印 + dev-plan 二十五）。
    if want("t9") and SDXL_CKPT:
        op("model.unload", target="all")
        _, h = call("/health")
        ref_name = os.environ.get("REF_IMG", "accept-m5-ref.png")
        print(f"[accept] t9 参考图：{ref_name}（INPUT_DIR）"
              f"，当前空闲显存 {h.get('vram_free_mb')}MB", flush=True)
        out, err = run("image.load", name=ref_name)
        if err:
            ref_cmd = (
                "docker -H tcp://<解析后IP>:<port> run --rm --gpus all "
                "--volumes-from flowx-inference-server "
                "-e CKPT=<模型名> -e SIZE=%d -e STEPS=%d -e CFG=%s "
                "-e SEED=%d -v <repo>/inference-server/scripts:/scripts "
                "--entrypoint python3 flowx-inference-server:pascal "
                "/scripts/accept-m5-pipe-ref.py" % (SDXL_SIZE, SDXL_STEPS,
                                                    SDXL_CFG, SDXL_SEED))
            skip("t9", f"参考图不存在（{str(err)[:80]}）；先跑：\n      "
                     f"{ref_cmd}")
        else:
            ref_png = fetch_image(out["image"])
            save("t9_pipe_ref", ref_png)
            tt = [t for t in ("t1_sdxl_txt2img",) if os.path.isfile(
                os.path.join(OUT_DIR, f"{t}.png"))]
            if not tt:
                skip("t9", "缺 t1 输出（先跑 t1）")
            else:
                from PIL import Image, ImageChops
                a = Image.open(os.path.join(OUT_DIR, f"{tt[0]}.png")).convert("RGB")
                b = Image.open(io.BytesIO(ref_png)).convert("RGB")
                same_px = a.size == b.size and \
                    ImageChops.difference(a, b).getbbox() is None
                same_sha = hashlib.sha1(open(
                    os.path.join(OUT_DIR, f"{tt[0]}.png"), "rb").read()
                ).hexdigest() == hashlib.sha1(ref_png).hexdigest()
                print(f"[accept] t9 vs 官方管道：像素"
                      f"{'一致' if same_px else '不一致'}"
                      f"，PNG sha1 {'一致' if same_sha else '不一致'}", flush=True)
                if not same_px:
                    failures.append("t9: 手动循环与官方管道出图不一致"
                                    "（先看 accept-m5-out/t1_* 与 t9_pipe_ref）")

    dt = time.time() - t_start
    print(f"\n[accept] 总耗时 {dt:.0f}s", flush=True)
    if skips:
        print("[accept] SKIPS:", *skips, sep="\n  - ", flush=True)
    if failures:
        print("[accept] FAILURES:", *failures, sep="\n  - ", flush=True)
        sys.exit(1)
    print("[accept] ALL PASS", flush=True)


if __name__ == "__main__":
    main()

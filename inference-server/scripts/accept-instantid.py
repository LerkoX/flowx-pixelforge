#!/usr/bin/env python3
"""InstantID 真机验收（GTX 1080 8GB，经 cpolar 隧道；dev-plan 二十八）。

前置：Phase 0 权重已就位（/models/ipadapter/instantid-ip-adapter.bin、
/models/instantid-controlnet/、/models/insightface/models/antelopev2/），
镜像含 insightface/onnxruntime，插件 instantid_ops.py 已热加载。

- t0 预检：/health、/ops 有 face.analyze/instantid.apply、权重文件就位
- t1 face.analyze：/input/test_person.jpg → 检出人脸、kps 图非黑有彩点
- t2 InstantID 端到端：SDXL(offload=model) + instantid.apply → sample 25 步
      → 出图内容核对；t2b 用 insightface 回测出图人脸与参考照的余弦相似度
      （量化身份保持，不纯靠肉眼——验收口径纪律）
- t3 消融：weight=0 & cn_strength=0 ≡ 同种子无 InstantID 基线逐字节一致
      （IPA scale=0 与 CN strength=0 的零影响既有结论的组合复验）
- t4 回归：SD1.5 默认采样 sha1=48e3225e8b38（M2/M6 基线，ops.py 改动不变性）
- t5 负面：SD1.5 模型 / 非 Resampler 权重走 instantid.apply → 明确报错
- t6 显存：model.unload 后余量回收

运行：
  BASE=http://x.tcp.cpolar.top:PORT TOKEN=xxx python3 scripts/accept-instantid.py
  # 只跑部分：ONLY=t1,t4 python3 scripts/accept-instantid.py
  # 相似度回测（容器内，需 docker）：另跑 scripts/accept-instantid-sim.py
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
SDXL_CKPT = os.environ.get("SDXL_CKPT", "sd_xl_base_1.0")
SD15_CKPT = os.environ.get("SD15_CKPT", "v1-5-pruned-emaonly-fp16")
IPA_NAME = os.environ.get("INSTANTID_IPA", "instantid-ip-adapter")
CN_NAME = os.environ.get("INSTANTID_CN", "instantid-controlnet")
FACE_REF = os.environ.get("FACE_REF", "test_person.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-instantid-out"))
BASELINE_M2 = os.path.join(os.path.dirname(__file__),
                           "accept-m2-out", "t4_euler_normal.png")
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}

SIZE = int(os.environ.get("SIZE", "1024"))
STEPS = int(os.environ.get("STEPS", "25"))
CFG = float(os.environ.get("CFG", "5.0"))  # 官方默认 guidance_scale=5.0
SEED = int(os.environ.get("SEED", "20260924"))
PROMPT = os.environ.get(
    "PROMPT", "portrait photo of an astronaut wearing a spacesuit, studio "
              "lighting, highly detailed, masterpiece, best quality")
NEG = os.environ.get(
    "NEG", "(lowres, low quality, worst quality:1.2), watermark, text, "
           "deformed, mutated, ugly, disfigured")

TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "2400"))
OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image",
             "control_net", "control", "ipadapter", "controlnet", "face"}

failures = []
skips = []


def call(path, payload=None, timeout=600, expect_error=False, retries=8):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    for attempt in range(retries):
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
        except (urllib.error.URLError, OSError) as e:
            # 隧道抖动：轮询类调用重试（隧道恢复后 job 还在服务端跑）
            if attempt == retries - 1:
                raise
            print(f"[accept] {path} 连接抖动，{5*(attempt+1)}s 后重试 "
                  f"({e!r:.80})", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


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
              f"diffusers={h.get('diffusers_version')} "
              f"vram_free={h.get('vram_free_mb')}MB "
              f"resident={h.get('resident_models')}", flush=True)
        _, ops = call("/ops")
        names = {o["name"] for o in ops.get("ops", [])}
        for need in ("face.analyze", "instantid.apply", "ipadapter.load",
                     "controlnet.load", "checkpoint.load", "sample"):
            if need not in names:
                failures.append(f"t0: 算子缺失 {need}")
        _, files = call("/models/files")
        fl = files.get("files", [])
        have = {f["name"] for f in fl}
        for need in (SDXL_CKPT, CN_NAME):
            if need not in have:
                failures.append(f"t0: 模型缺失 {need}（现有: {sorted(have)}）")
        ipa_files = [f["name"] for f in fl if f.get("kind") == "ipadapter"
                     or IPA_NAME in f.get("name", "")]
        print(f"[accept] t0 ipadapter 相关: {ipa_files}", flush=True)

    # ---------------- t1 face.analyze ----------------
    face_id = None
    kps_png = None
    if want("t1"):
        try:
            img = op("image.load", name=FACE_REF)
        except RuntimeError as e:
            skip("t1", f"参考照 {FACE_REF} 不在 /input：{e}")
        else:
            out, err = run("face.analyze", image=img["image"], face_index=-1)
            if err:
                failures.append(f"t1: face.analyze 失败：{err}")
            else:
                face_id = out["face"]
                kps_png = fetch_image(out["kps"])
                save("t1_kps", kps_png)
                # kps 图应稀疏（黑底 + 5 点 + 4 短线）；不走 check_img
                # 的亮度门禁（kps 图设计即近全黑，mean<2 是正常的）
                from PIL import Image
                im = Image.open(io.BytesIO(kps_png)).convert("RGB")
                px = list(im.getdata())
                lit = sum(1 for p in px if sum(p) > 30)
                ratio = lit / len(px)
                print(f"[accept] t1 kps 非黑像素占比 {ratio:.4%} "
                      f"size={im.size}", flush=True)
                if not 0.0001 < ratio < 0.05:
                    failures.append(f"t1: kps 像素占比异常 {ratio:.4%}")
                if len(set(p for p in px if sum(p) > 30)) < 3:
                    failures.append("t1: kps 颜色种类过少（五色关键点未生效）")
    else:
        try:
            img = op("image.load", name=FACE_REF)
            out = op("face.analyze", image=img["image"], face_index=-1)
            face_id = out["face"]
        except RuntimeError:
            pass

    # ---------------- t2 InstantID 端到端 ----------------
    if want("t2"):
        if face_id is None:
            skip("t2", "t1 未产出 face 对象")
        else:
            m = op("checkpoint.load", ckpt=SDXL_CKPT, dtype="fp16",
                   offload="model")
            pos = op("clip.encode", clip=m["clip"], text=PROMPT,
                     width=SIZE, height=SIZE)
            neg = op("clip.encode", clip=m["clip"], text=NEG,
                     width=SIZE, height=SIZE)
            lat = op("latent.empty", width=SIZE, height=SIZE, batch_size=1)
            ipa = op("ipadapter.load", name=IPA_NAME)
            cn = op("controlnet.load", name=CN_NAME)
            ap = op("instantid.apply", model=m["model"],
                    ipadapter=ipa["ipadapter"], controlnet=cn["control_net"],
                    face=face_id, weight=0.8, cn_strength=0.8)
            t0 = time.time()
            out, err = run("sample", model=ap["model"], pos=pos["cond"],
                           neg=neg["cond"], latent=lat["latent"], seed=SEED,
                           steps=STEPS, cfg=CFG, control=ap["control"])
            dt = time.time() - t0
            if err:
                failures.append(f"t2: sample 失败：{err}")
            else:
                print(f"[accept] t2 sample {dt:.0f}s（{STEPS} 步 + CN）",
                      flush=True)
                png = fetch_image(op("vae.decode", vae=m["vae"],
                                     latent=out["latent"])["image"])
                check_img("t2_instantid", png, expect_size=(SIZE, SIZE))
            # t2b：出图里应能检出人脸（身份相似度容器内回测，见 sim 脚本）
            if not err:
                up = urllib.request.Request(f"{BASE}/images", data=png,
                                            method="POST")
                if TOKEN:
                    up.add_header("Authorization", "Bearer " + TOKEN)
                with urllib.request.urlopen(up, timeout=120) as r:
                    gen_id = json.loads(r.read())["id"]
                out2, err2 = run("face.analyze", image=gen_id, face_index=-1)
                if err2:
                    failures.append(f"t2b: 出图未检出人脸（身份注入失败？）："
                                    f"{err2}")
                else:
                    print("[accept] t2b 出图人脸检出 OK（相似度回测走 "
                          "accept-instantid-sim.py）", flush=True)

    # ---------------- t3 消融：weight=0 & cn_strength=0 ≡ 基线 ----------------
    # （行为不变性命题与分辨率/步数无关，用快档跑：768px/10 步）
    T3_SIZE, T3_STEPS = 768, 10
    if want("t3"):
        if face_id is None:
            skip("t3", "t1 未产出 face 对象")
        else:
            m = op("checkpoint.load", ckpt=SDXL_CKPT, dtype="fp16",
                   offload="model")
            pos = op("clip.encode", clip=m["clip"], text=PROMPT,
                     width=T3_SIZE, height=T3_SIZE)
            neg = op("clip.encode", clip=m["clip"], text=NEG,
                     width=T3_SIZE, height=T3_SIZE)
            lat = op("latent.empty", width=T3_SIZE, height=T3_SIZE, batch_size=1)
            # 基线：无 InstantID
            base_out = op("sample", model=m["model"], pos=pos["cond"],
                          neg=neg["cond"], latent=lat["latent"], seed=SEED,
                          steps=T3_STEPS, cfg=CFG)
            png_base = fetch_image(op("vae.decode", vae=m["vae"],
                                      latent=base_out["latent"])["image"])
            sha_base = check_img("t3_baseline", png_base,
                                 expect_size=(T3_SIZE, T3_SIZE))
            # 消融：InstantID 全挂但强度全 0
            ipa = op("ipadapter.load", name=IPA_NAME)
            cn = op("controlnet.load", name=CN_NAME)
            ap = op("instantid.apply", model=m["model"],
                    ipadapter=ipa["ipadapter"], controlnet=cn["control_net"],
                    face=face_id, weight=0.0, cn_strength=0.0)
            abl_out = op("sample", model=ap["model"], pos=pos["cond"],
                         neg=neg["cond"], latent=lat["latent"], seed=SEED,
                         steps=T3_STEPS, cfg=CFG, control=ap["control"])
            png_abl = fetch_image(op("vae.decode", vae=m["vae"],
                                     latent=abl_out["latent"])["image"])
            sha_abl = check_img("t3_ablation_zero", png_abl,
                                expect_size=(T3_SIZE, T3_SIZE))
            if sha_base != sha_abl:
                failures.append(f"t3: weight=0/cn_strength=0 应 ≡ 基线，"
                                f"sha {sha_abl[:12]} != {sha_base[:12]}")
            else:
                print("[accept] t3 零强度 ≡ 基线 逐字节一致", flush=True)
            # 消融后再跑一次裸基线：验证 IPA 处理器/投影层恢复无残留污染
            # （140 vs 144 恢复失败事故就是靠这类复跑抓的）
            re_out = op("sample", model=m["model"], pos=pos["cond"],
                        neg=neg["cond"], latent=lat["latent"], seed=SEED,
                        steps=T3_STEPS, cfg=CFG)
            png_re = fetch_image(op("vae.decode", vae=m["vae"],
                                    latent=re_out["latent"])["image"])
            sha_re = check_img("t3_baseline_rerun", png_re,
                               expect_size=(T3_SIZE, T3_SIZE))
            if sha_re != sha_base:
                failures.append(f"t3: 消融后裸基线复跑被污染 "
                                f"sha {sha_re[:12]} != {sha_base[:12]}")
            else:
                print("[accept] t3 复跑零污染（处理器/投影恢复干净）",
                      flush=True)

    # ---------------- t4 SD1.5 回归 ----------------
    if want("t4"):
        if not os.path.exists(BASELINE_M2):
            skip("t4", f"基线文件缺失 {BASELINE_M2}")
        else:
            m = op("checkpoint.load", ckpt=SD15_CKPT)
            pos = op("clip.encode", clip=m["clip"],
                     text="a cat sitting on a windowsill, warm sunset light, "
                          "high quality")
            neg = op("clip.encode", clip=m["clip"],
                     text="lowres, blurry, worst quality")
            lat = op("latent.empty", width=512, height=512, batch_size=1)
            out = op("sample", model=m["model"], pos=pos["cond"],
                     neg=neg["cond"], latent=lat["latent"], seed=424242,
                     steps=20, cfg=7.0)
            png = fetch_image(op("vae.decode", vae=m["vae"],
                                 latent=out["latent"])["image"])
            sha = check_img("t4_sd15_regression", png,
                            expect_size=(512, 512))
            want_sha = hashlib.sha1(open(BASELINE_M2, "rb").read()).hexdigest()
            if sha != want_sha:
                failures.append(f"t4: SD1.5 回归 sha1={sha[:12]} != "
                                f"基线 {want_sha[:12]}")
            else:
                print("[accept] t4 SD1.5 与 M2 基线逐字节一致", flush=True)

    # ---------------- t5 负面用例 ----------------
    if want("t5"):
        m15 = op("checkpoint.load", ckpt=SD15_CKPT)
        ipa = op("ipadapter.load", name=IPA_NAME)
        cn = op("controlnet.load", name=CN_NAME)
        img = op("image.load", name=FACE_REF)
        fa = op("face.analyze", image=img["image"], face_index=-1)
        _, err = run("instantid.apply", model=m15["model"],
                     ipadapter=ipa["ipadapter"],
                     controlnet=cn["control_net"], face=fa["face"])
        if not err or "SDXL" not in err:
            failures.append(f"t5a: SD1.5 模型应报 SDXL 限定错误，got {err!r}")
        else:
            print(f"[accept] t5a SD1.5 明确拒绝：{err[:80]}", flush=True)
        ipa15 = op("ipadapter.load", name="ip-adapter_sd15")
        mxl = op("checkpoint.load", ckpt=SDXL_CKPT, dtype="fp16",
                 offload="model")
        _, err = run("instantid.apply", model=mxl["model"],
                     ipadapter=ipa15["ipadapter"],
                     controlnet=cn["control_net"], face=fa["face"])
        if not err or "Resampler" not in err:
            failures.append(f"t5b: 非 InstantID 权重应报 Resampler 错误，"
                            f"got {err!r}")
        else:
            print(f"[accept] t5b 标准 IPA 权重明确拒绝：{err[:80]}", flush=True)

    # ---------------- t6 显存回收 ----------------
    if want("t6"):
        _, before = call("/health")
        call("/model/unload", {})
        call("/gc", {})
        _, after = call("/health")
        print(f"[accept] t6 resident {before.get('resident_models')} -> "
              f"{after.get('resident_models')}，可用 "
              f"{before.get('vram_free_mb')} -> {after.get('vram_free_mb')}MB",
              flush=True)
        if after.get("resident_models"):
            failures.append(f"t6: unload 后仍有常驻 {after['resident_models']}")

    # ---------------- 汇总 ----------------
    dt = time.time() - t_start
    print(f"\n==== 汇总（{dt:.0f}s）====")
    for s in skips:
        print(f"SKIP: {s}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

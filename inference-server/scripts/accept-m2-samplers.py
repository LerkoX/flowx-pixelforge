"""M2 验收：sampler × scheduler 解耦（sampler_name 更新公式 × sigma 曲线自由组合）。

用法：
  # 容器外（隧道 / 局域网）
  BASE=http://<host>:8100 TOKEN=<token> python3 scripts/accept-m2-samplers.py

流程与判定（ckpt 默认 v1-5-pruned-emaonly-fp16，OUT_DIR 存抽验图）：
  1. 确定性对照：同 seed 下旧一体名 dpmpp_2m_karras 与新写法 dpmpp_2m+karras
     出图必须逐字节一致（兼容映射不改变行为）
  2. 新组合出图：euler+karras / euler+exponential / dpmpp_2m+beta /
     uni_pc+exponential，亮度均值落在正常区间（非黑非爆）且图片下载存盘供抽图核对
  3. 非法组合 euler_a+karras 必须明确报错（job failed 且错误信息点明不支持）
  4. 回归：默认 euler(+缺省 scheduler) 出图正常
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
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-m2-out"))
STEPS = int(os.environ.get("STEPS", "20"))
SEED = 424242
PROMPT = "a cat sitting on a windowsill, warm sunset light, high quality"
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

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


def run_job(name, **inputs):
    """提交异步 job 并轮询到终态，返回 outputs dict；失败返回 None + 记 error。"""
    wrapped = {k: ({"$id": v} if k in ("vae", "model", "clip", "cond", "latent",
                                      "pos", "neg", "image") else v)
               for k, v in inputs.items()}
    jid = call("/jobs", {"name": name, "inputs": wrapped})["job_id"]
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


def op(name, **inputs):
    out, err = run_job(name, **inputs)
    if err:
        raise RuntimeError(f"op {name} failed: {err}")
    return out


def fetch_image(image_id):
    req = urllib.request.Request(f"{BASE}/images/{image_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def sample_and_decode(model, vae, pos, neg, latent0, tag, sampler, scheduler,
                      save=True):
    """一轮 sample+decode，返回 (png_bytes, mean)；失败记 failures 返回 None。"""
    label = f"{sampler}+{scheduler}" if scheduler else sampler
    t0 = time.time()
    lat = op("sample", model=model, pos=pos, neg=neg, latent=latent0,
             seed=SEED, steps=STEPS, cfg=7.0, sampler_name=sampler,
             scheduler=scheduler or "normal", denoise=1.0)["latent"]
    img_id = op("vae.decode", vae=vae, latent=lat)["image"]
    png = fetch_image(img_id)
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(png)).convert("L")
    mean = ImageStat.Stat(im).mean[0]
    dt = time.time() - t0
    print(f"[accept] {tag} {label}: {dt:.0f}s mean={mean:.1f} "
          f"sha1={hashlib.sha1(png).hexdigest()[:12]}", flush=True)
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f"{tag}_{sampler}_{scheduler or 'normal'}.png"),
                  "wb") as f:
            f.write(png)
    if not (10.0 < mean < 245.0):
        failures.append(f"{tag} {label}: 亮度异常 mean={mean:.1f}")
    return png, mean


def main():
    print(f"[accept] base={BASE} ckpt={CKPT} steps={STEPS} seed={SEED}", flush=True)

    out = op("checkpoint.load", ckpt=CKPT)
    model, clip, vae = out["model"], out["clip"], out["vae"]
    print(f"[accept] checkpoint loaded: model={model}", flush=True)
    pos = op("clip.encode", clip=clip, text=PROMPT)["cond"]
    neg = op("clip.encode", clip=clip, text="")["cond"]
    latent0 = op("latent.empty", width=512, height=512, batch_size=1)["latent"]

    # 1. 确定性对照：旧一体名 vs 新写法，同 seed 必须逐字节一致
    png_old, _ = sample_and_decode(model, vae, pos, neg, latent0,
                                   "t1a", "dpmpp_2m_karras", "normal")
    png_new, _ = sample_and_decode(model, vae, pos, neg, latent0,
                                   "t1b", "dpmpp_2m", "karras")
    if hashlib.sha1(png_old).hexdigest() == hashlib.sha1(png_new).hexdigest():
        print("[accept] t1 PASS: 旧名 dpmpp_2m_karras ≡ dpmpp_2m+karras（逐字节一致）",
              flush=True)
    else:
        failures.append("t1: 旧名兼容结果不一致")

    # 2. 新组合出图（抽图存盘）
    combos = [("euler", "karras"), ("euler", "exponential"),
              ("dpmpp_2m", "beta"), ("uni_pc", "exponential")]
    for i, (s, sc) in enumerate(combos):
        sample_and_decode(model, vae, pos, neg, latent0, f"t2{i}", s, sc)

    # 3. 非法组合必须明确报错
    _, err = run_job("sample", model=model, pos=pos, neg=neg, latent=latent0,
                     seed=SEED, steps=4, cfg=7.0, sampler_name="euler_a",
                     scheduler="karras", denoise=1.0)
    if err and "euler_a" in err and "karras" in err:
        print(f"[accept] t3 PASS: euler_a+karras 明确报错: {err[:100]}", flush=True)
    else:
        failures.append(f"t3: 非法组合未按预期报错（err={err!r}）")

    # 4. 回归：默认 euler（scheduler 缺省）
    sample_and_decode(model, vae, pos, neg, latent0, "t4", "euler", "normal")

    if failures:
        print(f"[accept] FAIL ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print(f"[accept] ALL PASS（抽验图已存 {OUT_DIR}，请人工看图确认内容）", flush=True)


if __name__ == "__main__":
    main()

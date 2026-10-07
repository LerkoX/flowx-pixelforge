#!/usr/bin/env python3
"""FLUX.1-schnell（GGUF 路线）真机验收（GTX 1080 8GB，经隧道；dev-plan 二十九）。

前置：Phase 0 权重就位（/models/flux/flux1-schnell-Q4_K_S.gguf、
t5xxl-Q4_K_S.gguf、ae.safetensors、clip_l、tokenizer*、t5 config/tokenizer、
scheduler/），镜像含 gguf/sentencepiece，flux 家族算子由节点 server_op.py 自注册（push_plugins.py 预热）。
首次运行 T5 会建 npy 分片缓存（~6min，只一次）。

- t0 预检：/health、/ops 有 flux.load/flux.encode/flux.sample
- t1 端到端：flux.load → flux.encode（招牌 case：多元素+文字渲染，FLUX 强项）
     → flux.sample → 出图存盘（语义级正确性无像素基线，人工核对元素齐全）
- t2 资源回收：model.unload 后 resident 清空、VRAM 回到空载水位
     （顺带覆盖"flux→SDXL 切换"编排前提：卸载后 SDXL 可正常加载采样）
- t3 回归：SD1.5 txt2img sha1 == 48e3225e8b38（M2/M6 基线）——验证插件
     全局 RMSNorm/SDPA shim 补丁对既有路径零影响
- t4 负面：非 FLUX 管道（SD1.5）喂 flux.encode / flux.sample → 明确报错

运行：
  BASE=http://x.tcp.cpolar.top:PORT TOKEN=xxx python3 scripts/accept-flux.py
  ONLY=t1 SIZE=512 STEPS=4 python3 scripts/accept-flux.py
"""
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8100")
TOKEN = os.environ.get("TOKEN", "")
SD15_CKPT = os.environ.get("SD15_CKPT", "v1-5-pruned-emaonly-fp16")
SDXL_CKPT = os.environ.get("SDXL_CKPT", "sd_xl_base_1.0")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__),
                                                 "accept-flux-out"))
os.makedirs(OUT_DIR, exist_ok=True)
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}

TRANSFORMER = os.environ.get("FLUX_TRANSFORMER", "flux1-schnell-Q4_K_S.gguf")
T5 = os.environ.get("FLUX_T5", "t5xxl-Q4_K_S.gguf")
SIZE = int(os.environ.get("SIZE", "512"))
STEPS = int(os.environ.get("STEPS", "4"))
SEED = int(os.environ.get("SEED", "20260925"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "256"))
# 招牌 case：多主体 + 精确文字渲染 + 版式（SDXL/SD1.5 做不出来的那类）
PROMPT = os.environ.get(
    "PROMPT", "a cozy coffee shop interior, a chalkboard menu on the wall "
              "with the text 'FLOWX COFFEE' written in white chalk, a cat "
              "sleeping on the counter, warm sunlight through the window, "
              "photorealistic, 4k")

TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "3600"))
OBJ_PORTS = {"model", "clip", "vae", "pos", "neg", "latent", "image", "cond", "control"}

failures = []


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


def fetch_image(obj_id):
    req = urllib.request.Request(f"{BASE}/images/{obj_id}")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def vram_free_mb():
    _, h = call("/health")
    return h.get("vram_free_mb"), h.get("resident")


def check(tid, desc, fn):
    if ONLY and tid not in ONLY:
        return
    print(f"[{tid}] {desc}", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"[{tid}] OK ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        failures.append(f"{tid}: {e}")
        print(f"[{tid}] FAIL: {e}", flush=True)


def t0():
    _, h = call("/health")
    assert h["status"] == "ok"
    _, ops = call("/ops")
    names = {o["name"] if isinstance(o, dict) else o for o in
             (ops.get("ops", ops) if isinstance(ops, dict) else ops)}
    for n in ("flux.load", "flux.encode", "flux.sample"):
        assert n in names, f"{n} 未注册"
    print("  ops: flux.* 已注册", flush=True)


def t1():
    m = op("flux.load", transformer=TRANSFORMER, t5=T5, offload="auto")
    model = m["model"]
    c = op("flux.encode", model=model, text=PROMPT, max_seq=MAX_SEQ,
           release_t5=1)
    r = op("flux.sample", model=model, cond=c["cond"], width=SIZE,
           height=SIZE, seed=SEED, steps=STEPS, guidance=0.0)
    assert r.get("seed") == SEED, f"seed 回传不符: {r}"
    png = fetch_image(r["image"])
    sha = hashlib.sha1(png).hexdigest()[:12]
    path = os.path.join(OUT_DIR, f"t1_flux_{SIZE}px.png")
    open(path, "wb").write(png)
    assert len(png) > 50000, f"图像过小（{len(png)}B）疑似黑图/坏图"
    print(f"  出图 {len(png)}B sha1={sha} -> {path}（人工核对："
          f"招牌文字 FLOWX COFFEE + 猫 + 咖啡馆元素）", flush=True)
    return model  # 供 t2 用


_T1_MODEL = []


def t1_wrap():
    _T1_MODEL.clear()
    _T1_MODEL.append(t1())


def t2():
    free0, resident0 = vram_free_mb()
    # 卸载 flux（t1 若未跑则先装再卸，独立可跑）
    if ONLY and "t1" not in ONLY or not _T1_MODEL:
        m = op("flux.load", transformer=TRANSFORMER, t5=T5, offload="auto")
        model = m["model"]
    else:
        model = _T1_MODEL[0]
    op("model.unload", model=model)
    time.sleep(5)
    free1, resident1 = vram_free_mb()
    assert resident1 in (None, [], {}), f"卸载后仍有常驻: {resident1}"
    assert free1 is None or free0 is None or free1 >= free0 - 200, \
        f"VRAM 未回收: {free0} -> {free1} MB"
    print(f"  vram free {free0} -> {free1} MB, resident={resident1}",
          flush=True)
    # 编排前提验证：flux 卸载后 SDXL 可正常加载采样（错峰共存验收）
    mm = op("sd.checkpoint.load", ckpt=SDXL_CKPT, offload="model",
            dtype="fp16")
    lat = op("sd.latent.empty", width=512, height=512, batch_size=1)
    pos = op("sd.clip.encode", clip=mm["clip"], text="a red apple",
             width=512, height=512)
    neg = op("sd.clip.encode", clip=mm["clip"], text="",
             width=512, height=512)
    lat2 = op("sd.sample", model=mm["model"], pos=pos["cond"], neg=neg["cond"],
              latent=lat["latent"], seed=1, steps=4, cfg=5.0)
    img = op("sd.vae.decode", vae=mm["vae"], latent=lat2["latent"])
    png = fetch_image(img["image"])
    assert len(png) > 30000, "SDXL 切换后出图异常"
    op("model.unload", model=mm["model"])
    print(f"  flux->SDXL 错峰切换出图 OK ({len(png)}B)，SDXL 已卸载",
          flush=True)


def t3():
    """SD1.5 回归：sha1 必须等于 M2 基线 48e3225e8b38（验证全局
    RMSNorm/SDPA shim 补丁零影响——shim 只剥离 enable_gqa=False，
    SD1.5/SDXL 注意力路径不传该参数，此处实证）。"""
    mm = op("sd.checkpoint.load", ckpt=SD15_CKPT)
    lat = op("sd.latent.empty", width=512, height=512, batch_size=1)
    pos = op("sd.clip.encode", clip=mm["clip"],
             text="a cat sitting on a windowsill, warm sunset light, "
                  "high quality")
    neg = op("sd.clip.encode", clip=mm["clip"], text="")  # M2 基线用空负向
    lat2 = op("sd.sample", model=mm["model"], pos=pos["cond"], neg=neg["cond"],
              latent=lat["latent"], seed=424242, steps=20, cfg=7.0)
    img = op("sd.vae.decode", vae=mm["vae"], latent=lat2["latent"])
    png = fetch_image(img["image"])
    op("model.unload", model=mm["model"])
    sha = hashlib.sha1(png).hexdigest()[:12]
    open(os.path.join(OUT_DIR, "t3_sd15_regression.png"), "wb").write(png)
    assert sha == "48e3225e8b38", \
        f"SD1.5 回归漂移: sha1={sha} != 48e3225e8b38"
    print(f"  SD1.5 sha1={sha} ✓ 与 M2 基线逐字节一致", flush=True)


def t4():
    mm = op("sd.checkpoint.load", ckpt=SD15_CKPT, dtype="fp16")
    try:
        _, err = run_job("flux.encode", model=mm["model"], text="hi")
        assert err, "SD1.5 管道走 flux.encode 居然成功？"
        assert "flux" in str(err).lower() or "Flux" in str(err), \
            f"错误信息不含 flux 提示: {err}"
        print(f"  flux.encode(SD1.5) 正确拒绝: {str(err)[:80]}", flush=True)
        _, err = run_job("flux.sample", model=mm["model"],
                         cond={"prompt_embeds": "x", "pooled": "y"})
        assert err, "SD1.5 管道走 flux.sample 居然成功？"
        print(f"  flux.sample(SD1.5) 正确拒绝: {str(err)[:80]}", flush=True)
    finally:
        op("model.unload", model=mm["model"])


def main():
    check("t0", "预检：健康/算子注册", t0)
    check("t1", f"端到端：招牌 case {SIZE}px/{STEPS}步出图", t1_wrap)
    check("t2", "资源回收 + flux→SDXL 错峰切换", t2)
    check("t3", "回归：SD1.5 sha1=48e3225e8b38", t3)
    check("t4", "负面：非 FLUX 管道拒绝", t4)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("ALL FLUX ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()

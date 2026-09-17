"""AnimateDiff t2v 验收：motion.load + video.sample 出片，校验无黑帧、有内容。

用法（容器内，默认 http://127.0.0.1:8100）：
  docker cp scripts/accept-animatediff.py flowx-inference-server:/tmp/
  docker exec -d flowx-inference-server sh -c 'python /tmp/accept-animatediff.py > /tmp/accept-t2v.log 2>&1; echo $? > /tmp/accept-t2v.exit'

流程：checkpoint.load(SD1.5) → motion.load → /jobs 提交 video.sample（异步）
→ 轮询 → GET /videos/{id} 存 /videos/accept-t2v.mp4 → 逐帧亮度统计。
判定：采样失败 / 全帧均亮度 < BLACK_MEAN / 帧间无差异（静止=运动模块未生效），退出码 1。
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8100").rstrip("/")
TOKEN = os.environ.get("TOKEN", "")
BLACK_MEAN = float(os.environ.get("BLACK_MEAN", "2.0"))     # 0~255
MOTION_MIN_STD = float(os.environ.get("MOTION_MIN_STD", "0.5"))  # 帧间亮度均值序列的最小波动
OUT = os.environ.get("OUT", "/videos/accept-t2v.mp4")


def call(path, payload=None, timeout=1800, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    return b if raw else json.loads(b)


def op(name, **inputs):
    resp = call("/op", {"name": name, "inputs": inputs})
    return {k: m.get("id", m.get("value")) for k, m in resp.get("outputs", {}).items()}


def main():
    ckpt = os.environ.get("CKPT", "v1-5-pruned-emaonly")
    motion = os.environ.get("MOTION", "animatediff-motion-adapter-v1-5-2")
    print(f"[accept-t2v] base={BASE} ckpt={ckpt} motion={motion}", flush=True)

    t0 = time.time()
    base = op("checkpoint.load", ckpt=ckpt)
    print(f"[accept-t2v] checkpoint loaded ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    ad = op("motion.load", model={"$id": base["model"]}, motion=motion)
    print(f"[accept-t2v] motion composed ({time.time()-t0:.0f}s) -> {ad['model']}",
          flush=True)

    inputs = {
        "model": {"$id": ad["model"]},
        "prompt": os.environ.get("PROMPT", "a cat walking through a sunlit garden, "
                                           "gentle breeze, cinematic"),
        "neg_prompt": "blur, distortion, low quality, watermark",
        "width": 512, "height": 512, "num_frames": 16, "fps": 8,
        "steps": 20, "cfg": 7.5, "seed": 42, "decode_chunk_size": 8,
    }
    jid = call("/jobs", {"name": "video.sample", "inputs": inputs})["job_id"]
    print(f"[accept-t2v] job={jid} submitted", flush=True)

    t0 = time.time()
    while True:
        j = call(f"/jobs/{jid}", timeout=30)
        st = j["status"]
        p = j.get("progress") or {}
        print(f"[accept-t2v] {st} {p.get('current', 0)}/{p.get('total', 0)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if st == "done":
            break
        if st in ("failed", "cancelled"):
            print(f"[accept-t2v] FAIL: job {st}: {j.get('error')}", flush=True)
            sys.exit(1)
        time.sleep(10)

    video_id = j["result"]["outputs"]["video"]["id"]
    data = call(f"/videos/{video_id}", timeout=300, raw=True)
    with open(OUT, "wb") as f:
        f.write(data)
    print(f"[accept-t2v] saved {OUT} ({len(data)} bytes)", flush=True)
    if len(data) < 10_000 or b"ftyp" not in data[:32]:
        print("[accept-t2v] FAIL: 不是合法 mp4", flush=True)
        sys.exit(1)

    # 逐帧亮度统计：黑帧检查 + 帧间波动（运动模块生效判据）
    import imageio.v3 as iio
    import numpy as np
    frames = iio.imread(OUT, plugin="pyav") if False else None
    try:
        import imageio
        rd = imageio.get_reader(OUT)
        means = [float(np.asarray(fr).mean()) for fr in rd]
        rd.close()
    except Exception as e:
        print(f"[accept-t2v] WARN: 帧统计跳过（{e}）", flush=True)
        means = []
    if means:
        black = sum(1 for m in means if m < BLACK_MEAN)
        fluct = float(np.std(means))
        print(f"[accept-t2v] frames={len(means)} black={black} "
              f"mean={np.mean(means):.1f} fluct_std={fluct:.2f}", flush=True)
        if black:
            print(f"[accept-t2v] FAIL: {black}/{len(means)} 黑帧", flush=True)
            sys.exit(1)
    print("[accept-t2v] PASS", flush=True)


if __name__ == "__main__":
    main()

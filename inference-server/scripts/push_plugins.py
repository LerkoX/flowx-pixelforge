#!/usr/bin/env python3
"""push_plugins.py：把节点包的 server_op.py 批量上传到推理服务（预热/验收前置）。

算子全面插件化后，服务端启动时注册表为空；节点运行时会经 ensure_plugin
自动上传（hash 幂等），但直接调 /op、/jobs 的验收脚本（accept-*.py）不走节点，
需先用本脚本预热。PLUGINS_DIR 持久化，同一服务端只需推一次（内容变化自动覆盖）。

用法：
  python3 scripts/push_plugins.py --base http://127.0.0.1:8100 --token $INFERENCE_TOKEN
  python3 scripts/push_plugins.py --base ... --token ... --nodes ksampler vae-decode
  python3 scripts/push_plugins.py --base ... --token ... --list   # 只看会推什么

默认推送 nodes/ 下所有含 server_op.py 的节点包（重复内容自动去重——
flux 家族 9 节点共享同一文件，只推一次）。
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # flowx-pixelforge/
NODES = ROOT / "nodes"


def post(base, path, payload, tok):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {tok}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {path} -> HTTP {e.code}: "
                           f"{e.read().decode()[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="推理服务地址，如 http://127.0.0.1:8100")
    ap.add_argument("--token", default=os.environ.get("INFERENCE_TOKEN", ""))
    ap.add_argument("--nodes", nargs="*", default=None,
                    help="只推指定节点目录名；缺省推全部含 server_op.py 的节点")
    ap.add_argument("--list", action="store_true", help="只列出将推送的内容")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # 按内容去重：同一文件被多节点共享时只推一次（flux/video 家族模式）
    seen = {}
    for d in sorted(NODES.iterdir()):
        f = d / "server_op.py"
        if not f.is_file():
            continue
        if args.nodes and d.name not in args.nodes:
            continue
        content = f.read_bytes()
        h = hashlib.sha256(content).hexdigest()
        if h in seen:
            print(f"skip {d.name:<24}（内容与 {seen[h]} 相同）")
            continue
        seen[h] = d.name
        if args.list:
            print(f"push {d.name:<24} sha256={h[:12]}")
            continue
        # 文件名按算子族归一：取该文件注册的第一个算子名（与 ensure_plugin 同约定）
        import re
        m = re.search(rb'registry\.register\(\s*["\']([\w.]+)["\']', content)
        fname = (m.group(1).decode() if m else d.name).replace(".", "_") + ".py"
        resp = post(base, "/admin/plugins",
                    {"filename": fname, "content": content.decode(),
                     "sha256": h}, args.token)
        print(f"push {d.name:<24} -> {resp['plugin']} ops={resp['ops']}")

    if not args.list:
        print("\n完成。GET /ops 可确认算子已注册。", file=sys.stderr)


if __name__ == "__main__":
    main()

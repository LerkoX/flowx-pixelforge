#!/usr/bin/env python3
"""purge_legacy_plugins.py：清理服务端 plugins.d 里的旧插件文件（算子全面插件化迁移配套）。

背景：镜像内置插件（cond_latent_ops / preprocess_ops / ipadapter_ops /
instantid_ops / upscale_ops / flux_ops）已迁移为节点包 server_op.py 自注册。
已部署服务端 PLUGINS_DIR 里的旧文件若残留：① 算子双源，重启按文件名字典序
扫描归属漂移；② flux_ops.py 是旧契约（flux.encode 收 model），会遮蔽节点版
新契约（收 clip）。本脚本列出并删除这些旧文件。

用法：
  python3 scripts/purge_legacy_plugins.py --base http://127.0.0.1:8100 --token $TOKEN
  python3 scripts/purge_legacy_plugins.py --base ... --token ... --yes   # 直接删
"""
import argparse
import json
import os
import urllib.request

# 迁移前部署到 plugins.d 的旧插件文件名（按历史部署脚本/DEPLOY 记录）
LEGACY = ["cond_latent_ops.py", "preprocess_ops.py", "ipadapter_ops.py",
          "instantid_ops.py", "upscale_ops.py", "flux_ops.py"]


def req(base, path, tok, method="GET"):
    r = urllib.request.Request(base + path, method=method)
    r.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--token", default=os.environ.get("INFERENCE_TOKEN", ""))
    ap.add_argument("--yes", action="store_true", help="直接删除，不询问")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    code, data = req(base, "/admin/plugins", args.token)
    if code != 200:
        raise SystemExit(f"GET /admin/plugins -> {code}: {data}")
    present = [f for f in data.get("plugins", {}) if f in LEGACY]
    if not present:
        print("无旧插件文件残留，无需清理。")
        return
    print("发现旧插件文件：")
    for f in present:
        ops = data["plugins"][f]["ops"]
        print(f"  {f}  注册算子: {', '.join(ops)}")
    if not args.yes:
        ans = input("\n删除以上文件？（对应算子将由节点包 server_op.py 重新自注册）[y/N] ")
        if ans.strip().lower() != "y":
            print("已取消。")
            return
    for f in present:
        code, d = req(base, f"/admin/plugins/{f}", args.token, method="DELETE")
        print(f"  DELETE {f} -> {code} unregistered={d.get('unregistered')}")
    print("\n完成。节点下次运行会自动重传新实现；或主动跑 "
          "scripts/push_plugins.py 预热。")


if __name__ == "__main__":
    main()

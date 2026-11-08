#!/usr/bin/env bash
# 构建并推送两个栈的镜像到 Docker Hub，并写出可锁定的 IMAGES.lock
#
# 用法：
#   DOCKERHUB_USER=xxx DOCKERHUB_TOKEN=xxx bash scripts/push-images.sh [--stack M|P] [--no-build]
#
# 凭据从环境变量读取，**不落盘、不写仓库**。推荐用 Docker Hub Access Token
# （Account Settings → Security → New Access Token，Read/Write 权限），别用登录密码。
#
# 推送后脚本把 「tag → digest」写入 IMAGES.lock，install.sh 可据此精确锁定版本。
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${REPO:-lerkobba/flowx-inference}
LOCK="$HERE/IMAGES.lock"
NO_BUILD=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) NO_BUILD=1; shift ;;
    --stack) ONLY="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

say() { printf '\033[32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31m[错误]\033[0m %s\n' "$*" >&2; exit 1; }

[ -n "${DOCKERHUB_USER:-}" ] || die "未设置 DOCKERHUB_USER"
[ -n "${DOCKERHUB_TOKEN:-}" ] || die "未设置 DOCKERHUB_TOKEN（建议用 Access Token，不要用密码）"

if ! printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin >/dev/null 2>&1; then
  # 无外网时 CLI 登录会失败（认证走 CLI），但 push 实际由 daemon 执行；
  # 此时回退到已有的 ~/.docker/config.json 凭据继续。
  if [ -f "$HOME/.docker/config.json" ]; then
    say "docker login 失败（CLI 侧网络受限），沿用已有 ~/.docker/config.json 凭据继续"
  else
    die "docker login 失败且没有本地凭据"
  fi
fi
say "登录用户 ${DOCKERHUB_USER}"

[ "$NO_BUILD" = 1 ] || bash "$HERE/scripts/build-images.sh" ${ONLY:+--stack "$ONLY"}

TAGS=()
for st in ${ONLY:-P M}; do
  case "$st" in
    M) TAGS+=("m-2.13.0-cu132" "modern" "latest") ;;
    P) TAGS+=("p-2.3.1-cu121" "pascal") ;;
  esac
done

: > "$LOCK"
{
  echo "# 镜像版本锁定（由 scripts/push-images.sh 生成，勿手工编辑）"
  echo "# 生成时间: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# 用法: install.sh 可指定 --image ${REPO}@sha256:... 精确复现"
} >> "$LOCK"

for t in "${TAGS[@]}"; do
  say "推送 ${REPO}:${t}"
  out=$(docker push "${REPO}:${t}" 2>&1) || { printf '%s\n' "$out" | tail -5; die "推送 ${t} 失败"; }
  digest=$(printf '%s\n' "$out" | grep -o 'digest: sha256:[a-f0-9]*' | head -1 | cut -d' ' -f2)
  printf '%s\n' "$out" | grep -E "Pushed|Mounted" | tail -2 | sed 's/^/    /'
  echo "${t} ${digest}" >> "$LOCK"
  say "  → ${digest}"
done

say "完成。已写入 $LOCK"
cat "$LOCK"

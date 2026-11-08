#!/usr/bin/env bash
# 构建两个栈的镜像（P = 10 系/V100 旧栈，M = 20/30/40/50 系现代栈）
#
# 用法：
#   bash scripts/build-images.sh                 # 自动：本地有基底镜像就快速重打包，否则全量构建
#   bash scripts/build-images.sh --full          # 强制全量（走完整 Dockerfile，慢但自包含）
#   bash scripts/build-images.sh --stack M       # 只构建某一栈
#   DOCKER_HOST=tcp://... bash scripts/build-images.sh
#
# 快速重打包（默认）的动机：P 栈的依赖层（torch 2.3.1 + 全部算子依赖，约 8GB）
# 与代码无关，重装一次要十几分钟并多占 8GB 磁盘。若本地已有同栈基底镜像，
# 只需 COPY 当前 app/ 覆盖即可得到「同一依赖栈 + 最新代码」的镜像（秒级、一层）。
# 基底镜像不存在时自动回退到全量构建。
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${REPO:-lerkobba/flowx-inference}
FULL=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --stack) ONLY="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

say() { printf '\033[32m==>\033[0m %s\n' "$*"; }

# 栈定义（与 install.sh / Dockerfile 保持一致）
stack_image() {
  case "$1" in
    M) echo "$REPO:m-2.13.0-cu132" ;;
    P) echo "$REPO:p-2.3.1-cu121" ;;
    *) echo "未知栈 $1" >&2; return 1 ;;
  esac
}
stack_dockerfile() { [ "$1" = "M" ] && echo "Dockerfile.modern" || echo "Dockerfile.pascal"; }

# P 栈的「快速重打包基底」候选（按优先级）：已有 p- 镜像 → 现役生产镜像（取版本号最大的）
p_basis() {
  if docker image inspect "$REPO:p-2.3.1-cu121" >/dev/null 2>&1; then
    echo "$REPO:p-2.3.1-cu121"; return 0
  fi
  local prod
  prod=$(docker images --format '{{.Repository}}:{{.Tag}}' \
         | grep -E '^flowx-inference-server:pascal' | sort -V | tail -1)
  [ -n "$prod" ] && { echo "$prod"; return 0; }
  return 1
}

build_stack() {
  local st=$1 img df
  img=$(stack_image "$st"); df=$(stack_dockerfile "$st")

  if [ "$FULL" = 0 ]; then
    local basis=""
    if [ "$st" = "P" ]; then basis=$(p_basis || true); fi
    if [ -n "$basis" ]; then
      say "快速重打包 $img（FROM $basis + COPY app/，仅一层）"
      local tmpdf; tmpdf=$(mktemp)
      cat > "$tmpdf" <<EOF
# 自动生成（scripts/build-images.sh --fast）：复用 $basis 的完整依赖层，
# 只替换应用代码。需要从零构建请加 --full。
FROM $basis
WORKDIR /app
COPY app/ ./app/
EOF
      docker build -f "$tmpdf" -t "$img" "$HERE"
      rm -f "$tmpdf"
      return 0
    fi
  fi

  say "全量构建 $img（$df）"
  [ -f "$HERE/$df" ] || { echo "缺少 $HERE/$df" >&2; return 1; }
  docker build -f "$HERE/$df" -t "$img" "$HERE"
}

for st in ${ONLY:-P M}; do
  build_stack "$st"
done

say "构建完成："
docker images --format '  {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep "^  $REPO" || true
echo
echo "下一步：bash scripts/push-images.sh   （或 bash install.sh 直接部署）"

#!/usr/bin/env bash
# 部署后健康验收：轮询 /health 并断言「装对了栈」
#
# 为什么必须验收而不是「起完就算」：装错栈的典型症状是容器 running 但第一次
# 推理才炸（no kernel image / 驱动不足）。这里在安装阶段就把它们挡下来。
#
# 用法：
#   bash scripts/verify.sh --url http://127.0.0.1:8100 \
#        --expect-torch 2.13.0 --expect-stack M \
#        [--container flowx-inference-server] [--docker-host tcp://...] [--timeout 180]
#
# 退出码：0=通过，1=超时/断言失败
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -uo pipefail

URL="http://127.0.0.1:8100"
TOKEN=""
EXPECT_TORCH=""
EXPECT_STACK=""
CONTAINER=""
DOCKER_HOST_ARG=""
TIMEOUT=180

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --expect-torch) EXPECT_TORCH="$2"; shift 2 ;;
    --expect-stack) EXPECT_STACK="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --python)    PYTHON="$2"; shift 2 ;;
    --docker-host) DOCKER_HOST_ARG="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '    %s\n' "$*"; }
good() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }

dump_logs() {
  [ -n "$CONTAINER" ] || return 0
  local d=(); [ -n "$DOCKER_HOST_ARG" ] && d=(-H "$DOCKER_HOST_ARG")
  echo "  ---- 容器日志（尾部 40 行）----"
  docker "${d[@]}" logs --tail 40 "$CONTAINER" 2>&1 | sed 's/^/  /'
}

json() { printf '%s' "$1" | grep -o "\"$2\":\"[^\"]*\"" | head -1 | cut -d'"' -f4; }
jsonnum() { printf '%s' "$1" | grep -o "\"$2\":[0-9]*" | head -1 | cut -d: -f2; }

# ── 架构兼容性硬校验 ──────────────────────────────────────────────
# 为什么需要：/health 只反映「服务进程活着 + torch 能初始化 CUDA」，
# **架构不匹配时它照样报 ok**（实测：modern 镜像跑在 sm_61 的 GTX 1080 上，
# /health = status:ok + cuda_available:true，但一算 matmul 就
# `no kernel image is available for execution on the device`）。
# 这里用 get_device_capability 与 get_arch_list 做纯本地比对（不需要下载模型）：
#   同 major 且 arch 的 minor <= 卡的 minor ⇒ 可跑（如 sm_86 cubin 跑 sm_89 的 40 系）
arch_check() {
  local pycheck out rc=0
  pycheck='import sys, torch
if not torch.cuda.is_available():
    print("SKIP:容器内 CUDA 不可用"); sys.exit(3)
maj, mnr = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
target = "%d%d" % (maj, mnr)
ok = any(a.startswith("sm_%d" % maj) and int(a[3:]) <= int(target) for a in arches)
print("card=sm_%s  arches=%s" % (target, ",".join(arches)))
sys.exit(0 if ok else 4)'

  if [ -n "$CONTAINER" ]; then
    local -a d=()
    if [ -n "$DOCKER_HOST_ARG" ]; then d=(-H "$DOCKER_HOST_ARG"); fi
    out=$(printf '%s' "$pycheck" | docker "${d[@]}" exec -i "$CONTAINER" python -W ignore - 2>&1) || rc=$?
  elif [ -n "$PYTHON" ]; then
    out=$(printf '%s' "$pycheck" | "$PYTHON" -W ignore - 2>&1) || rc=$?
  else
    return 0
  fi

  # torch 在栈不匹配时会往 stdout/stderr 喷大段 UserWarning，只保留结果行
  local clean
  clean=$(printf '%s\n' "$out" | grep -E "^(card=|SKIP)" | head -1)
  if [ -n "$clean" ]; then out="$clean"; fi

  case "$rc" in
    0) say "  ✓ 架构兼容  $out" ;;
    3) warn "跳过架构校验（CUDA 不可用）" ;;
    4)
      echo
      bad "栈与显卡不匹配！"
      echo "  镜像内 kernel 架构：$out"
      echo "  服务能起来、/health 也报 ok，但**一推理就崩**（no kernel image is available）。"
      echo "  处理：重跑 install.sh（会按探测结果自动换栈），或检查 --image/--stack 是否强制指定错了。"
      exit 1 ;;
    *) warn "架构校验未完成（${out:0:300}）" ;;
  esac
}

echo "==> 健康检查 $URL/health（最长 ${TIMEOUT}s）"
deadline=$((SECONDS + TIMEOUT))
body=""
# 注意：不能写 ${TOKEN:+-H "Authorization: Bearer $TOKEN"} —— 扩展结果里的引号
# 不会被重新解释，curl 会收到拆开的参数。必须用数组。
hdr=()
if [ -n "$TOKEN" ]; then hdr=(-H "Authorization: Bearer $TOKEN"); fi
while [ $SECONDS -lt $deadline ]; do
  body=$(curl -fsS -m 5 "${hdr[@]}" "$URL/health" 2>/dev/null) && break
  sleep 3
done
if [ -z "$body" ]; then
  bad "服务在 ${TIMEOUT}s 内没有响应 /health"
  dump_logs
  exit 1
fi

status=$(json "$body" status)
torch_v=$(json "$body" torch_version)
diff_v=$(json "$body" diffusers_version)
gpu=$(json "$body" gpu_name)
cc=$(json "$body" compute_capability)
cuda_avail=$(printf '%s' "$body" | grep -o '"cuda_available":[a-z]*' | head -1 | cut -d: -f2)

fail=0
[ "$status" = "ok" ]        && good "status=ok"            || { bad "status=${status:-<空>}"; fail=1; }
[ "$cuda_avail" = "true" ]  && good "cuda_available=true"  || { bad "cuda_available=${cuda_avail:-<空>}（GPU 没进容器？）"; fail=1; }
say "GPU=$gpu  torch=$torch_v  diffusers=$diff_v  ${cc:+compute_capability=$cc}"

if [ -n "$EXPECT_TORCH" ]; then
  case "$torch_v" in
    "$EXPECT_TORCH"*) good "torch 版本符合预期 $EXPECT_TORCH（实际 $torch_v）" ;;
    *) bad "torch 版本不符：期望 $EXPECT_TORCH.*，实际 ${torch_v:-<空>}（装错栈）"; fail=1 ;;
  esac
fi

# 栈与 torch 大版本的一致性交叉校验
if [ -n "$EXPECT_STACK" ]; then
  want_prefix=""
  case "$EXPECT_STACK" in
    M) want_prefix="2.1" ;;   # modern = torch 2.1x
    P|P-lite) want_prefix="2.3" ;;
  esac
  case "$torch_v" in
    "$want_prefix"*) good "栈 $EXPECT_STACK 与 torch $torch_v 一致" ;;
    "") ;;
    *) bad "栈 $EXPECT_STACK 期望 torch $want_prefix.*，实际 $torch_v"; fail=1 ;;
  esac
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "  常见原因："
  echo "    · 'no kernel image is available' → 栈选错（Pascal 用了 modern / Blackwell 用了旧栈）"
  echo "    · 'CUDA driver version is insufficient' → 宿主机驱动太老，需升级到对应栈的最低版本"
  echo "    · /health 里 cuda_available=false → nvidia-container-toolkit 未装或未重启 docker"
  dump_logs
  exit 1
fi

arch_check
echo "  ✅ 验收通过"
exit 0

#!/usr/bin/env bash
# 显卡/驱动探测与「栈」决策（纯只读，零副作用）
#
# 输出：人可读摘要到 stderr，`KEY=VALUE` 到 stdout（供 install.sh 解析）；
#       --json 时 stdout 只有一行 JSON。
#
# 决策表（依据 NVIDIA 官方 release notes + PyPI/官方 wheel 索引实测，见
# DEPLOY.md「多代显卡一键安装」与 UPGRADE-MULTI-ARCH.md）：
#
#   栈      目标卡                        算力          最低驱动   Python  torch    CUDA/cuDNN
#   M       20/30/40/50 系、T4、A100…     sm_75~sm_120  >=580      3.12    2.13.0   13.2 / 9
#   P       10 系 Pascal、V100（旧栈）    sm_60/61/70   >=530      3.10    2.3.1    12.1 / 8
#   P-lite  同上但驱动 450~529            sm_60/61/70   >=450      3.10    2.3.1    11.8 / 8
#   X       不支持（Maxwell 及更老 / 50 系驱动不足 / 驱动过老）
#
# 为什么 10 系不能升到 cu13.2：
#   CUDA 13.x release notes "Dropped support for pre-Turing architectures
#   (Maxwell, Volta, and Pascal)" —— sm_60/61/70 在 CUDA 13 上无 kernel。
#
# 用法：
#   bash scripts/detect.sh                 # 本机 nvidia-smi 探测
#   bash scripts/detect.sh --json          # 机器可读
#   bash scripts/detect.sh --docker-host tcp://192.168.5.60:2375
#                                          # 探测该 daemon 宿主机上的 GPU
#   bash scripts/detect.sh --probe-image flowx-inference-server:pascal-partA13
#                                          # 远端探测用的镜像（默认拉 nvidia/cuda:12.1.0-base-ubuntu22.04）
#   bash scripts/detect.sh --gpus 0        # 多卡时只看指定卡（默认取算力最低的那张）
#
# 退出码：0=可用（STACK=M/P/P-lite），2=不支持（STACK=X），1=探测失败/环境错误

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

DOCKER_HOST_ARG=""
JSON=0
PROBE_IMAGE="nvidia/cuda:12.1.0-base-ubuntu22.04"
GPUS_SEL=""
ALLOW_PULL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --docker-host) DOCKER_HOST_ARG="$2"; shift 2 ;;
    --probe-image) PROBE_IMAGE="$2"; shift 2 ;;
    --gpus) GPUS_SEL="$2"; shift 2 ;;
    --no-pull) ALLOW_PULL=0; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

log() { printf '%s\n' "$*" >&2; }

# ---------- 探测：本机 ----------
probe_local() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  # compute_cap 字段需要较新驱动；老驱动回退到「名称推断」
  local csv
  if csv=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
             --format=csv,noheader,nounits 2>/dev/null); then
    printf '%s\n' "$csv"
    return 0
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '{printf "%s,,%s,%s\n", $1, $2, $3}'
}

# ---------- 探测：通过 docker daemon（GPU 在 daemon 宿主机上） ----------
probe_docker() {
  local pull_opt=()
  [ "$ALLOW_PULL" = 1 ] || pull_opt=(--pull=never)
  local args=(run "${pull_opt[@]}" --rm --gpus all
              --entrypoint nvidia-smi "$PROBE_IMAGE"
              --query-gpu=name,compute_cap,memory.total,driver_version
              --format=csv,noheader,nounits)
  local d=()
  [ -n "$DOCKER_HOST_ARG" ] && d=(-H "$DOCKER_HOST_ARG")
  docker "${d[@]}" "${args[@]}" 2>/dev/null
}

# ---------- 型号 → 算力 推断（compute_cap 字段不可用时的兜底） ----------
cc_from_name() {
  local n
  n=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  case "$n" in
    # Blackwell（50 系 / B100-B200 / RTX PRO Blackwell）= sm_120
    *rtx*50*|*b100*|*b200*|*blackwell*) echo "12.0" ;;
    # Hopper = sm_90
    *h100*|*h200*|*h800*) echo "9.0" ;;
    # Ada（40 系 / L40 / L4 / RTX A6000 Ada）= sm_89
    *rtx*40*|*l40*|*l4*|*ada*|*rtx*6000*|*rtx*a*) echo "8.9" ;;
    # Ampere（30 系 / A100 / A40 / A10 / A5000-A2000）= sm_80/86
    *a100*|*a30*|*a40*|*a10*|*a5000*|*a4000*|*a3000*|*a2000*|*rtx*30*) echo "8.6" ;;
    # Turing（20 系 / GTX16xx / T4）= sm_75
    *rtx*20*|*gtx*16*|*t4*|*quadro*t*) echo "7.5" ;;
    # Volta = sm_70
    *v100*|*v100*) echo "7.0" ;;
    # Pascal（10 系 / P4 / P40 / P100 / MX）= sm_61
    *gtx*10*|*p4*|*p40*|*p100*|*mx1*|*mx2*|*mx3*) echo "6.1" ;;
    # Maxwell 及更老
    *gtx*9*|*gtx*7*|*gtx*6*|*k80*|*k40*|*m40*|*m60*) echo "5.2" ;;
    *) echo "" ;;
  esac
}

# ---------- 栈决策 ----------
# 入参：cc_major cc_minor driver_major  → 输出 "STACK<TAB>REASON"
decide() {
  local ccm=$1 ccn=$2 drv=$3
  # 50 系及更新（sm_120+）必须 modern
  if [ "$ccm" -ge 12 ]; then
    if [ "$drv" -ge 580 ]; then printf 'M\tBlackwell(sm_%s%s) 仅 CUDA 13.x 支持，驱动 %s >= 580\n' "$ccm" "$ccn" "$drv"
    else printf 'X\t50 系(sm_%s%s) 必须 CUDA 13.x（驱动 >= 580），当前驱动 %s 太老——请升级驱动\n' "$ccm" "$ccn" "$drv"; fi
    return
  fi
  # Maxwell 及更老：两栈都不支持
  if [ "$ccm" -lt 6 ]; then
    printf 'X\t算力 sm_%s%s（Maxwell 及更老）不受支持：CUDA 13 已移除 sm<75，旧栈要求 >= sm_60\n' "$ccm" "$ccn"; return
  fi
  # Volta（sm_70）及更老的 Turing 之前架构（Pascal sm_60/61）：CUDA 13 已全部移除 → 只能旧栈
  if [ "$ccm" -lt 7 ] || { [ "$ccm" -eq 7 ] && [ "$ccn" -lt 5 ]; }; then
    if [ "$drv" -ge 530 ]; then printf 'P\t算力 sm_%s%s 早于 Turing，CUDA 13.x 已移除该架构支持，走旧栈 cu121（驱动 %s >= 530）\n' "$ccm" "$ccn" "$drv"
    elif [ "$drv" -ge 450 ]; then printf 'P-lite\t算力 sm_%s%s 早于 Turing（CUDA 13 已移除）且驱动 %s < 530，走 cu118 变体\n' "$ccm" "$ccn" "$drv"
    else printf 'X\t算力 sm_%s%s 早于 Turing，且驱动 %s < 450，无法运行\n' "$ccm" "$ccn" "$drv"; fi
    return
  fi
  # sm_75+：优先 modern
  if [ "$drv" -ge 580 ]; then
    # 注意措辞：镜像的 arch 列表只到 sm_86/sm_120，没有 sm_89，
    # 4090 靠 sm_86 cubin 兼容运行 ⇒ 只有 bf16，拿不到 fp8。别写「fp8 可用」误导用户。
    if [ "$ccm" -eq 8 ] && [ "$ccn" -eq 9 ]; then
      printf 'M\tsm_%s%s >= sm_75 且驱动 %s >= 580，走 CUDA 13.2 栈（bf16 可用；fp8 需原生 sm_89 kernel，本镜像无，不启用）\n' "$ccm" "$ccn" "$drv"
    else
      printf 'M\tsm_%s%s >= sm_75 且驱动 %s >= 580，可吃 CUDA 13.2 栈（bf16%s）\n' "$ccm" "$ccn" "$drv" \
        "$([ "$ccm" -ge 9 ] && echo '/fp8 可用' || echo '，fp8 需 sm_89+')"
    fi
  elif [ "$drv" -ge 530 ]; then
    printf 'P\tsm_%s%s 支持 modern，但驱动 %s < 580 上不了 CUDA 13——降级旧栈 cu121（无 bf16/fp8）；升级驱动后重跑可得 modern\n' "$ccm" "$ccn" "$drv"
  elif [ "$drv" -ge 450 ]; then
    printf 'P-lite\tsm_%s%s 且驱动 %s < 530，走 cu118 变体（无 bf16/fp8）\n' "$ccm" "$ccn" "$drv"
  else
    printf 'X\t驱动 %s < 450，无法运行任何栈\n' "$drv"
  fi
}

# ---------- 主流程 ----------
RAW=""
if [ -n "$DOCKER_HOST_ARG" ]; then
  log "[探测] 通过 docker daemon ${DOCKER_HOST_ARG} 读取 GPU 信息"
  RAW=$(probe_docker) || {
    log "[错误] 无法通过 docker 探测 GPU：请确认 daemon 可达、nvidia-container-toolkit 已装、镜像 ${PROBE_IMAGE} 可拉取"
    log "       可用 --probe-image <已有镜像> 避免重新拉取（如 flowx-inference-server:pascal-partA13）"
    exit 1
  }
else
  log "[探测] 本机 nvidia-smi"
  RAW=$(probe_local) || {
    log "[错误] 本机没有可用的 nvidia-smi。若 GPU 在别的机器上，用 --docker-host tcp://<host>:2375"
    exit 1
  }
fi

# 解析多卡：取算力最低的那张定栈（保守）
BEST=""; BEST_CCM=99; BEST_CCN=99; BEST_MEM=""; COUNT=0; DRV=""; NAMES=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  IFS=',' read -r name cc mem drv <<<"$line"
  name=$(printf '%s' "$name" | sed 's/^ *//;s/ *$//')
  cc=$(printf '%s' "$cc" | tr -d ' ')
  mem=$(printf '%s' "$mem" | tr -d ' ')
  drv=$(printf '%s' "$drv" | tr -d ' ')
  [ -z "$cc" ] && cc=$(cc_from_name "$name")
  [ -z "$cc" ] && { log "[错误] 无法判定 $name 的算力，请升级驱动后重试（nvidia-smi 缺 compute_cap 字段）"; exit 1; }
  ccm=${cc%%.*}; ccn=${cc#*.}; [ "$ccn" = "$cc" ] && ccn=0
  COUNT=$((COUNT + 1))
  NAMES="${NAMES:+$NAMES|}${name}"
  [ -z "$DRV" ] && DRV=$drv
  if [ "$ccm" -lt "$BEST_CCM" ] || { [ "$ccm" -eq "$BEST_CCM" ] && [ "$ccn" -lt "$BEST_CCN" ]; }; then
    BEST=$name; BEST_CCM=$ccm; BEST_CCN=$ccn; BEST_MEM=$mem
  fi
done <<<"$RAW"

[ -z "$BEST" ] && { log "[错误] 未解析到任何 GPU"; exit 1; }

DRV_MAJOR=${DRV%%.*}
DEC=$(decide "$BEST_CCM" "$BEST_CCN" "$DRV_MAJOR")
STACK=${DEC%%$'\t'*}
REASON=${DEC#*$'\t'}

case "$STACK" in
  M)      PY="3.12"; TORCH="2.13.0"; CUDA="13.2"; CUDNN="9"; MIN_DRV="580"; IMAGE_TAG="m-2.13.0-cu132" ;;
  P)      PY="3.10"; TORCH="2.3.1";  CUDA="12.1"; CUDNN="8"; MIN_DRV="530"; IMAGE_TAG="p-2.3.1-cu121" ;;
  P-lite) PY="3.10"; TORCH="2.3.1";  CUDA="11.8"; CUDNN="8"; MIN_DRV="450"; IMAGE_TAG="p-2.3.1-cu118" ;;
  *)      PY="-";    TORCH="-";      CUDA="-";    CUDNN="-"; MIN_DRV="-";   IMAGE_TAG="-" ;;
esac

# WSL2 检出
WSL=0
{ [ -e /dev/dxg ] || grep -qi microsoft /proc/version 2>/dev/null; } && WSL=1

log ""
log "════════ 探测结果 ════════"
log "  GPU       ${NAMES}（共 ${COUNT} 张）"
log "  定栈依据  ${BEST}  sm_${BEST_CCM}${BEST_CCN}  ${BEST_MEM} MiB"
log "  驱动      ${DRV}（最低要求 ${MIN_DRV}）"
log "  环境      $( [ "$WSL" = 1 ] && echo "WSL2" || echo "Linux" )"
log "  ──────────────────────"
log "  栈        ${STACK}"
log "  说明      ${REASON}"
if [ "$STACK" != "X" ]; then
  log "  版本      Python ${PY} / torch ${TORCH} / CUDA ${CUDA} / cuDNN ${CUDNN}"
  log "  镜像      lerkobba/flowx-inference:${IMAGE_TAG}"
fi
log "══════════════════════════"

if [ "$JSON" = 1 ]; then
  cat <<EOF
{"gpu_count":${COUNT},"gpu_names":"${NAMES}","gpu_sm":"sm_${BEST_CCM}${BEST_CCN}","gpu_cc":"${BEST_CCM}.${BEST_CCN}","gpu_vram_mb":"${BEST_MEM:-0}","driver":"${DRV}","driver_major":"${DRV_MAJOR}","min_driver":"${MIN_DRV}","wsl":$([ "$WSL" = 1 ] && echo true || echo false),"stack":"${STACK}","reason":"${REASON}","python":"${PY}","torch":"${TORCH}","cuda":"${CUDA}","cudnn":"${CUDNN}","image_tag":"${IMAGE_TAG}"}
EOF
else
  cat <<EOF
GPU_COUNT=${COUNT}
GPU_NAMES=${NAMES}
GPU_SM=sm_${BEST_CCM}${BEST_CCN}
GPU_CC=${BEST_CCM}.${BEST_CCN}
GPU_VRAM_MB=${BEST_MEM:-0}
DRIVER=${DRV}
DRIVER_MAJOR=${DRV_MAJOR}
MIN_DRIVER=${MIN_DRV}
WSL=${WSL}
STACK=${STACK}
REASON=${REASON}
PYTHON=${PY}
TORCH=${TORCH}
CUDA=${CUDA}
CUDNN=${CUDNN}
IMAGE_TAG=${IMAGE_TAG}
EOF
fi

[ "$STACK" = "X" ] && exit 2
exit 0

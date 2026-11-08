#!/usr/bin/env bash
# flowx-pixelforge 推理服务 —— 多代显卡一键安装（Linux / WSL2）
#
# 支持显卡：10 系(Pascal) / 20 系(Turing) / 30 系(Ampere) / 40 系(Ada) / 50 系(Blackwell)
# 两种模式：docker（容器）/ native（直接装到本机，systemd 常驻）
#
# 为什么有「两个栈」：CUDA 13.x 的 release notes 明确移除 Maxwell/Pascal/Volta
# （"Dropped support for pre-Turing architectures"），因此 10 系与 V100 永远上不了
# cu13.2；而 50 系(sm_120)只能跑 CUDA 13。故 10 系走旧栈、20~50 系走新栈，
# 分档只认 compute capability + 驱动版本，不认型号名（GTX 16xx 是 sm_75 走新栈）。
#
# 用法：
#   bash install.sh --dry-run              # ★先跑这个：只打印判定与将执行的命令
#   bash install.sh                        # docker 模式（默认）
#   bash install.sh --mode native          # 裸机模式（micromamba + systemd）
#   bash install.sh --docker-host tcp://192.168.5.60:2375
#   bash install.sh --uninstall --keep-data
#
# 详细参数见 --help。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DETECT="$HERE/scripts/detect.sh"
VERIFY="$HERE/scripts/verify.sh"

# ---------- 默认值 ----------
MODE="docker"            # docker | native
DRY_RUN=0
ASSUME_YES=0
PORT=8100
TOKEN=""
DATA_DIR=""              # native 默认 /var/lib/flowx-inference
BIND_DIR=""              # docker 绑定挂载根目录（默认 $HERE）
MAX_RESIDENT=""
GPUS_SEL=""
DOCKER_HOST_ARG=""
IMAGE_OVERRIDE=""
FORCE_BUILD=0
NO_PULL=0
OFFLOAD=""
STACK_OVERRIDE=""
PREFIX="/opt/flowx-inference"
FORCE=0
UNINSTALL=0
KEEP_DATA=0
DEPS_ONLY=0              # 只装依赖，不起服务（无 GPU 的机器上预装用）
DOCKER_REPO="lerkobba/flowx-inference"
PIP_INDEX_TORCH=""

# ---------- 输出 ----------
C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_0=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$C_OK" "$*" "$C_0"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_WARN" "$C_0" "$*"; }
die()  { printf '%s[错误]%s %s\n' "$C_ERR" "$C_0" "$*" >&2; exit "${2:-1}"; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '%s[dry-run]%s %s\n' "$C_DIM" "$C_0" "$*"; else "$@"; fi; }
confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  [ "$DRY_RUN" = 1 ] && return 0
  printf '%s? %s [y/N] ' "$C_WARN" "$*$C_0"
  local a; read -r a || true
  case "$a" in y|Y|yes|YES) return 0 ;; *) die "已取消" 3 ;; esac
}

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; cat <<'EOF'

参数：
  --mode docker|native    安装形态（默认 docker）
  --dry-run               只打印决策与命令，不改动系统
  --yes                   跳过交互确认
  --port N                对外端口（默认 8100）
  --token T               推理服务令牌（默认随机生成）
  --stack M|P|P-lite      强制栈（默认按显卡自动判定）
  --data-dir DIR          native 数据目录（默认 /var/lib/flowx-inference）
  --bind-dir DIR          docker 绑定挂载根目录（默认脚本所在目录）
  --models-dir DIR        docker 模式：模型目录（默认 <bind-dir>/models）
  --max-resident N        常驻模型上限（默认按显存推算）
  --gpus N                只用指定 GPU（默认 all）
  --offload none|model|sequential  显存治理档位（默认按显存推算）
  --docker-host URL       远程 docker daemon（自动改用命名卷）
  --probe-image IMG       探卡用的镜像（默认 nvidia/cuda:12.1.0-base-ubuntu22.04）
  --image REPO:TAG        直接用指定镜像（跳过 pull/构建）
  --build                 强制本地构建镜像
  --no-pull               禁止 pull，只用本地镜像或本地构建
  --micromamba-url URL    native 模式：micromamba 下载地址
  --deps-only             只装依赖（micromamba + torch + 算子依赖），不起服务、不验收。
                          用于「先开无卡机器把依赖装好，等有卡再启动」；隐含 --mode native。
                          栈按 --stack 指定，未指定默认 M
  --uninstall             卸载（容器/服务与安装目录）
  --keep-data             卸载时保留模型数据
  --force                 允许覆盖同名容器/已有配置
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --stack) STACK_OVERRIDE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --bind-dir) BIND_DIR="$2"; shift 2 ;;
    --models-dir) MODELS_DIR_ARG="$2"; shift 2 ;;
    --max-resident) MAX_RESIDENT="$2"; shift 2 ;;
    --gpus) GPUS_SEL="$2"; shift 2 ;;
    --offload) OFFLOAD="$2"; shift 2 ;;
    --docker-host) DOCKER_HOST_ARG="$2"; shift 2 ;;
    --probe-image) PROBE_IMAGE="$2"; shift 2 ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    --build) FORCE_BUILD=1; shift ;;
    --no-pull) NO_PULL=1; shift ;;
    --micromamba-url) MICROMAMBA_URL="$2"; shift 2 ;;
    --deps-only) DEPS_ONLY=1; MODE=native; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --keep-data) KEEP_DATA=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1（--help 查看用法）" ;;
  esac
done

case "$MODE" in docker|native) ;; *) die "--mode 只能是 docker 或 native" ;; esac
MODELS_DIR_ARG=${MODELS_DIR_ARG:-}
# 下载抗断线：云厂商/国内机器直连 GitHub 经常传到一半 EOF（实测 19MB 断在 8.3MB）
CURL_RETRY='--retry 6 --retry-delay 3 --retry-all-errors --connect-timeout 20'

# 默认优先用 GitHub Releases 的**单文件静态二进制**（免解压，避免 minimal 系统缺 bzip2 的坑）；
# 若该地址不可达，自动回退到 micro.mamba.pm 的 .tar.bz2 包（会先装 bzip2）。
MICROMAMBA_URL=${MICROMAMBA_URL:-https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64}
PROBE_IMAGE=${PROBE_IMAGE:-}

# ---------- 通用前置 ----------
check_os() {
  # 远程 daemon 模式下，真正跑服务的是 daemon 宿主，架构要查 daemon 而不是本机
  # （本机 CLI 可能跑在 Termux/macOS 上做远程部署）
  if [ -n "$DOCKER_HOST_ARG" ]; then
    local arch
    arch=$(docker -H "$DOCKER_HOST_ARG" info --format '{{.Architecture}}' 2>/dev/null || true)
    case "$arch" in
      x86_64|amd64) info "daemon 架构 $arch（x86_64）" ;;
      "") warn "读不到 daemon 架构，跳过 x86_64 校验" ;;
      *) die "docker daemon 架构为 $arch，只支持 x86_64（ARM64/Jetson 不在支持范围）" ;;
    esac
    WSL=0
    return 0
  fi

  [ "$(uname -s)" = "Linux" ] || die "只支持 Linux / WSL2（当前 $(uname -s)）"
  case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die "只支持 x86_64（当前 $(uname -m)）。ARM64/Jetson 不在支持范围" ;;
  esac
  if grep -qi microsoft /proc/version 2>/dev/null || [ -e /dev/dxg ]; then
    WSL=1; info "检测到 WSL2"
  else
    WSL=0
  fi
}

# 解析 detect.sh 的 KEY=VALUE 输出到环境变量
load_detect() {
  local out
  local -a dargs=()
  if [ -n "$DOCKER_HOST_ARG" ]; then dargs+=(--docker-host "$DOCKER_HOST_ARG"); fi
  if [ -n "$GPUS_SEL" ]; then dargs+=(--gpus "$GPUS_SEL"); fi
  if [ -n "${PROBE_IMAGE:-}" ]; then dargs+=(--probe-image "$PROBE_IMAGE"); fi
  out=$(bash "$DETECT" "${dargs[@]}") || {
    local rc=$?
    [ "$rc" -eq 2 ] && die "当前显卡/驱动不在支持范围（详见上方输出）" 2
    die "显卡探测失败" 1
  }
  while IFS= read -r line; do
    case "$line" in
      *=*) export "${line%%=*}=${line#*=}" ;;
    esac
  done <<<"$out"
  if [ -n "$STACK_OVERRIDE" ]; then
    STACK="$STACK_OVERRIDE"
    warn "已按 --stack 强制覆盖判定：$STACK"
  fi
}

apply_stack_defaults() {
  case "$STACK" in
    M)      PY="3.12"; TORCH="2.13.0"; CUDA="13.2"; IMAGE_TAG="m-2.13.0-cu132"; TORCH_INDEX="cu132"; DOCKERFILE="Dockerfile.modern"; REQ="requirements-modern.txt"; CONTAINER="flowx-inference-server"; BASE_IMAGE="pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime" ;;
    P)      PY="3.10"; TORCH="2.3.1";  CUDA="12.1"; IMAGE_TAG="p-2.3.1-cu121";  TORCH_INDEX="cu121"; DOCKERFILE="Dockerfile.pascal"; REQ="requirements-pascal.txt"; CONTAINER="flowx-inference-server"; BASE_IMAGE="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime" ;;
    P-lite) PY="3.10"; TORCH="2.3.1";  CUDA="11.8"; IMAGE_TAG="p-2.3.1-cu118";  TORCH_INDEX="cu118"; DOCKERFILE="Dockerfile.pascal"; REQ="requirements-pascal.txt"; CONTAINER="flowx-inference-server"; BASE_IMAGE="pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime" ;;
    *) die "内部错误：未知栈 $STACK" ;;
  esac
  IMAGE="${IMAGE_OVERRIDE:-$DOCKER_REPO:$IMAGE_TAG}"
  # 容器名按栈区分：P 栈沿用现役名（不惊动生产），M 栈加 -modern 后缀，
  # 这样两栈可以同时存在于一台 daemon 上（不同端口）而互不覆盖。
  if [ "$STACK" = "M" ]; then CONTAINER="flowx-inference-server-modern"; fi
}

# 按显存推算默认值（与 DEPLOY.md 的显存预算式淘汰配合）
tune_by_vram() {
  local vram=${GPU_VRAM_MB:-0}
  if [ -z "$MAX_RESIDENT" ]; then
    if   [ "$vram" -lt 12288 ]; then MAX_RESIDENT=1
    elif [ "$vram" -lt 24576 ]; then MAX_RESIDENT=2
    else MAX_RESIDENT=3; fi
  fi
  if [ -z "$OFFLOAD" ]; then
    if [ "$vram" -lt 12288 ]; then OFFLOAD="model"; else OFFLOAD="none"; fi
  fi
  # 必须用 if：本行是函数最后一条语句，若写成 `[ -z "$TOKEN" ] && ...`，
  # 用户传了 --token 时整条返回 1，`set -e` 会让脚本静默退出。
  if [ -z "$TOKEN" ]; then
    TOKEN=$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 28)
  fi
}

env_pairs() {
  # 输出 KEY=VALUE 行（docker 与 native 共用）
  cat <<EOF
MODELS_DIR=/models
LORAS_DIR=/loras
INPUT_DIR=/input
VIDEO_DIR=/videos
MAX_RESIDENT_MODELS=${MAX_RESIDENT}
OFFLOAD_MODE=${OFFLOAD}
VRAM_STRICT=1
OBJECT_TTL_ORPHAN_SECONDS=1800
INFERENCE_TOKEN=${TOKEN}
PORT=8100
EOF
}

# ================= docker 模式 =================
mode_docker() {
  step "docker 模式（栈 $STACK / 镜像 $IMAGE）"
  local d=()
  if [ -n "$DOCKER_HOST_ARG" ]; then d=(-H "$DOCKER_HOST_ARG"); fi

  command -v docker >/dev/null 2>&1 || die "未找到 docker CLI，请先安装 docker 20.10+"
  if [ -n "$DOCKER_HOST_ARG" ]; then
    info "daemon: $DOCKER_HOST_ARG"
    docker "${d[@]}" version --format '{{.Server.Version}}' >/dev/null 2>&1 \
      || die "连不上 docker daemon：$DOCKER_HOST_ARG"
  else
    docker version --format '{{.Server.Version}}' >/dev/null 2>&1 \
      || die "本机 docker daemon 未运行（或需要 sudo）"
  fi

  # 1) 容器已存在？
  if docker "${d[@]}" container inspect "$CONTAINER" >/dev/null 2>&1; then
    if [ "$FORCE" != 1 ] && [ "$UNINSTALL" != 1 ]; then
      if [ "$DRY_RUN" = 1 ]; then
        warn "已存在容器 $CONTAINER：真实执行时会被拒绝（需 --force）。此处继续演示后续步骤"
      else
        die "已存在容器 $CONTAINER。重装请加 --force（会重建容器，模型卷保留）"
      fi
    fi
  fi

  # 1.5) 端口占用（两栈并存时最容易踩）
  local busy
  busy=$(docker "${d[@]}" ps --filter "publish=${PORT}" --format '{{.Names}}' 2>/dev/null | grep -v "^${CONTAINER}$" | head -1 || true)
  if [ -n "$busy" ]; then
    if [ "$DRY_RUN" = 1 ]; then
      warn "端口 $PORT 已被容器 $busy 占用（换 --port 或先停它）"
    else
      die "端口 $PORT 已被容器 $busy 占用。请用 --port 换端口，或先停止该容器"
    fi
  fi

  # 2) 镜像来源
  local have_local=0
  docker "${d[@]}" image inspect "$IMAGE" >/dev/null 2>&1 && have_local=1
  if [ "$FORCE_BUILD" = 1 ]; then
    build_image "${d[@]}"
  elif [ "$have_local" = 1 ]; then
    info "本地已有镜像 $IMAGE"
  elif [ "$NO_PULL" = 1 ]; then
    warn "本地无镜像且指定了 --no-pull，改为本地构建"
    build_image "${d[@]}"
  else
    info "拉取 $IMAGE ..."
    if [ "$DRY_RUN" = 1 ]; then
      run docker "${d[@]}" pull "$IMAGE"
    elif ! docker "${d[@]}" pull "$IMAGE"; then
      warn "pull 失败（离线或被墙），改为本地构建"
      build_image "${d[@]}"
    fi
  fi

  # 3) 数据卷 / 目录
  local vols=()
  if [ -n "$DOCKER_HOST_ARG" ] || [ -n "$DATA_DIR" ]; then
    # 远程 daemon：绑定挂载指向 daemon 机器上的路径，必须用命名卷
    local -a vnames=(flowx-inference-models flowx-inference-loras inference-input inference-videos)
    local -a mounts=(/models /loras /input /videos)
    for i in "${!vnames[@]}"; do
      run docker "${d[@]}" volume create "${vnames[$i]}" >/dev/null
      vols+=(-v "${vnames[$i]}:${mounts[$i]}")
      info "命名卷 ${vnames[$i]} → ${mounts[$i]}"
    done
    warn "命名卷模式（远程 daemon 必须如此）：模型放进卷 flowx-inference-models，"
    warn "  下载方法见 scripts/download-model.sh（TARGET=flowx-inference-models ...）"
  else
    local b=${BIND_DIR:-$HERE}
    [ -n "$MODELS_DIR_ARG" ] && b="$MODELS_DIR_ARG"
    for pair in "models:/models" "loras:/loras" "input:/input" "videos:/videos"; do
      local src="${pair%%:*}" dst="${pair#*:}"
      local dir="$b/$src"
      [ "$src" = "models" ] && [ -n "$MODELS_DIR_ARG" ] && dir="$MODELS_DIR_ARG"
      run mkdir -p "$dir"
      vols+=(-v "$dir:$dst")
      info "绑定挂载 $dir → $dst"
    done
  fi

  # 4) 启动
  local envs=()
  while IFS= read -r kv; do envs+=(-e "$kv"); done < <(env_pairs)

  step "启动容器 $CONTAINER"
  run docker "${d[@]}" rm -f "$CONTAINER" 2>/dev/null || true
  local gpu_arg=(--gpus "${GPUS_SEL:-all}")
  run docker "${d[@]}" run -d \
    --name "$CONTAINER" \
    --restart always \
    "${gpu_arg[@]}" \
    -p "0.0.0.0:${PORT}:8100" \
    "${envs[@]}" \
    "${vols[@]}" \
    "$IMAGE"

  # 5) 验收（daemon 在远端时访问远端主机 IP，不是本机 127.0.0.1）
  if [ "$DRY_RUN" != 1 ]; then
    local vhost="127.0.0.1"
    if [ -n "$DOCKER_HOST_ARG" ]; then
      vhost="${DOCKER_HOST_ARG#*//}"; vhost="${vhost%%:*}"
    fi
    local -a vopts=()
    if [ -n "$DOCKER_HOST_ARG" ]; then vopts=(--docker-host "$DOCKER_HOST_ARG"); fi
    bash "$VERIFY" --url "http://${vhost}:${PORT}" --token "$TOKEN" \
      --expect-torch "$TORCH" --expect-stack "$STACK" \
      "${vopts[@]}" --container "$CONTAINER" \
      || die "健康验收未通过（日志：docker logs $CONTAINER）" 1
  fi
  print_summary "docker"
}

build_image() (
  local d=("$@")
  step "本地构建镜像 $IMAGE（基座 $BASE_IMAGE，$DOCKERFILE）"
  [ -f "$HERE/$DOCKERFILE" ] || die "缺少 $HERE/$DOCKERFILE —— modern 栈镜像文件尚未创建"
  run docker "${d[@]}" build -f "$HERE/$DOCKERFILE" -t "$IMAGE" "$HERE"
)

# ================= native 模式 =================
mode_native() {
  step "native 模式（直接装到本机，栈 $STACK）"
  [ "$WSL" = 1 ] && info "WSL2：所有数据放在 Linux 侧，勿用 /mnt/c（IO 极慢）"
  if [ "$(id -u)" -ne 0 ] && [ "$DRY_RUN" != 1 ]; then
    die "native 模式需要 root（sudo bash install.sh --mode native ...）"
  fi
  local data=${DATA_DIR:-/var/lib/flowx-inference}
  local py="$PREFIX/py"

  # 1) micromamba
  if command -v micromamba >/dev/null 2>&1; then
    info "使用已有 micromamba：$(command -v micromamba)"
  elif [ -x /usr/local/bin/micromamba ]; then
    info "使用 /usr/local/bin/micromamba"
  else
    step "安装 micromamba（自包含 Python，不动系统 python）"
    # 两种形态都支持：
    #  ① 单文件静态二进制（默认，GitHub releases）→ 直接 chmod +x
    #  ② .tar.bz2 包（micro.mamba.pm）→ 需要 bzip2；Ubuntu minimal / Debian slim 常常没有，
    #     实测 22.04 云镜像就缺 bzip2，解压会报 "bzip2: Cannot exec"。
    case "$MICROMAMBA_URL" in
      *.tar.bz2|*/api/micromamba/*)
        if ! command -v bzip2 >/dev/null 2>&1; then
          step "安装 bzip2（解压 micromamba 包所需）"
          run bash -c 'apt-get update -qq && apt-get install -y -qq bzip2 || (dnf install -y bzip2 || yum install -y bzip2)'
        fi
        run bash -c "curl -fsSL $CURL_RETRY '$MICROMAMBA_URL' -o /tmp/mm.tar.bz2 && tar -xjf /tmp/mm.tar.bz2 -C /tmp bin/micromamba && install -m755 /tmp/bin/micromamba /usr/local/bin/micromamba && rm -f /tmp/mm.tar.bz2"
        ;;
      *)
        run bash -c "curl -fsSL $CURL_RETRY '$MICROMAMBA_URL' -o /tmp/micromamba && install -m755 /tmp/micromamba /usr/local/bin/micromamba && rm -f /tmp/micromamba"
        ;;
    esac
  fi
  local mm=micromamba
  command -v micromamba >/dev/null 2>&1 || mm=/usr/local/bin/micromamba

  # 2) 创建环境
  if [ ! -x "$py/bin/python" ]; then
    step "创建 Python $PY 环境 → $py"
    run "$mm" create -y -p "$py" -c conda-forge "python=$PY"
  else
    info "复用已有环境 $py（python $("$py/bin/python" -V 2>&1 | awk '{print $2}')）"
  fi

  # 3) torch + 依赖
  # pascal 栈的 insightface 0.7.3 没有预编译 wheel，需要 g++ 编译 face3d 的 cython 扩展
  # （modern 栈用 insightface 2.1，已有 py3-none-any wheel，不需要）
  if [ "$STACK" = "P" ] || [ "$STACK" = "P-lite" ]; then
    if ! command -v g++ >/dev/null 2>&1; then
      step "安装编译工具链（insightface 0.7.3 需要 g++）"
      run bash -c 'apt-get update -qq && apt-get install -y -qq build-essential'
    fi
  fi
  step "安装 torch $TORCH（$TORCH_INDEX）与算子依赖"
  run "$py/bin/pip" install --upgrade pip
  run "$py/bin/pip" install "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
  [ -f "$HERE/$REQ" ] || die "缺少依赖清单 $HERE/$REQ"
  run "$py/bin/pip" install -r "$HERE/$REQ"

  # controlnet_aux / insightface 会拉非 headless opencv（容器无 libGL 时 import 崩），
  # 裸机同样处理：卸载后强制重装 headless。
  # 版本按栈选：modern 栈 numpy2 ⇒ opencv 5.x；pascal 栈 numpy<2 ⇒ opencv 4.11。
  local cv_ver="5.0.0.93"
  if [ "$STACK" = "P" ] || [ "$STACK" = "P-lite" ]; then cv_ver="4.11.0.86"; fi
  step "修正 opencv（非 headless → headless $cv_ver）"
  run "$py/bin/pip" uninstall -y opencv-python || true
  run "$py/bin/pip" install --force-reinstall --no-deps "opencv-python-headless==$cv_ver"
  run "$py/bin/python" -c "import cv2, torch; print('smoke ok: cv2', cv2.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

  if [ "$DEPS_ONLY" = 1 ]; then
    step "依赖预装完成（无卡模式，未启动服务）"
    cat <<EOF
    栈          $STACK（Python $PY / torch $TORCH / CUDA $CUDA）
    Python      $py/bin/python
    环境前缀    $PREFIX

    已装好 torch 与全部算子依赖。等这台机器有卡后，执行下面任意一条即可起服务：

      bash install.sh --mode native          # 自动探卡、写 systemd、跑验收
      bash install.sh --mode native --stack $STACK   # 强制沿用本栈

    注意：预装用的是 $STACK 栈。若实际接的卡需要另一套栈（如 10 系要 P 栈），
    两个栈 torch 版本不同，需要重装依赖（install.sh 会自动处理）。
EOF
    exit 0
  fi

  # 4) 数据目录
  step "数据目录 $data"
  run mkdir -p "$data"/{models,loras,input,videos}
  if [ "$data" = "/var/lib/flowx-inference" ] && [ "$DRY_RUN" != 1 ]; then
    info "把模型文件放到 $data/models（safetensors/ckpt 或 diffusers 目录）"
  fi

  # 5) 环境文件
  local envf="/etc/flowx-inference/env"
  step "写入环境配置 $envf"
  if [ -f "$envf" ] && [ "$FORCE" != 1 ]; then
    warn "$envf 已存在，保留原配置（--force 可覆盖）"
  else
    run mkdir -p /etc/flowx-inference
    if [ "$DRY_RUN" = 1 ]; then
      printf '%s[dry-run]%s 写入 %s（含 INFERENCE_TOKEN 等）\n' "$C_DIM" "$C_0" "$envf"
    else
      {
        env_pairs | sed "s|^MODELS_DIR=.*|MODELS_DIR=$data/models|;s|^LORAS_DIR=.*|LORAS_DIR=$data/loras|;s|^INPUT_DIR=.*|INPUT_DIR=$data/input|;s|^VIDEO_DIR=.*|VIDEO_DIR=$data/videos|;s|^PORT=.*|PORT=$PORT|"
      } > "$envf"
      chmod 600 "$envf"
    fi
  fi

  # 6) systemd
  local use_systemd=0
  if [ -d /run/systemd/system ]; then use_systemd=1; fi
  if [ "$DRY_RUN" = 1 ]; then use_systemd=1; fi
  if [ "$use_systemd" = 1 ]; then
    step "写入 systemd 服务 flowx-inference.service"
    if [ "$DRY_RUN" = 1 ]; then
      printf '%s[dry-run]%s 写入 /etc/systemd/system/flowx-inference.service\n' "$C_DIM" "$C_0"
    else
      cat > /etc/systemd/system/flowx-inference.service <<EOF
[Unit]
Description=FlowX inference server (diffusers runtime)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/flowx-inference/env
WorkingDirectory=$HERE
ExecStart=$py/bin/uvicorn app.main:app --host 0.0.0.0 --port \${PORT}
Restart=always
RestartSec=5
# 显存治理相关：避免 OOM killer 误杀
OOMScoreAdjust=-200

[Install]
WantedBy=multi-user.target
EOF
      systemctl daemon-reload
      systemctl enable flowx-inference.service
      systemctl restart flowx-inference.service
    fi
    if [ "$DRY_RUN" != 1 ]; then
      bash "$VERIFY" --url "http://127.0.0.1:${PORT}" --token "$TOKEN" \
        --expect-torch "$TORCH" --expect-stack "$STACK" --python "$py/bin/python" || {
          die "健康验收未通过（日志：journalctl -u flowx-inference -n 100）" 1; }
    fi
  else
    warn "本机没有 systemd（容器/WSL 未启用 systemd）。跳过服务注册。"
    warn "手动启动：$py/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT"
    if [ "$WSL" = 1 ]; then
      warn "WSL2 启用 systemd：在 /etc/wsl.conf 加 [boot]\\nsystemd=true 后 wsl --shutdown 重启"
    fi
  fi
  print_summary "native"
}

uninstall_all() {
  step "卸载"
  if [ "$MODE" = "docker" ]; then
    local d=(); if [ -n "$DOCKER_HOST_ARG" ]; then d=(-H "$DOCKER_HOST_ARG"); fi
    run docker "${d[@]}" rm -f "$CONTAINER" 2>/dev/null || true
    run docker "${d[@]}" rmi "$IMAGE" 2>/dev/null || true
    if [ "$KEEP_DATA" != 1 ]; then
      warn "保留模型卷（--keep-data 语义默认生效）；如需删除：docker volume rm flowx-inference-models flowx-inference-loras"
    fi
  else
    if [ -d /run/systemd/system ]; then
      run systemctl disable --now flowx-inference.service 2>/dev/null || true
      run rm -f /etc/systemd/system/flowx-inference.service
      run systemctl daemon-reload || true
    fi
    run rm -rf "$PREFIX"
    if [ "$KEEP_DATA" = 1 ]; then
      info "保留数据目录 ${DATA_DIR:-/var/lib/flowx-inference}"
    else
      warn "数据目录 ${DATA_DIR:-/var/lib/flowx-inference} 未删除（含模型权重），确认后手动 rm -rf"
    fi
  fi
  step "卸载完成"
  exit 0
}

print_summary() {
  local mode=$1
  local host="127.0.0.1"
  [ -n "$DOCKER_HOST_ARG" ] && host="${DOCKER_HOST_ARG#*//}"; host="${host%%:*}"
  step "安装完成"
  cat <<EOF
    栈          $STACK（Python $PY / torch $TORCH / CUDA $CUDA）
    GPU         ${GPU_NAMES:-?}（${GPU_SM:-?}，驱动 ${DRIVER:-?}）
    服务地址    http://${host}:${PORT}
    令牌        $TOKEN
    模式        $mode

    FlowX 侧：把节点参数 service_url 指向上面地址、api_token 填上面的令牌。
    健康检查：curl -H "Authorization: Bearer $TOKEN" http://${host}:${PORT}/health
EOF
  if [ "$STACK" = "P" ] || [ "$STACK" = "P-lite" ]; then
    warn "当前为旧栈（cu$CUDA）：不支持 bf16/fp8。升级驱动到 >=580 后重跑本脚本可切到 modern 栈"
  fi
}

# ================= 主流程 =================
check_os

if [ "$UNINSTALL" = 1 ]; then
  # 卸载不需要探测显卡
  STACK=${STACK_OVERRIDE:-M}; apply_stack_defaults; uninstall_all
fi

step "环境探测"
if [ "$DEPS_ONLY" = 1 ]; then
  # 无卡预装：装 wheel 不需要 GPU，跳过探卡。栈必须能确定——
  # 未指定 --stack 时默认 M（modern 栈覆盖 20~50 系，是最常见目标）。
  if [ -z "$STACK_OVERRIDE" ]; then
    STACK_OVERRIDE="M"
    info "无卡预装：未指定 --stack，默认按 M 栈（modern）准备依赖"
  fi
  STACK="$STACK_OVERRIDE"
  GPU_NAMES="（无卡预装，未探测）"; GPU_SM="n/a"; DRIVER="n/a"; GPU_VRAM_MB=16384
  info "跳过显卡探测（--deps-only）：装完依赖后，在有卡时直接 bash install.sh 起服务即可"
else
  load_detect
fi
apply_stack_defaults
tune_by_vram

info "栈 $STACK：Python $PY / torch $TORCH / CUDA $CUDA / 镜像 $IMAGE"
if [ "$DEPS_ONLY" != 1 ]; then
  info "显存 $(( ${GPU_VRAM_MB:-0} / 1024 )) GB → MAX_RESIDENT_MODELS=$MAX_RESIDENT, OFFLOAD_MODE=$OFFLOAD"
fi
[ "$DRY_RUN" = 1 ] && warn "dry-run 模式：以下命令只会打印，不会执行"

case "$MODE" in
  docker) mode_docker ;;
  native) mode_native ;;
esac

exit 0

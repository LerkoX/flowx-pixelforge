#!/bin/sh
# 下载模型到推理服务的 MODELS_DIR（下载容器跑在 daemon 所在机器上，
# 远程 daemon 场景同样适用）。只拉 fp16 权重 + 配置。
#
# 用法：./scripts/download-model.sh <preset|repo-id>
#   svd        SVD-XT 1.1 图生视频（ModelScope 镜像，fp16 ~4.5GB）
#   其他 repo  走 HF（hf-mirror）：./scripts/download-model.sh Org/some-model
#
# 目标位置用 TARGET 环境变量指定（docker -v 语法，默认 D:/flowx-data/models）：
#   TARGET=flowx-inference-models ./scripts/download-model.sh svd   # 命名卷
#   TARGET=D:/flowx-data/models ./scripts/download-model.sh svd     # Windows 绑定挂载
#
# 前提：daemon 能创建容器 + 目标盘有足够空间。
set -e

TARGET=${TARGET:-D:/flowx-data/models}
case "$1" in
  svd)
    # stabilityai 官方 HF 仓库是门控的（需申请），走 ModelScope 公开镜像
    MS_REPO="stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    DIR="stable-video-diffusion-img2vid-xt-1-1"
    echo ">> [modelscope] $MS_REPO -> $TARGET/$DIR (fp16-only)"
    docker run --rm -v "$TARGET":/models python:3.11-slim sh -c "
      pip install -q modelscope &&
      modelscope download --model '$MS_REPO' \
        --include '*.json' '*.fp16.safetensors' \
        --local_dir '/models/$DIR'
    "
    ;;
  "")
    echo "usage: $0 <preset|hf-repo-id>" >&2; exit 1 ;;
  *)
    REPO="$1"
    DIR=$(basename "$REPO")
    echo ">> [hf-mirror] $REPO -> $TARGET/$DIR (fp16-only)"
    docker run --rm -v "$TARGET":/models python:3.11-slim sh -c "
      pip install -q huggingface_hub &&
      HF_ENDPOINT=https://hf-mirror.com hf download '$REPO' \
        --include '*.json' '*.fp16.safetensors' \
        --local-dir '/models/$DIR'
    "
    ;;
esac
echo ">> done. 模型名（ckpt_name）: $DIR"

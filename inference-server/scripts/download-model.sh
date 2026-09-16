#!/bin/sh
# 下载模型到推理服务的 MODELS_DIR（下载容器跑在 daemon 所在机器上，
# 远程 daemon 场景同样适用）。国内网络走 hf-mirror，只拉 fp16 权重 + 配置。
#
# 用法：./scripts/download-model.sh <preset>
#   svd        SVD-XT 1.1 图生视频（stabilityai/stable-video-diffusion-img2vid-xt-1-1，fp16 ~9.6GB）
#   任意 HF repo id 也可以直接传，如 ./scripts/download-model.sh Org/some-model
#
# 目标位置用 TARGET 环境变量指定（docker -v 语法，默认 D:/flowx-data/models）：
#   TARGET=flowx-inference-models ./scripts/download-model.sh svd   # 命名卷
#   TARGET=D:/flowx-data/models ./scripts/download-model.sh svd     # Windows 绑定挂载
#
# 前提：daemon 能创建容器 + 目标盘有足够空间。
set -e

TARGET=${TARGET:-D:/flowx-data/models}
case "$1" in
  svd) REPO="stabilityai/stable-video-diffusion-img2vid-xt-1-1" ;;
  "")  echo "usage: $0 <preset|hf-repo-id>" >&2; exit 1 ;;
  *)   REPO="$1" ;;
esac
DIR=$(basename "$REPO")

echo ">> downloading $REPO -> $TARGET/$DIR (via hf-mirror, fp16-only)"
docker run --rm -v "$TARGET":/models python:3.11-slim sh -c "
  pip install -q 'huggingface_hub[cli]' &&
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download '$REPO' \
    --include '*.json' '*.fp16.safetensors' \
    --local-dir '/models/$DIR' --local-dir-use-symlinks False
"
echo ">> done. 模型名（ckpt_name）: $DIR"

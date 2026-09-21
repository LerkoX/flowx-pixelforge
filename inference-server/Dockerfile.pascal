# inference-server — Pascal 兼容镜像（GTX 10xx / Tesla P4/P40/P100，sm_60/61）
#
# 为什么需要单独的镜像：
#   - PyTorch 2.7+（CUDA 12.8+ 构建）的官方二进制不再包含 sm_60/61，
#     在 Pascal 上会报 "no kernel image is available"。
#   - cuDNN 9 在 Pascal 上存在兼容性风险，cudnn8 是最稳的组合。
#
# 基础镜像自带 torch 2.3.1（含 sm_60/61），requirements-pascal.txt 固定
# torch==2.3.1，防止 pip 把 torch 升级到不含 Pascal 支持的新版本。
#
# 若宿主机驱动 < 525（跑不动 CUDA 12.x 镜像，旧 Pascal 机器常见），
# 改用 CUDA 11.8 基础镜像（驱动 >= 450 即可），同时把
# requirements-pascal.txt 里的 torch 固定为 2.3.1+cu118：
#     FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements-pascal.txt .
RUN pip install --no-cache-dir -r requirements-pascal.txt
# controlnet_aux 会拉入 opencv-python（非 headless），抢占 cv2 命名空间且容器无
# libGL 导致 cv2 导入崩溃；卸载之，headless 提供同一 cv2 API（真机已验证）。
RUN pip uninstall -y opencv-python || true

COPY app/ ./app/

# Pascal 卡显存多为 8~11 GB（Tesla P40 为 24 GB），
# SD1.5 单模型约占 4 GB 显存，默认常驻数降为 1 更稳妥，可按卡调整。
ENV MODELS_DIR=/models \
    MAX_RESIDENT_MODELS=1 \
    PORT=8100

EXPOSE 8100

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

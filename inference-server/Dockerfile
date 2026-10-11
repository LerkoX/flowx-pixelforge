FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import importlib.metadata as im; import timm, kornia, einops; print('deps smoke ok, timm', im.version('timm'), 'kornia', im.version('kornia'))"

COPY app/ ./app/

ENV MODELS_DIR=/models \
    MAX_RESIDENT_MODELS=2 \
    PORT=8100

EXPOSE 8100

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

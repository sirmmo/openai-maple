# CPU image by default. For CUDA, build with:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t openai-maple:cuda .
# and run with --gpus all. The model code then still uses the pure-torch
# attention shim unless a flash-attn wheel is added (see README).
FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt

COPY openai_maple ./openai_maple
COPY pyproject.toml README.md ./

# Weights (~40GB of bf16 safetensors) are pulled from HuggingFace on first
# start and cached here. Mount a volume or you will download them every run.
RUN mkdir -p /cache/huggingface
VOLUME ["/cache"]

EXPOSE 8000

ENV MAPLE_HOST=0.0.0.0 \
    MAPLE_PORT=8000

# Loading 40GB from a cold cache takes minutes; give the start period room.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

ENTRYPOINT ["python", "-m", "openai_maple"]

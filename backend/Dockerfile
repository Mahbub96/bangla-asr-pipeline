# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Bangla & English ASR Studio — FastAPI backend
#
# Build (CPU, default):
#   docker build -f backend/Dockerfile -t bangla-asr-backend .
# Build (CUDA 12.1 wheels):
#   docker build -f backend/Dockerfile \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#     -t bangla-asr-backend:gpu .
# Build with the fine-tuning stack (transformers/datasets/peft/accelerate):
#   docker build -f backend/Dockerfile --build-arg INSTALL_TRAINING=true ...
#
# Build context is the REPOSITORY ROOT, not backend/.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_TRAINING=false

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# torch first, from the index matching the target accelerator, so the generic
# requirements resolve against an already-satisfied torch instead of pulling the
# multi-gigabyte default CUDA build.
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" "torch>=2.0.0"

COPY requirements.txt requirements_gpu.txt ./
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt
RUN if [ "${INSTALL_TRAINING}" = "true" ]; then \
        pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements_gpu.txt ; \
    fi

# --------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

# ffmpeg  -> audio sanitising (backend/services/audio.py)
# libsndfile1 -> soundfile/librosa
# libgomp1 -> ctranslate2 / faster-whisper OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    OMP_NUM_THREADS=4 \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=8000

WORKDIR /app

# Application code only — models/ and data/ arrive as volumes (see compose file).
COPY backend/ ./backend/
COPY scripts/ ./scripts/

RUN mkdir -p /app/models /app/data /app/.tmp/api \
    && useradd --create-home --uid 1000 asr \
    && chown -R asr:asr /app
USER asr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${ASR_PORT}/api/health" || exit 1

CMD ["sh", "-c", "exec uvicorn backend.main:app --host \"$ASR_HOST\" --port \"$ASR_PORT\""]

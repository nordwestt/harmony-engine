# --- builder: install deps with CPU-only PyTorch (avoid PyPI's CUDA mega-wheel) ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY harmony ./harmony

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv sync --extra db --extra embed --extra api-minimal --no-dev \
        --no-install-package torch --no-install-package torchaudio \
    && uv pip install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && find /app/.venv -type d -name __pycache__ -exec rm -rf {} + \
    && find /app/.venv -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache/uv

# --- runtime: slim image with only the venv + app ---
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY harmony ./harmony
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV HARMONY_DATA_DIR=/data
ENV HF_HOME=/data/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# --- builder: install deps with CPU-only PyTorch (avoid PyPI's CUDA mega-wheel) ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY docker/pyproject.cpu.patch.toml /tmp/pytorch.patch
COPY harmony ./harmony

# [tool.uv.sources] must live in pyproject.toml (not uv.toml). Patch at build time only.
RUN cat /tmp/pytorch.patch >> pyproject.toml

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Install CPU torch first so muq's transitive torch dep does not pull PyPI's CUDA wheel.
# Reinstall after sync to ensure torchaudio matches (libcudart errors = CUDA build).
RUN uv venv \
    && uv pip install torch torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv sync --extra db --extra embed --extra api-minimal --no-dev \
        --no-install-package torch --no-install-package torchaudio --no-install-package torchvision \
    && uv pip install --reinstall \
        torch torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /app/.venv/lib/python3.12/site-packages/nvidia \
    && /app/.venv/bin/python -c "\
import torch, torchaudio, torchvision; \
import torchvision.transforms; \
from muq import MuQMuLan; \
assert not torch.backends.cuda.is_built(); \
print('pytorch', torch.__version__, 'torchvision', torchvision.__version__)" \
    && find /app/.venv -type d -name __pycache__ -exec rm -rf {} + \
    && find /app/.venv -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache/uv

# --- runtime: same bookworm Python ABI as builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl libgomp1 \
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

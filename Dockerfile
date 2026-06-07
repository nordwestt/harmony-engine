FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY harmony ./harmony

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv sync --extra db --extra embed --extra api --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV HARMONY_DATA_DIR=/data
ENV HF_HOME=/data/huggingface

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

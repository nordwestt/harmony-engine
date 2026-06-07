#!/usr/bin/env bash
# Bootstrap a local Harmony development environment.
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync --extra db --group dev

uv run harmony init
uv run harmony status

echo ""
echo "Done. Run tests with:       uv run pytest"
echo "Index music with:          uv run harmony index /path/to/music"

#!/usr/bin/env bash
# Bootstrap a local Harmony development environment.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[db,dev]"

harmony init
harmony status

echo ""
echo "Done. Run tests with: pytest"
echo "Index music with:    harmony index /path/to/music"

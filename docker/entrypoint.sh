#!/bin/sh
set -e

DATA_DIR="${HARMONY_DATA_DIR:-/data}"

if [ ! -f "$DATA_DIR/config.yaml" ]; then
  harmony --data-dir "$DATA_DIR" init --local
fi

exec harmony --data-dir "$DATA_DIR" serve \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"

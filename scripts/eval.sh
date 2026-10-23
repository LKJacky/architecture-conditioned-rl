#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EVAL_URL="${EVAL_URL:-http://127.0.0.1:8000/v1}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-ALLOYLM}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-work_dirs/eval}"
EVAL_CONFIG="${EVAL_CONFIG:-swarr/configs/eval.py}"

mkdir -p "$EVAL_WORK_DIR"
echo "evaluating: URL=$EVAL_URL MODEL=$EVAL_MODEL_NAME WORK_DIR=$EVAL_WORK_DIR"
exec "$PYTHON" -m alloylm.algorithm.eval.run "$EVAL_CONFIG" \
  --url "$EVAL_URL" \
  --model_name "$EVAL_MODEL_NAME" \
  --work_dir "$EVAL_WORK_DIR" \
  "$@"

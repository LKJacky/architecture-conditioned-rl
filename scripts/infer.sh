#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-work_dirs/default/best_ckpt/hf}"
TOKENIZER="${TOKENIZER:-$MODEL}"
MODEL_NAME="${MODEL_NAME:-ALLOYLM}"
PORT="${PORT:-8000}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-8192}"
MEMORY_USAGE="${MEMORY_USAGE:-0.7}"
SAMPLER_BATCH_SIZE="${SAMPLER_BATCH_SIZE:-8}"

exec "$PYTHON" scripts/infer.py \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --model-name "$MODEL_NAME" \
  --port "$PORT" \
  --max-prefill-length "$MAX_PREFILL_LENGTH" \
  --memory-usage "$MEMORY_USAGE" \
  --sampler-batch-size "$SAMPLER_BATCH_SIZE" \
  "$@"

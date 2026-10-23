#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
GPU="${GPU:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
SP="${SP:-1}"
WORK_DIR="${WORK_DIR:-work_dirs/default}"

if [[ "$GPU" -lt 1 ]]; then
  echo "error: no GPU detected; set GPU=<number of GPUs>" >&2
  exit 1
fi

if (( SP < 1 || GPU * WORLD_SIZE % SP != 0 )); then
  echo "error: SP must divide GPU * WORLD_SIZE (got SP=$SP, GPU=$GPU, WORLD_SIZE=$WORLD_SIZE)" >&2
  exit 1
fi

export GPU WORLD_SIZE SP WORK_DIR

mkdir -p "$WORK_DIR"
echo "training: GPU=$GPU WORLD_SIZE=$WORLD_SIZE SP=$SP WORK_DIR=$WORK_DIR"
exec "$PYTHON" -m alloylm.algorithm.rl.run swarr/configs/rl.py "$@"

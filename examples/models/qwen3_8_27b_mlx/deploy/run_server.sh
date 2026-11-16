#!/usr/bin/env bash
# Run MLX-VLM in its own environment. CUDA_VISIBLE_DEVICES is set by the caller on Linux.
set -euo pipefail

WORK_DIR=${WORK_DIR:?the directory containing the MLX server environment}
MLX_PYTHON=${MLX_PYTHON:-$WORK_DIR/envs/mlx_server/bin/python}
MODEL=${MODEL:-mlx-community/Qwen3.8-27B-4bit}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CONTEXT_LIMIT=${CONTEXT_LIMIT:-16384}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-4096}
PREFILL_STEP_SIZE=${PREFILL_STEP_SIZE:-512}

exec uv run --no-project --python "$MLX_PYTHON" -m mlx_vlm.server \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --enable-thinking --max-num-seqs 1 \
  --max-kv-size "$CONTEXT_LIMIT" --max-tokens "$MAX_OUTPUT_TOKENS" \
  --prefill-step-size "$PREFILL_STEP_SIZE"

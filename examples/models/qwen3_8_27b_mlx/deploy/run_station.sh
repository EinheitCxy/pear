#!/usr/bin/env bash
# Use an MLX-VLM endpoint from an existing PEAR robot station.
set -euo pipefail

WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
STATION_CONFIG=${STATION_CONFIG:?the configuration file of this station}
SERVER=${SERVER:?the MLX-VLM server, host:port}
RUN=${RUN:?the run directory}
MODEL=${MODEL:-mlx-community/Qwen3.8-27B-4bit}
STATION_PYTHON=${STATION_PYTHON:-$WORK_DIR/envs/station/bin/python}
CONTEXT_LIMIT=${CONTEXT_LIMIT:-16384}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-4096}
MODEL_LABEL=${MODEL_LABEL:-Qwen3.8-27B-4bit (MLX)}

exec uv run --no-project --python "$STATION_PYTHON" -m pear.deploy.station.cli gui --config "$STATION_CONFIG" \
  run_dir="$RUN" \
  default_model=served \
  models.served.label="$MODEL_LABEL" \
  models.served.client_class=pear.deploy.station.agent_loop.clients.mlx.MLXClient \
  models.served.client.base_url="http://$SERVER/v1" \
  models.served.client.model="$MODEL" \
  models.served.client.context_limit="$CONTEXT_LIMIT" \
  models.served.client.max_output_tokens="$MAX_OUTPUT_TOKENS"

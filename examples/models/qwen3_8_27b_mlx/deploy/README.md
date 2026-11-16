# Qwen3.8-27B-4bit with MLX-VLM

Use [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) through a separate
MLX-VLM server and PEAR's `MLXClient`. The existing station supplies observations, validates final JSON actions,
and controls its YAM or SO-101 robot. Training and the vLLM deployment recipes use their existing backends.

This checkpoint is the generic Qwen model converted to MLX; it is **not a PEAR-trained robot policy**. Connecting
its API does not establish manipulation success or reproduce PEAR's published results. A PEAR-trained checkpoint
must be converted separately and evaluated before claiming equivalent behavior.

## Install the model server

Run from the repository root with [uv](https://docs.astral.sh/uv/) installed. Keep this environment separate from
`envs/server`, `envs/verl`, and `envs/station`.

The protocol is based on [MLX-VLM v0.7.12](https://github.com/Neywa-Labs/mlx-vlm/releases/tag/v0.7.12).
Install the backend appropriate to the **server** machine; the station can run on another Linux machine.

```bash
export WORK_DIR=/path/to/pear-work
uv venv --python 3.12 "$WORK_DIR/envs/mlx_server"
```

On Linux with NVIDIA CUDA 12:

```bash
uv pip install --python "$WORK_DIR/envs/mlx_server/bin/python" \
  'mlx-vlm==0.7.12' 'mlx[cuda12]==0.32.4'
```

For a CUDA 13 installation, replace `cuda12` with `cuda13`. MLX's
[official requirements](https://ml-explore.github.io/mlx/build/html/install.html#cuda) include glibc >= 2.35 and
a supported driver, GPU architecture, and Python. Rocky Linux 9's glibc 2.34 cannot install this Linux wheel
natively; use the explicit container setup below or a supported server OS. The existence of a CUDA backend does
not by itself verify this model's full vision inference on an RTX 4090 D.

On Apple Silicon, use an ARM Python on a supported macOS version:

```bash
uv pip install --python "$WORK_DIR/envs/mlx_server/bin/python" 'mlx-vlm==0.7.12' 'mlx==0.32.4'
```

The weight files occupy approximately 16.1 GB. This is not the peak memory requirement: vision encoding,
KV/recurrent state, activations, and the runtime require additional memory.

### Linux container

For hosts with older glibc, `Dockerfile` provides Ubuntu 24.04 and NVIDIA's published
[`12.9.2-runtime-ubuntu24.04`](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md)
image. The host needs a compatible NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image does not install or change the host driver. Build and start it explicitly:

```bash
docker build -t pear-mlx:0.7.12 examples/models/qwen3_8_27b_mlx/deploy
docker run --rm --gpus 'device=1' \
  -p 127.0.0.1:8080:8080 \
  --mount type=volume,src=pear-mlx-hf-cache,dst=/cache/huggingface \
  -e MODEL=mlx-community/Qwen3.8-27B-4bit \
  -e CONTEXT_LIMIT=16384 -e MAX_OUTPUT_TOKENS=4096 \
  pear-mlx:0.7.12
```

This exposes only GPU 1 to the container and preserves downloaded weights in the named volume. It starts the
server described below; do not start a second native server on the same port. Image building, container GPU
access, model loading, and inference remain target-machine checks; protocol tests do not validate them.

## Start the server

On NVIDIA, select **one** authorized GPU before starting; for example:

```bash
export CUDA_VISIBLE_DEVICES=1
```

On either platform:

```bash
export MODEL=mlx-community/Qwen3.8-27B-4bit
export CONTEXT_LIMIT=16384 MAX_OUTPUT_TOKENS=4096
bash examples/models/qwen3_8_27b_mlx/deploy/run_server.sh
```

The first start downloads the model if it is not cached. `MODEL` can instead name a local MLX checkpoint.
The defaults are one concurrent sequence, 512-token prefill steps, and a 16,384-token context budget including
up to 4,096 generated tokens. These are starting settings, not measured VRAM guarantees. This recipe does not
implement multi-GPU tensor parallelism; setting `CUDA_VISIBLE_DEVICES=1,2` alone does not combine their memory.

The server listens at `127.0.0.1:8080`. Set `HOST` and `PORT` explicitly for another address, or forward the port
over SSH to the station machine. Verify the loaded name with:

```bash
curl --fail http://127.0.0.1:8080/v1/models
```

## Connect a station

Set up the station and its robot using the existing [station guide](../../qwen3_8_27b_pear/deploy/README.md#drive-a-robot).
Use the same `MODEL`, `CONTEXT_LIMIT`, and `MAX_OUTPUT_TOKENS` as the server:

```bash
export WORK_DIR=/path/to/pear-work
export STATION_CONFIG=/path/to/station.yaml
export SERVER=127.0.0.1:8080 RUN=/path/to/records
export MODEL=mlx-community/Qwen3.8-27B-4bit
export CONTEXT_LIMIT=16384 MAX_OUTPUT_TOKENS=4096
bash examples/models/qwen3_8_27b_mlx/deploy/run_station.sh
```

The page offers the MLX model through the same station workflow. The server's `/v1/models` listing must contain
the exact configured `MODEL`; for a local checkpoint use the name the endpoint reports.
To configure an existing station file directly, merge this model section into it:

```yaml
default_model: qwen_mlx
models:
  qwen_mlx:
    label: Qwen3.8-27B-4bit (MLX)
    client_class: pear.deploy.station.agent_loop.clients.mlx.MLXClient
    timeout_s: 300
    client:
      base_url: http://127.0.0.1:8080/v1
      model: mlx-community/Qwen3.8-27B-4bit
      context_limit: 16384
      max_output_tokens: 4096
```

`api_key_file` is available for an authenticated endpoint or proxy; relative paths resolve beside the station file.
Optional client settings are `max_images` (128), `temperature` (1), `top_p` (0.95), `top_k` (0, disabled),
`repetition_penalty` (1), and `enable_thinking` (true). Output tokens include thinking tokens. If generation reaches
the output limit, the client refuses that turn even if a JSON fragment appears complete.

## Protocol and validation

The client re-sends complete data URLs for all historical images; it never uses vLLM's UUID-only image references.
Reference images join the first successful turn only. Assistant reasoning travels separately from final content;
the two MLX reasoning aliases are consumed once. Every request disables MLX-VLM automatic compaction with
`context_management: []`, so the server does not silently summarize the station's history. Clear the task when
the context or image capacity is reached.

Only a normally terminated response (`finish_reason: stop`) enters history and proceeds to the existing action
parser. Cancellation, timeout, `length`, connection loss, and server errors fail the turn without committing a
partial response or retrying. Token counts come from the endpoint's usage chunk; missing counts remain unknown.
The client does not treat MLX-VLM's estimated timings as a measured server decode interval.

CPU protocol tests use a local HTTP/SSE stand-in and do not download model weights or open robot hardware:

```bash
uv run --no-project --python "$WORK_DIR/envs/station/bin/python" -m pytest \
  tests/deploy/station/test_mlx_in_station.py tests/deploy/station/test_clients_in_station.py
```

Passing these tests verifies the adapter protocol. Actual model loading, vision inference, GPU memory peaks,
latency, and robot task success require validation on the target machine.

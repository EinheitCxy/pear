FROM nvidia/cuda:12.9.2-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libglib2.0-0t64 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --python /usr/bin/python3 /opt/mlx \
    && uv pip install --no-cache --python /opt/mlx/bin/python 'mlx-vlm==0.7.12' 'mlx[cuda12]==0.32.4'

ENV WORK_DIR=/opt/pear MLX_PYTHON=/opt/mlx/bin/python HOST=0.0.0.0 HF_HOME=/cache/huggingface
WORKDIR /opt/pear
COPY run_server.sh /opt/pear/run_server.sh
EXPOSE 8080
CMD ["bash", "/opt/pear/run_server.sh"]

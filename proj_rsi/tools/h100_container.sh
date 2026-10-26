#!/usr/bin/env bash
# Install as /root/okc_run.sh on the iteration H100. The GPU lock lives inside
# the container, so loss of the SSH client cannot release it before CUDA exits.
set -euo pipefail
touch /tmp/gpu.lock
container=${PROJ_GPU_CONTAINER:-okc-proj-$$}
exec docker run --rm --init --name "$container" --gpus all --ipc=host --shm-size=8g \
  --label okc.scheduler=container-flock \
  -v /root/okc:/okc -v /root/okc_cache:/root/.cache \
  -v /tmp/gpu.lock:/gpu.lock -w /okc \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e TORCH_CUDA_ARCH_LIST=9.0a --entrypoint flock \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel -w 7200 /gpu.lock "$@"

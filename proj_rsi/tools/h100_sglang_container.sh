#!/usr/bin/env bash
# Install as /root/okc_run_sglang.sh on the iteration H100. The GPU lock lives inside
# the container, so loss of the SSH client cannot release it before CUDA exits.
set -euo pipefail
touch /tmp/gpu.lock
container=${PROJ_GPU_CONTAINER:-okc-proj-$$}
profile_options=()
# The driver restricts performance counters to CAP_SYS_ADMIN. Opt in only for a
# named, bounded Nsight diagnostic; leave the driver policy and ordinary jobs alone.
if [[ ${PROJ_GPU_PROFILE:-0} == 1 ]]; then profile_options+=(--cap-add SYS_ADMIN); fi
exec docker run --rm --init --name "$container" "${profile_options[@]}" --gpus all --ipc=host --shm-size=8g \
  --label okc.scheduler=container-flock \
  -v /root/okc:/okc -v /root/okc_cache/sglang-cu130:/root/.cache \
  -v /tmp/gpu.lock:/gpu.lock -w /okc \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e TORCH_CUDA_ARCH_LIST=9.0a --entrypoint flock \
  lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a -w 7200 /gpu.lock "$@"

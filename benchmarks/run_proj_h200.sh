#!/usr/bin/env bash
# Time a muse_bf16_proj submission on the local H200. Run this on the GPU host.
# Refuses to start when a compute process already holds the device.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image='8cfc5a72cb29'
# Shared with the serving RSI executor; hold through check and container exit.
# The busy check alone has a race while another launcher is still initializing.
mkdir -p "$workspace/.gpu-cache"
exec 9>"$workspace/.gpu-cache/rsi-device0.lock"
if ! flock -n 9; then
    echo "GPU device lock held by another benchmark; not starting" >&2
    exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "GPU is busy; not starting" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 2
fi
mkdir -p "$workspace/.gpu-cache" "$workspace/results/proj-rsi"
docker run --rm --gpus '"device=0"' --shm-size=16g \
    --volume "$workspace:/workspace" --workdir /workspace \
    --volume "$workspace/.gpu-cache:/root/.cache" \
    --env PYTHONPATH=/workspace/muse_bf16_proj \
    --entrypoint /bin/bash "$image" -lc \
    'python3 benchmarks/proj_time.py "$@"' \
    benchmark "$@"

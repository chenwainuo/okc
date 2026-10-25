#!/usr/bin/env bash
# Run from any directory on a Docker host with NVIDIA Container Toolkit.
# No model downloads or challenge login needed: these tests use synthetic weights.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image='lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a'
runner=benchmarks/moe_local.py
if [[ ${1:-} == --matrix ]]; then
    runner=benchmarks/run_matrix.py
    shift
elif [[ ${1:-} == --loop ]]; then
    runner=benchmarks/optimize_loop.py
    shift
elif [[ ${1:-} == --execution ]]; then
    runner=benchmarks/sweep_execution.py
    shift
elif [[ ${1:-} == --tiles ]]; then
    runner=benchmarks/sweep_tiles.py
    shift
else
    set -- --baseline benchmarks/serving_stock_moe.py --candidate benchmarks/serving_moe.py --output results/serving-stack.json "$@"
fi
docker run --rm --gpus '"device=0"' --shm-size=16g \
    --volume "$workspace:/workspace" --workdir /workspace \
    --volume "$workspace/.gpu-cache:/root/.cache" \
    --env PYTHONPATH=/workspace/serving/sglang/python \
    --entrypoint /bin/bash "$image" -lc \
    'python3 benchmarks/preflight.py && python3 benchmarks/check_local_alignment.py && python3 benchmarks/check_finalize.py && python3 "$@"' \
    benchmark "$runner" "$@"

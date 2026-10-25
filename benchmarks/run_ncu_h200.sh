#!/usr/bin/env bash
# Run only after clean benchmark jobs stop. Profiling perturbs execution.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
batch=${1:-128}
[[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch must be a positive integer' >&2; exit 2; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'GPU compute process detected; finish existing jobs before profiling.' >&2
    exit 1
fi
mkdir -p "$workspace/results/hardware"
image='lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a'
docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN --shm-size=16g \
    --volume "$workspace:/workspace" --workdir /workspace \
    --volume "$workspace/.gpu-cache:/root/.cache" \
    --env PYTHONPATH=/workspace/serving/sglang/python \
    --entrypoint /bin/bash "$image" -lc \
    'ncu --profile-from-start off --kernel-name "regex:.*Marlin.*" --launch-count 2 \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
        --section SchedulerStats --section LaunchStats --clock-control none \
        --cache-control all --force-overwrite --export "results/hardware/bs${1}" \
        python3 benchmarks/profile_hardware.py --batch "$1" && \
     ncu --import "results/hardware/bs${1}.ncu-rep" --page raw --csv > "results/hardware/bs${1}.csv"' \
    profile "$batch"

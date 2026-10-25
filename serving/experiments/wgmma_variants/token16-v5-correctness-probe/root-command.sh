#!/usr/bin/env bash
# ROOT ONLY: requires matching canonical600s lease; no execution by preparer.
set -euo pipefail
cd /root/okc
experiment=${1:-WGMMA-TOKEN16-v5-CORRECTNESS}
control=/root/okc/results/rsi/candidates/wgmma-token16-v5-probe
source=/root/okc/results/rsi/candidates/wgmma-token16-v5
baseline=/root/okc/results/rsi/immutable-trees/baseline-a263b86eba62c39fbfa55b6ad580d28fc3119deec5a655919ab8cadfc4673eab
output=/root/okc/results/rsi/wgmma-token16-v5-correctness
cache=/root/okc/.gpu-cache/wgmma-token16-v5
image=lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a
name="okc-wgmma-token16-v5-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
mkdir "$output" # refuse any evidence overwrite
exec 9>/root/okc/.gpu-cache/rsi-device0.lock
flock -n 9
gpu_pids=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_pids" ]] || exit 2
python3 "$control/verify_inputs.py" --baseline "$baseline" \
  --manifest /root/okc/results/rsi/bundles/baseline-v2.tar.gz.manifest.json \
  --source "$source" --runner /root/okc/benchmarks --lease /root/okc/results/rsi/gpu-lease.json \
  --experiment "$experiment" --output "$output/input-pins.json"
mkdir -p "$cache"
cleanup() {
    original_code=$?
    trap - EXIT
    trap '' TERM INT
    python3 "$control/teardown.py" "$name" "$output" "$original_code" || exit 42
    exit "$original_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Private standalone layout preserves /baseline and /cache paths. CPU preseed,
# if desired, is separate before lease admission; never copied from /candidate.
timeout --signal=TERM --kill-after=15s 540s docker run --rm --init --name "$name" \
  --gpus '"device=0"' --shm-size=16g --network=none \
  -v "$baseline:/baseline:ro" -v /root/okc/benchmarks:/runner:ro \
  -v "$source:/probe:ro" -v "$control:/control:ro" -v "$output:/output" \
  -v "$cache:/cache" -e PYTHONPATH=/runner -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e XDG_CACHE_HOME=/cache -e SGLANG_JIT_CACHE_DIR=/cache/sglang \
  -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  --entrypoint bash "$image" /control/inside-command.sh > "$output/run.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
cd /okc
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=2
metrics=$(cat proj_rsi/candidates/hw_compression_byteplanes/metrics.txt)
exec /usr/local/cuda/bin/ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include raw_generic/ --nvtx-include raw_none/ \
  --nvtx-include planes_generic/ --nvtx-include planes_none/ \
  --launch-count 4 --kill no --metrics "$metrics" \
  --export /okc/logs/hw-byteplanes01 -f \
  python3 proj_rsi/candidates/hw_compression_byteplanes/driver.py \
  --json /okc/logs/hw-byteplanes01.json

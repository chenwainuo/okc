#!/usr/bin/env bash
set -euo pipefail
cd /okc
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=2
metrics=$(cat proj_rsi/candidates/hw_compression_signlow_zero/metrics.txt)
exec /usr/local/cuda/bin/ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include raw_generic/ --nvtx-include raw_none/ \
  --nvtx-include signlow_generic/ --nvtx-include signlow_none/ \
  --nvtx-include zero_generic/ --nvtx-include zero_none/ \
  --launch-count 6 --kill no --metrics "$metrics" \
  --export /okc/logs/hw-signlow-zero01 -f \
  python3 proj_rsi/candidates/hw_compression_signlow_zero/driver.py \
  --json /okc/logs/hw-signlow-zero01.json

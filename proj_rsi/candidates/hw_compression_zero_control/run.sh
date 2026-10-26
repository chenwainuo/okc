#!/usr/bin/env bash
# Same profiler/cache/clock/metric policy as frozen real-weight counter diagnostic.
set -euo pipefail
cd /okc
metrics=$(cat proj_rsi/candidates/hw_compression_zero_control/metrics.txt)
exec /usr/local/cuda/bin/ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include compression_generic/ --nvtx-include compression_none/ \
  --launch-count 2 --kill no --metrics "$metrics" \
  --export /okc/logs/hw-zero-counters01 -f \
  python3 proj_rsi/candidates/hw_compression_zero_control/driver.py \
  --json /okc/logs/hw-zero-counters01.json

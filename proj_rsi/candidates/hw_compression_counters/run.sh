#!/usr/bin/env bash
# Root launches only after review in the exclusive profiling container. No clock mutation.
set -euo pipefail
cd /okc
metrics=$(cat proj_rsi/candidates/hw_compression_counters/metrics.txt)
exec /usr/local/cuda/bin/ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include compression_generic/ --nvtx-include compression_none/ \
  --launch-count 2 --kill no --metrics "$metrics" \
  --export /okc/logs/hw-native-counters01 -f \
  python3 proj_rsi/candidates/hw_compression_counters/driver.py \
  --json /okc/logs/hw-native-counters01.json

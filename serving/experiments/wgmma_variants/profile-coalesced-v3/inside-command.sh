#!/usr/bin/env bash
set -euo pipefail
ncu --version > /output/ncu-version.txt
# Small explicit metric set; do not use --set full. Validate metric availability
# with installed tool; unsupported names must fail, never silently become zeros.
metrics='launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block,sm__warps_active.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'
ncu --profile-from-start off --kernel-name 'regex:.*routed_nvfp4_down.*' \
  --launch-count 1 --replay-mode kernel --metrics "$metrics" \
  --clock-control none --cache-control all \
  --export /output/down-m128 python3 /probe/profile_down.py
ncu --import /output/down-m128.ncu-rep --page raw --csv > /output/down-m128.csv

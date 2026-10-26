#!/bin/bash
# One locked GPU session of CUPTI timing experiments (LATENCY_DBG: 1 no x, 2 no math, 4 no exceptions).
cd /okc
D=proj_rsi/cleanroom/latency
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu --format=csv
for dbg in 0 1 2 4 7; do
  echo "=== dbg $dbg"; LATENCY_DBG=$dbg python $D/prof.py --layers qkv_gate,pooler,router,o --ms 1,32 2>&1 | grep -v "^PACK\|^PLAN"
done
for smax in 2 4 8; do
  echo "=== smax $smax"; LATENCY_SMAX=$smax python $D/prof.py --layers qkv_gate,pooler,router --ms 1 2>&1 | grep -v "^PACK\|^PLAN"
done
echo "=== mb_load 216KB"; python $D/mb_load.py --kb 216 2>&1 | grep "cpa\|ld unroll"
echo "=== done"

#!/bin/bash
cd /okc
D=proj_rsi/cleanroom/latency
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu --format=csv
for dbg in 16 23; do
  echo "=== stamps dbg $dbg"; LATENCY_DBG=$dbg python $D/stamps.py 2>&1 | grep -v "^PACK\|^PLAN"
  echo "=== prof dbg $dbg"; LATENCY_DBG=$dbg python $D/prof.py --layers qkv_gate,router,o --ms 1,32 2>&1 | grep -v "^PACK\|^PLAN"
done
echo "=== done"

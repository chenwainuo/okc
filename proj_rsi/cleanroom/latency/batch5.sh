#!/bin/bash
cd /okc
D=proj_rsi/cleanroom/latency
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu --format=csv
echo "=== check producer P=1"; LATENCY_DBG=32 python $D/lab.py --notime --layers qkv_gate,router,o --ms 1,8,32 2>&1 | grep -v "^PACK\|Warning\|torch.sum"
for dbg in 32 96 224 39; do
  echo "=== prof dbg $dbg"; LATENCY_DBG=$dbg python $D/prof.py --layers qkv_gate,pooler,router,o --ms 1,32 2>&1 | grep -v "^PACK\|^PLAN"
done
echo "=== stamps dbg 48"; LATENCY_DBG=48 python $D/stamps.py --layers qkv_gate,o --ms 1 2>&1 | grep -v "^PACK\|^PLAN"
#!/bin/bash
cd /okc
D=proj_rsi/cleanroom/latency
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu --format=csv
for dbg in 16 23; do
  echo "=== stamps dbg $dbg"; LATENCY_DBG=$dbg python $D/stamps.py 2>&1 | grep -v "^PACK\|^PLAN"
  echo "=== prof dbg $dbg"; LATENCY_DBG=$dbg python $D/prof.py --layers qkv_gate,router,o --ms 1,32 2>&1 | grep -v "^PACK\|^PLAN"
done
echo "=== done"

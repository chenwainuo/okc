#!/bin/bash
cd /okc
D=proj_rsi/cleanroom/latency
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu --format=csv
SPLIT="qkv_gate=32:36x2;pooler=32:64x2;router=32:16x8"
SPLIT2="qkv_gate=32:72x4;router=32:8x4;unpooler=32:64x1"
echo "=== check default"; python $D/lab.py --notime 2>&1 | grep "m=\|rror"
echo "=== check $SPLIT"; LATENCY_PLAN="$SPLIT" python $D/lab.py --notime --layers qkv_gate,pooler,router 2>&1 | grep "m=\|rror"
echo "=== check $SPLIT2"; LATENCY_PLAN="$SPLIT2" python $D/lab.py --notime --layers qkv_gate,router 2>&1 | grep "m=\|rror"
echo "=== prof default"; python $D/prof.py 2>&1 | grep "m="
echo "=== prof $SPLIT"; LATENCY_PLAN="$SPLIT" python $D/prof.py --layers qkv_gate,pooler,router 2>&1 | grep "m="
echo "=== prof $SPLIT2"; LATENCY_PLAN="$SPLIT2" python $D/prof.py --layers qkv_gate,router 2>&1 | grep "m="
echo "=== prof dbg 6 (no math, no exceptions)"; LATENCY_DBG=6 python $D/prof.py --layers qkv_gate,o,pooler --ms 1,32 2>&1 | grep "m="
echo "=== stamps"; LATENCY_DBG=16 python $D/stamps.py --layers qkv_gate,o,pooler --ms 1,32 2>&1 | grep "m="
echo "=== done"

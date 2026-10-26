#!/bin/bash
cd /okc
P=proj_rsi/cleanroom/bandwidth/lab/sweep2.py
python $P qkv_gate 1,16,32 "S=3,D=4;S=3,D=4,diag=4;S=3,D=4,diag=8"
python $P o 1,32 "S=1,D=4;S=1,D=4,diag=4;S=2,D=4"
python $P pooler 1,32 "S=2,D=4;S=2,D=4,diag=4"
python $P unpooler 8,32 "S=8,D=4;S=1,D=4;S=2,D=4"
python $P router 1,8 "kind=rfma,S=16,D=4;kind=rfma,S=16,D=3;kind=rfma,S=8,D=4"
python $P router 16,32 "S=16,D=4;S=8,D=4"
python proj_rsi/tools/h100_bench.py proj_rsi/cleanroom/bandwidth/submission.py --official --reps 20 --json logs/bandwidth/v9.json

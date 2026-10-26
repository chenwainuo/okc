#!/bin/bash
# one lock acquisition, several sweeps (each ~10-20 s)
cd /okc
P=proj_rsi/cleanroom/bandwidth/lab/sweep.py
python $P qkv_gate 1,16,32 "S=3,D=4;S=4,D=4;S=7,D=4;S=7,D=6;S=7,D=8;S=4,D=8" 15
python $P pooler 1,32 "S=2,D=4;S=4,D=4;S=4,D=8;S=8,D=4" 15
python $P o 1,32 "S=1,D=4;S=2,D=4;S=2,D=8;S=4,D=4" 15
python $P unpooler 1,8 "S=4,D=4;S=8,D=4;S=8,D=8;S=16,D=4" 15
python $P unpooler 16,32 "S=1,D=4;S=2,D=4;S=2,D=8" 15
python $P router 1,8 "kind=rfma,S=8,D=4;kind=rfma,S=8,D=6;kind=rfma,S=4,D=4;kind=rfma,S=16,D=4" 15
python $P router 16,32 "S=8,D=4;S=4,D=4;S=16,D=4" 15
python $P qkv_gate 128,256 "S=7,D=12;S=4,D=12;S=8,D=12" 15
python $P pooler 128,256 "S=4,D=12;S=2,D=12;S=8,D=12" 15
python $P o 128,256 "S=2,D=12;S=1,D=12;S=4,D=12" 15
python $P router 128,256 "S=8,G=4,D=12;S=8,G=2,D=12;S=4,G=4,D=12;S=8,G=1,D=12" 15

#!/bin/bash
cd /okc
P=proj_rsi/cleanroom/bandwidth/lab/sweep2.py
python $P qkv_gate 16,32 "S=3,D=4;S=3,D=4,diag=1;S=3,D=4,diag=2;S=3,D=4,diag=3;S=3,D=4,diag=4;S=3,D=4,diag=8;S=2,D=4;S=3,D=6;S=4,D=4" 
python $P qkv_gate 1 "S=3,D=4;S=2,D=4;S=3,D=6;S=4,D=4;S=3,D=4,diag=4"
python $P pooler 1,32 "S=2,D=4;S=1,D=4;S=2,D=6;S=4,D=4"
python $P o 1,32 "S=1,D=4;S=1,D=6;S=2,D=4"
python $P unpooler 1,8 "S=8,D=4;S=16,D=4;S=4,D=4"
python $P unpooler 16,32 "S=1,D=4;S=2,D=4;S=1,D=6"
python $P router 1,8 "kind=rfma,S=8,D=4;kind=rfma,S=16,D=4;kind=rfma,S=16,D=3"
python $P router 16,32 "S=8,D=4;S=16,D=4"
python $P qkv_gate 128,256 "S=3,G=2;S=2,G=3;S=6,G=1;S=7,G=1;S=1,G=6"
python $P pooler 128,256 "S=4,G=1;S=2,G=2;S=8,G=1"
python $P o 128,256 "S=2,G=1;S=1,G=2"
python $P unpooler 128,256 "S=2,G=1;S=1,G=2"
python $P router 128,256 "S=8,G=4;S=4,G=4;S=8,G=2;S=4,G=8"

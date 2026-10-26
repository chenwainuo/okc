#!/bin/bash
cd /okc
P=proj_rsi/cleanroom/bandwidth/lab/sweep2.py
python $P unpooler 128 "S=2,G=1,KK=2;S=2,G=1,KK=2,diag=1;S=2,G=1,KK=2,diag=2;S=2,G=1,KK=2,diag=3;S=2,G=1,KK=2,diag=4;S=2,G=1,KK=2,diag=8;S=2,G=1,KK=2,diag=16;S=2,G=1,KK=2,diag=32;S=2,G=1,KK=2,diag=63;S=2,G=1,KK=1;S=2,G=1,KK=1,diag=3"
python $P qkv_gate 16,32 "S=3,D=4;S=3,D=4,diag=4;S=3,D=4,diag=3;S=3,D=8;S=3,D=8,diag=3;S=6,D=4;S=6,D=4,diag=3;S=2,D=8"
python $P qkv_gate 1 "S=3,D=4;S=3,D=8;S=6,D=4"
python $P o 1,32 "S=1,D=4;S=2,D=4;S=1,D=8;S=1,D=4,diag=4"
python $P pooler 1,32 "S=2,D=4;S=2,D=8;S=4,D=4"
python $P unpooler 8,32 "S=8,D=4;S=16,D=4;S=1,D=4;S=2,D=4"
python $P router 1,8 "kind=rfma,S=16,D=3;kind=rfma,S=16,D=4;kind=rfma,S=8,D=4"
python $P router 16,32 "S=16,D=4;S=8,D=4"
python $P qkv_gate 128,256 "S=2,G=3,KK=2;S=2,G=3,KK=1;S=3,G=2,KK=2"
python $P router 128,256 "S=2,G=8,KK=2;S=2,G=16,KK=2;S=2,G=8,KK=1"

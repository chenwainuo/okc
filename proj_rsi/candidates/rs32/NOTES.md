# rs32: rq (RS128 swap-AB register-A kernel) at m = 32 / 16, and a 64-row variant for o

Seed: pw34 @ 49b22d8, not rs128. The rq code is byte-identical in the two files, and pw34 carries the incumbent
m=16/32 logic (pw31 router, pw33 qkv gv geometries) that the bake has to beat. All numbers below are from the H100 box
(1980 MHz SM clock).

## Kernel changes (all inside the rq template)
- NT = 32 / 16: wgmma m64n32k16 / m64n16k16 specialisations; x box NT x 64 (4 / 2 KB per slot), SW128 B descriptor
  unchanged (SBO 1024), ND = NT/2 accumulators, epilogue loops NT/16 stmatrix.x4.trans, reduce box {64, NT}.
- FL bit 4 HALF: 64 W rows per CTA (one math WG; tiles 4h..4h+3 = a contiguous 6,144 B half of each 12,288 B chunk),
  grid x = 2 NB. Bit 2 + KS 1 = plain TMA store (no zeroing). Counters are indexed by blockIdx.x, so cnt = 4 NB.
- FL bit 5 EARLY: the W-lane thread inits the barriers and issues the first ST W chunks and x tiles before
  __syncthreads / setmaxnreg.
- Every instance: 168 regs (the launch-bounds cap; setmaxnreg 40/232), 0 local, 0 spill (ptxas -v).

## Ladder (CUPTI kernel us, cold L2; cuBLAS kernel in the same run)
| step | qkv m32 | o m32 | pooler m32 | unpooler m32 | qkv m16 |
|---|---|---|---|---|---|
| cuBLAS (+ splitK reduce z) | 15.68 (+2.05) | 8.32 | 25.06 (+2.05) | 25.02 | 15.42 (+1.89) |
| t1 NT=m, KS grid, ST 8 | 13.76 | 8.00 | 20.94 | 20.82 | 13.71 |
| ST 5 (best ring; 3/4 and 12 worse) | 13.25 | 7.74 | 20.29 | 20.27 | 13.06 |
| + L2 hints (hz) | 13.06-13.33 | 7.74-7.81 | 19.98-20.00 | 19.73-19.81 | 13.02-13.07 |
| + EARLY (hze) | 13.01 | 7.58 | 19.97 | 19.60 | 13.09 |
| qkv KS 6 (108 CTAs) | 12.90 | - | - | - | 12.51 |
| GS 2 / WM 2 | 14.72 / 14.08 | 8.64 / 8.34 | worse | worse | worse |
| HALF KS 1 (o, unpooler) | - | 7.97 | - | 19.90 | - |

INS (qkv m32, ST 5): stage loop ~985 clk, of which 534 is the wfull wait, i.e. W-arrival bound at ~3.1 TB/s steady.
Fixed terms: entry -> first wgmma median 2.05 us, epilogue ~0.55 us (1100 clk), CTA exit spread 10.5-12.3 us. NT=16
runs at the same speed as NT=32: the kernel is W-stream plus fixed-cost bound, so rq loses to gv at m=16 everywhere.

o m=128 (in-kernel): cuBLAS 8.90, s4s5hz 9.49, s4s5hze 9.02, h128s5-k1 8.80, h128s8e-k1 8.54 (graph 10.66 vs 10.98).
In the bake it reads 11.0 vs r6's tuned cuBLASLt at 10.8-10.9, so it is not picked.
HALF + KS 2 on o = 256 CTAs: not co-resident, so the zero handshake serialises (19 us). Dropped.

## Bake (after gx; incumbent = st["rq"] at m=128, else st["sel"], else _r6; one warm-up discarded, timed before and
after the variants, lower kept). Box final run, in-prepare us:
| shape | incumbent (before / after) | rs32 pick |
|---|---|---|
| qkv m32 | r6 18.27 / 18.18 | rq:t32s6hze-k6 14.88 |
| pooler m32 | r6 23.94 / 23.42 | rq:t32s5hze-k4 21.86 |
| unpooler m32 | r6 23.87 / 23.39 | rq:t32s5hze-k2 21.82 |
| o m32 | r6 9.98 / 9.89 | r6 (rq 9.79-9.86) |
| m16 all | r6 wins (qkv 14.66 vs 14.85; o 8.90 vs 9.66) | r6 |
| o m128 | r6 11.01 / 10.94 | r6 (h128s8e 11.01) |

## Harness (h100_bench, one locked session: pw34 then rs32; judge-style replay us)
decode32: qkv 21.31 -> 18.27, pooler 26.93 -> 24.88, unpooler 26.67 -> 24.59, o 13.44 / 13.82 (same r6 path).
On paths both files share, run-to-run drift is up to ~1 us (decode16 pooler 25.62 / 24.59; verify16 o 15.22 / 14.14).
rs32: 65/65 CHECK ok incl. --official; prepare 49 s on the box; file 254,567 B.

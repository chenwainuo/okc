# 01 — State, board, score arithmetic

## Board progression (all judge-confirmed, scored runs)
| candidate | job | score µs | speedup | what changed |
|---|---|---|---|---|
| r6-c12 | (baseline lineage) | 3428 | 1.41 | 12-bit fixed-width register-direct gv kernels for m ≤ 32, tuned cuBLASLt above |
| pw23 | 17f4484b97644673 | 3399.3 | 1.4211 | + split-K router at m=128 |
| pw31 | bbb732a78955425b | 3337.1 | 1.4476 | + pw27h small-shape self-zeroing split-K, pw29h wgmma router |
| pw33 | 8eeaca43bf43467f | 3332.1 | 1.4498 | + P2 shorter-chain qkv geometries at m=1/8/16 |
| **pw34** | **846e25d9df09433f** | **3293.9** | **1.4666** | + RS128 12-bit register-A split-K kernel at m=128 (pooler, qkv picked) — incumbent |
| pw34 @ 49b22d8 | 15514e8241db4d09 | 3295.5 | 1.4659 | prepare reorder; same within drift (board keeps 3293.9) |
| pw35 | (scored run launched 19:08 UTC, report → results/proj-rsi/jobs/pw35.scored.txt) | ? | ? | + rtk one-node router at m=128/256 (box: −2.4 / −0.5 µs) |

Judge drift between runs is ~3% overall and up to ±0.6 µs per shape; only within-run bake comparisons are evidence.
The board keeps the best scored run, so a worse scored run costs only the slot.

## The judge's per-shape table (run 846e25d9, H200, µs) with CUPTI kernel time and the non-kernel gap
| load | layer | m | ours | sglang | sol | kernel | gap | served by |
|---|---|---|---|---|---|---|---|---|
| decode1 | qkv | 1 | 9.06 | 14.34 | 7.87 | 9.4 | −0.3 | gv |
| decode1 | o | 1 | 5.09 | 6.78 | 3.50 | 4.2 | +0.9 | gv |
| decode1 | pooler | 1 | 13.53 | 21.18 | 13.99 | 14.2 | −0.7 | gv |
| decode1 | unpooler | 1 | 3.96 | 6.72 | 1.75 | 3.3 | +0.7 | gv (1024-row slice) |
| decode1 | router | 1 | 3.20 | 9.89 | 1.75 | 2.6 | +0.6 | rr (1024-thread) |
| decode16 | qkv | 16 | 10.28 | 14.05 | 7.93 | 10.1 | +0.2 | gv |
| decode16 | o | 16 | 5.52 | 6.93 | 3.56 | 5.2 | +0.3 | gv |
| decode16 | pooler | 16 | 14.17 | 21.39 | 14.06 | 14.5 | −0.3 | gv |
| decode16 | unpooler | 16 | 14.82 | 18.61 | 14.06 | 14.6 | +0.2 | gv |
| decode16 | router | 16 | 4.87 | 15.07 | 1.81 | 3.7 | +1.2 | gv |
| decode32 | qkv | 32 | 14.27 | 14.40 | 8.00 | 11.8 | +2.5 | **cuBLAS splitK (2 kernels)** |
| decode32 | o | 32 | 6.73 | 6.69 | 3.62 | 4.7 | +2.0 | **cuBLAS** |
| decode32 | pooler | 32 | 16.96 | 22.05 | 14.14 | 16.5 | +0.5 | gv |
| decode32 | unpooler | 32 | 16.86 | 18.75 | 14.14 | 16.3 | +0.6 | gv |
| decode32 | router | 32 | 5.17 | 15.34 | 1.86 | 4.3 | +0.9 | gv |
| verify1 | qkv | 8 | 9.14 | 14.10 | 7.90 | 9.3 | −0.2 | gv |
| verify1 | o | 8 | 5.15 | 6.48 | 3.53 | 4.4 | +0.8 | gv |
| verify1 | pooler | 8 | 13.65 | 21.12 | 14.02 | 14.4 | −0.8 | gv |
| verify1 | unpooler | 8 | 4.49 | 7.17 | 1.76 | 3.6 | +0.9 | gv (slice) |
| verify1 | router | 8 | 4.63 | 26.42 | 1.78 | 3.4 | +1.2 | gv |
| verify16 | qkv | 128 | 12.79 | 16.72 | 8.42 | 12.4 | +0.4 | **rq (RS128)** |
| verify16 | o | 128 | 7.71 | 7.58 | 3.99 | 5.5 | +2.2 | **cuBLAS** |
| verify16 | pooler | 128 | 18.16 | 25.44 | 14.64 | 18.8 | −0.6 | **rq (RS128)** |
| verify16 | unpooler | 128 | 18.98 | 19.98 | 14.64 | 18.8 | +0.2 | cuBLAS (rq ties: 22.66 vs 22.69 in-prepare) |
| verify16 | router | 128 | 9.42 | 16.54 | 2.21 | 8.4 | +1.0 | rsk2 + elementwise (2 nodes) → pw35: rtk |
| verify32 | qkv | 256 | 17.96 | 19.74 | 9.77 | 17.2 | +0.8 | cuBLAS |
| verify32 | o | 256 | 8.99 | 9.33 | 4.48 | 8.1 | +0.9 | cuBLAS |
| verify32 | pooler | 256 | 24.91 | 28.86 | 17.37 | 24.9 | 0.0 | cuBLAS |
| verify32 | unpooler | 256 | 24.56 | 24.51 | 17.37 | 25.0 | −0.4 | cuBLAS |
| verify32 | router | 256 | 11.33 | 17.79 | 2.68 | 9.8 | +1.5 | cuBLAS splitK + reduce + elementwise (3 nodes) → pw35: rtk |

Load sums (ours): decode1 34.84, decode16 49.66, decode32 59.99, verify1 37.06, verify16 67.06, verify32 87.75.

## Score arithmetic (proj_rsi/score.py)
score_us = 62 × geomean over the six loads of (sum of the load's five layer times). Speedup = geomean of SGLang/ours.
Marginal value of 1 µs saved in a load = score / (6 × load sum): **decode1 15.8, verify1 14.8, decode16 11.1, decode32 9.2,
verify16 8.2, verify32 6.3 score-µs**. To reach 3000 from 3294 we need ln-gain 0.094 (≈ −9%); top-10 needs ≈ −5.3%.

## Where the remaining score-µs are (realistic, not SOL)
| pool | shapes | realistic µs | score-µs | owner / track |
|---|---|---|---|---|
| m=128 rq rate | pooler/qkv/unpooler m=128: rq streams W at 2.7 TB/s on the H200 vs ~3.4 achievable | −3.3 / −4.3 / −3.6 | ≈ −90 | RS128 author (04) + P4 (07) |
| m=32 off cuBLAS | qkv 14.27 → ~11.0, o 6.73 → ~5.1 (gap of 2–2.5 recovered + bytes) | −3.3 / −1.6 | ≈ −45 | rs32 (05) |
| router one node | m=128 9.42 → ~8.5, m=256 11.33 → ~9.3 (box −0.5 / −2.4) | −0.9 / −2.0 | ≈ −20 (judge est.) | pw35 (06), scored run in flight |
| small loads | router/unpooler/o at m ≤ 16: fixed latency, one-node everything-in-flight kernels | −1 … −3 per load | −30 … −90 | latency school (08) |
| o m=128 | cuBLAS 7.71 incl. 2.2 gap; rq loses 2.1 in-prepare (short K: 8 rounds per CTA) | −2.5 | ≈ −20 | rs32's o decomposition (05) |
| m=256 | compute-bound (floor 17.4 pooler; cuBLAS ≈ 70–79% of peak); rq variants lose 3 µs; closed for rq | −13 if a new design hits 85% | ≈ −85 | nobody; see 04/10 |

Observations that matter: (a) every shape served by our own single-node kernel shows a non-kernel gap of −0.7…+0.5 µs,
while cuBLAS-served qkv m=32 / o m=32 / o m=128 carry +2.0…+2.5 µs and the multi-node router paths +1.0…+1.5 µs — moving a
shape to a one-node kernel recovers that gap for free (the transfer rule says fixed delays carry to the H200 unchanged);
(b) the gv kernel costs +2.3…+4.4 µs more at m=32 than at m=16 (issue-bound at four token groups), so the bake leaves qkv/o
m=32 on cuBLAS; (c) pooler at m=1 is already under the bf16 SOL because of the 12-bit pack — the small shapes are latency-bound.

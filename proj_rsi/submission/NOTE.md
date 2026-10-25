# muse_bf16_proj — where the work stands

Date: 2026-10-25. Account: chenwainuo.

This file is the note for the submission in this directory. `submission.py` is a byte-for-byte copy of the board kernel `proj_rsi/candidates/r6-c12/submission.py`. It was not posted again. The board already has it.

## The submission

| | |
|---|---|
| Problem | `muse_bf16_proj` |
| File | `proj_rsi/submission/submission.py` |
| Source | `proj_rsi/candidates/r6-c12/submission.py` |
| Scored job | `23cecdef1bb54c9e` |
| Result | 9/9 correct, speedup 1.409×, **3428.0 µs** |
| Rank when submitted | 58 |
| Content hash reported with that job | `5f5076c24c2ce71a71c95661d209a3a219ffa3a9131349ec20d1d8aeca79c07b` |
| 95% interval | about ±1.5%, roughly ±51 µs around 3428 |

Official displayed time is `4830.8 / speedup`. The published SGLang reference is 4830.8 µs. The leader is about 3000.8 µs (1.610×). A new point counts only when the top of its interval is under 3428, which means the point itself has to be under about 3377 µs.

Quota after the last check of the day: scored submits 1 used, 9 left. Off-board checks 15 used, 5 left. Reset is 00:00 UTC.

`muse_bf16_proj/submission.py` is still the starter. It was not replaced.

## What r6 actually does

`out = x[m, k] @ W[n, k].T`. Five layers, each weighted 62. The score is the geomean of six loads (decode and verify at m = 1, 16, 32, and the verify rows are really m = 8, 128, 256).

- m ≤ 32, except a few cuBLAS cases: a register GEMV (`gv`) or the router microkernel. Pooler m = 1 already sits on its byte floor (13.95 vs 13.99 µs).
- Fat BF16 layers (m = 128 and 256): one cuBLASLt nvjet kernel, algo 66, picked by a prepare-time search. Pooler m = 128 is `nvjet_sm90_tst_64x64_64x13_1x2_v_bz_coopB`, 20.3 µs, 100% of that profile. Split-K is 1. There is no partials kernel.
- Router, m > 32: the FP32 weight is split into BF16 hi and lo, concatenated on N, one BF16 cuBLASLt GEMM with FP32 output, then an add of the two halves. Profile at m = 128: GEMM 5.4 µs + split-K reduce 1.6 + add 1.6. Scored yours is about 10.3 µs. Floor is 2.2 µs.
- 12-bit lossless weight pack (`c12`) is used only for m ≤ 32.

r6 yours vs the published floor, verify loads, µs:

| shape | m=128 yours | floor | m=256 yours | floor |
|---|---:|---:|---:|---:|
| qkv | 15.4 | 8.4 | 17.4 | 9.8 |
| o | 7.2 | 4.0 | 9.6 | 4.5 |
| pooler | 20.3 | 14.6 | 24.5 | 17.4 |
| unpooler | 19.3 | 14.6 | 24.1 | 17.4 |
| router | 10.3 | 2.2 | 11.1 | 2.7 |

Pooler m = 128 moves 70.25 MB. At 4.8 TB/s that is 14.64 µs. 20.3 µs is 3.5 TB/s, 39% above the floor. The bytes are read once. A second read would be over the HBM peak.

## Tried, and where each one is

### On the board

r6-c12. 3428 µs. This is the submission.

### Closed before this session's Fable loop

- Register-A WGMMA. Six checks. Best pooler m = 128 raw about 57.7 µs against r6 at about 24 µs. The cause was 16 `ld.shared.b32` per K tile. Dead.
- r7 bulk `gemm_k`. Dispatch is off. A bulk copy of the working set hit about 3.8 TB/s. The full GEMM did not beat nvjet. `proj_rsi/candidates/r7-bulk/submission.py` is not called.
- Lossless 12-bit pack on the fat layers. Already a wash at m = 1: pooler m = 1 uses the pack and lands on the uncompressed floor (13.95 vs 13.99). Fewer bytes did not make a faster kernel. The swarm proposed it again for pooler m = 128. Not accepted.
- Direct-global `mma.sync` for m = 16 and 32. See exp 1 below.

### Fable loop, checked

| check | what | result | status |
|---|---|---|---|
| `2d9ba3f9ed534f34` | exp 1, first compile | died in prepare. `-std=c++17` overrides the host and ATen wants C++20 | fixed, not a kernel result |
| `140fb59bbf2c490b` | exp 1, bad A fragment | timing fail, rel L2 1.111 | layout wrong |
| `8f4c78fd88f5469c` | exp 1, swapped A order | timing fail, rel L2 1.008 | layout still wrong |
| `e094654ff1f346cb` | exp 1, PTX `m16n8k16` order, scalar global loads into `mma.sync`, m = 16 and 32 only | 9/9, **5611.9 µs**, 0.861×. Decode m = 32 pooler 95.8 µs | correct and slow. File `candidates/fable-e1/`. Do not retune |
| `0a2f1bbff69148c5` | exp 2, descriptor WGMMA | prepare `TypeError`. Extension cache `_SS = None` overwrote r6's shift table | renamed to `_SSEXT`. Not a kernel result |
| `b43dd09edfd442f2` | exp 2, `ss_kern`. BM 128, BN 256, BK 64, 4 stages, 128 B swizzle, 1D bulk B, TMA A, cluster split-K. Owns o, pooler, unpooler at m = 128 and 256 | 9/9, **4692.4 µs**, 1.029×. Pooler m = 128 kernel 65.8 µs, yours 76, r6 yours 20.2. About 1.2 µs per K tile plus ~35 µs of epilogue | numerically exact. Mainloop is fine. Epilogue is the miss. Not submitted |
| `ec5f1c56b86c4e0f` | same kernel, epilogue rewritten: compile-time fragment stores, reduce-scatter with `ld.shared.v4`, direct BF16 store | 9/9, **4975.9 µs**, 0.971×. Pooler m = 128 kernel **90.9 µs** vs a 25 µs continue bar. Extra time scaled with peer count (+25 µs at 7 peers, +2 µs at 1 peer) | correct and slower. `float d[128]` stayed in local memory. Kernel stopped |
| `84b9630256584be6` | cuBLASLt sweep on pooler and unpooler, m = 128 and 256. New axes: in-place reduction, split-K up to 16, 256 MB workspace | 9/9, **3414.4 µs**, 1.415×, interval 1.394–1.436×. Every shape kept the incumbent. 396–660 configs, finished in about 1 second, so cuBLAS rejected the new combos. Pooler m = 128 yours 19.7 vs r6 20.3 | noise. Interval covers 3428. Not submitted. File `candidates/fable-e4/` |
| `ca7ac81d552d4a70` | algo-66 sweep for qkv m = 16 and 32. The old search below m = 128 only tried layout-0 heuristics | 9/9, **3425.5 µs**, 1.410×, interval 1.389–1.431×. m = 16 stayed on `gv` (yours 10.7 vs 11.2). m = 32 switched to nvjet split-K 6: GEMM 8.6 µs + reduce 2.2 µs, scored yours **13.9** vs r6 14.21, floor 8.0 | the GEMM is near the floor. The second kernel puts the score back. Interval covers 3428. Not submitted. File `candidates/fable-e5/` |

### Assigned and not checked

- Router with K tripled (`x_hi·W_hi + x_hi·W_lo + x_lo·W_hi`). `x` is already BF16, so `x_lo` is zero. CPU check: rel L2 2.4e-6, same answer as r6's two-product GEMM. K = 2K is the same math with `x` stored twice. Expected gain is the 1.6 µs add, inside the noise band. Router at its floor on both fat loads, everything else left at r6, scores about 3308 µs, not 3000.
- L2 prefetch of the pooler weight during an earlier "idle" part of the layer. The judge times one `kernel_fn` per shape and flushes L2 before every replay. A prefetch inside the call is inside the timer. The weight is already read once.
- Replacing nvjet's B load with a 16 KB bulk copy from a pre-swizzled pack. nvjet is closed source. `ss_kern` already loaded B that way and lost.

## What would move the board

The interval top is under 3428 only if the point score is under about 3377 µs.

One shape, brought all the way to its published floor, does that:

- Router m = 128, 10.28 → 2.21 µs: score about **3361**, interval top about 3412.
- qkv m = 32, 14.21 → 8.00 µs: score about **3367**, interval top about 3418.

We do not have a kernel that reaches either floor. The qkv m = 32 GEMM is already at 8.6 µs. Fusing its 2.2 µs reduce ceilings near 3385 µs, and that interval still covers 3428.

Under 3200 µs needs four shapes on their floors. Under 3000 µs needs eight. Flooring every router and every m ≤ 32, and leaving qkv, o, pooler, and unpooler at m = 128 and 256 on r6, would score about 2723 µs. That is a ceiling across many shapes, not one check.

## Stop line

No further check until there is a mechanism that can move the score by more than about 51 µs. The fat nvjet layers, `ss_kern`, the router relayout, the in-place split-K sweep, the prefetch, and the qkv m = 32 sweep are closed. Five off-board checks and nine scored submits are still unused today.

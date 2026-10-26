# Pipeline diff: our m ≥ 128 pipelines vs a cuBLAS/DeepGEMM-grade mainloop

Written 2026-10-26 from code and the in-prepare ladders only (no judge run, no GPU). Sources: `candidates/pw24/submission.py`
(bg: bf16 SS wgmma at cuBLAS's 64x64 geometry), `candidates/pw19/submission.py` (pwt: decode warpgroup → bf16 A ring → SS
wgmma), `candidates/pw10/submission.py` (pwg: register-A wgmma, decode in the MMA warpgroup), `audits/pw10-ladder.md`,
`audits/m128-study.md`, `audits/cutlass-mixed-input.md`, `audits/wgmma-layout.md`, LOOP.md iterations 7-8, 11, 13, 17, 22-24;
DeepGEMM `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` + `csrc/jit_kernels/heuristics/sm90.hpp` +
`scheduler/gemm.cuh` (scratchpad `ref/DeepGEMM`); CUTLASS 3.9.2 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` (scratchpad `cut/y`).

Conventions. Stage = 64 k. Pooler m=128: N 4096, K 8192, 128 stages over K, W 67.1 MB bf16 / 50.3 MB 12-bit, x 2 MB.
Judge clock taken as 1.98 GHz; kernel µs = in-prepare µs − 4.5 (empty-launch/replay floor measured 4.9 in pw9, 4.0 used in
m128-study; I use 4.5 and quote both inputs). clk/stage = kernel µs × 1980 / stages-per-CTA. Smem port = 128 B/clk/SM.
m64n64k16 bf16 = 32 tensor clk, m64n128k16 = 64, m64n256k16 = 128 (2048 MAC/clk/SM).

New evidence produced for this audit (scratchpad `bgchk/`): pw24's BGCU kernel block extracted verbatim and compiled with
nvcc 12.8 `-arch=sm_90a -O3 -Xptxas -v`: all 7 bg instances 0 spills, 58 registers for the MMA variants, **no C7510
"wgmma.mma_async instructions are serialized" warning** (the pw24 critic recorded registers but not this check; pw19's
critic did check pwg/pwt: none there either). SASS of the +wgmma probe (`_Z2bgILi12ELi1ELi3ELi1E…`): the steady-state
consumer loop is 2 × `SYNCS.PHASECHK.TRANS64.TRYWAIT` (wfull, xfull), `WARPGROUP.ARRIVE`, 4 × `HGMMA.64x64x16.F32.BF16
R24, gdesc[UR8], R24` each preceded by 4 × `R2UR` (descriptors rebuilt in vector registers every stage; only the
prologue batch uses the uniform datapath), the last HGMMA with `gsb0`, `WARPGROUP.DEPBAR.LE gsb0, 0x1`, then
`SYNCS.ARRIVE.TRANS64.A1T0` ×2 + `SYNCS.ARRIVE.TRANS64.RED.A1T0` (remote), ~60 instructions per stage, no bar.sync. So
the bg +10.9 µs rung is **not** a compiler artefact and not instruction count; it is hardware-side (port / issue /
cluster), see §4.

## 1. The reference mainloops, as written

**DeepGEMM sm90_bf16_gemm.cuh** (one kernel, used for all bf16 GEMMs on Hopper):
- Warp roles: `kNumTMAThreads = 128` (one full warpgroup, only one elected lane issues) + `kNumMathThreads` = 128 for
  BLOCK_M ≤ 64, else 256 (`heuristics/sm90.hpp:192`). `__launch_bounds__(threads, 1)` (`:38`).
- Registers: `warpgroup_reg_dealloc<48>` for the TMA warpgroup, `warpgroup_reg_alloc<248>` (1 math WG) or 224 (2 WGs)
  (`:127-128, :149, :212`).
- Ring: `kNumStages = min((232448 − D tile − barriers) / (A+B per stage), 16)` (`sm90.hpp:149, :180`): at 64x64x64 bf16 that
  is **13 stages** (the same number cuBLAS's nvjet kernel name carries). Then the **stage merge** (`sm90_bf16_gemm.cuh:46-55`):
  if ≥ 10 stages, Normal GEMM, both K-major, one math WG → `BLOCK_K = 64 × (stages/5)`, i.e. 13 stages become **6 stages of
  128 k (32 KB each, 192 KB)** with 8 wgmmas per batch. The comment states the purpose: "for reducing the
  `warpgroup_wait<0>()` overhead".
- Consumer loop per stage (`:254-281`): `full_barriers[stage]->wait(phase)` (all 128 threads poll) → fence operands →
  `warpgroup_arrive` → BLOCK_K/16 wgmmas (SS, both descriptors advanced by uniform adds) → `commit` → fence →
  **`warpgroup_wait<0>()`** (`:278`) → `empty_barrier_arrive(stage)` = one `arrive` per warp from lane 0 (`:240`), or
  `kNumTMAMulticast` arrives (one per destination CTA) under multicast. Empty barrier count = `kNumTMAMulticast × math
  warps` (`:116`). Zero wgmma groups are left in flight across stages; the merge amortises the drain.
- Producer loop per stage (`:171-200`): `empty_barriers[stage]->wait(phase ^ 1)` → 1-2 TMA loads (A and B, each one
  `cp.async.bulk.tensor` per 64-wide swizzle atom) → `arrive_and_expect_tx(A+B bytes)`. No `__syncthreads` anywhere in
  either loop.
- Scheduling (`scheduler/gemm.cuh:186`): **persistent** grid = `num_sms` CTAs; `block_idx = iter × kNumSMs + blockIdx.x`,
  group-swizzled 8 or 16 blocks per group for L2 locality (`:18`). The producer runs ahead into the next tile while the
  math WG does the epilogue.
- Epilogue (`:299-386`): `tma_store_wait<0>` (previous tile's store), named barrier of the math threads, accumulators →
  `STSM` into a swizzled smem D tile, `tma_store_fence`, named barrier, one thread issues `SM90_TMA_STORE_2D` (or
  **`SM90_TMA_REDUCE_ADD_2D` when accumulating into C/D, bf16 or fp32**, `:382`), `tma_store_arrive`. The store is
  asynchronous and overlaps the next tile's mainloop.
- Multicast policy (`sm90.hpp:234`): **"Disable multicasting if only one wave exists"** — a cluster layout is rejected
  outright whenever `num_blocks ≤ num_sms`. Our m=128 grids are all one wave.
- Its cost model (`sm90.hpp:212-213`): L2 at `min(64 B/clk/SM, 6153 B/clk)` aggregate and L1/smem at 128 B/clk/SM; for
  64x64 on pooler m=128 the L2 term wins: 262 MB / 6153 = 42.6k clk ≈ **21.5 µs**, i.e. DeepGEMM's own model predicts
  cuBLAS-level time (not HBM SOL) for a 64x64 bf16 pipeline, because of L2→SM bytes.
- `cudaGridDependencySynchronize()` (`:131`): PDL, prologue overlapped with the previous kernel. Not something our
  harness exploits.

**CUTLASS SS warp-specialised mainloop** (`sm90_mma_tma_gmma_ss_warpspecialized.hpp`): `K_PIPE_MMAS = 1` (`:265`);
prologue issues 1 batch; steady state (`:534-552`): `consumer_wait` → fence → `warpgroup_arrive` → batch → `commit` →
**`warpgroup_wait<K_PIPE_MMAS>` (=1)** → `consumer_release(previous stage)`. One batch in flight, release lags one stage.
Load warpgroup 40 regs, MMA warpgroups 232 (cooperative: 2 math WGs on one 128-row tile; ping-pong: 2 math WGs on
alternate tiles).

## 2. Our kernels, as written

**pw24 bg** (`pw24/submission.py:1703-1817`): 64 W rows × 64 tokens, full K, no split-K, grid (m/64, N/64) = 128 CTAs at
m=128. Two rings of ST × 8 KB (W and x, 2D TMA SW128 boxes 64k × 32 rows / 64k × 64 tokens), ST = 12 (192 KB, 1 CTA/SM) or 6
(2 CTAs/SM). Two producer lanes in warps 4 and 5 (`:1726, :1736`), each blocking only on its own ring. Consumer = one
warpgroup: `wait wfull(s), wait xfull(s)` → fence → 4 × SS m64n64k16 → commit → `wait_group 1` → release(s−1) (`:1782-1783`):
local wempty arrive, remote wempty arrive via `mapa` when clustered, xempty arrive; wempty count 8/4 (`:1718`). CL=1 =
cluster (2,1,1) over the two token tiles, each CTA multicasting one 32-row half of W (`:1732`). 192 threads, 58 regs, no
setmaxnreg (the producer warps are not a whole warpgroup, so it cannot be used). Epilogue: `wait_group 0`, `bar.sync 1,128`,
32 × `st.shared.u16` per thread into a [64 tok][64 n] tile in x slot 0, bar, 16-B row stores; `barrier.cluster` at exit
(`:1794-1816`). Ladder (bg12c, pooler m=128): W-only 22.4 / W+x 25.6 / +wgmma 36.5 / full 38.2; bake cuBLAS 24.2 | bg12 28.9 |
bg6 33.0 | bg6c 37.9 | bg12c 38.3.

**pw10 pwg** (`pw10/submission.py:636-830`): 64·NWG/KSW W rows × NT tokens, K split over gridDim.y = KS as a cluster
(1,KS,1), grid (N/BN, KS) = 128 CTAs at BN 64 / KS 2. One ring of ST slots, each `[x tile NT×128 B][W chunk RTB×1536 B]`
= 22 KB at NT 128 (`:598`), one full barrier per slot (expect_tx = both, `:653-655`), one producer lane in a 5th warpgroup
that exits after its last issue; empty count = consumer warps (`:658`). Consumer warpgroup per stage: `decode(u)` = full
wait + 3 × `ld.shared.v4` + `pw_unit` (4 × pw_dec: 2 LOP/IADD + 4 PRMT each) + `pw_fix3` escape loop → `mma(u)` = fence +
4 × RS m64n128k16 (A from registers, B = x descriptor) + commit → **`wait_group 1`** → `release(u−1)` (fence A regs,
`__syncwarp`, lane-0 arrive) (`:743-747`). A double-buffered (A0/A1, 32 regs). setmaxnreg 40/232 only at NWG 2
(`:668, :679`). Epilogue: `pw_cbar`, fp32 partials parked in the drained ring, cluster sync, each rank sums its rectangle
over DSMEM (`pw_ldc`, `:782-798`), transpose through smem, 16-B row stores, cluster sync. Ladder (w1d4, 4 stages): bare
stream 17.2 / W-only 20.9 / W+x 25.4 / +decode 24.6 / +wgmma 30.1 / full 34.6; w1d8 32.3, w1d8z (wait_group 0) 38.7, wk2
31.3-31.7, torch 28.9-29.5. No ladder was ever taken at depth 8.

**pw19 pwt** (`pw19/submission.py:945-1130`): same geometry as pw10 (64 rows × 128 tokens, KS 2 cluster), three rings:
W (WST 8 × 6 KB, bulk), x (XST 4 × 16 KB, TMA), A (AST 4 × 8 KB bf16, written by a 4-warp decode warpgroup with
`stmatrix.x4` into the SW128 layout, `:1026`), 144 KB total; one producer lane polling both empty rings with
`mbarrier.test_wait` (`:985`). Decode warp per stage: wait wfull + wait aempty, LDS + pw_unit + fix3, 4 stmatrix,
`fence.proxy.async` **per thread**, lane-0 arrive on wempty, **arrive on afull from every one of the 128 decode threads**
(afull count NWD×32, `:974, :1034`). MMA warpgroup per stage: wait afull + wait xfull → fence → 4 × SS m64n128k16 →
commit → `wait_group 1` → release(u−1) on aempty + xempty (`:1070`). 416 threads, 90 regs. Ladder (tw64): W-only 19.9 /
W+x 30.4 / +decode 32.5 / +wgmma 39.5 / full 43.8.

## 3. Per-stage arithmetic (pooler m=128)

| kernel | tile / K per CTA | CTAs | stages per CTA | smem in flight | smem writes per stage | smem reads per stage | port clk/stage (128 B/clk) | tensor clk/stage | measured clk/stage (kernel = in-prepare − 4.5) |
|---|---|---|---|---|---|---|---|---|---|
| cuBLAS nvjet 64x64x64, 13 st, cluster 1x2 | 64 n × 64 tok, full K | 128 | 128 | 208 KB | 16 KB (TMA) | 16 KB (4 × SS A2+B2) | 256 | 128 | full 19.7 µs → **305** |
| DeepGEMM would pick (sm90.hpp) | 64x64, 13 → 6 × 128-k, 1 math WG, no cluster (1 wave) | 128 persistent of 132 | 64 (128-k) | 192 KB | 32 KB | 32 KB | 512 per 128-k = 256 | 128 | model 341 (L2 term) |
| pw24 bg12c | 64 n × 64 tok, full K, W multicast | 128 (64 clusters) | 128 | 192 KB | 16 KB | 16 KB | 256 | 128 | W-only 277 / W+x 326 / +wgmma **495** / full 521 |
| pw24 bg12 | same, no cluster | 128 | 128 | 192 KB | 16 KB | 16 KB | 256 | 128 | full 24.4 µs → **377** (no ladder) |
| pw10 w1d4 | 64 n × 128 tok, K/2, RS | 128 | 64 | 88 KB (≈ 44 KB effective: 2 slots pinned) | 22 KB (x 16 + W 6) | 22 KB (B 4 × 4 + LDS 6) | 352 | 256 | W-only 507 / W+x 647 / +decode 622 / +wgmma **792** / full 931 |
| pw10 w1d8 | same, 8 stages | 128 | 64 | 176 KB (≈ 132 effective) | 22 KB | 22 KB | 352 | 256 | full 27.8 µs → 860 (no ladder) |
| pw10 wk2 | 2 WGs each 2 k16 steps of every stage | 128 | 64 | 176 KB | 22 KB | 28 KB | 400 | 256 | full 26.8 µs → 829 |
| pw19 tw64 | 64 n × 128 tok, K/2, SS via A ring | 128 | 64 | 144 KB (x: 4 × 16, ≈ 2 effective) | 30 KB (x 16 + W 6 + A 8) | 30 KB (LDS 6 + A 8 + B 16) | 480 | 256 | W-only 476 / W+x 801 / +decode 866 / +wgmma **1083** / full 1216 |

Delivery rates (SM ingest counts multicast copies; L2 reads do not): cuBLAS 256 MB SM ingest / 19.7 µs = 13.0 TB/s
(192 MB of L2 reads if W is multicast → 9.7); bg12c W+x 256 MB / 21.1 = 12.1 (192 MB L2 → 9.1); bg12 full 256 MB / 24.4
≥ 10.5 (incl. wgmma); pw10 w1d4 W+x 178 MB / 20.4 = 8.7 (34 B/clk/SM); tw64 W+x 178 / 25.9 = 6.9. **bg's producer
structure (two rings, two lanes, 12 deep) already delivers at ≥ 90% of cuBLAS's rate; pw10's single 4-deep ring does not.**

Three ceilings sit on a 64x64 bf16 SS pipeline at this shape, all near cuBLAS's measured 305 clk/stage: smem port 256
clk/stage (16.5 µs at 128 stages), DeepGEMM's L2 term 341 clk/stage (21.5 µs), HBM 67 MB at 3.35-4.3 TB/s = 15.6-20 µs
(245-310 clk/stage). cuBLAS at 305 is at all three at once; that is why the judge shows cuBLAS at 20.3 vs an HBM SOL of
14.6 and why **a bf16 64x64 SS pipeline of ours cannot beat cuBLAS by more than noise even if it were perfect** (pw24's
day-1 premise was wrong before it was built). Beating cuBLAS needs a geometry that lowers all three: 12-bit W (HBM 50 MB:
11.7-16.7 µs), register-A (smem 22 KB per 64x64-equivalent instead of 32), and fewer x re-reads (x is 128 MB of the L2
traffic at BN 64; 64 MB at BN 128 or with x multicast).

## 4. Item by item

### (a) wgmma groups in flight; where wait_group sits relative to slot release

| | groups in flight | batch | wait | release |
|---|---|---|---|---|
| DeepGEMM | 0 across stages | 8 × m64n64k16 (128-k merged) | `wait<0>` every stage (:278) | after the wait, same stage (:281) |
| CUTLASS SS WS | 1 | 4 × (per 64-k) | `wait<1>` after commit (:548) | stage s−1 after the wait (:552) |
| bg | 1 | 4 × m64n64k16 | `wait_group 1` (:1782) | s−1 (:1783) |
| pwg | 1 | 4 × m64n128k16 RS, A double-buffered | `wait_group 1` (:743) | u−1 |
| pwt | 1 | 4 × m64n128k16 SS | `wait_group 1` (:1070) | u−1 (A and x slots) |

Ours is CUTLASS's protocol exactly; this is not where the difference is. What differs is **batch length relative to the
single warpgroup's bookkeeping**. With one batch in flight, after `DEPBAR.LE 1` returns exactly one batch (bg: 128 clk,
pwg: 256 clk) remains queued, and the warpgroup must get back to the next `HGMMA` within that window or the tensor pipe
idles. bg's window is 128 clk against ~100-150 clk of two barrier polls, descriptor rebuild (4 R2UR per HGMMA), three
arrives and `WARPGROUP.ARRIVE`; pwg's window is 256 clk against the full-barrier poll + 3 LDS.128 + 16 PRMT/LOP with
dependent chains + escape check + 4 RS issues (≈ 300-400 clk of latency for one warp with no cross-stage ILP). Evidence:
w1d8z (`wait_group 0` each stage) is +6.4 µs = +200 clk/stage over w1d8, the size of one drain-and-reissue bubble; wk2
(two warpgroups, each issuing 2 wgmmas of every stage) was the best variant despite doubling LDS and barrier traffic — a
second issuer hides the first's bookkeeping; DeepGEMM merges to 128-k batches precisely "for reducing the
warpgroup_wait<0>() overhead"; cuBLAS's kernel runs 64 k per stage (its name; its warpgroup count is not known) and
lands at 305 clk/stage, so if it is one warpgroup its inter-batch path is shorter than ours (uniform-register
descriptors, one barrier per stage), and if it is two they alternate.
Fix: (i) 128-k batches per barrier (8 wgmmas per fence/commit/wait); (ii) two batches in flight (`wait_group 2`, triple
A buffer, release u−2; needs depth ≥ 8); (iii) two consumer warpgroups on alternating whole stages (stage ping-pong,
separate accumulators summed once in smem at the end), which also halves decode per warpgroup.
**Experiment:** pw10 w1d8 (or bg12) with variants (i), (ii), (iii) and the +wgmma rung re-measured with `clock64()`
stamps per stage written per CTA (time in full-wait / decode / issue / wait_group / release). Expect the rung to fall from
+170 (pwg) / +170 (bg) clk/stage to ≤ +40.

### (b) Barrier protocol per stage per warp

| | consumer warp: waits | consumer warp: arrives | producer: per stage | `__syncthreads` / bar.sync in loop |
|---|---|---|---|---|
| DeepGEMM | 1 (full) | 1 (lane 0; ×multicast CTAs) | 1 empty wait + 1-2 TMA + 1 expect_tx | none |
| CUTLASS | 1 | 1 per warpgroup per dest CTA | same | none |
| bg | 2 (wfull, xfull) | 2-3 (wempty local, wempty remote, xempty), after `__syncwarp` | 2 lanes, each 1 wait + 1 expect + 1-2 TMA | none; `bar.sync 1,128` only in the epilogue |
| pwg | 1 | 1 (lane 0) after `__syncwarp` | 1 lane: 1 wait + expect + TMA + bulk | none |
| pwt decode warp | 2 (wfull, aempty) | 1 lane-0 (wempty) + **32 per warp = 128 per CTA on afull** + 32 `fence.proxy.async` | 1 lane polling two rings with test_wait | none |
| pwt MMA warp | 2 (afull, xfull) | 2 | | |

Cost evidence: pw4/pw7 skeleton probes put the per-stage mbarrier round trip at ≤ 0.5 µs total; bg's W+x rung (two rings,
two producers) reaches 326 clk/stage at 12.1 TB/s SM ingest, so two waits + three arrives per stage are affordable.
pwt's 128 arrives + 128 proxy fences per stage on one barrier word are serialised atomics in the smem pipe (order of
1-2k clk per stage if not aggregated; the hardware may aggregate a warp's arrives, unmeasured) and are the likely bulk of
tw64's +2 decode rung and part of its +7 wgmma rung. Fix for pwt: fence per thread, `__syncwarp`, lane-0 arrive (count 4);
but pwt is dominated by (c) anyway. For bg: merge the two full barriers into one with expect_tx = 16 KB (both producers
can complete_tx on one barrier; only one lane may post the expect_tx), saving one poll per stage; keep separate empties.
**Experiment:** none needed on its own; the clock64 stamps of (a) report the poll cost directly.

### (c) Smem traffic per stage vs 128 B/clk

Per 64-k stage per CTA: cuBLAS/DeepGEMM 64x64 SS: 16 KB TMA writes + 16 KB wgmma reads = 32 KB = 256 clk (84% of
cuBLAS's measured 305). bg: identical 32 KB. pwg (RS, 64 × 128 tok): 22 + 22 = 44 KB = 352 clk per 128 tokens (176 per
64-token-equivalent, 31% less than cuBLAS per unit of work; 38% of its measured 931). pwt: 60 KB = 480 clk (with 1 CTA/SM;
the x2 variant doubles it per SM). The SS m64n64k16 reads 4 KB per 32 clk, the whole port, during its batch; the
TMA writes of later stages must interleave with it; wider N dilutes this (n128: 6 KB per 64 clk = 96 B/clk; n256: 10 KB
per 128 clk = 80 B/clk) and register-A removes the A read entirely (B only: n128 RS 4 KB per 64 clk = 64 B/clk).
Evidence that the port binds the SS kernels: bg +wgmma = +168 clk/stage with a minimal, unserialised loop (verified
above) and the port minimum (256) already above the W+x period's TMA share; cuBLAS itself sits at 84% of the port; pwt
+216 clk/stage at 60 KB/stage. Evidence that it does not bind the register-A kernels: pwg uses 352 of its 931 clk, and
pw15 q1b (register-A at full K, one warpgroup, 2 CTAs/SM) still paid +10.3 µs on its wgmma rung with the same low port
load, which points at (a), not the port.
Fix: register-A (W in registers after decode, pw10's form) at 128 tokens per CTA; or 128-k merged stages do not help the
port, only (a). **Experiment (H100, 1 hour, settles this for every geometry):** a synthetic kernel with a resident 16 KB
smem tile and no TMA that loops `fence → 4 wgmma → commit → wait` for SS m64n64k16, SS m64n128k16, RS m64n128k16; then the
same with a concurrent TMA producer writing 16 KB/stage into a separate ring (never read). Report clk per batch. If SS n64
alone measures > 32 clk/instr or the TMA-concurrent run slows it by ≥ 50%, the port is the bg/tw rung and bf16 SS at
64x64 is closed for good.

### (d) x delivery: bytes in flight, L2 re-read volume, multicast operand

Bytes in flight per SM: cuBLAS 208 KB; DeepGEMM 192 KB; bg12 192 KB (two 12-deep rings, ~2 slots pinned → ≈ 160 KB
effective); pw10 w1d4 88 KB nominal, ≈ 44 KB effective (slot u−1 pinned until `wait_group 1` after mma(u), x and W in the
same slot so W cannot be released at decode); w1d8 ≈ 132 KB effective; tw64 x ring 4 × 16 KB with 1-2 pinned ≈ 32-48 KB
effective (W ring 48 KB, A ring 32 KB separate). Little's law at ≈ 1.5 µs loaded latency: 34 B/clk/SM (pw10's W+x rate)
needs ≈ 100 KB in flight; cuBLAS's 65 B/clk/SM needs ≈ 190 KB. The measured ingest rates line up with effective depth:
tw64 6.9 TB/s (≈ 40 KB) < pw10 w1d4 8.7 (≈ 44 KB, but 73% of each slot is x so the x bytes are what is short) < bg12c
12.1 (≈ 160 KB) ≈ cuBLAS 13.0 (≈ 190 KB).
L2 re-read volume (x is re-read once per n-block, independent of split-K): BN 64 → 64 × 2 MB = 128 MB (bg, pwg, pwt);
BN 128 → 64 MB (w2c2 / NWG 2, 32 n-blocks); x multicast over an n-pair cluster → L2 reads halve, SM ingest unchanged.
W: bf16 128 MB SM ingest at 64-token tiles (67 HBM + 61 L2 hits; 67 L2 reads if multicast); 12-bit 50 MB at 128-token
tiles. Totals: cuBLAS 256 MB ingest / 192 L2 reads; bg12 256 / 256; bg12c 256 / 192; pwg 178 / 178; pwg + x multicast
178 / 114; BN 128 + 12-bit 114 / 114.
Which operand to multicast: at 64x64 bf16 both W-over-token-pair and x-over-n-pair save 64 MB of L2 reads, so cuBLAS's
choice is not informative; for the 12-bit 64 × 128-token geometry x is 72% of the traffic and the only candidate, and
multicasting x keeps the two CTAs' HBM streams (different W rows) independent — only the L2-resident, low-latency operand
is shared, so pair skew costs little, whereas sharing W (bg12c) couples the two HBM streams and cost +9.4 µs. DeepGEMM
refuses any multicast on a one-wave grid (sm90.hpp:234); the judge measured +10 µs per cluster-size step (pw7, pw9) and
+9.4 for bg12c vs bg12; on the H100 this must be measured directly before any multicast design is built.
**Experiment:** a TMA-ingest-only probe (no MMA, no decode: wait full → arrive empty), sweeping ring depth {4, 8, 12} ×
x tile {64, 128, 256 tokens} × producer lanes {1, 2} × cluster {none, (1,2) x-multicast, (2,1) W-multicast}, reporting
B/clk/SM and µs; plus an empty-kernel launch with and without a cluster attribute (cluster launch cost on the H100). This
gives the delivery curve, the knee in bytes-in-flight, whether multicast time tracks L2 reads or SM ingest, and whether
the +10 µs cluster cost is the judge's or the hardware's.

### (e) Epilogue

DeepGEMM: STSM → swizzled smem D → one-thread TMA store (or TMA reduce-add), asynchronous, overlapped with the next
tile's mainloop by persistence; waits only before smem D is reused. cuBLAS: same family (no split-K at pooler m=128).
bg: `wait_group 0`, bar, 32 × 2-byte `st.shared` per thread (4-way bank conflicts, trivial), 16-B row stores, cluster exit
barrier: +1.8 µs including the drain (36.5 → 38.2). pw15 (no split-K): +1.0. pwg: cluster DSMEM reduce +4.5 (w1d4);
pw13's no-cluster last-arriver +8.2 (a dependent chain of L2 round trips on a one-wave grid where every CTA hits the tail
together); pwt: pwg's. At m=128 with one tile per CTA the epilogue cannot overlap anything in any of these designs,
including cuBLAS's; what makes ours 3-7 µs worse is purely the cross-CTA reduction. The reduction is needed only because
64 rows × 128 tokens at full K is 64 CTAs; it is the price of the geometry that has smem and L2 headroom (§3).
Fix: keep split-K but make the reduction a fire-and-forget **TMA reduce-add** (`cp.reduce.async.bulk.tensor.2d …add`
into bf16 `out`, the exact instruction DeepGEMM uses for `kWithAccumulation`, sm90_bf16_gemm.cuh:382): each CTA STSMs its
tile and issues one reduce-add, no cluster, no counter, no last arriver; `out` pre-zeroed by a 1 MB memset node (≈ 1 µs) or
by the KS=0 CTA's plain store gated by a per-n-block flag the KS>0 CTAs spin on before issuing their reduce-add (no chain,
one-way). Precision: pw27 simulated a 4-way bf16 split at 3.2e-3 relative L2 against the 1% rule; a 2-way split is one
extra rounding. **Experiment:** pwg w1d8 with the DSMEM epilogue replaced by STSM + TMA reduce-add (KS 2, no cluster
attribute at all); measure full − (+wgmma) on the ladder. Expect ≤ 1.5 µs vs 4.5, and the cluster launch gone from the
full path as a side effect.

### (f) Register budget per warp role; setmaxnreg

DeepGEMM: TMA WG 48, math 248 (1 WG) / 224 (2 WGs); CUTLASS: 40 / 232; both need it because a 64 × 256 accumulator is
128 regs and the stock builds spill otherwise. bg: 58 regs × 192 threads, no setmaxnreg (cannot: the producer warps 4-5
are not a full warpgroup), 0 spills. pwg: NWG 1 → no setmaxnreg, 64 accumulators + 32 A + decode temps under the
`__launch_bounds__(256,1)` cap of 255, 0 spills; NWG 2 → 40 / 232. pwt: 90 regs × 416 threads, 0 spills, no setmaxnreg.
No variant spilled and none is register-limited in occupancy at 1 CTA/SM (smem limits first). Register budget is not a
cause of either rung. It becomes one only under fix (a)(ii) (triple A buffer: +16 regs) or at NT 256 (128 accumulators),
where the pwg layout (producer = a whole 5th warpgroup) already allows 40/232.
**Experiment:** none; read `-Xptxas -v` of the (a) variants.

### (g) CTA count vs SM count; tile scheduling

DeepGEMM: persistent `num_sms` CTAs, swizzled tile order, the producer prefetches the next tile's stages during the
epilogue. cuBLAS: 128 CTAs at pooler m=128 (one wave, 4 SMs idle), persistent at larger grids. Ours: all non-persistent
one-CTA-per-tile grids: pooler/unpooler m=128 128 CTAs; bg at m=256 256 CTAs = two cold waves at 1 CTA/SM (bg6 39 vs cuBLAS
27.8); o (K 1024) 256 CTAs × 16 stages, ramp-dominated; qkv (N 2304): 36 n-blocks → bg 72 CTAs at 64x64 (55% of the
SMs; bg 28.7 vs cuBLAS 19.4) and pwg 36 × KS 2 = 72 or × 4 = 144 (12-tile second wave). Evidence: every m=256 and qkv
variant is far worse than its pooler counterpart in every bake, which is a tile-count/wave effect, not a mainloop one.
Fix: persistent CTAs (grid = 132, tile loop, ring and barrier phases carried across tiles, producer runs ahead) for the
two-wave shapes; for qkv, stream-K-style straddled tiles (pw28's P2 idea) rather than split-K 4.
**Experiment:** bg or pwg made persistent (tile loop around the mainloop, phases continuing) at unpooler m=128 (256 tiles)
and pooler m=256; expect −3 to −5 µs on each, zero change at pooler m=128.

## 5. Ranked: what is most likely behind the +5..11 µs wgmma rung

1. **Smem port saturation by SS 64x64 wgmma (bg +10.9, tw64 +7).** 32-60 KB per 64-k stage against 128 B/clk; cuBLAS's
   own 64x64 runs at 84% of the port. Loop verified minimal and unserialised. Fix: register-A at 128 tokens (22 KB per
   64x64-equivalent). Experiment: the resident-tile wgmma-rate probe of (c).
2. **One consumer warpgroup whose inter-batch bookkeeping exceeds the batch's tensor time (all three; pwg's +5.5 at
   depth 4, probably ≈ +2.3 at depth 8).** wait_group 1 leaves one batch queued; bg has 128 clk to get back to the next
   HGMMA, pwg 256 clk minus a 300-400 clk decode+poll+issue chain. w1d8z's +200 clk/stage shows the bubble size; wk2's win
   shows a second issuer hides it; DeepGEMM's merge exists for it. Fix: 128-k batches / wait_group 2 with triple A /
   two warpgroups on alternating stages. Experiment: (a)'s three variants with clock64 stamps.
3. **Ring slot pinned by the MMA (pwg w1d4, pw13, tw64): effective lookahead = depth − 2.** w1d4 → w1d8 −2.3 µs; tw64's
   x ring of 4 cost +5 on the W+x rung. Fix: depth ≥ 8 with x and W in separate rings, W released at decode. Experiment:
   the pw10 ladder at depth 8 and 10 (never measured; only the full path was).
4. **Cluster lockstep of the multicast W pair (bg12c only, +9.4 of the 14 µs gap to cuBLAS).** DeepGEMM refuses
   multicast on one-wave grids. All bg rungs were taken on the cluster variant. Experiment: the identical ladder on bg12
   (no cluster) — the first thing to run; bg12's full path is only 72 clk/stage behind cuBLAS, which bounds items 1-2
   for a bf16 SS kernel at ≈ 4.7 µs total.
5. **pwt's per-thread afull arrives + proxy fences + stmatrix (tw64).** 128 serialised arrives per stage on one word.
   Fix: lane-0 arrives; but the design loses on (c) regardless. Experiment: none beyond 1.

## 6. Ranked: what is behind the +3..5 µs x rung

1. **Too few x bytes in flight per SM (pwg: ≈ 44 KB effective at depth 4; tw64: 32-48 KB).** Rates 6.9-8.7 TB/s vs bg's
   12.1 at ≈ 160 KB and cuBLAS's 13.0 at ≈ 190 KB. Fix: bg's producer structure (two rings, two lanes, ≥ 8-12 deep) under
   the register-A consumer. Experiment: the ingest probe of (d), depth × tile sweep.
2. **x re-read volume: 128 MB at BN 64, whatever the split-K.** Even at cuBLAS's delivery rate 178 MB (12-bit + x) is
   ≈ 14-15 µs of L2 traffic, parity with cuBLAS's kernel, not a win; 114 MB (x multicast or BN 128) is the first total
   that leaves room. Fix: x multicast over an n-pair cluster, or BN 128 with KS 4 and the reduce-add epilogue (no cluster
   anywhere). Experiment: the ingest probe's cluster axis + the empty cluster-launch probe (is the judge's +10 µs per
   cluster step real on an H100?).
3. **A single in-order producer lane serving two rings (pw13 merged schedule; pwt's polling loop).** Head-of-line blocking
   of W behind a pinned x slot. bg's two lanes already fix it. Experiment: the probe's producer-lanes axis.
4. **x via TMA box 64 k × 128 tokens (16 KB per op) vs cuBLAS's 8 KB ops.** No evidence either way; the probe's tile axis
   covers it.

## 7. Consequences for the program

- bf16 SS at 64x64 (pw24, pw25/26 CUTLASS stock configs) is capped at ≈ cuBLAS by the smem port and L2 bytes; the CUTLASS
  sweep (pw26) can at best reach parity. The mixed-input RS fork (step B) is the only CUTLASS route with headroom.
- The in-house design with headroom is pw10's register-A 64 × 128-token geometry with: two rings ≥ 8 deep and two producer
  lanes (bg's producer), 128-k batches or stage ping-pong across two consumer warpgroups (a), TMA reduce-add split-K
  epilogue with no cluster (e), and x multicast only if the H100 shows cluster launch is cheap (d). Per-SM budget at
  judge rates: HBM 50 MB ≈ 11.7 µs (H200) / 15-16.7 (H100); L2 114-178 MB at 9-13 TB/s ≈ 9-20 µs; smem 352 clk × 64 stages
  = 11.4 µs; tensor 256 × 64 = 8.3 µs. The L2 line is the one that decides between parity and a 3-5 µs win.
- Order of H100 work: (1) ladder bg12 without the cluster and the empty cluster-launch probe (half a day; calibrates
  items 5.4 and 6.2); (2) the ingest probe (d) and the wgmma-rate probe (c) (one day; converts every "suspect" above into a
  number); (3) pw10 w1d8 ladder + the (a) variants with clock64 stamps; (4) reduce-add epilogue; (5) persistence for
  m=256/unpooler.

## 8. Measuring on the H100 rather than the H200

- HBM 3.35 vs 4.8 TB/s: every HBM-bound rung (W-only, the bare stream) is ≈ 25-30% slower in µs on the H100; the bf16
  pooler SOL on the H100 is ≈ 20-22 µs, i.e. cuBLAS's H200 time. Compare rungs as clk/stage differences, not µs, and
  treat a rung whose µs does not change between the two GPUs as not HBM-bound.
- Same 132 SMs, 50 MB L2, 228 KB smem, 128 B/clk port, same cluster hardware: items (a), (b), (c), (f) and the cluster
  launch cost transfer directly; (d)'s L2 egress ceiling transfers in B/clk if the clocks match. Record `nvidia-smi -q -d
  CLOCK` (SM and memory) with every number and lock the SM clock if permitted; the stage-clk arithmetic above assumes 1.98
  GHz.
- cuBLAS on the H100 picks its own kernel (possibly not nvjet 64x64x13); capture the kernel name with nsys or
  cupti before using it as the bar, and prefer our own bg12 and the probes as the within-run references.

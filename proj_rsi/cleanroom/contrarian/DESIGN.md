# contrarian — one m-adaptive packed-weight GEMM for muse_bf16_proj

Clean-room design (no look at the incumbent). Files: `submission.py` (prepare + kernel_fn, single file),
`sweep.py` / `test_red.py` (box-side tools), `bake_h100_shared.log` (last bake, see "Measured").

## Thesis

Every shape is the same kernel: the weights are re-encoded losslessly to 12 bits in a layout that decodes
*directly into tensor-core A fragments*, so W never touches shared memory as bf16; x is the B operand and is the
only thing staged in smem. Small m (decode loads, verify1) is then a pure weight stream with 4 ALU ops per weight
pair; large m (verify16/32) is the same stream feeding `wgmma` through descriptors. Split-K is a cooperative
reduction (no atomics on the main path) and the router is one kernel over the hi|lo bf16 pair (K = 16384).

## Encoding (lossless, 12 bits/weight)

* Per segment (a layer; the router has two: hi and lo) all weights are scaled by 2^j so that the 16 most
  common exponent codes land exactly on [112, 127]; j is undone by one FMUL in the epilogue (exact).
* Code = byte plane `B = [s d3 d2 d1 d0 M6 M5 M4]` + nibble plane `[M3..M0]`, with E = 112 + d.
  Decode of a pair (two bf16 in one register): `w = prmt(B-word, N-word, sel)` -> `[B1 Nb B0 Nb]`;
  `v = (w & 0x8000800F) | 0x38003800`; `v |= (w >> 4) & 0x07FF07F0`. **4 instructions per pair** (prmt, lop3,
  shf, lop3). This layout was chosen so a single 4-bit shift serves both halves of the register.
* Weights outside the window (E < 112 after scaling, ~1 in 10^4; 1268 in qkv, 2318 in pooler, 2254 in the
  router's lo plane) are stored as the all-zero escape code (decodes to +2^-15 in the scaled domain) plus an exact
  fp32 correction `w*2^j - 2^-15` in a CSR keyed by (64-row band, 256-wide k block). The kernel applies the
  corrections from the x tile already in smem for that k group, so the computation uses every weight at its own
  precision; probe rows come out exact (checked: "probe rows exact to 0.0e+00").
* Stream layout: `[64-row band][k64 group][warp][1536 B]` where a warp's 1536 B are three 512 B planes
  (`[plane][lane][16 B]`), i.e. each lane's 48 bytes are exactly its 32 weights of a 16x64 A block in
  `mma.m16n8k16` fragment order. Loads are three 16 B `cp.async` per lane per group, fully coalesced.
* Packing runs on the GPU in prepare with torch scatter (about 1.5 s for all layers); compile is ~40 s.

## Kernels

1. `gemm_kernel<NMF, NW, STAGES>` (mma.sync, MT = 8*NMF <= 32 in practice): CTA = NW warps = NW*16 n-rows;
   each warp decodes its 16 rows x 64 k per stage straight into A registers; x tile `[MT][64]` bf16 in smem with a
   128 B XOR swizzle, read by `ldmatrix`; fp32 accumulators in mma C layout; corrections accumulate into a small
   smem fp32 tile `C`; epilogue: fragments + C -> smem tile -> output or partials.
2. `gemm_wg_kernel<NMF, STAGES>` (wgmma, MT = 64..256): 2 warpgroups = 128 n-rows; decoded A is written with 16
   conflict-free `st.shared.b32` into a double-buffered SW128 K-major tile and both operands go to
   `wgmma.m64nMTk16` via descriptors (the same swizzle as the x tile, so `fence.proxy.async` + a named barrier
   is all the synchronisation). Corrections `red.add` into an fp32 scratch that the finaliser folds in.
   Hard-won ptxas facts: (a) register-A wgmma gets serialised (`WARPGROUP.DEPBAR` after every HGMMA) because
   ptxas reuses the A registers between k-steps; (b) a *runtime* scale-d predicate also serialises the whole
   stream (C7515) — peel the first step and use constant predicates; (c) `wait_group 1` must follow the commit
   in the same step. With all three fixed the SASS shows 4 HGMMA per `DEPBAR.LE 0x1`.
3. Split-K: each CTA writes its fp32 partial tile `[splits][MT][N]` with plain stores, arrives on a per-tile
   counter, spins until all `splits` CTAs of the tile arrived (the host guarantees `grid <= occupancy * SMs`
   via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, so all are co-resident), and reduces 1/splits of the tile
   with float4 loads; the last to exit resets both counters, so no memset node is needed and the graph replays
   cleanly. The router's fp32 output is written directly by the finaliser.
4. prepare() bakes: for every (layer, m-tile in 8..256) it times each valid (warps, splits, stages, kernel)
   candidate as a CUDA graph with an L2 flush, verifies it against a torch reference, times cuBLAS the same way,
   and keeps the fastest; cuBLAS wins -> `torch.mm` for that shape. m > 256 runs in 256-row slices.

## Measured (H100 box, shared with other agents' GPU jobs; graph-timed with L2 flush, same minute, same noise)

From `bake_h100_shared.log` (prepare's own timings; the box's empty-graph replay was 11-13 us while a foreign
soak process ran outside the lock, so absolute numbers are inflated; ratios within a row are the signal):

| layer    | mt  | torch us | mine us | ratio | config           |
|----------|-----|---------:|--------:|------:|------------------|
| qkv_gate | 8   | 23.30 | 20.93 | 1.11 | (4,8,8) mma      |
| qkv_gate | 16  | 23.52 | 21.50 | 1.09 | (4,8,8) mma      |
| qkv_gate | 32  | 23.74 | 27.97 | 0.85 | (4,4,8) mma      |
| qkv_gate | 128 | 26.72 | 45.06 | 0.59 | (8,4,4) wgmma    |
| o        | 8   | 11.78 | 12.13 | 0.97 | (4,1,8) mma      |
| o        | 32  | 11.94 | 14.66 | 0.81 | (4,1,8) mma      |
| pooler   | 8   | 35.68 | 30.21 | 1.18 | (4,4,8) mma      |
| pooler   | 16  | 36.03 | 30.59 | 1.18 | (4,4,8) mma      |
| pooler   | 32  | 36.16 | 31.39 | 1.15 | (4,4,8) mma      |
| pooler   | 128 | 37.15 | 51.42 | 0.72 | (8,4,4) wgmma    |
| unpooler | 8   | 11.10 | 12.45 | 0.89 | (4,8,8) mma      |
| unpooler | 16  | 33.57 | 30.05 | 1.12 | (4,1,8) mma      |
| unpooler | 32  | 33.95 | 31.49 | 1.08 | (4,2,8) mma      |
| router   | 8   | 17.76 | 15.74 | 1.13 | (4,64,8) mma     |
| router   | 32  | 18.82 | 17.82 | 1.06 | (4,64,8) mma     |
| router   | 128 | 19.65 | 27.23 | 0.72 | (8,64,4) mma     |

Correctness: the harness checked m = 1, 32, 128 on o and qkv_gate (relative L2 ~1e-4, probe rows exact); the
bake verifies every candidate config (all 30 shapes x all candidates) against a torch fp32 reference at 1e-2
(router 1e-4) — none failed in the final version. The `--official` harness run is queued behind other agents'
jobs at the time of writing (see the structured result for its status).

Verdict vs the bar: not there. The bar reads 12-bit at the HBM rate at m <= 8 (1.4x cuBLAS on qkv); this kernel
is 1.1-1.2x cuBLAS at m <= 16 and loses at m >= 32, so prepare will route most shapes to cuBLAS on the judge.

## Why it is not at the HBM rate yet (what I measured, in order of confidence)

1. **Issue-bound at small m.** SASS of the m=1 main loop: ~156 instructions per 1536 B group per thread, of which
   64 are decode, 12 memory ops, and ~80 pipeline bookkeeping (slot/address math, barriers, exception check).
   At 12.6 MB per layer that is ~10K warp-instructions per SM against a 3.8 us memory time: the SM cannot hide
   it at 8-16 warps/SM. Earlier versions were at 370/group (8-op decode + generic loaders); cutting to 156 moved
   the o layer from 0.86x to 0.97x cuBLAS. The remaining 80 need to go: TMA 1D bulk copies (one instruction per
   6 KB stage instead of 3 LDGSTS + address math per lane), compile-time slot addressing with an mbarrier ring,
   and dropping the per-group `__syncthreads` by giving each warp its own x tile at MT <= 8.
2. **Cooperative split-K on a shared GPU.** Every split > 1 config spins until the tile's CTAs are all resident;
   with other agents' kernels occupying SMs that wait is tens of microseconds here, which is why all wgmma and
   split configs look catastrophic in the table (o mt=256: 121 us). The judge's GPU is exclusive, but I could not
   measure that regime on this box, so the design decision is unverified. A safe alternative that keeps the
   benefit: partials + "last CTA of the tile reduces" (no spinning), at the cost of a serialised tail.
3. **wgmma path is correct but unmeasured in clean conditions.** The pipeline is now 4 HGMMA per wait (SASS
   verified); whether it reaches the memory floor at m = 128 (expected ~17 us for the pooler on H100) is open.

## What I would do next (in order)

1. Replace the per-lane `cp.async` ring with `cp.async.bulk` (1D TMA) + mbarrier per stage, one elected thread,
   compile-time slots; target <= 100 instructions per group per thread, 2 CTAs/SM x 8 warps.
2. Measure the wgmma path alone on the GPU (no sharing) at m = 128/256; if it is memory-bound, add a third
   warpgroup as a dedicated producer and switch the x tile to a 2D TMA box so the SW128 layout is free.
3. Switch the split-K to last-arriver reduce (deadlock-free under sharing); only keep the cooperative variant if
   measured to matter.
4. Bake on the judge: the current prepare already times cuBLAS vs each candidate and routes per shape, so any of
   the above lands incrementally without risking a regression.

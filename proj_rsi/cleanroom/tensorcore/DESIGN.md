# tensorcore — a TMA + mbarrier + wgmma GEMM over a 12-bit lossless packing of W

Clean-room design for `muse_bf16_proj` (out = x @ W.T, x bf16 [m][K], W bf16 [N][K]; router fp32 as a bf16 hi+lo pair).
Judge: H200.  Measurement box: H100 SXM (shared with other agents; absolute numbers below carry a ~10-13 us replay floor
that is 2-3x the usual one because the box was contended, so compare `yours` with `torch` within one run).

## 1. Orientation: W is the wgmma A operand, fed from registers

`out[m][n] = sum_k x[m][k] W[n][k]` is computed transposed, `D[n][m] = W[n][:] . x[m][:]`, so that

* **W is the A operand** (M = 64 weight rows per warpgroup, K-major).  wgmma allows A to come from registers in the
  `m16n8k16`-style fragment layout, so the packed weight stream is decoded straight into the fragment registers and never
  touches shared memory as bf16.  The packing is done in `prepare()` in exactly that fragment order, so every thread's 12
  bytes per k16 step are one contiguous, conflict-free 16-byte-vector read from the TMA-landed stage buffer.
* **x is the B operand** (N = tokens, K-major): x[m][k] row-major *is* the K-major B layout.  A 2-D TMA with 128-byte
  swizzle lands a `[MP tokens][64 k]` tile per stage in the canonical wgmma layout (SBO 1024, swizzle mode 1); out-of-range
  token rows are zero-filled by the TMA, which is how m = 1, 5, 7, 9, 33, 130 are served by the MP = 8/16/64/256 variants.
* The accumulator D[n][token] is `m64n{MP}` f32 in registers: MP/2 registers per thread, 128 at MP = 256.

## 2. Encoding: 12 bits per weight, exponent window + exact escapes

bf16 = sign(1) | exp(8) | mant(7).  The weights are N(0, 1/sqrt(K)) so the exponent lives in ~16 consecutive codes.
Per layer, prepare() picks the 16-code window `[base, base+16)` holding the most weights (base = 107..109 here) and stores

* a **byte plane**: `sign<<7 | mant7` (one byte per weight), and
* a **nibble plane**: the 4-bit exponent code `e4 = exp - base`,

laid out per (128-row n-block, 64-wide k-chunk) *unit* of 12288 bytes as `[3 x 16B vectors][256 threads]`
(vector 0-1: the 32 bytes of the thread's 4 k16 steps, vector 2: four 32-bit nibble words).  Decode of one bf16x2 register
pair is 5 instructions: `PRMT` with sign-replication (puts the sign at bit 15 and the mantissa at bits 0-6), a shift of the
nibble word, `LOP3` (exponent nibble | 0x60 into bits 7-14), `LOP3` merge, and one `HMUL2.BF16` by the constant
2^(base-96), which is exact.  So ~2.5 ALU ops per weight; at the HBM rate the SM has ~13 issue slots per weight.

Weights outside the window (6.7e-5 .. 1e-4 of them, |w| < 2^-20 or exactly 0) are stored as the placeholder
`+2^(base-127)` and listed in a per-unit CSR escape list `(unit, n_local, k_local, delta = w - placeholder)`.  After a CTA
flushes an n-block, its warps apply `ws[tok][n] += delta * x[tok][k]` for the escapes of its own units: exact
reconstruction, ~10 escapes per CTA.  The router stores hi and lo (lo = fp32 - hi, both bf16, each with its own window,
base 107 / 98) interleaved per unit (24576 B) and runs two wgmmas per k16 step into the same accumulator.

## 3. Pipeline

* Grid: 132 persistent CTAs (one wave), **stream-K** over the linearised (n-block, k-chunk) units: CTA c owns units
  `[c*U/132, (c+1)*U/132)` — balanced to one unit (qkv: 17-18 units each, 3% max imbalance), every CTA touches 1-2
  n-blocks.
* Producer: one elected thread issues per stage one 1-D `cp.async.bulk` (the 12288-byte W unit) and one 2-D
  `cp.async.bulk.tensor` (the x tile), both completing on the stage's `full` mbarrier (`expect_tx`).  Below MP = 128 it is
  a dedicated 9th warp (288 threads); at MP = 128/256 it is thread 0 of consumer warpgroup 0 (256 threads), because ptxas
  caps registers at 168 for > 256 threads and the MP = 256 accumulators need 190.
* Consumers: 2 warpgroups, 64 weight rows each.  Per unit: wait `full`, LDS 3 x 16 B, decode 4 k16 steps into a
  double-buffered A register set, `wgmma.fence`, 4 x `wgmma.mma_async`, `commit`, `wait_group 1` (one group in flight while
  the next is decoded; at MP = 256, groups of 2 steps, single-buffered, the other warpgroup covers the tensor pipe), then
  release the *previous* stage (`empty` mbarrier, count 2).
* Stages: 12 (MP <= 32, 13-16 KB each) .. 7 (MP = 128, 28 KB) .. 4 (MP = 256, 44 KB): 170-200 KB in flight per SM.
* Epilogue per n-block (after `wgmma.wait_group 0`):
  1. **escape fix-ups in registers**: at the start of the n-block the 256 consumer threads prefetch the block's escape
     list (<= 64 entries: index + delta, and the bf16 column `x[0..m)[k]` of each, 2 dependent L2 loads that overlap the
     pipeline fill) into smem; at the flush the quad owning accumulator row n adds `delta * x[tok][k]` into its own
     registers (a few dozen FMAs).  An n-block with > 64 escapes (never seen; P ~ 1e-4) falls back to scalar atomics.
  2. **2-CTA cluster pre-reduction (DSMEM)**: consecutive CTAs form clusters of 2; for an n-block held by both, the
     higher rank writes its fragment-layout accumulator tile with `st.shared::cluster.v4` into the leader's pipeline smem
     (free: a leader's shared block is always its last one), then arrives on the leader's mbarrier; the leader (which
     set a `ready` flag after its own `wait_group 0`) adds the slot into its registers.  Halves the global atomic traffic.
  3. accumulators -> per-warpgroup smem transpose (64 n x 8 tokens) -> `red.global.add.v4.f32` into the fp32 workspace
     `ws[256][N]` (256-byte contiguous runs).
* A tiny second kernel (`convert_kernel`, m*N/4 threads) turns `ws` into bf16 `out[m][N]` (fp32 copy for the router) and
  re-zeroes it, so the workspace is zero between calls: no memset node, no counters, no fences in the GEMM.  (The fused
  "last CTA converts" variant was measured and cost 7-10 us at m = 128/256: the `__threadfence` after 64-128 KB of
  atomics plus one CTA streaming the block serialised at the kernel tail.)

## 4. Measured (H100 box, graph replay with L2 flush, median of 10-20; us, box replay floor 12-14 us inside every number)

The box was shared with other agents the whole time (the GPU lock is not honoured by everyone: the empty-graph replay
floor was 12-14 us instead of ~4), so only differences within one run are meaningful.  `torch` is cuBLAS on bf16 W
(the router: two cuBLAS GEMMs on the hi/lo split), i.e. the same thing the judge's bar runs at m >= 128.

### 4.1 Ladder (qkv_gate, v23; `mode` = what the kernel does)

| m | torch | stream+decode+wgmma (mode 3) | +convert kernel (8) | +fp32 red.add flush (4) | +escapes (5) | full (0) |
|---|---|---|---|---|---|---|
| 16  | 23.0 | 19.4 | 20.6 | 21.6 | 22.5 | 23.9 (v22) |
| 128 | 22.6 | 21.7 | 23.6 | 25.7 | 27.2 | 30.1 (v22) |
| 256 | 27.3 | 26.3 | 29.0 | 34.3 | 36.1 | 42.6 (v22) |

Earlier ladder steps (v13): no-x-tile (mode 6) and no-wgmma (mode 7) are within 0.5 us of mode 3 at every m, i.e. the
mainloop is purely W-stream-bound: the x tile re-reads from L2 (37 MB at m = 128, 75 MB at m = 256) and the tensor work
are hidden.  **The whole gap to cuBLAS at m >= 128 is the split-K epilogue**: the fp32 `red.add` tail (every CTA flushes
BN x MP x 4 B = 64-128 KB at the same moment: 8.6-17 MB of L2 atomics, 2-8 us), the escape path (1-3 us: a chain of
dependent L2 loads at the start plus the register apply) and the convert kernel (1-3 us).

### 4.2 Design experiments (all measured, see git log)

* one wgmma per k16 step with `wait_group 1` after each: tensor pipe serialised, ~2x slower at m = 128 -> groups of 4
  steps per commit, double-buffered A registers.
* 384 threads + `setmaxnreg` (producer WG 40 / consumers 232): ptxas still capped the allocation at 168 and emitted
  C7515 "wgmma serialized" for every variant; 288 threads (dedicated producer warp, no setmaxnreg) is clean but also
  capped at 168, which spills at MP = 256; 256 threads with thread 0 as producer gives 255 registers -> used from MP = 64 up.
  A dedicated producer warp at MP = 128 (288 threads) was 6% slower than the integrated one.
* 4-7 vs 12 stages: no difference (the stream is latency-tolerant at 170-200 KB in flight).
* fused "last CTA per n-block converts" epilogue: 7-10 us at m = 128/256 (fence after the atomics + one CTA streaming
  the block) -> separate convert kernel, launched with programmatic dependent launch (saves ~0.5 us vs a plain launch).
* escape fix-ups by scalar atomics: 7-14 us (dependent-load chains at the tail) -> prefetched smem buffer + register apply.
* 2-CTA cluster DSMEM pre-reduction of the partials: measured +1-2 us *worse* at m = 16/128.  Analysis: with ~7 CTAs per
  n-block, a CTA pair (c, c+1) shares c's last and c+1's first block; the push only replaces a mid-stream (already hidden)
  flush, while both CTAs still end with a 64 KB atomic flush of their own last block, so the tail volume is unchanged and
  the DSMEM push, the slot sum and the cluster sync are pure overhead.  Kept in the code (`CLUSTER = 2`) but off.
* router grid 64 instead of 128 CTAs: slower (the 8 MB stream wants all SMs more than the atomics want fewer partials).

### 4.3 Harness (h100_bench.py --official, v23 policy: packed kernel for m <= 64 on the four bf16 layers, all m on the router; torch above)

(table from logs/tensorcore_bench6.log on the box, appended below when the queued run completes)

## 5. What next

1. **Kill the atomic tail.**  The partial-sum volume is G x BN x MP x 4 B, independent of how K is split, and every CTA
   flushes at the same moment.  Options, in order of expected payoff: (a) bf16x2 `red.add` partials for the *mid-stream*
   flushes only (hidden anyway, so no), (b) align the stream-K partition so that CTA pairs/quads share exactly one
   n-block and reduce through DSMEM *before* the tail (the pair's higher rank must have that block as its *last* block,
   which the current consecutive partition does not give), (c) a K-split of 2 with BN = 64 for m >= 128 (halves the
   tail bytes, doubles the x L2 traffic that is currently free at m = 128), (d) write partials with plain 16-byte
   stores to a [split][m][N] buffer and reduce in the convert kernel with all SMs (no atomics; 8.6 MB write + read).
2. Serve m = 128 with the packed kernel once the tail is < 1 us: the mainloop alone (mode 3) is already 4% faster than
   cuBLAS on the H100 at m = 128 and 32% faster on the H200 by the byte ratio.
3. The router could use a 13-bit lo plane with the exponent relative to hi's (4-bit j) instead of its own window
   (escape rate 6e-4 -> ~1e-5), and the K=1024 `o` layer could use BK = 128 units (fewer, larger TMA transactions).
4. Measure on an idle box: every number above carries a 12-14 us contended replay floor.

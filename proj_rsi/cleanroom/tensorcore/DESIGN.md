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

## 4. Measured (H100 box, graph replay with L2 flush; us)

(filled in below from `h100_bench.py` and the ladder; see the table at the end of this file)

## 5. What next

(see the end of this file)

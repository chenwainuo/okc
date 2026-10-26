# bandwidth — information-and-bandwidth design for muse_bf16_proj

Clean-room design (no reference to the incumbent's code). Thesis: every load of this problem is a weight stream
whose only free variable is bytes/weight, so re-encode the weights losslessly at ~12 bits, lay the stream out in
tensor-core fragment order so decode is a handful of integer ops per pair, and use one register-direct mma.sync
kernel family for every m from 1 to 256 — including m = 128/256, where cuBLAS still reads 16-bit weights.

## 1. The weights, as information

`W ~ N(0, 1/sqrt(K))` rounded to bf16. Per weight: sign (1 bit, uniform), mantissa (7 bits, uniform —
incompressible), exponent (8 bits but ~2.5 bits of entropy). Measured on the task's generator (seed 1234):

| layer | exponent window (16 codes) | escapes | rate |
|---|---|---|---|
| qkv_gate | base 107 .. 122 | 1268 / 18.9M | 6.7e-5 |
| o | 109 .. 124 | 819 / 8.4M | 9.8e-5 |
| pooler | 107 .. 122 | 2318 / 33.6M | 6.9e-5 |
| unpooler | 108 .. 123 | 3307 / 33.6M | 9.9e-5 |
| router (fp32 top half) | 107 .. 122 | 144 / 2.1M | 6.9e-5 |

So a 4-bit exponent code covers all but ~1 weight in 14,000. The escapes are the tiny weights (|w| < 2^-12 sigma);
the large side is Gaussian and never escapes. A 4-bit code + sign + 7-bit mantissa = **12 bits/weight, 25% fewer
HBM bytes than bf16**, and the decode needs no table.

Entropy-coding the exponent (~2.5 bits) would give ~10.5 bits/weight (−12% more bytes) but costs a variable-length
decode that cannot be done in 5 integer ops per pair; it is the obvious next step only if the kernels are shown to be
purely HBM-bound with ALU slack (see §6).

### Encoding (prepare, GPU, lossless)

* bf16 bits = `s e[7:0] m[6:0]` → `SM byte = s<<7 | m`, `E nibble = e − base` (base = argmax of the 16-wide
  exponent histogram window, per layer).
* Escapes (e outside the window) are stored as SM=0, E=0 — a placeholder that decodes to `+2^(base−127)` — and
  listed per output column as `(k, w_true, w_placeholder)`. The epilogue fixes them exactly:
  `acc = fma(x, w_true, fma(−x, w_placeholder, acc))`. For a probe row (one-hot x) the accumulator is exactly
  `x·w_placeholder` before the fix, so the fix leaves exactly `x·w_true`; the checker reports probe rows exact to
  0.0 on all five layers.
* Router (fp32): the top 16 bits (= bf16 truncation) go through the same 12-bit path, the low 16 mantissa bits are
  stored verbatim in a third plane: 28 bits/weight, decoded to the exact fp32 for m ≤ 8 (FMA), or to the
  bf16 hi + bf16 lo = rn(w − hi) pair for m > 8 (two mma.sync per fragment, fp32 accumulate), as SGLang does.

### Stream layout: fragment order

The stream is written in `mma.sync.m16n8k16` B-fragment order. For one (64-column block, 16-k step) "warp step"
of 1024 weights, lane t holds columns `8j + t/4` (j = 0..7) at k = `2(t%4) + {0,1,8,9}`:

```
[ SM plane: 32 lanes × 32 B ][ E plane: 32 lanes × 16 B ][ LO plane (router only): 32 lanes × 64 B ]
```

so each lane's own 16-byte loads are exactly its own fragment bytes (no shuffles, no cross-lane dependency, so the
per-warp cp.async ring needs no block barrier). Decode of one bf16x2 fragment register:

```
t  = prmt(SM word, 0x9180)        // byte, signrep(byte) per halfword
u  = t & 0x807F807F               // sign at 15, mantissa at 0..6
ex = (E word << / >> c) & 0x07800780   // the 4-bit code of each half, at bit 7
b  = u + ex + (base<<7 | base<<23)     // one IADD3
```

5 integer ops per two weights (2.5 ops/weight). Budget check for the hardest case (m = 128 on the judge, 12-bit
stream at ~3.7 TB/s): 19M weights in 7.6 µs = 11 lane-ops/weight of issue budget per SM, decode uses 2.5.

## 2. Kernels

| m | kernel | tile / CTA | grid | B operand | A operand |
|---|---|---|---|---|---|
| 1..32 (bf16 layers, router 9..32) | `k_small<MT=1,2>` | 64 cols × K/S | NBLK × S (cluster S along K) | decoded in registers, straight from the lane's own smem ring bytes | x tile per warp via cp.async + ldmatrix |
| router 1..8 | `k_rfma` | 64 cols × K/8 | 4 × 8 | decoded to exact fp32 | x staged as fp32 in smem |
| 33..256 | `k_big<MT=4,8,16>` | 16·MT rows × 128 cols × K/(S·G) | NBLK × G × S | decoded once per step into a shared B tile (uint2 per lane per n-tile) | x tile ring via cp.async (swizzled), ldmatrix |
| > 256 | torch.mm | | | | |

* **k_small**: 8 warps split the CTA's K range; each warp streams its contiguous slice of the packed stream into a
  private D-stage cp.async ring and keeps the full m × 64 accumulator (MT·8·4 regs). Warps reduce through shared
  memory, then the S CTAs of a cluster reduce through DSMEM (each CTA owns 64/S columns), apply the escape fix and
  store bf16. Cluster sizes: qkv 36×7 = 252 CTAs, pooler 64×4, o / unpooler 128×2, unpooler-sliced 16×8, router 4×8.
* **k_big**: warps are a 4 (m) × 2 (n) grid; each k16 step all 8 warps decode one n-tile each of the two packed
  64-column blocks into the B tile (double-buffered), one `__syncthreads` per step, x via a 12-stage ring. Cluster
  split-K through DSMEM; for the router an outer split G = 4 is reduced through a ticket counter: every cluster
  writes its fp32 slice to a scratch buffer, the last arriver (atomic ticket) sums all G slices in order — no memset
  node, no spinning, deterministic.

## 3. Measured (H100 SXM box, harness graph replays, cold L2)

(filled in below from `logs/bandwidth/*.log`)

## 4. Why it should beat the bar on the judge

## 5. Open issues / what I would do next

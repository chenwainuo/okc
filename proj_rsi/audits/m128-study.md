# m=128 study: what reaches ~17 judge-µs on pooler m=128

All times are judge-µs; a stage is 64 k.

## The binding constraint is L2→SM delivery, not decode or HBM

- **Volumes.** Packed W is 50 MB. x is re-read (N/BN) × (m/TM) × TM·K·2 B in total, where TM is the tokens per CTA.
- **Reference rates.** cuBLAS's nvjet 64×64, 13-stage kernel moves 262 MB at 10.7 TB/s. Our skeletons reached 7.0–8.7 TB/s.
- **What 17 µs needs.** At 10 TB/s it allows about 170 MB of L2→SM traffic. The HBM floor is 50 MB / 4.2 TB/s (bulk probe) ≈ 12 µs plus about 1.5 µs of ramp.
- **Split-K has to go.** Our best paths paid about 4.5 µs of cluster epilogue (KS 2). The no-cluster last-arriver cost 8.2. cuBLAS uses no split-K.
- **Decode is not the problem.** Register-A decode was hidden in every ladder (+0 to +2).

## (b) Recommended: cuBLAS geometry, 12-bit W, register-A wgmma, W multicast

**Grid and tiles.**
- Tile is BN 64 W rows × TM 64 tokens, the full K per CTA, no split-K.
- 64 n-blocks × 2 token tiles = 128 CTAs at 1 CTA/SM.
- The two token tiles of an n-block form a cluster (2,1,1). Each CTA issues half of every packed W chunk with `.multicast::cluster`, so W leaves L2 once.
- This is the only cluster, and it is the shape cuBLAS uses. There is no cluster reduction.

**Warp roles and rings.**
- **Producers:** two lanes in separate warps, one for the W ring and one for the x ring, so there is no head-of-line blocking (pw13).
- **W ring:** 12 × 6 KB packed slots, released right after decode.
- **x ring:** 12 × 8 KB TMA SW128 tiles, released after wait_group(1).
- **In flight:** 168 KB per SM, which matches cuBLAS's depth.
- **Consumer:** one warpgroup decodes with r6's dec1 into register A, double-buffered (pw10 wk2's decode), then issues 4 × wgmma m64n64k16.
- **Epilogue:** bf16 is stored directly from the D layout, 64 × 64 per CTA. There is no reduction.

**Per-stage budget (128 stages; 17 µs gives 263 clk per stage).**

| Resource | Cost | Fits? |
|---|---|---|
| Tensor (wgmma) | 4 × 32 clk = 128 clk | yes |
| Decode issue | ~45 instr/thread ≈ 50 clk | yes |
| Shared memory | fills 14 KB + B reads 8 KB + W LDS 6 KB = 28 KB = 219 clk | yes, tight |
| L2→SM | W 3 KB (multicast halves it) + x 8 KB = 11 KB/SM/stage ≈ 10.5 TB/s | yes, at cuBLAS's level |

Total L2→SM is 50 + 128 = 178 MB. That closes only at cuBLAS-grade delivery.

- **Expected:** 17–19 µs.
- **Main risk:** our skeletons never exceeded 8.7 TB/s, which would give about 21 µs. The fix is depth plus two producers, untested together.
- **Build cost:** 3–4 days, including a ladder check.

**Variant (b2): x multicast.** A (2 tokens × 4 n) cluster of 8 also multicasts x, cutting x to 32 MB and total L2→SM to about 82 MB. That makes the kernel HBM-bound at about 14 µs. The risk is cluster-8 launch and placement: pw9 measured +10 µs per cluster step. Bake it only after (b) works.

## (a) CUTLASS mixed-input as the transform warpgroup

- **Escape-free codec.** Store 8-bit (sign + mantissa) and 4-bit (exponent index) planes. Both are CuTe-supported widths. The 0.01% escapes (about 3.4 K on pooler) become a sparse correction pass: out[:, n] += x[:, k] · ΔW, about 0.4 M FMA, around 1 µs.
- **Custom converter.** It needs a LUT for the exponent plane: a forked NumericArrayConverter, but not a forked collective.
- **Expected:** about 16–18 µs, since CUTLASS's pipeline is cuBLAS-grade.
- **Risks:** embedding about 1.5 MB of CuTe headers, and 30–90 s of compile per configuration within the judge's 10-minute run. The 1 µs correction kernel eats into the gain.
- **Build cost:** 5–6 days.

## (c) What the leaderboard cluster suggests

- **The m≥128 halfway-to-SOL arithmetic is not specific to packed weights.** A bf16 kernel at 4.0–4.2 TB/s on 67 MB is about 16–17 µs. Seven teams at about 1.55× may simply have a better m=128/256 GEMM than cuBLAS's 20.3 (3.3 TB/s), with no codec.
- **Cheapest test of that reading:** (b) with bf16 W (no decode, TMA SS wgmma, cuBLAS geometry, W multicast). It shares 90% of (b)'s code, its ladder reaches W-only first, and it isolates whether delivery or decode is our gap.

## Recommendation (9 days to 2026-11-04)

| Days | Work |
|---|---|
| 1–2 | Build (b) in its bf16 form first (no decode): two producers, 12 stages, cluster-2 W multicast, direct store, with a W-only / W+x / full ladder. Kill if W+x > 20 in-prepare. |
| 3–4 | Swap in the 12-bit register-A decode. Bake both the bf16 and 12-bit paths on pooler, unpooler and qkv at m=128 and 256. |
| 5–6 | Try (b2) x multicast if W+x is still the limit; otherwise tune stages and swizzle. |
| 7–9 | Hardening and the final scored run. Keep (a) as the fallback only if (b) cannot pass 9 TB/s by day 4. |

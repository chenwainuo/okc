# m=128 study: reaching ~17 judge-µs on pooler m=128

All times are judge-µs; a stage is 64 k.

**The binding constraint is L2→SM delivery, not decode or HBM.**
- x is re-read (N/BN)·(m/TM) times. cuBLAS's nvjet (64×64 tiles, 13 stages, no split-K) moves 262 MB at 10.7 TB/s; our skeletons reached 7.0–8.7.
- At 17 µs and about 10 TB/s the budget is about 170 MB. The HBM floor for 50 MB of packed W is about 12 µs at the 4.2 TB/s bulk rate, plus 1.5 µs of ramp.
- Split-K cost 4.5 µs (cluster DSMEM) or 8.2 µs (last-arriver), so it must go. Register-A decode was hidden in every ladder (+0 to +2).

## (b) Recommended: cuBLAS geometry, 12-bit W, register-A wgmma

**Design.**
- Tile 64 W rows × 64 tokens over the full K; 128 CTAs at 1/SM; no split-K.
- Cluster (2,1,1) over the two token tiles. Each CTA multicasts half of every packed W chunk, so W leaves L2 once. This is cuBLAS's own cluster size, with no reduction.
- Two producer lanes in separate warps, one for W and one for x (no head-of-line blocking).
- W ring 12 × 6 KB, released at decode. x ring 12 × 8 KB TMA SW128, released after wait_group(1). 168 KB in flight per SM.
- One consumer warpgroup decodes with dec1 into double-buffered A and issues 4 × m64n64k16.
- bf16 stored straight from D.

**Per-stage budget.** 128 stages in 17 µs is 263 clk each.

| Resource | Cost |
|---|---|
| Tensor | 128 clk |
| Decode | about 50 issue clk |
| Shared memory | 28 KB (fills 14, B reads 8, W LDS 6) = 219 clk: tight |
| L2→SM | 3 KB W (halved by multicast) + 8 KB x per SM-stage = 10.5 TB/s: cuBLAS's level |

Total L2→SM traffic is 178 MB.

- **Expected:** 17–19 µs.
- **Risk:** our pipelines never passed 8.7 TB/s, which would give about 21 µs. Depth plus two producers is untested.
- **Cost:** 3–4 days.

**(b2) x multicast as well.** A cluster of 2 tokens × 4 n cuts x to 32 MB and total traffic to 82 MB, making the kernel HBM-bound at about 14 µs. The risk is cluster-8 launch: pw9 measured +10 µs per step. Build it only after (b) works.

## (a) CUTLASS mixed-input as the transform warpgroup

- Make the codec escape-free: an 8-bit plane (sign + mantissa) plus a 4-bit exponent-index plane, both CuTe widths, with a custom LUT converter (no collective fork).
- The 0.01% escapes (about 3.4 K) become a sparse correction pass of about 1 µs.
- **Expected:** 16–18 µs.
- **Risks:** about 1.5 MB of embedded headers and 30–90 s of compile per config inside the 10-minute run; the correction eats part of the gain.
- **Cost:** 5–6 days.

## (c) What the leaderboard cluster suggests

The halfway-to-SOL arithmetic does not need a codec. A bf16 GEMM at 4.0–4.2 TB/s on 67 MB is 16–17 µs against cuBLAS's 3.3 TB/s. Seven teams at about 1.55× may simply have a better m≥128 bf16 pipeline. The bf16 form of (b) (TMA SS wgmma, no decode) tests this with about 90% shared code.

## Recommendation (to 2026-11-04)

| Days | Work |
|---|---|
| 1–2 | Build (b) bf16-first, with a ladder (W-only / W+x / full). Kill if W+x > 20 in-prepare. |
| 3–4 | Add the 12-bit decode. Bake bf16 and 12-bit on all four layers at m=128/256. |
| 5–6 | Add (b2) if delivery still binds; otherwise tune stages. |
| 7–9 | Harden and run the final scored check. (a) only if (b) is still below 9 TB/s on day 4. |

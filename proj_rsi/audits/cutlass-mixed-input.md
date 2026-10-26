# Step B plan: r6's 12-bit code as the CUTLASS A-operand transform

Source read: CUTLASS v3.9.2, `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` and `detail/collective/mixed_input_utils.hpp`.

## How the stock mixed-input mainloop works

**Pipeline.**
- The producer warp TMA-loads the narrow A (plus optional per-group scale/zero tensors) and the 16-bit B into an N-stage ring.
- Each consumer warpgroup, per k-block:
  - `copy_tensors_MK`: LDSM of A into registers, 2 k-blocks ahead;
  - `dequantize_A_kblock`: convert to bf16, 1 ahead;
  - register-A (RS) wgmma.
- This is pw10's structure (decode in the MMA warpgroup), but with CUTLASS's pipelining, setmaxnreg and schedulers.

**Supported transforms.**
- Modes: DirectConvert, ConvertAndScale, ConvertAndScaleWithZero.
- A types: int8, uint8, int4, uint4 (bit-trick converters), fp8 e4m3/e5m2, and int4 with a LUT scale. B is bf16/fp16.
- A must be K-major, and there is **one** narrow A tensor.

## Option (i): two planes, 8-bit + 4-bit

**Data.**
- `P8[n][k]` is r6's low byte (exponent LSB + mantissa7).
- `P4[n][k]` is sign + the 3-bit exponent offset.
- Both are plain K-major planes: 12 bits/weight, 50 MB on pooler.

**Custom code: a forked collective (about 300 lines), not just a functor.**
1. A second TMA stream and smem buffer for P4. This mirrors the zero-tensor plumbing, but at full M×K size.
2. A second smem→register copy in the same CuTe partition as A.
3. A new dequantize: `bf16 = sign << 15 | (off + base) << 8 | P8`. That is r6's dec1 arithmetic (PRMT/LOP3) on CuTe's fragment order.
4. A row-major plane packer.

**Escapes (about 0.01%).** Fixing them inside CuTe fragments is risky. Instead:
- Decode escapes as base+0.
- A post-GEMM kernel adds Σ x·ΔW per column's escape list, with ΔW = W − W_dec exact in fp32.
- About 3.4 K entries × 128 tokens on pooler, about 1–1.5 µs.
- One-hot rows stay exact, and random rows take one extra bf16 rounding, within tolerance.

## Option (ii): 16-bit A via a pre-decode into smem

This is pw19's SS design. pw19/pw24 measured smem-port saturation (+7 to +11 µs), and CUTLASS's RS path exists to avoid exactly that. **Not recommended.**

## Expected pooler m=128

- If pw25's bf16 CUTLASS reaches cuBLAS parity (about 20.3 judge-µs), 50 MB instead of 67 MB at the same efficiency gives about 15.2 µs.
- Plus 1–1.5 µs of correction: **about 16.5 µs (−4 µs)**.
- Go only if pw25 shows parity first.

## Cost and milestone

**Cost:** about 5 days:
- fork plus converter: 2 days;
- packer, correction kernel, emulator: 1.5 days;
- validation and bake: 1.5 days.

**Milestone 0, done locally: the stock RS mixed-input collective compiles for both plane widths.** Swap-AB setup, 128×64×64 cooperative, bf16 B and D, sm_90a:
- uint8: 16 stages, 168 regs, 0 spills.
- uint4: 12 stages, 168 regs, 0 spills.
- Both together compile in 36.8 s locally.
- SASS: LDSM.16.MT88 A loads, PRMT/LOP3/I2FP conversion, HGMMA.64x64x16 with register A, USETMAXREG, UTMALDG.3D.
- One pitfall: a non-dependent epilogue alias trips a host-gcc bug in the builder. All aliases must depend on template parameters.

**Next milestone:** the forked two-plane collective, 0 spills, plus a CPU emulation of its fragment order.

## pw25 result (check 04dc0a41413f45f3): stock CUTLASS bf16 is about 2× off cuBLAS on pooler m=128

**(1) Why.** 128×64 tiles on swap-AB pooler (M 4096 × N 128) give only 64 tiles, so the persistent grid streams W from 64 of 132 SMs (cut128 46.0, about 2× cuBLAS's 23.8). Stream-K (37.9) fills the SMs but pays a fixup reduction. Cluster 2×1 (47.8) keeps 64 tiles. nvjet's pick (64×64, 13 stages, cluster 1×2) gives 128 CTAs streaming W, with W shared between the two token tiles. Sweep, with expected in-prepare µs on pooler m=128:

| # | Config | Expected |
|---|---|---|
| a | 64×64×64 ping-pong, 13 fixed stages, cluster 1×1 | 25–28 |
| b | a + cluster 1×2 (the two token tiles of a W tile; W multicast), nvjet's shape | 24–27 |
| c | 64×64×64 basic warp-specialised (one consumer warpgroup), 7 stages, 2 CTAs/SM | 24–28 |
| d | 64×64×128, ping-pong, 6 stages (fewer, larger TMA boxes) | 25–29 |
| e | Non-swapped (M = tokens 128, N = W rows) 64×64 ping-pong, row-major out | Same tile count as a; the control |

Ping-pong spilled 36 B at 64×64 with auto stages. With fixed stages it must be re-checked before baking.

**(2) Compile budget.** 3 instantiations took 52 s on the judge, about 10 s fixed plus about 14 s each. r6 (118 s) + router (6 s) + 4 instantiations (about 66 s) + a bake of 8 shapes × 4 paths (about 30 s) ≈ 220 s, inside the 240 s guard. Four per check is safe and five is tight, so two checks of four cover a–e.

**(3) Odds (honest).**
- A CUTLASS bf16 config within 10% of cuBLAS: about 50% at m=128, about 40% at m=256 (nvjet's m=256 tiles are compute-tuned).
- Given that, the two-plane mixed-input fork beats cuBLAS by ≥ 3 µs on pooler by 2026-11-04: about 20–25%. The byte cut is worth −5 µs ideal; the RS conversion and the escape correction eat 1.5–2 µs, and the fork is 5 days of work.
- Overall about 10–12%.

**Correction (critic, pw26):** pw25 was built without -DNDEBUG. CUTLASS's device asserts left a function call inside the mainloop, and ptxas reported C7510 ("wgmma.mma_async instructions are serialized") for all 3 pw25 kernels. pw25's 2× gap therefore mixes the tile-count problem with a serialised wgmma pipeline, and overstates the configuration problem. With -DNDEBUG, rebuilt from pw26's embedded bundle with its own flags: 0 serialisation warnings, 4 × 64 regs, 0 spills, 29 s local.

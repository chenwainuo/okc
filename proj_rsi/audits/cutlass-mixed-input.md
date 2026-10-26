# Step B plan: r6's 12-bit code as the CUTLASS A-operand transform

Source read: CUTLASS v3.9.2, `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` (1,037 lines) and `detail/collective/mixed_input_utils.hpp`. The milestone below is compiled locally.

## How the stock mixed-input mainloop works

**Pipeline.**
- The producer warp TMA-loads the narrow operand, plus scale and zero tensors if used, and the 16-bit operand into an N-stage smem ring.
- The consumer warpgroups (cooperative 2×64 rows, or ping-pong) per k-block do the following:
  - `copy_tensors_MK`: smem → registers for the narrow A (LDSM), prefetched 2 k-blocks ahead.
  - `dequantize_A_kblock`: convert to the MMA type, prefetched 1 ahead.
  - `cute::gemm` with **register A** (RS wgmma), inside warpgroup_arrive / commit / wait.
- This is exactly pw10's structure (decode in the MMA warpgroup, register A), but with CUTLASS-grade pipelining, setmaxnreg and the stream-K/persistent schedulers.

**Supported transforms.**
- `ConversionMode` is DirectConvert, ConvertAndScale or ConvertAndScaleWithZero, with per-group scale/zero tensors (MN-major, group along K).
- A types: int8, uint8, int4, uint4 (bit-trick converters in `LayoutAwareConvertImpl`), fp8 e4m3/e5m2, and an int4 LUT-scale path. The 16-bit side is bf16/fp16.
- The narrow operand must be K-major.
- There is **one** narrow A tensor.

## Option (i): two planes, 8-bit + 4-bit

**Data.**
- `P8[n][k]` is r6's low byte (exponent LSB + 7 mantissa bits), K-major uint8.
- `P4[n][k]` holds the sign plus the 3-bit exponent offset from the per-matrix window, K-major uint4.
- 12 bits/weight in total, 50 MB on pooler.

**Custom code: a forked collective (about 300 changed lines), not just a functor.**
1. A second TMA stream and smem stage buffer for P4. This mirrors the "zero" tensor plumbing, but at full M×K size.
2. A second smem→register copy, with the same CuTe partition as A so element order matches.
3. A new `dequantize_A_kblock`: `bf16 = (P4 bits & 0x8) << 12 | ((P4 & 7) + base) << 8 | P8`. That is r6's dec1 arithmetic done with PRMT/LOP3 on 4 lanes at a time, on CuTe's fragment order rather than r6's.
4. A packer that writes plain row-major planes. This is simpler than r6's fragment pack.

**Escapes (h7 outside the 8-exponent window, about 0.01%).** In-register fixup inside CuTe's fragment layouts is the risky part. The proposed alternative:
- Decode escapes as base+0 and record ΔW = W − W_dec (exact in fp32).
- Apply a post-GEMM correction kernel: `out[tok][n] = bf16(float(out[tok][n]) + Σ x[tok][k]·ΔW)` over each column's escape list.
- Size: about 3.4 K entries × 128 tokens on pooler, one small launch of about 1–1.5 µs.
- One-hot rows stay exact: bf16(8·W_dec) + 8·ΔW = 8·W in fp32. Random rows take one extra bf16 rounding, within the 1% tolerance.

## Option (ii): 16-bit A fed by a pre-decode into smem

This is pw19's transform-warpgroup design with SS wgmma. pw24/pw19 measured smem-port saturation (+7 to +11 µs), and CUTLASS's own RS path exists to avoid exactly that round trip. **Not recommended.**

## Expected pooler m=128

- If bf16 CUTLASS (pw25) reaches cuBLAS parity at about 20.3 judge-µs, 50 MB instead of 67 MB at the same efficiency gives about 15.2 µs.
- Plus 1–1.5 µs of escape correction: **about 16.5 µs (−4 µs)**.
- Across the big four at m=128/256 that is about +0.07 ln, only if pw25 shows parity first.

## Cost and first milestone

**Cost:** about 5 days:
- fork plus converter: 2 days;
- packer, correction kernel, emulator: 1.5 days;
- validation and bake: 1.5 days.

**Milestone 0, done locally: the stock collective compiles for both plane widths.** Swap-AB setup, 128×64×64 cooperative, bf16 B and D:
- `MI<uint8_t>`: 16 stages, 168 regs, 0 spills.
- `MI<uint4b_t>`: 12 stages, 168 regs, 0 spills.
- Both together compile in 36.8 s locally.
- SASS: LDSM.16.MT88 A loads, PRMT/LOP3/I2FP conversion, HGMMA.64x64x16 with register A, USETMAXREG, UTMALDG.3D.
- One host-compiler pitfall: a non-dependent epilogue alias trips a gcc bug in the builder. All aliases must depend on template parameters.

**Next milestone:** the forked collective with both planes and the r6 merge, 0 spills, plus a CPU emulation of the fragment order.

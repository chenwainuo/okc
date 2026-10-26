# P4c: which pipe bounds the 12-bit decode, and the cheapest formulation (H100 box, pooler m=128)

**Kernel.** pw24h's register-A pipeline (rsk2x): 64×64 tile, BLOCK_K 128, 7 slots, merged barriers, no cluster, 128 CTAs.
- A is produced per k64 per thread by 3 `LDS.128` of the slot (12 words: the 12-bit payload of 32 weights) followed by the decode formulation.
- The decoded registers are the wgmma's A operand, so the data flow is real. The data is the bf16 tile reinterpreted, so values are garbage and these rungs are timing-only.
- ptxas: 0 serialization. SASS op counts checked per formulation: dec1 per 8 weights = 4 PRMT + 2 LOP3 + 1 SHF + 2 IMAD.IADD; ptxas emits the "+Tb" adds as IMAD.IADD, which offloads them from the ALU pipe.
- In-prepare µs, median of 31 graph replays. Runs pipe, pipe2 and pipe3 (`/root/okc/logs/pw24h-pipe*.log`); spread ±0.2.

## Formulations

| formulation | ALU-pipe ops / weight | FMA-pipe ops / weight | pooler m=128 µs (pipe / pipe2 / pipe3) | vs LDS-only |
|---|---|---|---|---|
| rsk2x: bf16 W, ldmatrix A (no 12-bit) | – | – | 28.38 / 28.58 / 28.35 | – |
| LDS only: 3 LDS.128 → A, no decode | 0 | 0 | 28.48 / 28.54 / 28.54 | 0 |
| PRMT merge floor (4 PRMT / 8 weights, high bytes as stored) | 0.5 | 0 | 28.54 / 28.64 / 28.77 | +0.1 |
| **dec1** (pw10/r6): (C&m)+Tb, SHF+LOP3+add, 4 PRMT | 0.875 | 0.25 (IMAD.IADD) | **28.51 / 28.86 / 28.70** | **+0.03 to +0.32** |
| aligned window (B % 8 == 0, codec change): (C&m)\|Tb as 1 LOP3, SHF+LOP3, 4 PRMT | 0.875 | 0 | 28.48 / 28.67 / 28.61 | +0.07 |
| OR-rebase + FP scale: (C&m)\|K8, SHF+LOP3, 4 PRMT, 4 HMUL2.BF16 × 2^(2B−16) | 0.875 | 0.5 (HMUL2, half rate) | 28.83 / 29.09 / 28.90 | +0.35 to +0.55 |
| as above, nibble shift by IMAD (C·16) instead of SHF | 0.75 | 0.125 IMAD + 0.5 HMUL2 | 28.77 / 29.09 / 28.80 | +0.3 to +0.55 |
| dec1 ×2 in sequence (real ILP, ~84 ops/thread/k64) | ~2.1 | 0.5 | 29.06 / 29.25 | +0.5 to +0.7 |
| dec1 ×3 | ~3.2 | 0.75 | 34.18 / 34.08 | +5.6 |
| dec1 ×4 | ~4.3 | 1.0 | 39.78 / 40.00 | +11.3 |

cuBLAS (torch.mm) in the same bakes: 28.61 / 28.70 / 28.58.

## Pipe test: dec1 plus R rounds of 16 independent ops per thread per k64 (high ILP)

| extra ops / thread / k64 | +32 | +64 | +128 | +192 | slope past the knee, clk per op per k64 |
|---|---|---|---|---|---|
| LOP3 (ALU) | 29.34 | 32.00 | 40.74 | 50.08 | **2.18**: INT32 half rate, 16 lanes per SMSP |
| HFMA2.BF16 | 29.18 | 30.78 | 41.28 | 50.30 | **2.36**: bf16x2 also half rate, no headroom |
| FFMA (FP32) | 28.99 | 29.28 | 33.73 | 38.59 | **1.12**: FP32 full rate, 2× the ALU headroom |

**Synthetic per-op chain rungs** (k2x SS, 4 dependent chains of N/4; slope over N = 64 → 128):

| op | IMAD | LOP3 | PRMT | SHF | HFMA2 | HMUL2 | FFMA | FMUL |
|---|---|---|---|---|---|---|---|---|
| clk per op | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 | 1.84 | 1.85 |

These are latency-bound: 4 chains deep, every op type the same. They do **not** measure pipe throughput. p4-decode-slack.md's synthetic curves were of this kind and overstated the cost of a real decode.

## Reading

1. **The real decode is not on the critical path on the H100.**
   - dec1 (36 instructions per thread per k64 at 64 rows) costs +0.03 to +0.32 µs over the no-decode data path, which is within noise of bf16 register-A and of cuBLAS.
   - The consumer absorbs about 2× dec1 (~84 high-ILP instructions per thread per k64) before the knee. Past it, cost is ALU throughput: ~2.1 clk per INT op per k64.
2. **Pipes.**
   - INT32 (LOP3/PRMT/SHF) and packed bf16x2 HFMA2/HMUL2 are both half rate on Hopper.
   - Only FP32 FFMA/FMUL is full rate.
   - Moving the exponent rebase to HMUL2 *costs* (+0.3–0.5). ptxas already moves dec1's adds to the FMA pipe as IMAD.IADD.
3. **Is ≤ 0.6 ALU ops per weight reachable?**
   - The PRMT byte merge alone is 0.5 per weight (one op per output register of 2 bf16) and is the floor for any byte-packed layout.
   - The high-byte build adds 0.25–0.375 (SHF + LOP3s, or adds moved to IMAD.IADD).
   - dec1 already sits at 0.875 ALU + 0.25 FMA-pipe. The aligned-window codec reaches 0.875 + 0. The 0.5 floor saves at most ~0.2 µs here.
   - An FP32 formulation would need one op per weight to build each fp32 plus a pack (F2FP), so it is worse than the 0.5 PRMT floor.
4. **H200 projection** (stream period ~0.7×, ≈ 110 clk less per k64):
   - The ALU budget shrinks from ~84 to ~30 instructions per thread per k64.
   - dec1 (36, of which ~7 per 8 weights on the ALU pipe ≈ 28 ALU + 8 IMAD) sits at the edge. The aligned-window variant (28 ALU) or the merge floor (16) leaves margin.
   - Expected judge cost: dec1 0–1 µs, aligned ≈ 0.
   - The 12-bit builds should keep dec1-class decode (or the aligned variant), keep ILP high (4 independent groups per k64), and avoid HFMA2-based rebasing.

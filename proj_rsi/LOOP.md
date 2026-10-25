# muse_bf16_proj loop — head log

Head: Fable 5.1 (high). Implementers/critics: Opus 5.5 (medium), one candidate directory each.
Goal: board under 3000 µs (speedup ≥ 1.6103×). Incumbent on board: r6-c12, job `23cecdef1bb54c9e`, 3428.0 µs (1.4092×).
Tools: `python3 -m proj_rsi {board,check,wait,result,status,lint,cpu-check,quota}`. Protocol: `proj_rsi/protocol.json`.
History before this loop: `proj_rsi/archive/fable-LOG-2026-10-25.md` and `proj_rsi/submission/NOTE.md` (another session's summary).

## Iteration 0 — 2026-10-25 18:00 UTC: board arithmetic and the plan

Per-load sums from the r6 scored profile (yours / SGLang, µs): decode1 38.2/59.8, decode16 51.2/76.0, decode32 60.4/76.2,
verify1 38.9/75.7, verify16 72.6/85.5, verify32 86.8/101.3. Summed ln(ratio) = 2.058; 3000 µs needs 2.859 (+0.80);
the published floors allow 4.236 (2384 µs). Leverage in score-µs per µs cut on one shape: decode1 14.9, verify1 14.7,
decode16 11.2, decode32 9.5, verify16 7.9, verify32 6.6.

What r6 leaves on the table, by mechanism:

| mechanism | shapes | r6 yours → plausible | score µs |
|---|---|---|---|
| packed-weight (12-bit lossless) tensor-core GEMM at m=128; r6 runs cuBLAS on 67 MB uncompressed at 3.5 TB/s | verify16 qkv/o/pooler/unpooler | 15.4/7.2/20.3/19.3 → ~9.5/5/15/15 | ~140 |
| same kernel at m=32; r6's SIMT GEMV is compute-bound there (16.9 vs 14.3 at m=16) | decode32 qkv/o/pooler/unpooler | 14.2/6.6/16.9/16.7 → ~8.5/4.5/13.8/13.8 | ~130 |
| same kernel at m=16 | decode16 | 51.2 → ~44 | ~80 |
| streaming efficiency + launch/tail overhead on tiny shapes (unpooler sliced 4.9 vs 1.75 floor, router 4.1 vs 1.75, qkv 10.0 vs 7.9) | decode1, verify1 | 38.2 → ~32.5 each | ~165 |
| router m=128/256 as one K=2K GEMM on packed hi/lo | verify16/32 router | 10.3/11.1 → ~4.5 | ~90 |
| wgmma at m=256 (compute-bound, cuBLAS at 71% of peak) | verify32 | 86.8 → ~78 | ~70 |

Everything together projects to roughly 2850–2950 µs; the goal needs about 430 µs of the ~675 listed. The previous
loop's failures (direct-global mma.sync at 0.4–0.7 TB/s; hand wgmma with a 35–60 µs epilogue; cuBLASLt sweeps that
changed nothing) were execution failures of specific designs, not evidence that the floor is unreachable; the leader
sits at 3000.75 with the same rules.

Design chosen for the rewrite ("pw"): one kernel family, packed weights in mma-fragment order streamed once from HBM
with deep bytes-in-flight, dequantized in registers, tensor-core MMA against x held in shared memory, K split across
CTAs to fill 132 SMs, partials reduced through fp32 atomics into a workspace zeroed inside kernel_fn, converted by the
last arriving CTA. Served shapes grow outward from m∈{16,32,128}; every unserved shape stays on the torch fallback.

Execution constraints: no local GPU. Measurement is `okc check` only (5 left today, 20/day from 00:00 UTC, 10 min a
run, prepare ≤ 300 s). Every candidate therefore carries its own bake-off (template: `proj_rsi/template/submission.py`)
so one check times every variant and never dies on a kernel bug.

## Iteration 1 — pw1: packed tensor-core streaming GEMM for m ∈ {16, 32, 128}, four BF16 layers

(brief, result and decision follow)

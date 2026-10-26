# 03 — Rules and lessons (things that cost us runs)

## Task rules (muse_bf16_proj)
- out = x @ W.T on one GPU; layers bf16 [N][K]: qkv_gate 2304×8192, o 8192×1024, pooler 4096×8192, unpooler 8192×4096 (rows 0–1023 only when m ≤ 8), router 256×8192 FP32 (tol 1e-4; computed as bf16 hi+lo with fp32 accumulation). Loads: decode1/16/32 = m 1/16/32, verify1/16/32 = m 8/128/256; official correctness cases also m = 5, 7, 9, 33, 130, 512, 1000. One-hot probe rows (≈1/8 of x) must be exact.
- Lossless weights only; no host sync in kernel_fn; no input-derived caching; no network/subprocess beyond load_inline; single file ≤ 1,000,000 bytes; prepare ≤ 300 s (we guard at 240 s; the judge's r6 phase alone takes 105–120 s vs 35 s on the box, so every extra bake sits behind a budget−30/−45 guard with timestamps); the judge regenerates weights per run, so packing happens in prepare on the GPU.
- Weights are normal(0, 1/sqrt(K)) rounded to bf16: mantissa incompressible, exponent ~2.5 bits entropy → lossless 12 bits/weight fixed-width (r6's format) is what every kernel streams; 10.5–11-bit variable codecs are issue-bound (closed, see 10).

## Measurement rules we learned the hard way
1. **Transfer rule** (measured across pw27h/pw29h/pw31/pw34): a gain from removing a FIXED delay (memset node, extra graph node, cluster launch, per-CTA latency) transfers H100 → H200 unchanged; a gain from better overlap with the memory stream shrinks or reverses on the H200 (faster HBM exposes fixed latencies). Byte savings (50 vs 67 MB) transfer well. Judge candidates must be judged by the judge's own within-run bake, not box totals.
2. **F1 baseline rule** (box bakes): a baseline timed right after heavy pack/unpack work reads ~3–4 µs high. Time r6 before the pack AND after the variants, keep the lower, print both; discard one warm-up timing after the pack. On the judge the inflation did not appear (run 15514e82: before 23.68 / after 23.65).
3. Judge drift: ~3% between runs, ±0.6 µs per shape; CIs of different runs do not overlap. Only within-run comparisons are evidence; a board number is one draw. The board keeps the best scored run.
4. Run position on the box shifts small shapes by ±1 µs; when comparing two submissions, run both orders or alternate.
5. The soak container (or any other GPU user) inflates everything; keep the GPU exclusive to the lock queue.
6. CUPTI kernel time vs measured: our one-node kernels show gap −0.7…+0.5; cuBLAS-served shapes +2.0…+2.5; multi-node graphs +1…+1.5. Moving a shape to one node recovers the gap.
7. ptxas drops wgmmas whose results are unread — probe/ladder kernels must keep accumulators live (the two-pass m=256 variant looked 3.7 µs faster until fixed).
8. `__float2bfloat16` on accumulators serialises register-A wgmma; use `cvt.rn.bf16x2.f32`.
9. In-prepare bake vs judge-measured offset ≈ −4 µs (no graph in the bake); never compare bake numbers to report numbers.

## Submission hygiene
- Upload limit 1,000,000 bytes (pw25 was rejected at 1,074,299; lint enforces).
- Every new path must be picked by an in-prepare bake against the incumbent pick for that (layer, m) (never against torch.mm — r6's tuned cuBLASLt is 1.6–4 µs faster than torch.mm), with a per-path budget guard and timestamp; failure of any new path must fall back to r6's choice (pw34 does: `_rqcall` returns nonzero → r6).
- kernel_fn dispatch is by exact m for rq paths (`state["rq"][(layer, m)]`), by bucket `_bk(m)` for gv paths; m=33 must never hit an m=32 path.
- The first pw34 scored run silently skipped the RS128 bake on the judge ("BAKE rq skipped-budget 160s"): always grep the judge `.log` for `skipped`/`BAKE`/`PICK` after a run.
- Keep `proj_rsi/LOOP.md` appended per iteration and the ledger updated (`python3 -m proj_rsi incumbent <job> --candidate … --board` when a scored run wins).

## Safety-classifier note
Two of the head's replies were withheld by a safety classifier while discussing SSH access to the first (dead) GPU box. Do not write about obtaining access to 162.243.109.198 or about SSH-key manipulation; just use 107.170.6.60 as set up.

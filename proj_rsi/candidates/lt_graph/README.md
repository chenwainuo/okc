# lt_graph — source audit and one untuned o32 layout policy

`state=CLOSE_POLICY_NO_SWEEP; GPU=lt-graph01_COMPLETE; incumbent_edits=NONE; new_CUDA_extension=NONE`

Final result: one legal, numerically valid layout2 candidate loses0.864us in both F1 and paired graph gates. No sweep/port/retry. See [RESULTS.md](RESULTS.md). Pretrial feasibility and hypotheses below are preserved as history.

**Feasibility:** NO-GO for an eager-vs-graph timing fix or another broad o128/Lt/CUTLASS sweep. GO for one narrow source-coverage hole: o32's prepared pretransposed-W layout2 is never tried by the current small-M Lt tuner. No claim that it is faster or legal on the target GPU until the existing Lt API verifies it.

Source evidence, frozen `pw36/submission.py` SHA256 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`:

| Location | Actual behavior | Consequence |
|---|---|---|
| `_tm`,1932 | captures `fn` in a CUDA graph; every timed replay follows a read-only flush over the supplied buffer | tuner measures graph replay, not eager call latency |
| `prepare`,2373–2403 | creates `w.t().contiguous()` for every BF16 projection; tuner gets64M float flush elements=256MiB | transpose already exists outside timing; flush exceeds L2 |
| `_tune`,2228–2259 | `m<128` takes only `lt_algos(...,lay=0)` and compares graph times; no all-layout search | o32 layouts1/2/3 have no candidate admission path in this branch |
| `_ltsearch`,2281–2355 | m128/256: all4 layouts' top8 heuristics,16 seeded configs, algo66 tile/cluster/stage/custom/splitK refinements under weighted22s budget | repeating broad o128 tuning is covered work; budget truncation alone is not a new mechanism |
| `LTCPP`,2075–2191 | lay bit0 row-major vs column-major; bit1 pretransposed W;64MiB process-owned workspace; public config/cap/check APIs | layout2 changes W orientation/transposition and may expose a different legal kernel family without added GPU nodes |
| `lt_add`,2151–2166 | explicit splitK>1 requires FP32 compute reduction; checks FP32 accumulation/no TF32, AlgoCheck/workspace limit | candidate can use the existing legal/numerical admission path |
| `lt_mm`,2179–2191 | one `cublasLtMatmul`, alpha1/beta0, same output dtype and existing workspace | source itself adds no explicit memset/reduce/transpose kernel |

The handover01 “o32 +2µs / o128 +2.2µs” gaps do not identify a responsible cluster/splitK attribute. Actual judge profiles for pw34 job846e25d9 show exactly one o32 kernel `nvjet_sm90_tst_128x16_64x11_1x2_h_bz_TNT` (4.7µs) and one o128 `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` (5.5µs). pw35 also shows one each. rs32's scored run now serves o32 with one rq kernel (5.6µs) and o128 with one nvjet kernel (5.5µs). Thus “remove an extra node” is unsupported for these o cases. Kernel names containing1x2 are not a causal measurement of cluster overhead. Benchmark/profile durations are different observation contexts; their difference cannot be treated as a free recoverable constant. Actual config fields and replay events are recorded by this lab before any attribution.

Closed-work check: handover10, LOOP's previous Lt/CUTLASS sweeps, `pw25/pw26` critiques, `deepgemm-host-embed`, and `cleanroom-measure` were read. They establish tunedLt as the comparator and close stock bf16/CUTLASS re-sweeps; they do not add a layout2 path to `_tune`'s m<128 branch. No precision change, alternate benchmark timing, or HBM/cache control is introduced.

One candidate policy, o32 only:

1. Prepare frozen pw36 once. Preserve its state and actual `kernel_fn` fallback (rq/sel/r6/Lt); capture that same dispatch throughout measurement.
2. Read layout2 heuristic metadata in rank order. Skip splitK>1. Take only the first config accepted by existing `lt_add` after explicitly setting splitK1/reduction0. This API checks FP32 accumulation/no TF32, AlgoCheck and64MiB workspace capacity. Verify returned splitK/reduction fields. No timing of a config grid, cluster sweep, or o128 candidate.
3. Use the existing prepared `o_t` and process-owned Lt workspace. No workspace reallocation, persistent cache trick, or overlapping replay. Algorithm append preserves all existing integer plan indices.
4. Require exact all1024 one-hot columns; random normal/outlier FP64 error checks; original/zero/fresh-outlier graph replay through the same X allocation; no input mutation. Only then compare cold-L2 graph replays.
5. F1 before/after baseline keeps the lower result; also21 rotated AB/BA graph pairs. Survive only with≥0.5µs gain in both. On failure close this policy with no tuning retries. No legal candidate is a normal close result.

Falsifiable prediction: pretransposed W may select a different Nvjet orientation/family with a faster short-K mainloop, while remaining one GPU matmul node. It does not predict that the old≈2µs gap disappears. If the candidate is the same family/config behavior, adds a reduce/copy/memset node, or fails the≥0.5µs actual-dispatch bar, the proposed mechanism has no measured ROI. Profiles are diagnostics after selection/timing; their durations never choose the winner. At0.5µs on o32 the old score sensitivity is only≈4.6score-µs, so no≥20score-µs aggregate or<3000 claim is made.

CPU/static proof and syntax gate (no CUDA calls):

```bash
python3 proj_rsi/candidates/lt_graph/preflight.py
```

This confirms actual tuner branches and graph/flush behavior, and proves layout0 transposed raw-W versus layout2 pretransposed-W index semantics against the same matrix product. No new CUDA source needs compilation; the lab calls the existing frozen public Lt extension.

After independent review, root can schedule one bounded exclusive trial:

```bash
python3 proj_rsi/candidates/lt_graph/lab.py --json /okc/logs/lt-graph01.json
```

`lab.py` SHA256 `8245f822dac5171aaf8a220899729c9ab5d5eacbb8a9bfae2e8404e4503578b3`. No GPU/remote operation or incumbent edit was performed by the author. Full correctness/runtime/API/config legality and performance remain unmeasured. The frozen fallback is the bar, never torch.

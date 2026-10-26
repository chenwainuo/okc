# pw36 takeover audit

Status: **static + official CPU/GPU pass; ready for an official check, judge evidence pending**. Reviewer: `/root/proj_integration`. No source change, remote mutation, extra GPU job, submission, ledger change, or commit performed by this reviewer.

## Source identity

Base: `49b22d8:proj_rsi/candidates/pw34/submission.py`. Parents: current `pw35/submission.py` and `rs32/submission.py`.

| source | SHA256 |
|---|---|
| pw35 | `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8` |
| rs32 | `a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad` |
| pw36 | `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf` |
| base | `958ad2a0197fe1f2bb57206494d9020cdc299b55f8c139a32befc7d76d4c547e` |

Occurrence-aware AST comparison covers all 176 top-level nodes, including repeated assignments. Every pw36 node matches one parent exactly except the single hand-merged `prepare` function. Embedded CUDA strings are compared as their full parsed string values.

- `RQCU`, `RQV`, small-m bake/compile helpers: exact rs32 parent.
- `RWCU`, router path tables and router bake/compile helpers: exact pw35 parent.
- `kernel_fn`: exact both parents; routed kernel picks use exact `(layer, m)` lookup before existing bucket/fallback logic. An m=33 input cannot use an exact-m=32 rq pick.
- RQ/RW extension flags explicitly include `compute_90a/sm_90a`; expected multi-`load_inline` lint warning remains.

## Combined prepare

`rq` and `rq32` background compile threads start before r6 tuning. Order: r6 → m128/256 rq bake → gx → small-m rq bake → router bake. Small-m plans compare the actual incumbent (`rq` exact pick, else selected gv path, else r6), retain lower before/after incumbent timing and require a 1% margin after numerical validation. Packs are reused, and temporary pack references are released after the small-m phase.

The small-m phase also tests o at m128; its baseline is the existing m128 rq pick when present. Router and small-m additions operate on different layers. Per-path counter tensors/module slots are disjoint. Exception paths retain correctness fallbacks; no new unguarded runtime dispatch was found.

Prepare gates (seconds after module start; total budget240): rq admission195, rq per-path210; rq32 admission/per-path210; router outer admission220; **router extension compile admission180**; router per-path240. The inner180 gate is inherited from pw35 and narrower than the hypothesis’s outer220 gate. Combined prepare may therefore preserve correctness but silently lose the rtk improvement if it reaches router too late. This is a measured-log admission condition, not a reason to change guards without evidence.

## Verification

- `python3 -m proj_rsi lint proj_rsi/candidates/pw36/submission.py`: pass, expected sm90a warning only.
- `python3 -m proj_rsi cpu-check proj_rsi/candidates/pw36/submission.py --official`: **9/9 pass**. This exercises the CPU fallback, not CUDA numerical paths.
- Existing H100 comparison: rootPID334430, `/root/okc/logs/pw36/run3.sh`, pw35 → rs32 → pw36 with `--official`; observed still queued, no duplicate launched.
- Parent rs32 judge job `fb548b5dc6f04696` passed at3200.615µs; pw35 `5d880cd0aa504209` at3278.638µs. These are parent evidence, not a measured pw36 score.

## Decision / remaining gates

No static correctness blocker found. Before spending a judge slot: collect the existing H100 run, require all65 GPU official cases, inspect `BAKE/PICK/PREP/skipped` lines for rq32 and rtk, and compare against both parents within the same quiet-GPU run. A later judge run must repeat those prepare/pick checks because the H100 build/tune time understates judge prepare time. No performance/promotion approval is implied by this audit.

Machine-readable evidence: `proj_rsi/candidates/pw36/takeover-audit.json`.

## Fresh exclusive H100 comparison — takeover-r1

Earlier `run3` was excluded because overlapping GPU contexts were discovered. Root stopped those runs and restarted the three-parent comparison under a container-owned shared lock. The replacement driver completed `BENCH_DONE` at2026-10-26T19:40:21Z. This reviewer read all three completed logs/JSONs and verified source hashes equal the static audit. Root owns the run’s concurrency evidence; this audit does not claim independent continuous ownership sampling.

All three candidates pass the exact65 official GPU cases (five layers × m1,5,7,8,9,16,32,33,128,130,256,512,1000). No `ERROR`, `INVALID`, traceback or budget-skipped path appears. JSON/log hashes, all30 timing rows, picks and exactcase checks are stored in `takeover-audit.json`.

| H100 shape | pw35 µs | rs32 µs | pw36 µs | interpretation |
|---|---:|---:|---:|---|
| qkv m32 |20.784|18.080|18.000|retains rs32 win|
| pooler m32 |26.304|25.040|24.864|retains rs32 win|
| unpooler m32 |26.768|24.688|24.704|retains rs32 win|
| o m32 |12.896|13.760|13.280|mixed; no claimed gain over pw35|
| router m128 |13.632|14.528|14.080|rtk selected; delta vs pw35 +0.448|
| router m256 |15.792|18.160|15.840|retains pw35 rtk gain|
| unpooler m128 |26.528|28.128|27.392|+0.864 vs pw35; disclose, not a causal regression proof|

The order is fixed pw35→rs32→pw36, not alternating paired trials. Empty-graph replay overheads11.536/13.600/12.144µs and differing picks show run-position effects. Do not subtract those overheads retrospectively, infer small-shape gains from these medians, or extrapolate a board score as established evidence.

### Positive bake evidence in pw36

- Prepare phases: rq39s, gx43s, rq3246s, router46s; total46s. Both extension paths executed and no gate was skipped.
- qkv m32: `t32s5hze-k6`14.85 vs r6 incumbent18.08; pooler21.63 vs23.26; unpooler21.50 vs23.10. Each passes local numerical/onehot validation and the1% margin.
- o m32: `t32s6hze-k2`9.60 vs9.73, a narrow bake choice; judge may choose differently.
- Router m128: `rt64k32b4z`10.46 vs older wgmma11.46 and F1 r6base16.00. Router m256: `rt64k32b3x2z`12.77 vs oldwgmma14.69 and r6base14.75.

### Readiness decision

**Ready for an official check of this exact source after the quota reset.** No source correction is justified by current evidence. Do not promote from this H100 screen. The official run must confirm both small-m and router bakes/picks under the judge’s longer prepare phase and report all shape regressions. In particular, retain the180s router compile admission warning and the m128 unpooler comparison above.

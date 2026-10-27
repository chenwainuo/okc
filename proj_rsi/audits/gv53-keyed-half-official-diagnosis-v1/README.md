# Keyed-half official transfer diagnosis

CHECK ac49231d41d441f6 passes 9/9 but scores 3225.893245548 us versus incumbent 9916d34357b745cf, 3157.038403653 us: +68.854841895 us worse. This is a valid performance-transfer regression, not a correctness/setup failure. Retain the local fixed-kernel M8 paired +0.224 us/F1 +0.176 us and mixed M1 paired +0.064 us/F1 -0.864 us; those measurements remain valid in their measured context.

## What is established

- PROVEN exposure: both official target cells execute `gv_keyed<1,1,1,2,16,8,1,1,0,3>`. Missing selection is ruled out.
- PROVEN source equivalence: the archived production239b GVCU helper and target kernel body are byte-identical to the measured lab source. The strict source inverse already proves original packers, escape construction, catalogs, search budgets and other selectors unchanged. Production prepares the same C12/geometry escape representation; no changed packing contract was found.
- PROVEN launch geometry equivalence at these two shapes: production gvl and lab binding yield grid(1,128,1), block1024, dynamic shared17408, no cluster attributes, current CUDA stream. G layout and actual helper are shared. Allocation alignment, fixture data, graph state and whole-module environment are not thereby proven equivalent.
- PROVEN local codegen equivalence: production-shaped GVCU prefix including the real gvl dispatch and target instantiation, compiled with CUDA12.8/O3/lineinfo for both sm90 and sm90a, matches the measured lab sm90a keyed target exactly in every instruction PC/predicate/128-bit encoding. Both have360 slots/345 nonNOP/64 registers/zero stack or local memory (1024 bytes compiler static shared). This target-only CPU check intentionally does not compile the full95-entry inventory.
- UNKNOWN realized official binary/compiler/architecture: actual remote SO/SASS and build environment were not archived. Production JIT infers architecture, whereas lab explicitly requests90a; under the tested local toolchain this configuration difference produces no target code difference. It is not evidence that architecture caused the loss.

## Calibration is separate from raw event timing

Every row satisfies calibrated_time=baseline_time/paired_ratio and baseline_time=baseline_raw_event-replay_overhead. Within-job paired_ratio compares reference versus submission, not keyed versus501d. Raw candidate event minus overhead is a useful separate diagnostic, not exactly the official calibrated median.

| Target | Raw candidate delta us | Raw-minus-overhead delta us | Calibrated delta us | Paired baseline/candidate ratio old -> new |
|---|---:|---:|---:|---|
| M1 pooler | -0.064000487 | +0.095999334 | +0.136924127 | 1.572588998 -> 1.566284746 |
| M8 pooler | +0.128000975 | +0.192001462 | +0.180757209 | 1.563717242 -> 1.562940659 |

M1 raw improvement flips after the overhead estimate falls by0.160 us. For M8, symmetric algebra decomposes its+0.180757209 calibrated delta into+0.173987839 from baseline level and+0.006769370 from ratio. M1 corresponding terms are+0.081557976 and+0.055366151. These are arithmetic explanations of reported numbers, not causal attribution to hardware or the kernel.

## All30 routes and collateral observations

`analysis.json` retains all30 old/new route names, raw/calibrated/baseline values, seeds, signed changes and six-load score reconstruction; `all30.csv` is the flat timing table. All30 seeds differ. Fifteen displayed routes changed: the two intended keyed targets and13 paths with unchanged source. Different fixture/prepare history/environment confound cross-job interpretation; unchanged source does not prove unchanged selected execution.

Largest calibrated losses: verify32 unpooler+1.627222 us, o+1.060633, pooler+0.994682 (different Lt routes); verify16 pooler+0.840075 (RQflag3->65539); decode32 qkv_gate+0.831142 (same displayed route); decode32 unpooler+0.623487 (RQST6->5). Preserve opposite signs including verify16 router-0.211 us, qkv_gate-0.155 us and o-0.073 us.

Replacing only the two target calibrated times in the old score vector gives3161.645776210 us, +4.607372557 us, versus the observed whole-job+68.854841895 us. This is an accounting counterfactual, not additive/causal attribution. Candidate six-load sums have geometric factor1.026005328; baseline-normalizer factor0.995910957; product1.021809948 matches the score ratio. Most whole-score movement therefore lies outside the two measured target entries, but why the other routes/timings moved remains UNKNOWN. Compilation/preparation stdout and same-seed control were not archived; a fresh module-cache prefix is a plausible contextual change, not a proven cause.

## Smallest next falsifier

Run one bounded M8 production-context diagnostic, not an unchanged official retry or a full30 gate. Compare parent501d and239b using the actual gvl/production G argument construction, one shared immutable C12/escape bank and identical default-aligned current X; include the existing lab keyed path as the third arm. Capture the exact selected native entry, SO/SASS and build args. Balance same-record three-arm orders and retain F1 endpoints, raw events, empty replay overhead and the same reference-relative calibration separately. Use existing numeric/current-input/own-terminal/cleanup checks; do not repeat full codec proofs.

Question: does the lab-keyed versus production-keyed equality hold in execution, and does production-keyed improve production-parent under matched workload/timing? If lab-keyed differs from production-keyed, investigate launch/packing/allocation context; if equal but both fail parent, the earlier local gain does not reproduce under this matched context. If both win parent, official transfer/selection variability remains the leading unresolved issue, and a later changed measured integration can be considered without rejecting keyed-half. No result is predicted.

Proposed cap120s inner/180s outer, single M8 cell, three arms, reuse the existing lab SO and compile only target production dispatch. Observed predecessor cost45.56s with42.19s compile; new cost is bounded, not measured. Conditional score value for0.1 us improvement on M8 alone is approximately1.4 us using incumbent weighting; actual attainable speedup remains UNKNOWN. Do not book68.85 us as recoverable keyed ROI, nor treat this information diagnostic as speed gain.

Loop: profile -> measure -> hypothesize -> implement -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](../../ITERATION.md).

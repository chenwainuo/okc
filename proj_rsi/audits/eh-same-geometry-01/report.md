# EH same-geometry trial01: valid small observations, no universal win

All four shapes pass the recorded runtime gates. Reconstructing **all1096raw calls** gives a small positive pooler EH-with-hints observation consistent in direction with full PW38's pooler result. QKV remains estimator-dependent. Every tiny positive and negative result is retained; no0.5us or other minimum-gain filter is applied.

Reproduce with `python3 proj_rsi/audits/eh-same-geometry-01/analysis.py`. [Program](analysis.py), [complete machine analysis](analysis.json), [original result](../../../results/proj-rsi/scheduler/artifacts/research.eh.samegeometry.gpu.01/collection-1793076695219438309/logs/scheduler/eh-same-geometry-01/result.json), [collection inventory](../../../results/proj-rsi/scheduler/artifacts/research.eh.samegeometry.gpu.01/collection-1793076695219438309/inventory.json). The program is CPU/std-library only, binds every input/program hash, rechecks actual JIT/canonical/physical gates and recomputes all arithmetic without trusting stored gain summaries. Independent07's [separate validation](../eh-same-geometry-01-independent.md) agrees with the signed outcomes.

## Exact scope and validity

Attempt `research.eh.samegeometry.gpu.01`, root guard412325, terminal2026-10-27 04:51:12UTC exit0. Host log spans04:49:32–04:51:12UTC. Collection records no missing files, no guard/compute PIDs or containers, container absent,0MiB GPU memory and no cleanup failure. Both JIT SASS files and result/log hashes match inventory; all32 dispatch source/dependency pins match. This is a valid fixed-geometry mechanism result, not an actual selected serving-policy or official H200 gain.

The source comparison is frozen PW36 `2a067059…` versus PW38 `29e557ea…`, with current main kernels M128/ST5/GS1/PRB0/WM0/INS0. Matched pairs are FL2→65538 (selfzero without hints) and FL3→65539 (selfzero with hints): only EHbit65536 differs within a pair. Same real task weights, lossless pack/unpack, KS7qkv/4pooler/2unpooler/2o; no selected-route or preparation-bake confound inside this trial. EPbit131072, EARLY32 and HALF16 are separate mechanisms.

All four actual main binaries match their frozen canonical PC/predicate/instruction streams. Independent re-execution of physical gates verifies WGMMA drain before STSM, first-half issue between STSM quartets for EH, publication fences/barriers, and final read completion after both TMA issues. Live attributes are168registers,0local,144520B shared,384threads/maxthreads, occupancy1; static rounded and dynamic register pools64512. Actual .so hashes are recorded by the driver; only their SASS dumps, not the .so files, are present in this collection, so this analysis independently rehashes/rechecks the dumped instruction streams.

All16 arm×shape validations record every K basis column exact, randomFP64 and normal/outlier task oracles. Source-bound RQTOL is0.008 atKS7,0.006 atKS4 and0.005 atKS2; it is not0.004 uniformly. Each shape has four distinct one-node kernel profiles, eight immediate actual terminal output oracles before replay/mutation/profile,20 NaN-poison/fixed-input mutations and final drain/reset of all four graphs: **32terminal oracles and80mutations total**. Immutable W/packedW/escape and zero counters are enforced by the frozen driver before success; the successful status attests those source assertions rather than serialized raw tensor contents. No raw hardware all-BF16-payload-bit proof is claimed.

## Every matched result

Positive gain means control minus EH, so positive is faster. Paired is the median of32same-fixture round differences. F1 uses the faster opening/closing control median minus its21-sample EH block median. W/T/L counts are strict paired wins/ties/losses. Values are microseconds, rounded here only; JSON retains full precision and all signed samples.

|Shape|EH flag / hints|F1 gain|Paired gain|W/T/L|Paired IQR|
|---|---|---:|---:|---|---|
|qkv_gate128|65538 / off|−0.096|+0.064|17/1/14|−0.232…+0.360|
|qkv_gate128|65539 / on|+0.352|−0.160|9/1/22|−0.672…+0.104|
|pooler128|65538 / off|−0.320|−0.176|12/0/20|−0.320…+0.200|
|pooler128|65539 / on|+0.768|+0.096|18/1/13|−0.136…+0.520|
|unpooler128|65538 / off|−0.160|+0.128|21/1/10|−0.184…+0.400|
|unpooler128|65539 / on|+0.224|+0.016|16/2/14|−0.736…+0.384|
|o128|65538 / off|−0.160|+0.096|19/0/13|−0.160…+0.320|
|o128|65539 / on|+0.224|+0.096|22/1/9|−0.032…+0.352|

This is three both-estimator-positive EH3 observations, one both-estimator-negative EH2 observation, and four sign disagreements. It does not justify averaging eight shape/flag outcomes into a universal EH speedup. Pooler without hints is a valid regression for this version, not a family ban. Hints-dependent results do not establish an isolated hardware cause.

All274calls per shape reconstruct exactly:20warmups+126F1+128paired. Across four shapes these are80+504+512=1096calls; none is dropped. Eight rotation/reverse orders occur four times each; every arm occupies every position eight times. Each matched control/EH precedence direction has16rounds. The positive EH3 pooler median is+0.208us when control runs first and+0.064us when EH runs first; o is+0.208/+0.080us. QKV EH3 is negative in both directions,−0.368/−0.048us. These sensitivity views retain all observations, not a favourable estimator selection.

Variance remains material. Unpooler EH3's first16/last16 median gains are−0.688/+0.320us despite its combined+0.016us; o EH2 changes+0.304→−0.096us. Pooler EH3 is+0.048/+0.112us in temporal halves and positive in5of8four-round order strata; o EH3 is+0.064/+0.208us and positive in7of8strata. IQRs include both signs for every comparison. These are descriptive partitions of one run, not extra independent experiments or a confidence guarantee. The small positives remain valid observations worth retaining, with uncertainty intact.

## Relation to the full PW38 stack

[Full prepared-policy result](../../../results/proj-rsi/scheduler/artifacts/pw38.stack.fullscope.01/collection-1793076533579348342/logs/scheduler/pw38-stack-01/result.json) chose FL3 controls for qkv128/pooler128 and EH65539 in PW38, same ST5/KS geometry as the EH3 rows above. Full preparation, allocation state, timing schedule, sample count and fixtures differ from this fixed-kernel trial, so magnitudes cannot be combined or transferred as an intrinsic epilogue latency.

|Selected shape|Full PW38 F1 gain vsPW36|Full paired gain vsPW36|Matched EH3 F1 / paired|
|---|---:|---:|---:|
|qkv128|+0.160|+0.080|+0.352 / −0.160|
|pooler128|+0.384|+0.224|+0.768 / +0.096|

Pooler's full+0.224us is directionally supported by the matched+0.096us, but the evidence does not establish that EH intrinsically saves exactly0.224us. Full pooler versus RS32 is+0.320us paired; all three observations retain their actual comparator/run. Full QKV is already mixed across identical-geometry controls: PW38 paired−0.032us versus RS32 but+0.080us versus PW36, while the matched EH3 paired result is−0.160us. Its F1 values remain positive. A stable QKV gain is therefore unresolved; the EH2 paired+0.064us cannot be substituted for selected EH3.

Timing context is visible even without an EH code change: in the full run the same-geometry FL3 PW36-minus-RS32 paired medians are+0.064us qkv and+0.048us pooler. Full unpooler selects the same non-EH s4s5hz in both PW36/PW38 yet reports+0.192us paired against PW36. Those observations preclude attributing every positive full-stack component delta to EH. They do not identify a specific clock/cache/scheduling cause; this trial has no causal counter or clock measurement supporting such attribution.

The fixed o EH3+0.096us is a preserved component observation against full-width ST5/KS2 RQ. Actual full PW36/PW38 o128 serves h128s8e HALF/ST8/KS1, while RS32 serves LT. Thus the fixed o gain is not a demonstrated improvement over the selected o route. Fixed unpooler EH remains optional and uncertain; its full selected route did not use EH. No cross-shape minima, cross-run gains, or parent-relative-to-incumbent substitutions are added into a production gain or score forecast.

## Retention and next decision

Keep the safe EH variants available to self-validating bake and preserve the full PW38 composition. Pooler EH3 has a small positive observation in both this matched run and the separate selected-policy run; retain it below any historical threshold. O EH3 also remains a measured small component opportunity, with the selected-route comparison still missing. QKV and unpooler need an uncertainty-aware selection/repeat decision rather than a universal enable/disable conclusion; the valid no-hint pooler loss remains attached to that exact flag/version.

This report changes no source, selector, queue, API, official job or admission ledger. Root's independent validation and any later full-stack/official decision remain separate. No automatic promotion or family retirement follows from these measurements.

# rtk-early01 — CLOSED_GAIN_BAR_FAILED

Decision: no port, official slot, tuning or follow-up GPU run. One MODE128 stage0-priority trial failed >=0.5us in BOTH graph and prepare-style F1 at BOTH target m. Source/design/manifest stay frozen.

Named root launch: PID347556, container `okc-proj-rtk-early01`; inner240s / outer300s / container-held exclusive flock. Harvest confirmed PID absent and container absent. Script complete normally in37.783s, reported extension compile33s (`build_s`35.897 including setup). No timeout/decline/fault.

## Production timing (us)

All baselines are shipped pw35 rtk, same hi+lo weights/FP32 accumulation, zero-counter semantics and current x/out. Each m has one candidate, same geometry as indicated parent, one scheduling mode only. Cold256MiB,40 paired alternating-order reps; no replay-overhead subtraction needed for same-run differences.

| m | candidate / parent | shipped graph medians | candidate graph | loss vs fastest shipped graph | shipped prepare F1 | candidate prepare | prepare loss |
|---|---|---|---:|---:|---:|---:|---:|
|128|301 /201 (KR256,3slots,2/SM)|201:10.720;210:10.688|11.168|+0.480|10.656|11.200|+0.544|
|256|300 /200 (KR512,6slots,1/SM)|200:13.136;201:13.056|13.152|+0.096|12.960|13.152|+0.192|

Even the direct m256 geometry comparison is graph +0.016us / prepare +0.096us versus v200's best pre/post13.056; none passes the bar. Do not promote a comparison against r6/torch or a slower baseline.

## Startup hypothesis resolved

Instrumented median CTA ns/globaltimer terms, us from earliest CTA; compare each new mode with its identical-geometry parent. Instrumented values are explanatory, not production latency.

|m / version|consumer ready|first full|last full|MMA done|zero ready|reds issued|
|---|---:|---:|---:|---:|---:|---:|
|128/201 shipped|0.320|2.752|4.896|5.504|5.696|6.496|
|128/301 gate|0.288|2.432|5.344|5.952|6.144|6.880|
|256/200 shipped|0.384|2.272|7.232|7.840|8.000|8.832|
|256/300 gate|0.320|1.888|7.360|7.968|8.160|8.928|

The gate advances first completion by0.320/0.384us, less than the0.5us gain target even before tail costs. It delays last completion by0.448/0.128us; reds issue later by0.384/0.096us. Removing later-stage requests from the initial queue creates a later-load bubble. Advancing the first stage alone does not improve this kernel's total critical path. Worst-CTA reds also regress7.200->7.744 at128,9.536->9.888 at256.

This bounds the benefit of this exact request-priority mechanism; it does not prove that all remaining~1.9–2.4us first-stage latency is physically irreducible. Handover's whole-wave queueing diagnosis was an inference: this experiment shows only a modest first-stage improvement, not the hypothesized~1us or end-to-end improvement. Do not reopen with another gate depth/delay or multicast.

## Correctness and convergence

All6 production shape/variant combinations passed:

- every8192 onehot column x8 bit-exact against FP32(hi+lo);
- shipped eager validator at m128/256 and odd121/250;
- normal + outlier input, fresh activations in place on captured pointers, persistent monotonic counters;
- actual measured graphs validated AFTER timing replays with changed X and output poisoned toNaN beforehand, four cases per shape/variant;
- timed-graph probe outputs exact against hi+lo, reference task checks, strict2e-5 relativeL2 /1e-4 row limits;
- zero-release counts complete multiples ofG, sentinel/poll counters unchanged, unused counters0;
- identical occupancy for each parent/new pair: v201/3012CTAs/SM; v200/3001CTA/SM; guards reject oversubscription and cluster self-zeroing unchanged.

Across actual timed-graph postchecks, maximum relL2 was2.794e-6 and max-row3.611e-6. GPU convergence and validation succeeded; speed killed the candidate.

## Immutable artifacts

Local copies in `results/`, copied from `/root/okc/logs/` read-only:

|artifact|bytes|SHA256|
|---|---:|---|
|rtk-early01.json|57534|a5091c2aed307af88b7896aef5c9c99e884654f174531113568f5f37f47a0854|
|rtk-early01.log|702|89eb599d6ca24a65763cfb43d6f70d3bd1063c53431bca6a645dd6b7fdff3e41|
|rtk-early01-host.log|0|e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855|

Remote lab/variant source hashes verified equal frozen SOURCE_MANIFEST before harvest. Generated CUDA SHA256 `f4ce5627db31635686c60a9caae1069c49dc084a1feaa3c02f6a16da3dd216f2`; frozen pw35 source `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8`. No source change, extra GPU operation or commit by harvesting agent.

# Active projection ROI: muse_bf16_proj below3ms

Official best **3157.038403653us**; gap to3000us **157.038403653us**. Root observed rank64 at22:14UTC; earlier62/1173 is archived, not current. [Machine view](roi_queue.json). Completed losses and every signed result remain in [history](HISTORY.md), outside the active slots.

|Active priority|Opportunity / scope|Actual readiness|5% target-cut conditional score value|Cost bound / next discriminator|
|---|---|---|---:|---|
|1|Quarter-stream GV rewrite: **poolerM8 first**, scalable design|CPU implementation assigned; not admitted/READY|**9.726540us**|Proposed first-screen180s cap/45CPUmin; not yet reviewed. Preserve codec/math/accumulator order; verify quarter decode/repair enables earlier firstMMA without liveness regression.|
|2|Lane-base: **unpoolerM16 only**|IndependentGO/root-admitted `research.gv.unpool16.lanebase.01`; exact READY/RUNNING not supplied|**7.712689us**|Reviewed150s outer/90s inner,10CPUmin. Compare actual501d control; S2R20→5 vs713c is static mechanism evidence, not a speed gain.|
|3|Thin C ABI build-cost validation|CPU POC; runtime-seam validation in preparation, not admitted/READY|**0 direct score credit**|Proposed180s cap/20CPUmin, preferably inside a useful authorized trial. Paired same-image compile/load and ABI/stream/graph lifetime validation; actual build saving unknown.|

**Attainable gains are UNKNOWN.** The5% values above are what-if arithmetic using actual9916 case/load sums: `S_new = S_9916 × product_load(1−f×target_fraction_in_load)^(1/6)`. Other cases and reference timings are held fixed. They are not measured gains, forecasts or a claim that5% is feasible.

Quarter-stream's firstM8 prototype covers6.1145% score elasticity. A hypothetical5% cut across all four selected predxor cases is35.357318us, or across eleven FMT1 GV cases65.741705us. These are **conditional expansion sets, not prototype coverage or proven eligibility**. If all eleven could improve uniformly, an approximately11.764419% cut would reach3000us with all other terms fixed; this is a target calculation, not attainable gain. Independent new shape/geometry validation is required before any family scope claim.

[Proposed low-confidence planning priors](audits/projection-active-roi-planning-priors-v1-20261027.json) are separate and **not adopted in the queue**; root decides. Quarter-stream proposes p_gain=.20/p_valid=.80, lane-base .10/.95 for their5%-case-cut scenarios. At their first-screen caps and$5/GPU-hour, prior-weighted scenario values are1.556246/0.732705score-us, or6.224986/3.516986score-us perGPU-dollar. These are subjective **first-step decision ratios**, not total expected ROI: integration/official follow-up costs remain unknown. Strategic ranking also values quarter-stream's scalable decision, without inventing numerical information credit.

Thin-ABI's localCUDA12.8 compile/link1.605662s and earlier unpairedTorch compile/load42.474651s do not establish a saving. A proposed10s saving across20future fresh builds would save200slot-seconds only if compilation occupies the exclusive GPU slot. With proposed p_save=.50/p_valid=.80, conditional avoided cost is$0.111111 versus a$0.25 first-probe cap (ratio.444444;45future builds to break even under those assumptions). Units differ from score-us/USD and are not directly ranked against kernel gains. Actual same-image savings, future build count and ABI runtime validity remain unknown.

Prior713c/robustness/RQ256 losses, keyed-choice mixed tuning effects and whole official CHECK regressions remain counterevidence in history. All-zero source-descriptor estimates mean **unpriced/unknown**, not a numerical ROI ordering. No serving/MoE entry is active; no source/review/DONE artifact or scheduler policy was rewritten.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](ITERATION.md).

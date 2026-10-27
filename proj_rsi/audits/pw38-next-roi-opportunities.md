# Five next serving experiments against scored PW38

**Five source-grounded CPU readiness briefs, ranked by an explicit conditional budget scenario. None is a measured gain or an executable GPU admission.** Actual new implementation/review/runtime costs remain null, so all five live scheduler ranks are null. This preserves uncertainty instead of pricing unknown work at zero. No GPU, remote/API, queue or serving-source change was performed.

Authority: scored job `a3f0ebf25a754c01`, **3198.617507982us**,9/9,30 timed cases; frozen source `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e` (PW36+EH, no cursor). The current actual-scored routes, not CHECK picks or old prototype winners, define every comparator.

Machine evidence and all30 component sensitivities: [pw38-next-roi-opportunities.json](pw38-next-roi-opportunities.json). Reproducer/checker: [pw38_next_roi_scan.py](pw38_next_roi_scan.py). The helper reproduces the current score within4.55e-13us, checks all30 route/profile identities and binds local PW38 JIT dumps to the exact same source.

## Exact value of small component cuts

Keep this job’s SGLang vector fixed. With C being the sum of five candidate component times in a load and delta the sum of hypothetical cuts in that load:

```text
S =4830.8 / exp(sum(log(B_load / C_load))/6)
S_new =S * exp(sum(log1p(-delta_load / C_load))/6)
```

Weight62 cancels. Each load counts equally; this is neither62×a component geomean nor a cross-run minimum. The following score reductions are for **one component** in that load, irrespective of which layer gets the cut. They are exact conditional arithmetic, not attainable forecasts.

|Load / actual M|Candidate sum us|Same-job SGLang sum us|0.1us cut → score us saved|0.5us|1us|
|---|---:|---:|---:|---:|---:|
|decode 1 /1|36.229055941|59.552002233|1.473174105|7.400063232|14.886934044|
|decode 16 /16|50.575408881|76.256001834|1.054944803|5.292218897|10.628657964|
|decode 32 /32|53.670758330|77.936000191|0.994055748|4.985807311|10.010839649|
|verify 1 /8|36.986198518|75.488000177|1.442982785|7.247714182|14.578674238|
|verify 16 /128|64.112571189|85.391999688|0.832051658|4.171128408|8.369668065|
|verify 32 /256|84.037607962|101.936001331|0.634677070|3.179703377|6.375306148|

A0.1us decode1 component improvement is worth1.473174score-us; a0.1us verify32 component improvement is worth0.634677score-us. Keep both. Multi-component/multi-load cuts below are recomputed with the exact law, never added independently.

## Ranking and full cost accounting

No author has quoted new engineering/proof time, new-version discriminator wall time, cumulative local validation time or official H200 work. Those expected costs are explicitly **null/unrankable** in JSON. The table below is a transparent **if-budget-fits** comparison only: each target component improves0.5us, common subjective P(gain)=.1 and P(valid)=.8, full local GPU reservation300s discriminator +3420s cumulative validation=$5.166667, CPU cap45min implementation +30min independent review +15min analysis=90min. The3420s guard comes from the existing reviewed PW39 fullscope job; it is not this proposal’s measured runtime. These common priors/caps expose opportunity arithmetic, not an authorization, calibrated success probability or fake expected-cost estimate. Official H200 cost/quota remains unknown and outside this local-dollar ranking.

|Conditional rank|Concrete proposal / target cases|0.1us each: score saved|0.5us each|1us each|Conditional expected score-us/$ under full local cap|Actual dispatch rank|
|---:|---|---:|---:|---:|---:|---|
|1|Move only P1 epoch-ticket allocation off the W/X launch prologue|6.406952|32.144328|64.569763|0.497719|UNRANKABLE / costs unknown|
|2|Weight-derived empty-escape specialization on current QKV GV1/GV8|2.915492|14.631010|29.397757|0.226545|UNRANKABLE / costs unknown|
|3|Replace Lt256 fixed1% rejection with bounded held-out finalist confirmation|2.542496|12.814917|25.892880|0.198425|UNRANKABLE / costs unknown|
|4|Decouple selected EH warpgroup epilogues with separate output slabs|1.665187|8.369668|16.850588|0.129595|UNRANKABLE / costs unknown|
|5|HALF+EARLY full-K plain-store o32, against its actual ST6/KS2 RQ|0.994056|4.985807|10.010840|0.077200|UNRANKABLE / costs unknown|

All stages count: quote readiness implementation, independent review, result analysis, full transfer/compile/prepare/validation/timing/cleanup reservation, cumulative stack check, then official work separately. Re-rank once an owner supplies those actual planning estimates. The first queue action for each row is a bounded CPU source/proof/cost-readiness task; no ready GPU descriptor or hidden agent spawn is created here.

## 1. Move only P1 epoch-ticket allocation off the W/X launch prologue

Brief ID: `pw38.next.p1-late-ticket`. Scope: router M8 (gv, scored median4.470569us), router M16 (gv, scored median5.104217us), router M32 (gv, scored median5.508452us), unpooler M1 (gv, scored median4.256899us), unpooler M8 (gv, scored median4.267541us).

**One mechanism.** For the existing selected z/x P1 instances, move the even-word atomicAdd ticket allocation from the prologue to immediately before its first epoch/acquire wait. Keep the zeroer, odd-word release publication, acquire poll, bounded trap, reduction, S and grid unchanged. On a single ordered CUDA stream each completed launch still consumes exactly S tickets/tile and one flag increment; prove this rather than reset counters.

**Current source/SASS basis.** Actual scored router8 is x/S16, router16 z/S16, router32 z/S8; unpooler1/8 use x/S4. The matched PW38 router8 SASS has ticket ATOMG at0x350 before X/W LDG0x410/0x460 and later acquire waits. Static PC order is not timing proof; this is a placement experiment, not an atomic-count reduction.

**CPU readiness / exact next test.** State-machine proof for arbitrary CTA arrival order and repeated graph launches, source-bound physical alias/counter transitions, then one new compile-time placement bit at fixed selected geometries. Explicitly model unsupported concurrent-state streams and hold them rather than assume one epoch.

**Required falsifier.** Ticket issue already overlaps all useful work, moving it delays the final acquire critical path, any epoch/alias test fails, or paired complete-kernel delta loses. Old midstream flag-prefetch no-gain evidence is not counted as support.

**Safety/comparator boundary.** Exact per-launch counter deltas and zero-before-reduction ordering must remain identical. No speculative output, removed acquire, extra graph node or host-maintained replay epoch. Compare untouched PW38 `kernel_fn` actual serving policy, exact matched old mechanism and new mechanism in the same GPU context; a later serving-selector change must be measured as a separate policy effect. Enforce complete one-hot/FP64/task/fresh-input/actual-timed-output/JIT/lifetime/counter/cleanup gates before any gain admission.

**Applicable lesson IDs.** `counter-contract-not-name@1`, `composition-not-additivity@1`. Apply invariant, comparator, pipeline and experiment-design lessons only. Different kernels/hardware/scopes do not transfer measured deltas or latency fractions.

Expected implementation/review/runtime/official cost: **unknown/null**. Use the common reservation table only as a conditional comparison. Preserve every tiny signed effect; do not retire a family from one version or combine cross-run minima.

## 2. Weight-derived empty-escape specialization on current QKV GV1/GV8

Brief ID: `pw38.next.gv-empty-escape-specialization`. Scope: qkv_gate M1 (gv, scored median8.998907us), qkv_gate M8 (gv, scored median9.343956us).

**One mechanism.** At entry, use the existing warp-uniform first escape sentinel to choose a templated escape-free step loop or the exact old escape loop. Empty-list warps execute identical12-bit decode/MMA/load ordering without the per-unit next-sequence test; nonempty lists preserve every correction. Do not change geometry, W depth, cache policy or arithmetic. Existing SASS already interleaves dec1 operations, so merely reordering four dec1 calls is NOT the proposal.

**Current source/SASS basis.** Scored QKV1 is GV(1,1,1,2,16,8,1,1,2,1),64registers; QKV8 is GV(1,1,1,1,16,8,1,1,2,8193). Current _gvesc emits32-entry sentinel lists; step checks nx on every K-unit. Exact CUDA empty-list prevalence is not archived and is the first discriminator.

**CPU readiness / exact next test.** Prove sentinel-uniform branch and identical corrections/math for both paths, including adversarial nonempty escapes and odd M. Compile only these two existing geometries; reject if64-register/two-CTA eligibility is lost or instruction-cache expansion removes the saved path. Count actual prepared empty lists in the next authorized trial before timing.

**Required falsifier.** Actual empty lists are rare, generated SASS already hoists the test equivalently, resource/code-size cost dominates, or complete current-policy F1 and paired signed deltas do not improve. Keep every small signed observation.

**Safety/comparator boundary.** No escape may be omitted based on sampled frequency; sentinel is exact weight-derived metadata. Both warp paths rendezvous through the existing CTA reduction unchanged. Compare untouched PW38 `kernel_fn` actual serving policy, exact matched old mechanism and new mechanism in the same GPU context; a later serving-selector change must be measured as a separate policy effect. Enforce complete one-hot/FP64/task/fresh-input/actual-timed-output/JIT/lifetime/counter/cleanup gates before any gain admission.

**Applicable lesson IDs.** `counter-contract-not-name@1`, `rq11-full-residency-ledger@2`, `prepare-selection-is-experiment@2`. Apply invariant, comparator, pipeline and experiment-design lessons only. Different kernels/hardware/scopes do not transfer measured deltas or latency fractions.

Expected implementation/review/runtime/official cost: **unknown/null**. Use the common reservation table only as a conditional comparison. Preserve every tiny signed effect; do not retire a family from one version or combine cross-run minima.

## 3. Replace Lt256 fixed1% rejection with bounded held-out finalist confirmation

Brief ID: `pw38.next.lt256-heldout-margin`. Scope: qkv_gate M256 (Lt/vendor, scored median17.212997us), o M256 (Lt/vendor, scored median9.089301us), pooler M256 (Lt/vendor, scored median24.838092us), unpooler M256 (Lt/vendor, scored median23.407247us).

**One mechanism.** Keep existing algorithms/layout/workspace/correctness inventory; change only final selection. Within unchanged22s aggregate Lt budget, compare torch default and at most two already-legal leaders in randomized paired brackets on fresh held-out fixtures. Replace the fixed best>torch*0.99 discard rule with a source-frozen no-forced-pick decision that preserves every raw tiny delta and validates selection regret separately.

**Current source/SASS basis.** All four BF16 M256 routes are vendor kernels in the scored job, not RQ. _ltsearch already enumerates all four layouts and seeded refinements; a new o32/layout2 sweep is not proposed. It retimes leaders while lowering t_torch by a minimum, then applies1% rejection. Scored stdout is truncated and selected Lt config IDs are not recovered; capture actual selected/default configurations before testing.

**CPU readiness / exact next test.** Host-only legality and configuration read-back, beta0/FP32 accumulation/workspace proofs, no data-dependent output caching, fixed global budget and independent decision/held-out seed sets. Reuse PW39 selection lesson, but do not duplicate its already-completed RQ confirmation run.

**Required falsifier.** No actual near-margin legal finalist exists, unbiased confirmation selects default with no held-out improvement, budget wall time grows or correctness fails. A particular small signed delta is not automatically dropped or called significant.

**Safety/comparator boundary.** Vendor SASS is not collected here; current scored actual vendor symbols are the route evidence. Unknown exact selected config stays null; no alternative is forced or selected by output values. Compare untouched PW38 `kernel_fn` actual serving policy, exact matched old mechanism and new mechanism in the same GPU context; a later serving-selector change must be measured as a separate policy effect. Enforce complete one-hot/FP64/task/fresh-input/actual-timed-output/JIT/lifetime/counter/cleanup gates before any gain admission.

**Applicable lesson IDs.** `prepare-selection-is-experiment@2`, `eh-tiny-context-bound@1`. Apply invariant, comparator, pipeline and experiment-design lessons only. Different kernels/hardware/scopes do not transfer measured deltas or latency fractions.

Expected implementation/review/runtime/official cost: **unknown/null**. Use the common reservation table only as a conditional comparison. Preserve every tiny signed effect; do not retire a family from one version or combine cross-run minima.

## 4. Decouple selected EH warpgroup epilogues with separate output slabs

Brief ID: `pw38.next.eh-private-output-slabs`. Scope: qkv_gate M128 (rq, scored median12.965924us), pooler M128 (rq, scored median18.187236us).

**One mechanism.** Keep selected EH ordering but allocate two dedicated16KiB output slabs outside live X/W rings. Compare (a)current reused-Xring EH, (b)dedicated slabs with old256-thread retirement barrier, (c)same dedicated slabs with only per-WG final-wait0 and128-thread slab rendezvous. The proposed gain is avoiding waiting for the other WG before its epilogue, not deleting per-WG WGMMA retirement or proxy publication.

**Current source/SASS basis.** PW38 qkv128 FL65539 and pooler128 FL65538 are actually selected. Current local-JIT EH has final WGMMA wait0 at0x3290,256-thread bar1 at0x3480, first STSM0x3550, then first TMA reduction0x3750. That cross-WG barrier protects reused Xring storage; it cannot simply be removed. Dedicated32768extra shared bytes are an explicit resource confound.

**CPU readiness / exact next test.** First prove disjoint dedicated slabs, producer drain and per-WG retirement for all ragged KS7/KS4 partitions. Freeze resource-control arm(b) before testing scheduling arm(c); require actual opt-in occupancy and zeroer co-residency. Historical EP early-poll had no gain and is deliberately NOT retried.

**Required falsifier.** Warpgroup completion is already balanced, extra32KiB prevents required occupancy/co-residency, equal-resource barrier removal is neutral/losing, or any X/W/output lifetime proof fails.

**Safety/comparator boundary.** This new lifetime topology has the largest unknown CPU proof cost. Do not treat source deletion of bar1 as safe or infer benefit from the SASS gap. Compare untouched PW38 `kernel_fn` actual serving policy, exact matched old mechanism and new mechanism in the same GPU context; a later serving-selector change must be measured as a separate policy effect. Enforce complete one-hot/FP64/task/fresh-input/actual-timed-output/JIT/lifetime/counter/cleanup gates before any gain admission.

**Applicable lesson IDs.** `eh-tiny-context-bound@1`, `composition-not-additivity@1`, `wring-mechanism-policy-separation@1`, `native-profile-x-handoff-accounting@1`, `qkv-split4-full-pipeline-regression@1`. Apply invariant, comparator, pipeline and experiment-design lessons only. Different kernels/hardware/scopes do not transfer measured deltas or latency fractions.

Expected implementation/review/runtime/official cost: **unknown/null**. Use the common reservation table only as a conditional comparison. Preserve every tiny signed effect; do not retire a family from one version or combine cross-run minima.

## 5. HALF+EARLY full-K plain-store o32, against its actual ST6/KS2 RQ

Brief ID: `pw38.next.o32-half-fullk-early`. Scope: o M32 (rq, scored median6.567761us).

**One mechanism.** Add one narrowly legal NT32/ST5/HALF+plain-store+EARLY instantiation (FL52, optional separately labelled HINT53) at KS1. Reuse the existing HALF and EARLY source paths rather than create a native-X prototype. N8192 gives2*NB=128 64-row CTAs, comparable grid count to current64*KS2=128; remove the current cross-CTA output zero/poll/reduce protocol by writing disjoint output tiles.

**Current source/SASS basis.** PW38 o32 actually serves rq<32,6,...,35>,KS2. RQV contains older h32s8/full-K controls and a successful-context h128s8e source alternative, but RQSBAKE never offers a shallow h32s5e. This is a new shallow-half/startup composition, not a claim old full-K/HALF losses disappeared. QKV16/32 remain unchanged controls; do not extrapolate to their much smaller grid.

**CPU readiness / exact next test.** Prove all8192x1024 W and32x8192 output tiles exactly once, full-K escape grouping and HALF offsets, EARLY startup/drain; compile current/control/NT32ST5FL52 before any GPU request. Report actual registers/occupancy: the exiting WG does NOT by itself prove two CTAs/SM.

**Required falsifier.** One-WG longer compute chain outweighs zero/reduce removal, resource occupancy deteriorates, actual single-main output/JIT/probe gates fail, or it loses the current complete o32 route. Keep a positive0.1us result even if below old status thresholds.

**Safety/comparator boundary.** No forced serving pick. Mechanism arms may be labelled explicitly, but shipping comparison must call untouched PW38.kernel_fn and candidate actual prepare. No claim this helps actual RQ16. Compare untouched PW38 `kernel_fn` actual serving policy, exact matched old mechanism and new mechanism in the same GPU context; a later serving-selector change must be measured as a separate policy effect. Enforce complete one-hot/FP64/task/fresh-input/actual-timed-output/JIT/lifetime/counter/cleanup gates before any gain admission.

**Applicable lesson IDs.** `prepare-selection-is-experiment@2`, `counter-contract-not-name@1`, `composition-not-additivity@1`, `qkv-split4-full-pipeline-regression@1`. Apply invariant, comparator, pipeline and experiment-design lessons only. Different kernels/hardware/scopes do not transfer measured deltas or latency fractions.

Expected implementation/review/runtime/official cost: **unknown/null**. Use the common reservation table only as a conditional comparison. Preserve every tiny signed effect; do not retire a family from one version or combine cross-run minima.

## What was read, retained and deliberately not repeated

The frozen catalog now contains316 positive observations,37 feature findings,75 selected-positive rows and10 local paired observations. All316 observation rows are indexed by original JSON pointer in the output, including97 below0.3us and138 below0.5us. They remain observations with comparator/source/context attached, not316 independent features or PW38 gains. All11 lesson versions and13 idea versions in the captured registry were read; applicability uses explicit version IDs.

- The current GV SASS already interleaves independent decode operations. A generic “interleave dec1” edit has no demonstrated missing mechanism and was rejected at CPU inspection.
- P2 tail balancing/deeper register rings, broad RT2/cluster changes and old o32 layout2 trials have losing or irrelevant comparator evidence. The GV proposal retains selected short-chain geometry; the Lt proposal concerns current M256 final selection, not repeating o32 layout search.
- Historical EP early zero polling was neutral/losing and is not the EH proposal. Current EH is already shipped and selected on qkv/pooler128; the new lifetime topology must compare against that actual EH parent.
- RQ11 byte ledgers do not establish speed; native-X/X4/STMatrix parent progress and source commutativity do not establish a win over the actual incumbent. None is offered as the default comparator here.
- PW39 close-finalist confirmation is already a distinct experiment; reuse its validated evidence rather than launch a duplicate RQ-selection trial. The Lt proposal targets an unchanged different selection function and requires current Lt config capture.

The inspected local JIT is the **PW38 parent arm** of the archived PW39 fullscope run, whose source hash equals the official frozen release. It is local H100 sm90a evidence, not an official H200 instruction dump or an instruction-to-time model. QKV1 GV has64registers/0stack/0local and129 static PRMT; router8 P1 has56registers/0stack/0local with ticket atomic before first LDG; RQ32ST6 and RQ128EH each have168registers/0stack/0local. Exact file hashes and selected instruction PCs/opcounts are preserved in JSON. Current vendor M256 symbols are in the scored profile, but exact vendor SASS and selected Lt config IDs are unavailable and remain unknown.

No minimum0.3/0.5us retention cutoff, forced production pick, reduced precision, timing/output hardcoding, activation-derived cached answer or automatic official promotion is proposed. A valid tiny effect is retained; its reproducibility, interaction and full cost determine the next decision.

Read-only `python3 -m proj_rsi.learning validate` also passes: registry3,11lesson versions/9latest lessons and13idea versions, GPU_calls0. Catalog coverage gaps remain explicit in the JSON; unavailable historical raw logs and failed/no-archive jobs are not silently promoted into measured mechanisms.

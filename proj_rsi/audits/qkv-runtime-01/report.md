# QKV128 split4 trial01: valid full-pipeline regression

The attempt is valid and the candidate is slower in **all30paired rounds against both actual selected controls**. Fresh pack+split4main+FP32reduce takes40.288us median, versus17.408us PW36 and17.088us RS32. No candidate-positive sample or tiny gain is hidden by a threshold. This is a result for the exact version/shape, not a family ban.

Reproduce: `python3 proj_rsi/audits/qkv-runtime-01/analysis.py`. [Program](analysis.py), [complete analysis](analysis.json), [original result](../../../results/proj-rsi/scheduler/artifacts/research.qkv.split4.harness.gpu.01/collection-1793077707233847694/logs/scheduler/qkv-runtime-01/run/result.json), [inventory](../../../results/proj-rsi/scheduler/artifacts/research.qkv.split4.harness.gpu.01/collection-1793077707233847694/inventory.json). Analysis uses CPU/std-library and offline cuobjdump only: no CUDA context, extension import/load, prepare, benchmark, API or queue change.

## Terminal and source validity

Job `research.qkv.split4.harness.gpu.01`, container `okc-proj-qkv-runtime-01`, terminal2026-10-27 05:07:53UTC exit0. Driver status is `VALID_LOCAL_COMPARISON_REVIEW_REQUIRED_NO_OFFICIAL_GAIN`. All12 inventory files and46 dispatch/source pins match; missing list empty. Collector confirms no guard/compute PIDs or containers, absent container,0MiB GPU memory and no cleanup failure. Driver records all captured graphs drained/reset. This was not a timeout/setup failure or invalid timing attempt.

Frozen runtime driver a218e398 and manifest efff5355 preserve the candidate device/host wrapper and original PW36/RS32 full prepares. Exact candidate extension source29f7f11b, copied realSO94865053 and dumpedSASS328f274a match recorded hashes. This analysis separately redumps the archived realSO **offline**, then compares complete PC/predicate/instruction streams for all three kernels against both collected SASS and frozen canonical hostSASS: main872instructions, pack56, reduce40. Physical A/D/fence gate passes two32word A banksR56..87/R88..119, D R24..55, two groups of8MMA per staticloop, two iterations, outlined paths, generic payload immutability while async and finalwait0 before epilogue. CPU source/coverage/CSR proofs remain bound through the manifest.

Main runtime registers122/local0,512threads/rounded65536pool, postoptin occupancy1; source-defined dynamicSMEM103600 and successful postoptin check are retained. Pack22 and reduce16registers,0local,256launch threads within reportedmax1024. The preoptin maxDynamicSharedSizeBytes was49152; hostoptin occurs before capture and checks adequate capacity. Runtime sharedSizeBytes0 and cuobjdump's SHARED1024 are kept as distinct reported fields, not forced into equality or converted into occupancy by guesswork.

Both controls are their **actual freshly prepared original kernel_fn**, with no forced route, reset preparation clock or extra output-copy node in timed graphs. Both selected `s4s5hz`, NT128/ST5/GS1/FL3/KS7, source9bdf5651 and identical actual selectedSO/SASS hashes,168regs/0local/144520shared. The analysis regenerates each captured selected CUDA source from its hash-bound submission AST and verifies exact equality, selected index/geometry and one-node profile. RQ decline/fallback would have invalidated the source-bound driver. Parent realSOs are recorded by hash rather than copied in this collection; their generatedsource/SASS/resources and selectedroute are archived.

## Numerical, graph and persistent-state gates

All8192basis columns are exact at x8 for each of PW36, RS32 and new. Normal/outlier task checks and FP64relativeL2<=0.004/maxrow<=0.02 pass for all three: candidateFP64 errors0.001606/0.001650, versus controls approximately0.00370–0.00379. Those accuracy results do not establish a performance benefit.

Actual task W roundtrips every BF16 bit; real CSR capacity32 passes guarded allocation, full2MiBXP and4.5MiBFP32 partial workspaces are pipeline-owned. Storage-only probing covers all65536BF16payload patterns through the actual pack; nonfinite arithmetic from that storage diagnostic is not timed or admitted as numerical evidence. Fresh normal inputs and workspaces are revalidated afterwards, and a separately allocated Pipeline passes ownership/poison/reconstruction gates. Every call performs fresh current-X pack, main writing all four partial planes, and ordered FP32 reduction.

All three actual last F1 outputs and all three actual last paired outputs are checked before any replay, input mutation, profile or repair: **six strict terminal oracles**. F1lastseeds are PW36720104/RS32720083/new720062; pairedlastseed730029 for all. CurrentX, exactXP and finite/all-written partials remain bound to the new arm's own recorded fixture; output equals ordered reduction of those partials. Forty-five graph mutation oracles cover original/zero/freshnormal/freshoutliers/allrowprobes at pre-timing, after-storage-bits and post-timing. NaN-poison/current-input checks pass. Profiles after both strict terminal boundaries show controls each oneRQnode and candidate exactly **pack→split4main→reduce**, not a resident/probe/main-only substitute.

Counter contracts are storage-based, not a blanket zero rule. Across54contexts, all49counter paths are checked:216active reset0 records and2430dormant-unchanged records. Both aliases of the selected QKV counter storage (m128 andm256) reset0; all other storages retain prepare digests/nonzero counts. Twenty-eight dormant paths legitimately retain nonzero values. No harness counter zeroing repairs them. Source-bound success also requires all original weights/immutable packed weights, escapes and metadata unchanged; raw tensor values are not serialized, so these remain successful runtime source assertions with saved digests.

## Unfiltered timing

All210raw records in result.json exactly equal raw.jsonl and the preregistered plan:15warmups+105F1+90paired calls. F1 uses21samples in PW36-before/RS32-before/new/RS32-after/PW36-after blocks. Thirty paired rounds use all six permutations exactly five times, same fresh fixture within each round; every arm occupies every position10times. No deletion, best-order selection or own-control-only comparison occurs.

Positive gain means control minus new, in microseconds:

|Actual control|F1 control / new|Conservative F1 gain|Paired median gain|Paired W/T/L|Median new/control ratio|
|---|---|---:|---:|---|---:|
|PW36 selectedRQ|17.440 /40.288|−22.848|−22.848|0/0/30|2.313×|
|RS32 selectedRQ|17.152 /40.288|−23.136|−23.184|0/0/30|2.358×|

PW36 paired gain IQR is−23.160…−22.544us, full range−24.000…−21.376. RS32 IQR is−23.536…−22.776us, full range−24.160…−21.664. When each control precedes/follows new, medians remain strongly negative: PW36−23.040/−22.816us, RS32−22.976/−23.328us. Every observation is retained; there is no small candidate-positive result to promote. Identical selected control binaries themselves differ by+0.352us median PW36-minus-RS32, which is a timing-context observation, not an optimization delta. Its scale does not account for the approximately23us candidate gap.

## What this establishes and what remains unknown

The version successfully increases split coverage to144CTAs and preserves a safe canonical8-MMA pipeline, but those properties did not make the full QKV pipeline competitive with the126CTA selectedRQ. A feasibility/coverage model and low register count cannot substitute for actual-policy timing.

A bounded source payload model gives candidate main72MiB logical X loads versus selected control36MiB X TMA payload for this geometry. Candidate also reads/writes2MiB each in freshpack and writes/reads4.5MiB each in partial/reduction. These are source-derived logical payloads, excluding metadata and cache response, **not measured DRAM traffic, request sector counts or elapsed costs**. The model motivates testing Xdelivery/front-end demand and auxiliary pipeline costs, but this attempt only records profile names, not per-node durations/counters. It cannot attribute the23us gap to pack, reduction, split tail, decoded-B publication, X scoreboard/fences, hints or instruction issue.

The next discriminating test, only if root assigns enough information value, is one bounded source-frozen full-pipeline diagnostic comparing the actual selectedRQ node with candidatepack/main/reduce. Retain real-output/JIT/immutability/counter/cleanup gates and whole-pipeline controls. Measure whether auxiliary pack/reduce cost could plausibly explain the gap; otherwise isolate mainloop Xrequests versus producer/Bpublication exposure using actual QKV-specific PC maps and denominators. Per-node counters/durations remain diagnostic, and PCfractions are not microseconds. This chooses a concrete next mechanism; it does not authorize an unchanged timing sweep or infer a gain from cold node replay.

Retain this valid regression and source as a control for later changed versions. Do not integrate or score it, assume smaller split/register/store changes recover23us, transfer its source-cost ratios to wall time, or reject the whole reverse/register-A family. Official H200 score remains unmeasured.

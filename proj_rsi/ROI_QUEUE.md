# Projection ROI queue: muse_bf16_proj below 3ms

Only `muse_bf16_proj` is active. Official incumbent `9916d34357b745cf` is **3157.038403653us**; the gap to3000us is **157.038403653us** (4.9742% score reduction). Latest archived board: **62/1173**. Source `501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9`. [Machine view](roi_queue.json).

This is the coordinator's projection planning priority view. The existing root-controlled exclusive GPU worker owns execution. Root observed **0 runtime READY jobs at21:33UTC**. CPU assignment/review readiness does not mean GPU ready; this observation is time-bound. Expected gains, probabilities and reviewed wall costs remain unknown where evidence is absent; numeric expected ROI is unrankable there. Conditional score scenarios use this exact official job's six-load arithmetic, not measured gains or forecasts. No quota is assumed.

|Priority|Projection work|Planning readiness|Actual runtime READY|Next falsifier / uncertainty|
|---|---|---|---|---|
|1|Rolled GV keyed-half repair screen|CPU_SOURCE_NATIVE_PROOF_READY_RUNTIME_STATUS_UNKNOWN|No; root observed0READY at21:33UTC|Matched current-source direct screen on poolerM1/M8 with exact tuple/current input/own timed output and source+resource evidence; preserve signed AB/BA/F1. Then test actual selected policy against GV9916 before any release.|
|2|RQ32 selected pooler/unpooler escape and loop scheduling|CPU_IMPLEMENTATION_ASSIGNED_NO_RUNTIME_READY_JOB|No; root observed0READY at21:33UTC|Pin actual selected ST6/FL35/KS4(pooler32) andKS2(unpooler32); implement one bounded changed-loop candidate versus current GV9916 source, preserving real timed graph numerics/mutation/output and every signed sample. Prior generic cursor results do not establish a win.|
|3|GV16 selected predxor repair/decode specialization|CPU_IMPLEMENTATION_ASSIGNED_NO_RUNTIME_READY_JOB|No; root observed0READY at21:33UTC|Compile and inspect only two exact selected M16 native functions against frozen501d source. Measure one supported change with actual controls; pooler at128registers makes liveness/spill risk explicit. Reuse unchanged codec/math proof and retain every signed runtime result.|
|4|Audit route-specific historical positives against current selections|VERSIONED_SUPPORTING_AUDIT_COMPLETE_NO_NEW_EXECUTABLE_READY|No; root observed0READY at21:33UTC|Use the completed versioned positive audit to scope an exact route-only current-control experiment. No executable or runtimeREADY job was produced.|

The keyed-half source is CPU/native-proof ready: three fewer non-NOP instructions,64 registers,zero local stack/spills in the targeted static comparison. **No performance gain is measured.** Its exact scored poolerM1/M8 routes account for **12.5762%** score elasticity. If both selected latencies fell1%, the conditional same-job score reduction would be **3.975372us**; this is an arithmetic scenario.

Priority2 targets actual selected RQ32 pooler/unpooler ST6/FL35/KS4/2, **9.7767%** combined score elasticity. Prior generic cursor losses remain attached to their actual source/control. Selected RQ overall covers **26.4267%** score elasticity across seven scored routes. A conditional10% cut to every selected RQ latency changes this job's score by85.263707us; it is not a gain estimate. Source availability, actual selected flags/KS, local result and official outcome must remain distinct.

Priority3 targets actual selected GV16 pooler/unpooler predxor, **9.6855%** combined score elasticity. Static2000/2080instruction and128/110register counts with16P2R instructions each define a small codegen question; they do not measure dynamic loop fraction or gain. CPU implementation is assigned; no runtimeREADY job is asserted.

The [priority4 completed supporting audit](../audits/projection-positive-recovery-20261027/independent-evidence.json) gives zero direct score credit. Its1,066 positive bake rows are observations, all feature_level=false;321match their own final bake label and745do not, which says nothing about current scored dispatch. It treats O32-only margins and private-EH observations as comparator-specific questions. Local O32 full-policy paired0.999842644 versus F11.001866895 conflicts; its local GV45 parent differs from the official RQ-ST5 route. Private-EH scored3207.568us lost. Complete d1ca3187.142400us and55f73186.744516us already scored and lost to3157.038404us. No isolated gains may be added or rebranded as unsubmitted wins. O32 target+.128us paired/+.080F1, pooler128 private-free target+.208/+.224us, and QKV128 fixed+.256/+.128us remain comparator-specific. QKV128 was actually unselected and its policy contrast lost−.128/−.512us. These are two selected local target positives plus one unselected diagnostic, not three deployable increments.

|Top selected scored cell|Latency us|Score elasticity|Conditional10% cell-cut score reduction us|
|---|---:|---:|---:|
|decode 1 / pooler M1 (gv_predxor)|13.714964|6.4617%|20.737335|
|verify 1 / pooler M8 (gv_predxor)|13.536974|6.1145%|19.605741|
|decode 32 / pooler M32 (rq)|15.522677|4.9625%|15.864680|
|decode 16 / unpooler M16 (gv_predxor)|14.345175|4.8563%|15.520992|
|decode 16 / pooler M16 (gv_predxor)|14.265114|4.8292%|15.433298|
|verify 32 / pooler M256 (Lt/vendor)|24.084212|4.8263%|15.424107|
|decode 32 / unpooler M32 (rq)|15.059097|4.8143%|15.385046|
|verify 16 / unpooler M128 (Lt/vendor)|18.543867|4.7752%|15.258710|
|verify 16 / pooler M128 (rq)|17.976854|4.6292%|14.786623|
|verify 32 / unpooler M256 (Lt/vendor)|23.087707|4.6266%|14.778369|

C11 frontier stays parked: the valid preloaded-metadata version improves only against slowerV2 (+2.192/+4.208us), while currentC12 remains faster by3.216/8.384us in the two fixed cells,0/24 candidate wins each. A new direct current-control hypothesis and bounded cost can reopen it; this is not family retirement.

All serving/MoE experiments are historical archive only and have no active queue entry here. Their sources, signed results and validity limits remain in [history](HISTORY.md). No new serving/MoE analysis or GPU work is authorized by this view.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](ITERATION.md).

# PW38 official unscored check: author analysis

Check `a3ed4ce5886a4415` passes9/9 official tests and all30timed shapes. Source remains `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`. Recomputed speedup is1.512632538066x and score is3193.637501793us, exactly reproducing the official result. This is an unscored check; no leaderboard promotion follows automatically.

The speedup95% interval is[1.489943050,1.535322026], equivalent to score[3146.440888,3242.271575]us. The point score is6.977507us below RS32 and18.777950us below PW36, but both older official intervals overlap this one. These cross-job differences are context, not incremental causal claims.

## Actual preparation and active stack

RQ and RQ32 background builds are ready at41s; r6 ends120s; RQ ends120s; GX122s; RQ32 ends123s; router build takes37s and final preparation ends161s. There are no skipped-budget, INVALID, ERROR or traceback lines. The original240s guards and documented300s prepare ceiling are respected in this official H200 run. Future cold/scored compilation remains a separate observation.

EH is genuinely active on qkv128 and pooler128: both profiles show `rq<128,5,1,0,0,0,65539>`. Router128/256 both show the inherited `rtk<256,3,128,128,2,4>`. All four projections atM32 plus qkv16 select the inherited small-NT RQ branch. Thus the actual checked file serves the merged branches and EH, rather than merely containing dormant source. Unpooler128/o128 and all four BF16 M256 projections retain r6/cuBLAS; EH is not forced.

|Same-job prepare comparison|Control us|Selected/new us|Printed gain us|
|---|---:|---:|---:|
|pooler128 hint RQ → EH|22.98|22.85|+0.13|
|qkv_gate128 hint RQ → EH|17.22|16.80|+0.42|
|unpooler128 hint RQ → EH|22.94|23.10|-0.16|
|o128 hint RQ → EH|12.86|13.38|-0.52|
|router128 old RW → RTK|12.93|11.49|+1.44|
|router256 old RW → RTK|16.03|13.18|+2.85|
|pooler32 R6 F1 → small RQ|20.96|19.39|+1.57|
|qkv_gate32 R6 F1 → small RQ|18.18|14.46|+3.72|
|qkv_gate16 R6 F1 → small RQ|14.56|14.05|+0.51|
|unpooler32 R6 F1 → small RQ|20.64|19.68|+0.96|
|o32 R6 F1 → small RQ|10.82|10.59|+0.23|

Pooler EH gains a printed0.13us versus the same-geometry hinted control; qkv EH gains0.42us versus its hinted control (0.35us versus the best old non-EH candidate). Those sequential prepare medians are not an isolated balanced production estimate. Unpooler/o hinted EH lose0.16/0.52us and are unselected; all tiny outcomes remain visible. The1% bake gate is against R6, not a separate1% EH-over-old-RQ requirement.

Router savings belong to PW35 RTK already included in PW36, not EH. The router R6 conservative F1 baselines are13.82us atM128 and15.14us atM256; RTK11.49/13.18 is below them by2.33/1.96us. The qkv/pooler/unpooler/o small-M RQ gains likewise establish inherited-package selection against this job’s actual R6 baselines, not that EH produced those savings.

## Original RS32 and PW36 component controls, each within its own job

These comparisons preserve the original raw-job controls instead of attributing cross-job changes to a source feature. RS32 final preparation is142s; PW36 is163s; the new check is161s. Original RS32 has no RTK or EH, while PW36 has RTK and no EH. Their small-NT choices are recorded exactly rather than assuming every prepare chooses the same plan.

|Original job / shape|Its own control|Control us|Its selected us|Its printed gain us|
|---|---|---:|---:|---:|
|rs32 / pooler32|same-job R6 F1|21.02|19.39|+1.63|
|rs32 / qkv_gate32|same-job R6 F1|17.95|14.05|+3.90|
|rs32 / qkv_gate16|same-job R6 F1|14.56|14.05|+0.51|
|rs32 / unpooler32|same-job R6 F1|20.96|19.17|+1.79|
|rs32 / o32|same-job R6 F1|10.94|10.50|+0.44|
|pw36 / pooler32|same-job R6 F1|20.93|19.49|+1.44|
|pw36 / qkv_gate32|same-job R6 F1|18.40|14.14|+4.26|
|pw36 / qkv_gate16|same-job R6 F1|14.78|14.11|+0.67|
|pw36 / unpooler32|same-job R6 F1|20.93|19.49|+1.44|
|pw36 / router128|same-job rw64k32s8z|13.06|11.84|+1.22|
|pw36 / router256|same-job rw64k32r3x2z|16.00|13.31|+2.69|

PW36 retains R6 for o32: its best RQ10.72us misses the1% threshold against R610.82us. Original RS32 and the new check select different same-family RQ alternatives there. Original RS32 router256 retains R615.33us against losing RW15.97us; PW36/new RTK wins against the old RW inside their respective jobs. These facts establish which compatible branches are active; differences between these bake deltas across jobs are not an isolated causal estimate. Full old PREP/BAKE/PICK arrays and line references are preserved in the author JSON.

## All30 actual profiles and calibrated times

Times are this check’s calibrated official microseconds. Profile durations are retained in JSON and are not substituted for these values. Each of the30profiles contains one kernel/call.

|Load|Layer|M|This check us|RS32 us, other job|PW36 us, other job|Actual kernel|
|---|---|---:|---:|---:|---:|---|
|decode 1|qkv_gate|1|9.231521|9.199055|9.159184|void gv<1, 1, 1, 2, 16, 8, 1, 1, 2, 1>(G)|
|decode 1|o|1|4.960973|4.967561|5.285687|void gv<1, 1, 2, 1, 8, 2, 1, 1, 0, 1>(G)|
|decode 1|pooler|1|13.703649|13.462036|13.907327|void gv<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|
|decode 1|unpooler|1|4.497523|3.996917|4.322471|void gv<1, 1, 2, 1, 16, 4, 1, 1, 0, 0>(G)|
|decode 1|router|1|3.699275|3.665705|3.750904|void rr<1, 1, 1024>(float4 const*, __nv_bfloat16 const*, ...|
|decode 16|qkv_gate|16|10.125109|9.663936|10.456028|void rq<16, 5, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|decode 16|o|16|5.536121|5.344963|5.201558|void gv<1, 2, 2, 1, 8, 2, 1, 1, 0, 0>(G)|
|decode 16|pooler|16|14.281962|14.166533|14.088955|void gv<1, 2, 2, 2, 8, 16, 1, 1, 0, 1>(G)|
|decode 16|unpooler|16|14.342607|14.420909|14.563461|void gv<1, 2, 2, 1, 4, 16, 1, 1, 2, 1>(G)|
|decode 16|router|16|4.722648|4.695848|4.868874|void gv<2, 2, 1, 1, 8, 1, 16, 1, 0, 536870912>(G)|
|decode 32|qkv_gate|32|10.414919|10.107945|10.281358|void rq<32, 5, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|decode 32|o|32|6.292522|6.365771|6.831209|void rq<32, 6, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|decode 32|pooler|32|15.081659|15.538102|15.498864|void rq<32, 5, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|decode 32|unpooler|32|15.287451|15.149120|15.209333|void rq<32, 6, 1, 0, 0, 0, 3>(RqP, RqTmap, RqTmap)|
|decode 32|router|32|4.964317|4.978521|5.217114|void gv<2, 4, 2, 1, 8, 2, 8, 1, 0, 134217728>(G)|
|verify 1|qkv_gate|8|9.547312|9.316762|9.431596|void gv<1, 1, 1, 2, 16, 8, 1, 1, 2, 1>(G)|
|verify 1|o|8|5.385221|5.274411|5.292240|void gv<1, 1, 2, 1, 8, 2, 1, 1, 0, 1>(G)|
|verify 1|pooler|8|13.983762|13.718591|13.877146|void gv<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|
|verify 1|unpooler|8|4.274938|4.222229|4.330138|void gv<1, 1, 2, 1, 8, 2, 4, 1, 0, 536870913>(G)|
|verify 1|router|8|4.588521|4.953340|4.909485|void gv<2, 1, 1, 1, 8, 1, 16, 1, 0, 536870912>(G)|
|verify 16|qkv_gate|128|13.361366|13.207008|12.922156|void rq<128, 5, 1, 0, 0, 0, 65539>(RqP, RqTmap, RqTmap)|
|verify 16|o|128|7.139444|7.550518|7.548773|nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNN|
|verify 16|pooler|128|18.560421|18.399035|18.151804|void rq<128, 5, 1, 0, 0, 0, 65539>(RqP, RqTmap, RqTmap)|
|verify 16|unpooler|128|18.625502|18.951375|19.103500|nvjet_sm90_tst_128x64_64x8_1x1_v_bz_coopB_NNN|
|verify 16|router|128|7.808118|9.403482|7.784168|void rtk<256, 3, 128, 128, 2, 4>(RtMap, RtMap, RtMap, flo...|
|verify 32|qkv_gate|256|18.653822|17.276527|17.768146|nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN|
|verify 32|o|256|9.599879|8.799157|9.184376|nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT|
|verify 32|pooler|256|25.126491|24.838240|24.369707|nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN|
|verify 32|unpooler|256|23.472195|24.693354|24.487153|nvjet_sm90_tst_128x128_64x6_2x1_v_bz_coopA_NNN|
|verify 32|router|256|9.473456|10.977570|9.271228|void rtk<256, 3, 128, 128, 2, 4>(RtMap, RtMap, RtMap, flo...|

The two comparator columns are separate official jobs, with separate SGLang calibrations and prepare choices. They are not a same-run causal ablation or stitched proposal. In particular o256 is slower than this check’s SGLang reference, and several other small-shape times exceed the RS32 job despite the lower aggregate point score.

## Exact score recomputation

|Load|Candidate sum us|Same-job SGLang sum us|Ratio|
|---|---:|---:|---:|
|decode 1|36.092941506|60.000000522|1.662374914|
|decode 16|49.008447856|76.288000215|1.556629593|
|decode 32|52.040868940|77.231999952|1.484064381|
|verify 1|37.779753815|75.343999080|1.994295660|
|verify 16|65.494851416|86.463999934|1.320164838|
|verify 32|86.325843104|102.272000629|1.184720554|

Use the geometric mean of those six same-job ratios, then4830.8/speedup. Multiplying62 by the geometric mean of candidate sums is not the exact official score. All30rows have60candidate/parent/baseline samples and60paired ratio records;1800candidate timing samples are retained in the source JSON.

## Identity, recovery and disposition

The immutable release snapshot is `results/proj-rsi/releases/pw38-v1-29e557ea53f3/submission.py`; before/after submission hashes match29e557…. The accepted check preceded a missing-experiment bookkeeping exception; root recovered the same API job id and archived the result without resubmission. The normalized report and recovered submitted descriptor name the exact snapshot. Server content_hash/payload_ref are preserved as opaque identifiers rather than falsely equated to local SHA256.

Author CPU gates verify source bytes, all30profile/benchmark pairs,9/9summary,60samples per shape, finite positive timing, no missing phases and exact score arithmetic. `pw38-official-check-author.json` retains all PREP/BAKE/PICK lines, same-job component comparisons, cross-job context and artifact hashes. No API request, GPU work, queue edit or scored release was performed by this analysis.

Conclusion: official correctness and execution feasibility pass for this exact cumulative file; the modest lower point score remains within reported uncertainty. Independent review and root’s separate scored-release decision are required. The incumbent remains RS32 until a scored result and explicit review justify changing it.

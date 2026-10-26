# pw36 official independent review — valid integration, no new incumbent

Job `e22a4b9d693d4902`, problem `muse_bf16_proj`. Source frozen `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`; archived launch handle records the same digest and current file still matches. Reviewer read raw JSON, sibling judge stdout, actual30 profile reports and ledger; no submission/GPU/ledger action performed.

**9/9 official tests pass;30/30 benchmark rows pass,60 runs each;30 actual profiles present.** Score3212.415452073451us, speedup1.503790550155013, CI[1.4812336919026878,1.526347408407338]. Independent recomputation from the six paired load-sum ratios reproduces1.5037905501550128 and4830.8/speedup=3212.415452073452us.

Incumbent remains rs32 job `fb548b5dc6f04696`,3200.6150084373335us. Pw36's point estimate is+11.800444us (+0.368693%) above it; the reported CIs overlap, and there is no demonstrated aggregate improvement or<3000us result. Do not promote, stitch per-shape minima across jobs, or use the approximate62*geomean shorthand as exact official arithmetic.

## Judge preparation and same-run bake evidence

Raw stdout confirms r6 done121s,rq122s,gx124s,rq32 begins124/finishes125s, router extension compilation38s at163s, router/final163s. No skipped-budget, failure, INVALID or ERROR lines. Both merged mechanisms were actually available within preparation budget; this result is not a skipped-phase fallback accident.

|shape|same-run comparator us|actual selected path us|evidence|
|---|---|---|---|
|qkv32|r6 F118.40|rq:t32s5hze-k7 14.14|selected; profile rq32/FL35|
|pooler32|r6 F120.93|rq:t32s6hze-k4 19.49|selected; profile rq32/FL35|
|unpooler32|r6 F120.93|rq:t32s5hze-k2 19.49|selected; profile rq32/FL35|
|qkv16|r614.78|rq:t16s5hze-k7 14.11|selected; profile rq16/FL35|
|o32|r6 F110.82|r6 retained|best rq10.72 misses sequential1%margin (threshold10.7118); profile one Nvjet kernel|
|router128|old rw13.06; r6 F114.02|rt64k32b3x2z11.84|actual pick;11.74 alternatives fail sequential1%margin after this pick|
|router256|old rw16.00; r6 F115.55|rt64k32b3x2z13.31|actual pick and profile|

Router F1 raw details: m128 r6 pre15.42/post14.98/initial14.02 ->14.02; m256 pre15.84/post15.55/initial15.55 ->15.55. Comparing the actual router pick to old rw gives-1.22/-2.69us inside this judge preparation. Bake times and calibrated scored/profile times are different observation contexts; they must not be substituted numerically for each other.

## Actual measured dispatch and aggregate

All30 profile reports show one GPU kernel/call per shape. Router128/256 are actual `rtk<256,3,128,128,2,4>` at6.2/8.0us profiler duration; calibrated scored times7.784168/9.271228us. O32 is one Nvjet TNT4.5us, o128 one Nvjet coopB NNN5.5us. These observations establish dispatch; benchmark-minus-profile differences are not a proven extra node or free recoverable latency.

|load|pw36 candidate sum us|same-run SGLang sum us|rs32 candidate sum us (cross-run context only)|pw36-rs32 us|
|---|---:|---:|---:|---:|
|decode1|36.425574|60.048000|35.291272|+1.134302|
|decode16|49.178875|75.519999|48.292190|+0.886685|
|decode32|53.037879|76.528001|52.139460|+0.898419|
|verify1|37.840604|75.600001|37.485333|+0.355272|
|verify16|65.510400|86.432001|67.511418|-2.001018|
|verify32|85.080610|102.192001|86.584849|-1.504239|

Cross-run router comparisons are9.403482->7.784168 (-1.619314) at128 and10.977570->9.271228 (-1.706342) at256, consistent with the same-run router bake mechanism. They are not a controlled rerun or an independent aggregate speedup proof. Smaller-load sums rise, offsetting the large-router gains under the official geometric aggregation. Dispatch can also change through fresh bakes: rs32 o32 profile was rq while pw36 retained r6. Therefore assigning every cross-run small-shape change to sampling drift alone is unsupported; source-equivalent paths, selection variation and environment variation are not separated by this evidence. SGLang load-sum geomean also changes+0.110938% across these jobs; raw own-sum geomean changes+0.480040%, reinforcing why paired official ratios matter.

Disposition: retain pw36 as a numerically validated merged reference; keep rs32 as score incumbent, preserve router/rq wins as mechanism evidence, and require new measured ROI for another scored run. Do not spend a slot on an unchanged rerun merely to harvest favorable drift. Ledger/queue ownership stays with root.

## Raw identity

- Job JSON `results/proj-rsi/jobs/e22a4b9d693d4902.json`, SHA256 `4d7aaadfdd8f212ac93273d51952227a3333ea2916a553180733c4f3fe02a0b6`.
- Actual judge stdout sibling `.log`, SHA256 `508db1c63274f49396993b79888ef7f2e4ccacaaab2a14a9b0c6895088fd0eb6`.
- Actual profile copies `proj_rsi/candidates/pw36/profiles/submission.e22a4b9d693d4902.{json,txt,log}`. The derived `pw36-scored-report.txt` was cross-checked against primary JSON/log; it does not replace them.
- Archived source handle `results/proj-rsi/takeover/20261026/pw36-scored-handle.json` binds the frozen SHA to this launch. Server result contains submission path rather than uploaded-source digest; execution lineage is supported by the frozen local artifact/handle, not a server-side source attestation.

# PW38 full-stack local result: author analysis

Valid full-scope local execution, with mixed shape results and a modest aggregate point gain. PW38 actually serves EH on qkv128 and pooler128 while retaining the merged RTK router. This is not an official H200 result, and it does not establish that every included mechanism improves every shape.

Frozen source `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`; driver `ddb48ef087790ceab114d651413999ca9a15b3c1bf5c24df99cfd2329ac9db15`. Job `pw38.stack.fullscope.01`, guard409660, exited0 at2026-10-27T04:48:31Z. The independent collected cleanup observation has no guard/container/compute process, zero GPU memory and no cleanup failure. Exit0 alone is not the validity basis.

CPU reanalysis reproduces every raw call and declared statistic without filtering:4350calls across30shapes,222task checks (65normal plus9official-distribution entries per arm),180immediate terminal-output oracles,360retained-graph mutations,15actualJIT archives and30per-shape graph drain/resets. The final empty graph-cleanup list means no graphs remained after those explicit resets. All artifact/source pins verify; no build/JIT errors or skipped-budget/INVALID/ERROR log lines appear.

Preparation import-clock totals were58.156s RS32,44.601s PW36 and59.020s PW38, without resetting clocks; each joins two compiler threads. This confirms this H100 run, not fresh H200 compilation/preparation feasibility. Runtime frozen-driver assertions cover current-X bytes, NaN output and floating-workspace overwrite, immutable weights/pack/escapes, physical counter-arena progress/reset/dormant semantics and distinct mutable workspaces. Successful tensor bytes are not reconstructed after process exit; the retained evidence is the reviewed instrumented execution and its complete records.

All three main RQ128 extension archives pass their targeted NT128/ST5 canonical PC/predicate/instruction and168-register/0-local gates; PW38 additionally passes the four-function EH/control set and EH physical checks. All six RQ extension archives are retained without gate errors. Additional CPU comparison proves all11common RQ32 and6common RQ128 functions are instruction-for-instruction identical across RS32/PW36/PW38 actual JITs; EH adds its two functions. Selected RQ resource queries report168registers and0local with exact expected shared bytes.

## Same-run full-policy observations

Positive deltas below mean PW38 is faster: control minus PW38, microseconds. Paired values are medians of30paired differences, not differences of marginal medians. F1 uses the lower control-before/after median minus the PW38 median. Every small win, tie and loss is retained.

|Shape|RS32 median|PW36 median|PW38 median|Paired vs RS32|Paired vs PW36|F1 vs RS32|F1 vs PW36|
|---|---:|---:|---:|---:|---:|---:|---:|
|decode1/qkv_gate|14.480|14.496|14.416|+0.032|+0.128|-0.064|+0.064|
|decode1/o|9.088|9.024|9.008|+0.144|+0.048|+0.128|+0.192|
|decode1/pooler|21.760|21.696|21.648|+0.000|+0.048|-0.160|+0.000|
|decode1/unpooler|7.472|7.616|7.456|+0.032|+0.064|+0.000|+0.064|
|decode1/router|6.976|7.024|7.072|-0.016|-0.032|+0.000|+0.000|
|decode16/qkv_gate|14.800|14.752|14.672|+0.096|+0.144|+0.160|+0.000|
|decode16/o|8.832|8.864|8.960|-0.096|-0.080|-0.096|-0.096|
|decode16/pooler|21.632|21.616|21.552|+0.144|+0.016|+0.384|+0.128|
|decode16/unpooler|21.920|22.048|21.888|+0.064|+0.192|+0.416|+0.256|
|decode16/router|7.984|8.016|8.064|-0.096|-0.048|-0.192|-0.096|
|decode32/qkv_gate|15.616|15.760|15.552|+0.160|+0.224|-0.448|-0.256|
|decode32/o|9.984|10.032|10.064|-0.096|-0.064|-0.160|+0.000|
|decode32/pooler|22.224|22.400|22.560|-0.240|-0.128|-0.192|-0.128|
|decode32/unpooler|22.208|22.144|22.128|+0.080|+0.000|+0.160|+0.000|
|decode32/router|8.256|8.288|8.224|+0.096|+0.032|+0.000|-0.064|
|verify1/qkv_gate|14.384|14.496|14.384|+0.000|+0.112|+0.256|+0.192|
|verify1/o|8.928|9.024|8.992|-0.032|+0.064|-0.128|-0.096|
|verify1/pooler|21.392|21.392|21.376|-0.080|+0.016|-0.224|-0.032|
|verify1/unpooler|7.152|7.344|7.312|-0.128|+0.064|+0.096|+0.032|
|verify1/router|7.696|7.744|7.712|+0.032|+0.032|-0.032|+0.000|
|verify16/qkv_gate|17.408|17.552|17.488|-0.032|+0.080|+0.416|+0.160|
|verify16/o|11.280|11.360|11.248|+0.048|+0.112|-0.256|-0.096|
|verify16/pooler|24.624|24.576|24.320|+0.320|+0.224|+0.288|+0.384|
|verify16/unpooler|24.160|24.352|24.240|+0.016|+0.192|+0.352|+0.160|
|verify16/router|11.776|10.704|10.784|+1.024|-0.048|+0.960|-0.032|
|verify32/qkv_gate|21.744|22.208|21.760|+0.096|+0.544|-0.064|+0.224|
|verify32/o|12.048|12.112|12.080|+0.032|+0.064|-0.064|-0.096|
|verify32/pooler|29.888|29.920|29.456|+0.384|+0.400|+0.608|+0.064|
|verify32/unpooler|28.000|28.320|27.856|+0.096|+0.384|+0.032|+0.320|
|verify32/router|15.152|13.280|13.328|+1.824|-0.048|+2.208|-0.032|

PW38 has19shape paired wins/2ties/9losses versus RS32, and22wins/1tie/7losses versus PW36. These counts do not weight load importance or prove repeatability.

## What can be attributed

- **Pooler128:** all arms choose NT128/ST5/KS4 with hints; PW38 alone enables EH. Paired gain is+0.320us versus RS32 and+0.224us versus PW36; F1 is+0.288/+0.384us. PW38 wins17/30 and20/30 pairs respectively. This is current same-geometry evidence favorable to EH, with noise: bootstrap median intervals are[-0.192,+0.464]us versus RS32 and[+0.048,+0.480]us versus PW36. One batch does not settle cross-run reproducibility.
- **QKV128:** same NT128/ST5/KS7 geometry, EH selected. Paired result is−0.032us versus RS32 and+0.080us versus PW36; F1 is+0.416/+0.160us. Both paired bootstrap intervals cross zero. Retain the F1 gains, but this is inconclusive paired EH evidence.
- **Router128/256:** PW38 retains PW35 RTK ids201/200. Against RS32 old RW ids126/107 it gains+1.024/+1.824us paired and+0.960/+2.208us F1, winning30/30 pairs for both shapes. Against PW36, which selects the same RTK variants, each paired difference is−0.048us and F1 about−0.032us. The router improvement belongs to the already merged RTK branch; it is not a new EH gain.
- **Pooler32 regression:** PW38 selects non-EARLY t32s5hz KS4 while both controls select t32s5hze KS4. It loses0.240us paired/0.192us F1 versus RS32 and0.128/0.128us versus PW36. The source contains no cursor patch; this is a concrete prepared-policy selection regression, not evidence that the rejected cursor was reintroduced.
- **QKV32 disagreement:** PW38 and RS32 both select unchanged t32s6hze KS6; paired favors PW38 by0.160us, while conservative F1 favors RS32 by0.448us. Do not choose only the favorable method or credit EH for this unchanged small-M body.
- **Other small observations remain:** qkv16 has+0.096us paired/+0.160us F1 versus RS32 on unchanged GV15. Pooler256 gains+0.384us paired/+0.608us F1 versus RS32 with different cuBLASLt selections. QKV256 chooses an existing RQ alternative only in PW38. These are actual full-policy observations, not isolated EH improvements. All per-shape picks and uncertainties are retained in analysis.json.

PW38 prepare actually chose s4s5hzEH at qkv128/pooler128. Its same-prepare bake values were17.06 versus old17.12us for QKV and23.84 versus old24.13us for pooler. The printed0.06us QKV bake difference was selectable: the1% admission gate is against R6, not an additional1% EH-over-old-RQ gate. Non-hint EH remains slower at those shapes; EH is not selected for unpooler128 or o128. Bake timing is a separate context from paired graph timing and is not substituted for it.

## Local six-load summaries, not official scoring

Each paired sum below is the sum of five marginal shape medians. They are descriptive summaries of this one local run, with replay overhead retained; they are not official calibrated load times or a formal aggregate confidence interval.

|Load|RS32 sum|PW36 sum|PW38 sum|RS32/PW38|PW36/PW38|
|---|---:|---:|---:|---:|---:|
|decode1|59.776|59.856|59.600|1.002953|1.004295|
|decode16|75.168|75.296|75.136|1.000426|1.002129|
|decode32|78.288|78.624|78.528|0.996944|1.001222|
|verify1|59.552|60.000|59.776|0.996253|1.003747|
|verify16|89.248|88.544|88.080|1.013261|1.005268|
|verify32|106.832|105.840|104.480|1.022511|1.013017|

The equal-load geometric means of these local control/PW38 ratios are1.005346664 versus RS32 and1.004939204 versus PW36 (roughly0.535%/0.494% ratios above unity). Conservative F1 local ratios are1.007462454/1.002510471. Decode32 and verify1 marginal-median load sums regress versus RS32; the aggregate must not erase them. Do not multiply these ratios into the H200 incumbent score or label them an official speedup.

## Reproduction and retained evidence

Run `python3 proj_rsi/audits/pw38-stack-01/analyze.py` for CPU-only hash, schedule, statistic, resource and physical-gate reanalysis. `analysis.json` retains every paired delta, IQR, bootstrap interval, win/tie/loss count, all30actual picks and all15JIT hashes. Inputs remain under `results/proj-rsi/scheduler/artifacts/pw38.stack.fullscope.01/collection-1793076533579348342/`; nothing there was modified.

Independent result review: `proj_rsi/audits/pw38-stack-independent.md`. Root owns the validity disposition and any official unscored-check release under `pw38-official-release-checklist.json`. The actual cumulative file has now passed this local full-stack run; official H200 checks/scoring remain separate. No GPU operation, API request, queue change or promotion was performed by this author analysis.

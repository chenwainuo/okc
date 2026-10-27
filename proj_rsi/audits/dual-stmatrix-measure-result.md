# Fixed-source DP/DS measurement: local diagnostic favors DS

Raw [JSON](../diagnostics/dual-stmatrix-measure01/dual-stmatrix-measure01.json)
SHAaa9c14d22deb9110a50e36e12b056658d1ba91b305ab7645ec052e0f47311378.
Root reports exit0/01:42:46UTC, container removed/node0MiB. Author independently
recomputed [all-record audit](../diagnostics/dual-stmatrix-measure01/analysis.py) →
[analysis.json](../diagnostics/dual-stmatrix-measure01/analysis.json).

Source frozen: measurement4812bbcf/plan2643a604, DP97dd5613/DS672dab44.
No kernel edits, parentprepare, incumbent or official comparison. Every timed
graph includes fresh1MiB pack+main; read256MiB before each replay. Fresh seeded
X per comparison, same X for both policies; separate XP/output/counters.

## Validity

PASS all350 raw records:320timed+30preservedwarmups; exact planned mode/order/
position,64comparisons32AB/32BA in32ADJACENTswapped-order blocks, six32sample
steady blocks, exact fixture seeds/shared pair fixtures, actual previous-policy
run lengths, monotone host timestamps and positive finite events. No sample
filtering, best-block selection or ratio estimator selection.

PASS both frozen manifests/files/generatedsources, both actual-JIT hashes and
exact canonical main+packer PC/instruction equality. JIT DPbebe4726/DS145c56d9
unchanged from earlier valid trial; A4/store/proxy/BOTHownerbarrier gates pass.
Both main122regs/local0/shared103600/512threads/rounded65536/occupancy1;
pack22/local0. Source/binary/attrs rechecked initial+aftereachmode.

PASS all62validation records:8all65536storage tests,40 graph mutation/byte-XP/
output-poison/oracle checks,8freshworkspace/output checks, **6 final ACTUAL timed
output/current-XP checks BEFORE any post-mode replay or input mutation**. Each
policy's last seed is used, including differing final fixtures in steady modes.
FP64/tasknormal+outliers/zero/probes, exact128onehot rows and weight/packed/escape
immutability/counters0; RESET_BOTH_AFTER_DRAIN. Earlier full4096basis suites of
these identical binaries remain valid; this diagnostic does not rerun them.

## All preregistered estimators

|measurement|result|
|---|---:|
|64 paired median DP−DS|**+1.167999581us**|
|DS wins/losses|48/16|
|32DP-first median gain|+1.264000311us|
|32DS-first median gain|+1.119999215us|
|median64 raw DP/DS ratio|1.034744060|
|median32 local swapped-geometric ratios|1.030983406|
|paired absolute DP/DS medians|34.495998174 /33.520000055us|
|before DP/DS/DP steady medians|34.591999 /33.776000 /34.511998us|
|after DS/DP/DS steady medians|33.551998 /34.607999 /33.280000us|
|before bracketed effect|+0.775998458us|
|after bracketed effect|+1.191999763us|

Paired gain Q1/Q3=+0.104002/+1.647998us, mean+0.960500us;
range−1.887999..+3.552001us. Sequential16-comparison epoch medians:
+1.328001/+0.703998/+0.960000/+1.168000us, wins13/11/11/13.
Thus direction agrees across both orders, both brackets and all four epochs.
This is stronger local DP/DS evidence than the earlier21-sample four-way result.
Median differences and median ratios are distinct estimates, all retained.

## Interpretation / limits

Supports selecting DS as the measured local producer-store parent for future
explicitly reviewed compositions. No production port or automatic source rebase;
root's separately frozen group8-from-DP experiment remains its own causal test.
No incumbent or officialscore win is claimed; official best remains unchanged.

The [earlier mixed result](dual-stmatrix-result.md) stays intact:
F1+0.832003us but paired−0.255998us/10of21wins. The new result does not invalidate
it or identify its cause. Two-way versus four-way kernel mix, fresh versus fixed
timing inputs, absent parentprepare/changed allocation state and epoch changed
together. Code residency, host event gaps and producer scheduling remain possible
explanations; no measured instruction-cache or clock evidence isolates one.

Paired taking-turns/freshX/coldL2/swapped order follows local TASK's structure.
Blocked timing is a sensitivity check. Raw ratios and the swapped-geometric
estimator are explicitly local; server ratio combination, replay-overhead
calibration and95%interval are not replicated. No officialCI or significance
claim. Numerical diagnostics run after each mode, not after every timed sample.

Separate instrumented profile DPpack2.112/main29.793us, DSpack2.016/main31.615us
does not agree with whole unprofiled ranking. Retain that fact: those calls are
not a decomposition of the unprofiled gain and do not prove producer store cost
or barrier latency changed by1.168us. No extra profile/rerun is required to invent
a unique cause; use future mechanism counters when that cause matters.

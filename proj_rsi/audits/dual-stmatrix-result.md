# DS01 terminal audit: VALID, comparison with DP unresolved

Raw [JSON](../candidates/operand_reverse_dual_stmatrix/operand-reverse-dual-stmatrix01.json)
SHAe5a745c41db06507879ada25f21726074d97eaccc316b98f8f003a8b68f2f87e.
Frozen source/manifest unchanged. Reproducible CPU
[analysis](../../results/proj-rsi/takeover/20261027/dual-stmatrix-analysis.py) →
[JSON](../../results/proj-rsi/takeover/20261027/dual-stmatrix-analysis.json).

Independent terminal checks: all17 frozen files/source identities; all3 actual
JIT artifact hashes and canonical PC/instruction equality with local binaries;
A4 retirement and store/proxy/barrier gates; main122/local0/occupancy1,
DP+DS512threads/rounded65536; pack22/local0; full4 numerical suites,
20actual graph mutations/15XPbytepoisons/3all65536bits/freshworkspaces;
W/packed/escape bytes immutable, counters0, RESET_ALL_FOUR. Actual incumbent
RQ s4s5hz/KS2 matches one-kernel profile. Each packed profile has rx_pack+its
exact main, no extra transform. VALID does not imply a timing win.

|same-job control|F1 control min|DS F1|F1 control−DS|paired median control−DS|DS wins/21|
|---|---:|---:|---:|---:|---:|
|incumbent|24.512000|33.599999|−9.087998|−10.592001|0|
|DP|34.432001|33.599999|+0.832003|−0.255998|10|
|STMX|38.431998|33.599999|+4.831999|+4.288003|21|

Decision: **HOLD robust DP parent; no DS-over-DP claim, no incumbent win/port.**
Beating STMX is not a new family win: DP already beat STMX robustly. Keep DS
valid as a measured version; do not discard F1 or the paired loss selectively.

## Where the estimates disagree

F1 has21 consecutive DS samples, bracketed by21-sample blocks for the controls;
paired has21 four-way rotations/reversals (period8). Both methods cold-read
flush256MiB before EVERY replay and include the complete freshpack+main graph.
Neither makes input prep free. Flush is outside event timing, not an instruction
cache reset. Five earlier warmups/label do not standardize every later code switch.

DS blocked median33.599999, paired median34.623999: **+1.024000us**.
DP blocked-before34.495998/after34.432001, paired34.368001: −0.064us
relative to conservative F1 control. Thus most divergence is DS's conditional
distribution, not a large DP before/after drift. Paired DS IQR33.568..35.104,
MAD0.992; DP IQR33.888..35.296/MAD0.672. Gains span−2.336..+3.104us;
mean gain+0.085333us, median−0.255998,10wins/11losses. Small effect versus
scatter; means do not replace the preregistered median decision.

Two COMPLETE eight-order cycles give paired-median gains+1.103999 then−0.559999;
last partial5 give+0.319999. Both complete cycles contain every order once, so
the sign reversal cannot be explained solely by extra repetitions of the first
five orders. Median-of-row-differences is used, not difference-of-medians.

## Order / position / predecessor

Descriptive cells, no sample deletion or causal adjustment:

|DS position in row|n|DS median|paired DP−DS median|
|---|---:|---:|---:|
|0|6|35.071999|−0.224000|
|1|5|34.623999|+0.415999|
|2|5|34.655999|−0.351999|
|3|5|33.567999|+1.247998|

Exact order cells have only2or3 repeats, often both signs. DSafterDP predecessor
n3 has DSmedian35.615999/gain−0.767998; afterSTMX n12 has34.335999/+0.032000;
afterincumbent n6 has34.639999/+0.032000. Predecessor includes the preceding
row's final case; all precede the mandatory flush. These groups overlap position,
direction and epoch, so one factor cannot be isolated. DSbeforeDP n11 gainmedian
+0.319999, DSafterDP n10 −0.303999: within-pair first/second effects possible,
but not established by this four-way experiment.

No clear monotone drift: first/middle/last7 DS medians34.336/35.104/33.632,
DP34.304/34.432/34.304. Global monotone clock drift alone is a weak explanation;
no per-sample clocks/power/wall timestamps were recorded. Broad distributions
and~0.5us sorted gaps do NOT establish bimodality from21 samples. F1's first DS
sample33.632 is not a distinct slow cold-start spike; simple one-cold-replay
explanation is unproven. Mixed-kernel instruction/code residency, producer
scheduling sensitivity and event/host dispatch gaps remain hypotheses. A read
flush guarantees none of those states, and no PC samples identify them here.

Separate profile: DPpack1.697/main30.080, DSpack2.113/main30.305us.
Those instrumented calls neither decompose F1 nor establish the source of the
unprofiled disagreement; no subtraction-based claim.

## ONE focused fixed-source measurement

Root authorized a DP/DS-only measurement driver. No kernel changes, parentprepare,
incumbent or official claims. The local [TASK timing contract](../../muse_bf16_proj/TASK.md:88)
says same fresh inputs, paired turns, adjacent swapped order, coldL2 and median
ratios. Thus two-policy AB/BA is primary; blocked timing is sensitivity evidence,
not automatically more faithful because it favors DS. Exact server timing code
is unavailable locally; report formula and raw values, no claim to duplicate its
interval/overhead calibration.

Predetermine64 comparisons in32 adjacent swapped-order blocks (32AB/32BA), with
fixed balanced block-direction seed. Same fresh X per comparison, graph pointer
stable,256MiB read flush outside each event interval, whole2node graphs. Surround
with DP/DS/DP and DS/DP/DS steady32-sample blocks; report ALL six block medians,
two bracketed effects and first/steady-run positions without selecting the best.
Keep raw order, position, immediate previous graph, host timestamps and seeds.
No clock forcing/Icache-flush workaround; no warm-block-only promotion.

Source hashes/JIT+A4+store/attrs+occupancy before math. Current-X/NaN-output/XPbyte
poison, all65536storage patterns, FP64 oracle/freshnormal/outlier/probes and weight/
counter/reset checks after each mode. Full4096basis suites already passed exact
frozen code; no costly repeat. One bounded cached-build normal root GPU run after
critic. Interpretation preregistered: paired primary, steadiness diagnosis;
unresolved disagreement retains DP. No new implementation from one noisy median.

Prepared CPU-only driver:
[dual_stmatrix_measure.py](../tools/dual_stmatrix_measure.py),
SHA4812bbcfa2314db3db9ca481dcfa052daabb54e6d332c02c2866705dbc3d4293;
[frozen plan](../../results/proj-rsi/takeover/20261027/dual-stmatrix-measure-plan.json)
SHA2643a6043239623bc6a67cfcf8e59753c2a9faa38afe3a245f071673284eb997.
320timed+30retainedwarmup samples, no parentprepare. Fresh fixtures are lab inputs,
never kernel state; both policies see the same fixture within each comparison.
After each mode, final timed outputs/XP are checked against each policy's last
recorded seed BEFORE any validation mutation/replay. Source/binary/attributes are
rechecked; then storage bits/mutations/poisons/freshworkspace oracles. ActualJIT
main+packer must match frozen PC/instruction sequences before any math. CPU plan,
syntax and existing actual-JIT gates passed; new GPU measurement is UNRUN.

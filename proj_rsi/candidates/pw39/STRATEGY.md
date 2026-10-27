# PW39: bounded prepare-selection confirmation

PW39 is a host-only selection experiment over immutable PW38. No measured gain
or reduction in selection error is established. The concrete failure motivating
it is an actual prepared-policy choice, rather than a changed kernel: in the
retained PW38 full-scope local run, pooler32 picked t32s5hz/KS4 while RS32/PW36
picked t32s5hze/KS4. PW38 lost0.240us paired versus RS32 and0.128us versus PW36.
All11common RQ32 and6common RQ128 actual JIT streams were identical across those
versions. See `../../audits/pw38-stack-01/report.md` and its analysis/artifacts;
that evidence remains frozen and is not an official H200 score or a forecast.

## Exact estimator and budget diagnosis

`_time_us` creates a side stream, runs two warmups, captures one graph, then
flushes256MiB before each event-timed replay. It returns the middle of15sorted
measurements by default; it does not return all15values. Every screened candidate
gets one such15-replay median after existing full-K/FP64 validation. Candidates
are tested in their fixed list order, then sorted by that single estimate.
The fastest is the only one considered by the old selection loop. There is no
second measurement of the top candidate or runner-up, and no penalty for a
candidate winning its screen through one unusually low median.

RQ128 measures R6 before packing for bothM128/M256, discards a7-replay warmup
postpack, and remeasures R6 after all variants. Selection requires candidate
latency below0.99times the lower R6 before/after estimate. RQ32 measures its
actual existing incumbent (RQ128, selectedGV, or R6), discards a5-replay warmup,
then takes before/after incumbent controls around candidate screening. Its
margin is also0.99against the lower incumbent measurement. This conservative
incumbent gate controls family admission, not ranking among validated RQ peers.
Thus a small or noisy difference among otherwise admitted RQ variants can change
the chosen prepared policy even when their kernel code has not changed.

The module's import-clock `_T0` is never reset. PREPARE_BUDGET_S is240. RQ128
entry/compiler join uses the195second threshold, layer/candidate screening
uses210seconds, RQ32 entry/join/layer/candidate screening uses210seconds.
Prepare starts both RQ compile threads before R6 tuning, then runs RQ128,
GX,RQ32 and late probes/router phases. All of those functions, joins, existing
guards, ordering and constants remain exact. PW38's local59.020s prepare is one
local observation; it does not prove fresh H200 cold-compile headroom.

## Small causal change

PW39 adds one helper and one call after the existing screen in each bake.
The screen, validation, lists, input fixture, incumbent callable, original margin
and all existing candidate timing calls remain intact. Only when screening
already admitted an RQ candidate does the helper time up to the two lowest
validated screen estimates with held-out confirmation blocks:

```
incumbent, A, B, incumbent, B, A, incumbent
```

Each block invokes unchanged `_time_us(...,it=5)`, so its recorded value is a real
five-replay median, not an individual replay or a kernel-only timing. A single
available candidate uses incumbent,A,incumbent,A,incumbent. Each candidate's two
values are divided by the lower adjacent incumbent block. Rank by the maximum
of the two ratios, retaining only a candidate whose worst ratio is strictly
below the unchanged0.99margin. This requires both local brackets to support
admission and prevents one downward confirmation fluctuation from dominating.
The runner-up need not clear its original screen margin individually; it can
replace the winner only when both fresh brackets clear0.99. This retests family
admission using the new controls without lowering the margin. It imposes no
extra1% gate between RQ alternatives or absolute microsecond gain
cutoff. Ties retain stable screen order. PICK's displayed microseconds remain
an actual (maximum) confirmation block median; the ranking ratios are separately
recorded, never disguised as an observed latency.

No layer or shape has a handpicked best candidate, old timing or fixed winner.
Layer/M identities are used only for telemetry. All existing options, including
EH, EARLY and the RTK router, remain available with byte-identical kernel bodies.
The preexisting candidate lists and compile selectors are unchanged.

## Budget and failure behavior

Both bakes share `st['rq_confirm']`. All elapsed helper time is charged using a
monotonic clock against a2second additional block-admission allowance. Before
every block the helper also checks the unchanged import-clock210second cutoff.
No new block starts after either limit. An admitted `_time_us` call cannot be
preempted safely; one in-flight five-replay block can cross the2second allowance
or210second cutoff. This is an explicitly bounded number of calls with soft
wall-time admission, not a hard real-time interrupt or a new240second guarantee.
The original prepare guards remain intact.

If confirmation cannot finish, preserve the exact old screen decision, retain
every completed block and the interrupted/unused budget reason, and mark it
unconfirmed. Partial confirmations never become selection evidence. A launch
failure, exception or nonfinite/nonpositive timing retains the existing incumbent
instead. Empty exception messages are safe. No output/input/weight values are
reconstructed or cached; new state is host-only timing metadata.

`st['rq_confirm']['records']` preserves each screen winner, top-two screen
medians, every confirmation block median and order, ratios when complete,
selection/fallback reason, failed block identity, elapsed and cumulative seconds.
No rejected timing is filtered away. Original BAKE/PICK logging is retained;
additional CONFIRM lines include raw block medians and fallback reasons.

There are8RQ128 and9RQ32 bake opportunities in current lists (o128 appears in
both stages), at most119confirmation timing calls/595measured replay events if
all17opportunities have two valid peers and enough budget. Actual count is
bounded further by the shared allowance and original guards. Unconfirmed paths
retain PW38's old behavior. Extra time can consume headroom of later phases;
full-policy preparation and selected-path evidence must quantify that risk.

## CPU evidence and limits

Three strict unique edits reverse to the entire immutable PW38 file byte for
byte. Existing named AST changes are exactly `_rqbake`/`_rqsbake`; the only new
named node is `_rqconfirm`. `prepare`, `_time_us`, compile threads, kernel_fn,
validation, every GPU literal and every generated production RQ translation
unit remain exact. Therefore all19PW38 RQ instances (8RQ128 including2EH,
11RQ32) retain the same generated CUDA source. This source identity reuses
PW38's existing kernel evidence; it is not a new actual JIT/GPU validation.
No production import, extension build, prepare or CUDA call is executed.

The extracted production helper passes CPU fake-clock/adversarial cases for
outlier correction, two-bracket admission, stable versus noisy competitors,
local drift, strict margin boundaries, tiny qualifying effects, top-two bounds,
single/empty candidate paths, all30task-shaped host identities, shared budget,
partial cutoff, wall-clock rollback, launch/empty-message errors and invalid
timings. These demonstrate policy behavior, not statistical optimality or a
lower real-world selection error rate.

## What the next GPU comparison must resolve

Compare complete PW38/PW39 policies from independent fresh prepares using the
same declared weights, import-clock240second guard and compile-thread strategy.
Retain allprepare clocks, BAKE/PICK/CONFIRM records, actual selected identities,
actual JIT/source/resource checks, full30shape correctness/current-input/
mutation/counter/immutability gates and every paired/F1 observation. Inspect
pooler32 as motivating evidence while retaining all wins/ties/losses; it is not
a privileged source branch or the only acceptance shape. Repeat fresh prepares
to separate selection variability from execution variability; one lucky prepare
cannot prove robustness. Official unscored correctness/scoring stays separate,
and PW38's currently awaiting official job is not replaced or edited.

Potential refutations: valid candidates fall outside the screen's top two;
worst-bracket ranking rejects real gains; shorter five-replay blocks are noisier;
extra capture/flush work changes cache or later budget availability; partial
budget fallback preserves the original wrong decision. Retain those outcomes
and revise the mechanism in a new version rather than declare the family dead.

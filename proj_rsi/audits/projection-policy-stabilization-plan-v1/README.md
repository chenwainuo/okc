# Stabilize the measurement before changing selection policy

The existing generic-policy-binding proposal solves runner binding duplication;
it intentionally leaves prepare and timing unchanged. It does not address the
observed policy drift. No finite timing scheme can guarantee identical choices
across jobs without a plan lock. Replaying incumbent9916's known plans would
suppress changes by construction, not establish valid runtime tuning, and is
not proposed.

Pinned archival counts are in observations.json: CHECK71 has14 visible route
changes,11 outside its three native-change shape domains; CHECKfe5 has15,
including14 outside M1/M8. Both differ in all30 benchmark seeds. Prepare
completes163/165s with no skipped-budget log lines, so blanket deadline failure
is unsupported. fe5's late matched confirmation chooses M1 on+.048000us and
retains M8 on−.016000us; earlier unrelated bakes have already completed. It
cannot retroactively explain their changed choices. Whole3196.414us still
loses3157.038us;71 is3182.349us. Raw, calibrated, bake and profile evidence
remain separate. Unchanged-kernel timing/calibration losses refute treating
all whole-score differences as a route-only effect.

The smallest useful next test is a **shadow finalist comparison**, not a broad
policy rewrite. Reuse the existing candidate objects and cold-cache graph
measurement at two existing decision surfaces: GV qkv_gateM1 and RQ
unpoolerM32. Choose their finalists from that actual run's ordinary screen,
never from a historical winning-plan list. Generate two private synthetic X
matrices prospectively on the actual current W; compare baseline and every
candidate on identical bank bytes. This bank is tuning data generated before
timing, not evaluator/benchmark X, seeds or outputs. No route depends on an
input seed/content key, and no bank is selected after observing favorable signs.

Use equal warmup and balanced A5/B5/B5/A5 groups for each fixture, preserving
all raw groups and the normal policy decision. Report paired effects and the
existing estimator/margin outcome separately. A shadow strict-positive decision
has no new minimum effect/significance gate; ties keep baseline. It initially
changes no production plan. The plan caps finalists at3GV/2RQ,200timed replay
observations and a12second aggregate soft allowance inside the original240s
prepare clock, reserving the existing final30s. A budget skip retains original
choices and is explicit. Captures/launches cannot be hard-preempted; the budget
is an upper planned allowance, not a guaranteed hard deadline or measured cost.

If the shadow comparison supports the mechanism, the implementation is one
small matched-finalist helper at `_gvprep` and `_rqsbake`, with private
per-cell fixtures, equal control/candidate sampling and current-W validators.
It would replace sequential decision evidence, not add a persistent winner
cache or framework. It would not stabilize LT/router choices automatically.
Actual selected full plans and every old signed result remain necessary
observations, not a new universal proof/admission gate.

Past attempts and exact reopening conditions are in plan.json. In particular,
28GV and21non-O32 RQ margin observations already passed the old hurdle; a
blanket margin change cannot explain those decisions. Old keepchoice had local
paired/F1 gains but did not demonstrate global stability. fe5 already shows
four target AB/BA pairs can yield opposite order-subgroup signs. A common
prospective fixture and balanced timing tests one confound; it is not certainty.
If rank flips persist, or a second prospectively frozen independent bank
contradicts them, do not expand the selector change. If decisions stabilize but
whole raw/calibrated losses persist, investigate transfer/context separately.

No active source, author file, queue or worker was changed. No GPU/API/SSH/git
operation occurred. Evidence source501d and every archived log/analysis pin are
in evidence-manifest.json. Later additions must be separately named supplements.

Loop: profile → measure → hypothesis → optimize → test → validate → stack →
submit measured gains → learn → reprioritize → repeat; idle ≤10s.
[Contract](../../ITERATION.md).

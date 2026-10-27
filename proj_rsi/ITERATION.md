# Current iteration contract

## Active objective and scope

User correction 2026-10-27: **muse_bf16_proj only**, official score target
**<3000 microseconds**. Serving and MoE optimization are out of scope. Preserve
their existing results as history, but do not launch further work for them or
count their gains toward this objective. All active ROI priorities, implementation
assignments and new GPU admissions must target projection. This overrides older
serving/MoE goals and handovers.

## Continuous coordinator loop

Authority: explicit user instructions, reaffirmed 2026-10-27. Read this section
on startup, takeover and context compaction. It governs coordination; the long
historical LOOP.md log is evidence, not the current work queue.

`profile -> measure -> hypothesis -> optimize/reimplement -> test -> validate -> stack -> submit measured gains -> generalize learning -> reprioritize -> repeat`

- During active authorized optimization, keep coordinating after answering status
  questions. A status answer is not a pause request. Do not end the turn merely
  because a worker or agent is running. Deliberate idle waits are at most 10s;
  inspect fresh state and perform useful work between waits. Tool/runtime
  latency is distinct from a deliberate idle wait.
- Every user-facing answer/update includes a compact restatement of this loop
  and a link to this document. Carry this requirement into every handover and
  context summary. A reminder is not evidence that coordination is running.
- Each coordinator cycle: inspect worker health, active GPU work, READY/BACKLOG
  and newly VALIDATING results; consume agent messages; validate completed
  results promptly; unblock dependencies; admit reviewed work by expected ROI;
  assign implementation/review/analysis concurrently. Poll mutation completion
  and verify effects rather than assuming a tool succeeded.
- Keep the GPU supplied with useful bounded work. A live polling worker with an
  empty eligible queue is not GPU utilization. Distinguish queue state, physical
  utilization, agent activity and coordinator activity in reports. If nothing
  is eligible, diagnose the blocker and build the next useful experiment; do not
  rerun unchanged work just to occupy the GPU.
- Preserve every source/result and signed effect. Diagnose failed attempts,
  repair and iterate; a losing version does not reject its family. Record facts,
  causal hypothesis, scope, counterevidence, next falsifier and ROI implications.
- Stack compatible changes explicitly and measure the actual combined package
  against its parent and incumbent. Source inclusion, selected execution,
  local gains and official gains are separate facts. Never add isolated gains
  arithmetically or call parallel branches a cumulative stack.
- Submit valid measured improvements within the authorized quota; no minimum
  effect/significance gate. Preserve mixed estimates and prioritize finite quota.
  Official claims require actual official receipts. See the promotion policy below.
- User cancellation/pause, explicit limits and genuine external blockers still
  apply. State an interruption honestly; never claim autonomous continuity after
  a turn has ended or a process has stopped.

Response reminder:
`Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle <=10s. [Contract](ITERATION.md)`

Queue details: [scheduling/OPERATIONS.md](scheduling/OPERATIONS.md) and
[scheduling/CONTRACT.md](scheduling/CONTRACT.md). Learning:
[LEARNING.md](LEARNING.md). Measured stack: [CURRENT_STACK.md](CURRENT_STACK.md).

## Experiment iteration

Authority: user2026-10-26 correction; supersedes historical family-wide
`CLOSED/NO_RETRY/kill` language. Execution authority: results/proj-rsi/scheduler/events.jsonl; queue.json is historical experiment context. Official incumbent and
score: protocol.json. Chronological measurements: LOOP.md and immutable artifacts.

`family != version != measurement_attempt`.

```text
profile -> measure -> hypothesis -> implement -> test -> validate -> compare -> iterate

measurement_attempt:
  INVALID_SETUP/CORRECTNESS/LIFETIME:
    retain raw failure; identify cause; repair source in a new version or repair
    environment with an explicit attempt id; rerun required validity gates
    no performance admission from invalid attempt
  VALID_REGRESSION / VALID_NO_GAIN / VALID_GAIN:
    retain source+hashes+all controls; explain observed vs predicted result
    choose next causal question/change; compare predecessor+actual incumbent
    independent validation before integration

family:
  ACTIVE: implementation or diagnosis has plausible marginal value
  PARKED: record remaining opportunity, resource/ROI reason and reopening evidence
  RETIRED: needs family-wide evidence; one losing version is insufficient

promotion:
  independent correctness/rules/graph/lifetime/source review
  valid same-run win over actual selected incumbent, not only failed predecessor
  official H200 validation/score required for official speed claims
```

Each next-action record should contain:

```yaml
family: stable lineage
version: unique immutable source identity
predecessor: source/result ids
observed_failure: numeric result or concrete invalidity
causal_hypothesis: one proposed explanation; identify confounded variables
change_or_diagnostic: smallest useful test of that explanation
predicted_observations: what supports/refutes it
comparison: [actual_incumbent, predecessor, new_version]
validity: [numerics, real_timed_graph, mutation, immutability, resources, cleanup]
forecast:
  attainable_gain: unknown unless evidence supports an estimate
  conditional_score_value: same-official-job arithmetic, not a measured forecast
  information_value: what next decision this resolves
cost: bounded runtime/compile/review; exclusive GPU queue position
next_after_failure: diagnosis/iteration, not automatic family rejection
```

Rank speed opportunity and information value separately. A diagnostic earns no
speedup. An improvement over a failed predecessor is learning even while the
incumbent remains faster. Conversely, a valid losing version does not become a
win because its counters or byte model look better. Preserve noisy/inconclusive
results and their limitations; do not replace unknowns with zero.

Version sources freeze during a run. New source gets a new version/hash; old
outputs stay attached to their actual source, environment, comparator and attempt.
Tests and profiling remain focused and bounded; those bounds control resource use
and measurement validity, rather than forbidding evidence-driven follow-ups.

User promotion policy: retain every measured positive; no minimum-size or statistical-significance gate for submission. Record uncertainty. Whole-source rewrites are allowed; exact old-control SASS identity is required only where the specific narrow experiment depends on that matched control.

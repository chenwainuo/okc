# Current iteration contract

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

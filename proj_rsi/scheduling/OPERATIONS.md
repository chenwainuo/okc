# ROI scheduling and GPU execution

Authority: user request2026-10-27. This package controls execution order. The
older `proj_rsi/queue.json` is experiment history/context, not a hardware queue.
All commands below run from `/home/ubuntu/okc`. Only the root controller dispatches
GPU work; implementation/review agents submit descriptors and CPU evidence.

`source version != attempt != family`. A failed attempt is retained and gets a
cause/next experiment; it never automatically rejects its family.

## Priority, units, uncertainty

```text
direct = conditional_score_reduction_us * P(gain) * P(valid)
information = conditional_downstream_score_reduction_us * P(resolve) * P(followup_gain)
value = max(direct, information)                 # avoid counting overlap twice
GPU_cost_USD = (expected_job_wall_s + expected_followup_wall_s) * 5/3600
GPU_priority = value/GPU_cost_USD * capped_age_multiplier
CPU_priority = value/expected_CPU_agent_minutes * capped_age_multiplier
```

The probabilities are explicit subjective priors. Conditional score scenarios
use the fixed same-official-job six-load arithmetic, not an extrapolation from
H100 to H200. Hypothetical score reduction is capped by the incumbent score.
A hardware profile has zero direct speed benefit; it needs a concrete decision,
linked follow-up IDs and the follow-up GPU cost to claim information value.
Unknown/null estimates are **unrankable/HELD**, not silently zero or rejected.
Every estimate names its evidence, confidence and assumptions. Cost includes
transfer, compilation, prepare, correctness and measurement wall time, not just
kernel execution. These ROI numbers are decision aids, not measured speedups.

Rank descending priority; ties use creation time then stable job ID. A 10% maximum
age bonus accrues over24h. Root may reprice priors after evidence; every old/new
value and reason is in the audit log. A changed CUDA/driver/command/review creates
a new immutable attempt descriptor; a forecast update does not rewrite results.

## Capacity and full queues

|resource|active cap|READY cap|overflow|
|---|---:|---:|---|
|CPU agents|10|20|persistent BACKLOG|
|local H100|1|10|persistent BACKLOG|

READY means eligible, independently reviewed where required, source-pinned and
inside the highest-ROI ready window. New high-ROI work displaces the lowest READY
item back to BACKLOG. It does not delete it or preempt RUNNING/VALIDATING work.
Dependencies and overlapping CPU write paths also constrain admission/claim.
CPU jobs require no GPU launch descriptor while code is being implemented.
Actual agent spawning remains a direct root collaboration action after a CPU
claim; the scheduler does not create hidden CLI/model sessions.

A GPU job in VALIDATING has already proven physical cleanup and no longer holds
the device. Its dependents wait for explicit DONE with an accepted disposition.
Independent READY work can use the device while CPU review runs. A RUNNING job
with unknown status continues to occupy the slot until actual handles reconcile.

## Descriptor and admission

Versioned descriptors live in `proj_rsi/scheduling/jobs/`. See CONTRACT.md and
those real jobs for the complete schema. Required GPU fields:

```yaml
id: family.version.attempt01                 # never reuse an attempt ID
resource: gpu
kind: optimization | validation | diagnostic | profile
state: BACKLOG
author: implementer_id
estimate: {direct: ..., information: ..., basis: ..., confidence: ..., evidence: [...]}
cost: {gpu_seconds: ..., followup_gpu_seconds: ..., cpu_minutes: ...}
depends_on: [job_id | {id: ..., accepted_dispositions: [...]}]
files: {repo/relative/path: sha256, ...}
review: {actor: independent_id, path: review.md, sha256: ..., decision: GO}
launch:
  host: root@107.170.6.60
  remote_repo: /root/okc
  container: okc-proj-unique-attempt-name
  inner_seconds: 240
  outer_seconds: 300                         # > inner+20
  profile: false
  env: {MAX_JOBS: '1', OMP_NUM_THREADS: '1'}
  argv: [python3, /okc/proj_rsi/tools/lab.py, --json, /okc/logs/result.json]
  artifacts: [/root/okc/logs/result.json, ...] # exact paths/directories; no glob
```

Pin the whole imported source/control/proof closure, including both
`proj_rsi/tools/h100_trial.sh` and `h100_container.sh`. The installed
`/root/okc_run.sh` must match the latter SHA and mode0755; mismatches HOLD for
root repair, never silent installation. The reviewer differs from the author;
the review SHA is checked locally and remotely. Source drift before launch HOLDs.
An identical descriptor can reference a reviewed manifest, but that does not
replace the individual files needed by imports and the exact loaded binary gate.

Normal runs receive no profiling capability. Explicit profile jobs use the
existing `PROJ_GPU_PROFILE=1` opt-in. Supported NCU flags are parsed with known
arity: the actual target must be pinned Python driver, never a shell followed by
a later Python-looking token. Unknown flags HOLD. NCU uses `--clock-control none`;
there are no automatic clock/driver changes. The existing named-container guard
and **inside-container** GPU flock remain authoritative; never wrap Docker with
a second outer flock on that same hardware lock, and never unlink it.

## Root command sequence

```bash
python3 -m proj_rsi.scheduling init
python3 -m proj_rsi.scheduling register proj_rsi/scheduling/jobs/JOB.json
python3 -m proj_rsi.scheduling status
python3 -m proj_rsi.scheduling next gpu
python3 -m proj_rsi.scheduling run-next --owner root
python3 -m proj_rsi.scheduling poll JOB_ID
python3 -m proj_rsi.scheduling collect JOB_ID
```

For bounded autonomous draining of already-approved work:

```bash
python3 -m proj_rsi.scheduling work --owner root --max-jobs 3 --max-wall-seconds 3600 --poll-seconds 10
```

When agents may register reviewed jobs after the queue becomes empty, use a
bounded idle window:

```bash
python3 -m proj_rsi.scheduling work --owner root --max-jobs 10 --max-wall-seconds 7200 --poll-seconds 10 --idle-wait-seconds 600
```

`--idle-wait-seconds` defaults to0, preserving immediate `NO_READY_JOB` exit.
With600, each empty eligible queue waits up to600seconds, or the remaining wall
budget, whichever expires first. The worker sleeps between checks, at most the
configured polling interval (never above60seconds), and selects from fresh ROI
ranking when eligible work arrives. Starting/resuming a real job and collecting
its terminal evidence reset the idle window. Queue polling alone does not reset
it. A wall deadline still prevents a new launch whose entire outer reservation
cannot fit; it never interrupts an active guard.

For handoff, root first confirms the prior retained worker process/session has
ended, retains its exit/log evidence, then starts the command above in a new
retained session. Do not launch duplicate worker processes to change flags, kill
an active guard, clear execution handles, or infer hardware death from an idle
timeout. An existing guard remains owned and observed under the original safety
rules; this flag changes only how an empty eligible queue is awaited.

Root runs this in a retained exec session/background process and monitors it.
Each selection rereads current ROI ranking. The worker refuses to start a guard
whose complete outer reservation exceeds the remaining worker budget; it never
kills an already running job to meet its own budget. It stops on unknown remote
state, external busy hardware, an expired empty-queue idle window or a budget limit. A new
arrival can change the next selection. Official H200 checks/submission and their
20/10 daily quotas remain under `python3 -m proj_rsi` and are never auto-promoted
by this local dispatcher.

CPU claim/assignment and reprice:

```bash
python3 -m proj_rsi.scheduling claim cpu --owner root
# Root now uses collaboration.spawn_agent/followup_task for that exact work item.
python3 -m proj_rsi.scheduling reprice JOB_ID updated-priors.json --reason 'new profile evidence'
python3 -m proj_rsi.scheduling hold JOB_ID --reason 'await changed-code review'
```

`updated-priors.json` contains only estimate/cost/annotations. All finite inputs,
probabilities, independent review, source pins, path bounds and deadlines are
validated. Holding or cancelling a queued item preserves its evidence. A live
job is not cancelled/demoted by changing priority.

## Execution and recovery contract

1. Atomically claim top eligible work; persist controller identity before SSH.
2. Verify local source/review; transfer exactly pinned files; verify remote hashes.
3. Verify installed runner SHA/mode and actual idle GPU/container/guard state.
4. Persist launch-requested handles, then nohup one bounded host guard. Store its
   authoritative PID, unique container, log and artifact paths.
5. Poll those same handles. A transient SSH error, stale heartbeat or expired
   lease is UNKNOWN, not evidence of completion and not permission to restart.
6. Require terminal guard record plus absent guard/container/compute PIDs and
   zero GPU memory before releasing the hardware slot to VALIDATING.
7. Collect raw files with SHA inventory. Missing/partial artifacts stay visible.
8. Independently validate the actual results, then root records DONE disposition.

```bash
python3 -m proj_rsi.scheduling reconcile JOB_ID
python3 -m proj_rsi.scheduling finish JOB_ID VALID_REGRESSION --validate --evidence validation.json
```

`--validate` is an explicit root controller action. `validation.json` must name
actual numerical/graph/terminal-output/source/JIT/resource/cleanup evidence and
independent review. Exit0, a favorable median or a stale JSON status alone is
insufficient. Required dispositions: CPU_PASS, VALID_GAIN, VALID_NO_GAIN,
VALID_REGRESSION, INVALID_SETUP, INVALID_CORRECTNESS, INVALID_TIMEOUT,
INVALID_INTERRUPTED. No local disposition is an official score claim.

For an uncertain execution, retain RUNNING and reconcile its original PID/name.
Do not launch another attempt merely because SSH observation timed out. An
interrupted/invalid run keeps all partial logs and needs a new attempt ID for a
repair. A regression needs a causal next question; a family is parked only with
an explicit marginal-ROI reason and reopening condition.

## Durable evidence and verification

Runtime store: `results/proj-rsi/scheduler/`; `.lock` is a stable local advisory
lock, `events.jsonl` is the fsynced checksummed authoritative WAL, and
`checkpoint.json` is a regenerable view. Torn final records are preserved before
repair; middle corruption fails closed. Never hand-edit runtime files or remove
locks. GPU artifacts are append-only collection directories under the runtime
store; versioned descriptors/reviews and code belong in Git. The local scheduler
lock and the existing remote hardware lock have separate roles.

```bash
python3 -m unittest discover -s proj_rsi/scheduling -t . -v
python3 -m unittest proj_rsi.audits.test_scheduling_adversarial -v
```

Tests cover competing processes claiming one GPU, CPU caps/write conflicts,
full-queue priority replacement, dependency/review/source drift, restart/WAL
corruption, unknown SSH launches, cleanup gates, safe literal commands, NCU
target/clock rules and exit0 remaining unvalidated. They use temporary stores and
fake transports: running the suite never starts a real GPU job.

## Learn from every result

After validation, use `python3 -m proj_rsi.learning validate`, `summarize`, and `query` to record and retrieve applicable evidence-bound lessons. Facts, causal explanations, transfer limits, conflicting estimators and the next discriminating test remain separate. Record a success, regression, mixed or invalid outcome; never erase a tiny positive or a failed parent. Link proposed ideas to lesson/version IDs and concrete ROI jobs. Preserve whole-registry snapshots before append. Learning uses CPU work after cleanup so the next GPU reservation can proceed. Measured improvements with correctness checks passing can be submitted without a minimum gain or significance gate.

For agent handoffs, `send_message` delivers an update but does not restart a completed worker. Use `followup_task` for a new assignment to a completed/idle agent and verify its actual status before reporting active work.

# Scheduling implementation contract

Owner: root. Runtime cap: 10 CPU agents, 1 H100. Price: USD5/GPU-hour.
This package becomes execution authority; `proj_rsi/queue.json` retains history.
Official H200 submission/quota remains outside this local hardware dispatcher.

Files/owners:
- model.py + policy.json + test_policy.py: Sol16.
- store.py + cli.py + test_store.py: Sol11.
- dispatch.py + test_dispatch.py: Sol13.
- OPERATIONS.md + migrated jobs/: root.
- Independent review/negative tests: Sol14.
No shared edits outside owned files without root coordination. Standard library,
Python3.10. `python3 -m proj_rsi.scheduling` is the standalone CLI.

Job descriptor (strict keys where execution matters):
```json
{
 "id":"family.version.attempt01", "resource":"gpu", "kind":"optimization",
 "state":"BACKLOG", "author":"sol08", "created_at":0,
 "depends_on":[], "write_paths":[],
 "estimate":{
  "direct":{"scenario_score_us":0,"p_gain":0,"p_valid":1},
  "information":{"scenario_score_us":30,"p_resolve":0.7,"p_followup_gain":0.2,"unblocks":["other.id"],"decision":"..."},
  "basis":"subjective conditional prior; not a measured forecast",
  "confidence":"low", "evidence":["repo/relative/file"]
 },
 "cost":{"gpu_seconds":120,"followup_gpu_seconds":0,"cpu_minutes":10},
 "files":{"repo/relative/file":"sha256"},
 "review":{"actor":"independent-agent","path":"repo/relative/review.md","sha256":"...","decision":"GO"},
 "launch":{"host":"root@107.170.6.60","remote_repo":"/root/okc",
  "container":"okc-proj-unique-attempt01","inner_seconds":240,"outer_seconds":300,
  "profile":false,"env":{"MAX_JOBS":"1"},
  "argv":["python3","/okc/proj_rsi/tools/lab.py","--json","/okc/logs/attempt01.json"],
  "artifacts":["/root/okc/logs/attempt01.json","/root/okc/logs/attempt01.host.log"]}
}
```

CPU jobs omit launch/review/files until implementation is frozen. Independent
CPU review is a separate CPU job/dependency; GPU admission needs GO and hashes.
`information.unblocks` may reference a planned job; require a concrete decision
and evidence, do not count an arbitrary diagnostic as an achieved speedup.
Unknown estimate values are null: hold for estimate, never silently treat as0.
Register historical terminal jobs without launch as imports, never execute them.

Model API:
- `score(job, resource=None, now=None, policy=None) -> dict` with `priority`,
  `direct_expected_us`, `information_expected_us`, `effective_value_us`,
  `cost_usd`/`cpu_minutes`, `age_multiplier`, `assumptions`.
- `validate_job(job, *, for_admission=False) -> list[str]` (empty means valid).
- `load_policy(path=None) -> dict`.
GPU ROI = max(direct expected score benefit, information expected score benefit)
/ ((gpu_seconds+followup_gpu_seconds)*5/3600). CPU ROI analogous per CPU-minute.
Max avoids counting the same learning/direct benefit twice. Probabilities are
explicit subjective priors; never label mathematical scenarios measured gains.
Capped age bonus policy; deterministic tie: older created_at, then id.

State API, class `QueueStore(directory, *, repo_root=None, policy=None)`:
- `register(job)`, `get(id)`, `snapshot()`, `rank(resource)`.
- `rebalance(resource)`: highest eligible top K become READY; displaced READY
  return BACKLOG with event. NEVER demote/preempt RUNNING/VALIDATING.
- `claim(resource, owner) -> job|None`: atomic highest READY only; enforce caps
  cpu10/gpu1 and no overlapping CPU write_paths. Records owner/lease/selection.
- `update(id, changes, *, actor, reason)` for reviewed estimates/annotations;
  reject mutation of executable hashes/argv while running.
- `attach(id, observation)`, `heartbeat(id, observation)`.
- `finish(id, disposition, *, evidence, actor)` moves RUNNING->VALIDATING first
  when execution ends; root validator sets DONE with explicit valid/invalid
  disposition. A successful subprocess is not a verified benchmark.
- `transition(id, state, *, actor, reason, evidence=None)` guarded transitions.
Use fields/state consistently and document concrete snapshot format in store.
Owner uses a local controller identity, not an alleged security boundary.

States: BACKLOG, READY, RUNNING, VALIDATING, DONE, HELD, CANCELLED.
Dispositions: CPU_PASS, VALID_GAIN, VALID_NO_GAIN, VALID_REGRESSION,
INVALID_SETUP, INVALID_CORRECTNESS, INVALID_TIMEOUT, INVALID_INTERRUPTED.
Dependencies default DONE + non-INVALID disposition; explicit dependency
objects may list accepted dispositions. Failed measurements stay immutable;
new attempt IDs for repairs/retries. Families never discarded automatically.

Persistence must use a stable advisory flock (never unlink), atomic writes,
and append-only audit. An append-only fsynced event WAL is authoritative;
materialized JSON is a regenerable checkpoint. Torn final event recovery must
preserve raw evidence and never ignore middle corruption. Test races/restarts.
Runtime directory default results/proj-rsi/scheduler/ (gitignored); immutable
job descriptors under proj_rsi/scheduling/jobs/ are versioned separately.

Dispatcher API class `GPUDispatcher(store, repo_root, *, transport=None)`:
- `launch_next(owner="root")`: claim->verify local hashes/review->transfer exact
  pinned files->verify remote hashes->probe GPU/container idle->nohup existing
  h100_trial.sh exact named container; persist authoritative guard PID/log paths.
- `poll(id)`: inspect actual guard PID/container, terminal TRIAL_EXIT/log/result;
  SSH timeout is UNKNOWN, not terminal. Refresh heartbeat with evidence.
- `collect(id)`: copy raw artifacts and hash inventory; never mark valid from
  exit0/status text. Transition to VALIDATING only after actual cleanup verified.
- `reconcile(id)`: stale lease is NOT a stopped job. Probe same remote handles;
  never auto retry/restart/kill unknown work. Root explicit new attempt needed.
- CLI one-shot `run-next`, `poll`, `collect`, `reconcile`; no daemon required.
Optional `--wait` may poll but must remain bounded and preserve restart handles.

Hardware guard remains /root/okc/proj_rsi/tools/h100_trial.sh, inside-container
flock via /root/okc_run.sh. Do not add an outer hardware flock or unlink it.
Normal optimization uses no extra capability; profile=true uses existing
PROJ_GPU_PROFILE=1 only. No clock/driver changes. Validate names/host/deadlines/
argv and use shlex quoting. Never expose auth material. Never launch when an
external GPU/container is live; leave queued/held and preserve observation.
Host stdout must yield only PID on launch; shell backgrounding must not capture
an entire validation AND-list. Artifacts/source hashes retained after failure.

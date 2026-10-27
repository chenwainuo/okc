# Scheduling independent review

Status: **GO for the reviewed queue/model/dispatcher implementation below.** Runtime descriptors still require their own frozen sources, independent GO, dependencies and readiness checks.

Scope: CPU and fake transport only. No GPU, real SSH, launch, queue mutation, author-file edits or commits. Reviewed scheduling/CONTRACT.md. Existing smallm v3 counter audit/frozen files remain untouched.

Required adversarial evidence before GO:

- Model: unknown/null/NaN/infinite priors never become zero-valued admissible work; positive finite costs; explicit subjective basis; information value requires a concrete decision, linked follow-up and follow-up cost; max(direct, information), capped age bonus and documented deterministic tie breaking.
- Queue: strict descriptor/review/source hashes/dependencies, top-K READY/backlog rebalancing, no RUNNING/VALIDATING preemption, CPU cap10 and overlapping-write exclusion, GPU cap1 under actual concurrent process claims.
- Persistence: stable lock, fsynced authoritative event WAL, restart replay, preserved torn-last-record evidence, fail-closed middle corruption, no ignored invalid event or checkpoint authority inversion.
- Dispatcher: source changes after admission block launch; local and remote pinned bytes verified; stale timer never treated as dead PID; SSH timeout/ambiguous launch is UNKNOWN and never auto duplicates; restart reuses persisted handles; named PID/container cleanup verified before VALIDATING; exit0 never implies validity.
- Guard: existing hardware guard only, inner container lock retained, no extra outer hardware flock/unlink; quoted validated argv/host/container/path; deadlines bounded; actor readiness/validation authority explicit; no undocumented manual priority override.

Source authors retain repair ownership. Final decision will identify reviewed hashes, exact commands/results, limitations and minimum repairs.

## Intermediate independent results

Current implementation passed 49 author tests and 45 independent tests before the new NCU launch surface was added. Independent tests include two real sixteen-process GPU claim races; each produces exactly one RUNNING job. They also cover CPU10 cap, CPU overlapping paths, READY displacement without RUNNING preemption, explicit invalid dependencies, source drift after admission, stale review invalidation, root-only DONE controller gate, cleanup requirement, checkpoint failure after durable claim, WAL restart/torn-tail recovery/middle-corruption failure, fake ambiguous SSH outcomes/restart without duplicate launch, external busy rejection, upload drift, installed runner mismatch, exit0 remaining VALIDATING, live handles blocking collection, literal argument quoting and absence of an extra hardware flock.

Resolved source-author repairs discovered by this review: huge numeric priors raised OverflowError; caller policy could raise hardware caps; author could self-validate a GPU result; editing launch could retain old GO; changing only reviewer actor could reuse the same review content; dispatch prelaunch rejection attempted forbidden RUNNING->HELD; guard was not pinned; installed inner runner was not verified. Current revisions address these issues. GPU VALIDATING releases physical capacity only after a full verified cleanup observation; CPU VALIDATING remains active. This is consistent with the root's operational clarification.

Remaining blocker at this checkpoint: the newly introduced NCU launcher searches for a later Python executable rather than proving it is NCU's actual target. Concrete accepted descriptor: `ncu bash -c 'arbitrary shell code' python3 /okc/proj_rsi/tools/pinned.py`. `validate_job(..., for_admission=True)` returns no errors, although NCU executes bash. Requested authors share one strict supported-NCU-option parser and require the actual target to be the pinned Python driver. Added independent negative test; final GO awaits repair and rerun.

NCU clock control is part of that repair: explicit `--clock-control none` is required by this repository's no-clock-changes contract. NVIDIA documents automatic clock control in its [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control). The review performs no profiling or hardware changes.

A second final policy repair is required: diagnostic/profile jobs must not evade the linked information-value contract by placing speculative benefit in the direct-gain branch. A diagnostic that only informs a future choice must use the concrete decision, follow-up IDs and follow-up GPU cost path. Added an independent negative test for positive direct benefit on diagnostic/profile kinds.


## Final independent decision

**GO** for the reviewed implementation. The shared `ncu_target` parser now accepts only known option arities before the actual first Python target, requires exactly one explicit `--clock-control none`, and binds its driver to pinned files. Shell/timeout targets, unknown options, missing clock policy and clock changes fail closed. `profile=true` requires profile/diagnostic kind; diagnostic/profile value cannot be placed in the direct-gain branch to evade the concrete information/follow-up-cost contract.

Independently reran:

```text
python3 -m unittest proj_rsi.scheduling.test_policy proj_rsi.scheduling.test_store proj_rsi.scheduling.test_dispatch
56 tests PASS
python3 proj_rsi/audits/test_scheduling_adversarial.py
51 tests PASS
```

The independent suite repeats base queue cases under its integrated dispatcher subclass; it includes two separate sixteen-process GPU-claim races, each admitting exactly one job. All test state and artifacts use temporary directories; fake transports never invoke SSH or a guard. Logs: [author suite](scheduling-author-tests.txt), [independent suite](scheduling-independent-tests.txt). Full reviewed source/test/log SHA256 inventory: [hash inventory](scheduling-independent-hashes.json).

| Reviewed file | SHA256 |
|---|---|
| scheduling/model.py | `13edc918de16a044eff2f9ea5e3b9e2bd687b7b82a524b1cdef1a03ca3e2b8f0` |
| scheduling/policy.json | `8f08bea6cfe27e25c78756ba7ff2bffd63b9e5bb7dcbc08bf4eed5a2946c86ab` |
| scheduling/store.py | `e1b82e5472a41dfbec7e1ee3b547c345de201e441f424557f9c9fef6b6c1cf1f` |
| scheduling/cli.py | `372e95e0b9d0ed84badfbe6db730054ec964494ce1ad4e3f64b163601b2ebb6c` |
| scheduling/dispatch.py | `2b81f30f30df54eee807c30bee45e1d3844a56fa628706eca3b02081ca8c9817` |
| tools/h100_trial.sh | `674036ba22edcc3c64dda2677271c58f27fe822339cf132c4f36ce58837ee9e5` |
| tools/h100_container.sh | `8b3529b5fa873e61e4e236e0cbcdaa0a94ad426a9a21b6aa6469af4bf6e29d67` |

The scheduler has no hidden manual priority override: score uses explicit subjective priors, max(direct, information), finite positive chain costs and a capped ten-percent age bonus; unknown/nonfinite inputs are unrankable. Top-K READY eviction retains BACKLOG and never preempts running work. Fixed caps are enforced even with a supplied policy. WAL audit is authoritative, locked using a stable inode, fsynced, replayed after restart and independent of checkpoint contents. Torn final bytes are preserved before truncation; complete corruption refuses execution. A durable claim survives checkpoint failure.

Dispatch persists LAUNCH_REQUESTED before the remote request. SSH loss remains UNKNOWN/RUNNING, with named container/log identity retained; unique actual guard PID observations can repair a missing response PID. Lease time never establishes death. Unlaunched release is owner-only and rejects any history beyond CLAIMED. Local and transferred sources/review/outer guard are checked, installed inner runner SHA and mode755 are checked, and the only hardware flock remains inside that runner's container. Shell rendering uses literal shlex quoting. Cleanup requires no actual guard/container/compute PID/memory and no cleanup-failure flag; repeated observation surrounds collection. Exit0/status text cannot mark a benchmark valid: collected work remains VALIDATING for explicit root disposition.

Limitations: this review does not establish a real SSH/rsync installation or GPU execution result, and makes no performance claim. Root-created job descriptors and operational migration were not present during this code review; their priors, dependency links and source/review pins still need their normal admission checks. The installed `/root/okc_run.sh` must match the pinned inner runner or dispatch holds for manual installation. These are enforced/runtime or remaining root integration requirements, not permission to bypass a gate. No frozen smallm-v3 files, historical queue records or commits were changed by this reviewer.

## Actual smallm v3 descriptor integration GO

**GO for root's bounded execution of `smallm.cursor.v3.all5.01`.** Reviewed descriptor `proj_rsi/scheduling/jobs/smallm-cursor-v3-01.json`, SHA256 `ab1da3227fbe0c5a2bcce1cd1a61b13cb56264840f0c5deb8933632461582a7a`, and `OPERATIONS.md`, SHA256 `5f0671e240caca433f41cd503bfd386b9a9fc393df5365b490e37ee9dbfee4d0`. This closes the earlier descriptor/migration integration limitation for this job.

Read the authoritative live WAL without modifying it: job READY, no RUNNING GPU jobs, every descriptor field matches the registered live job except its expected BACKLOG->READY state. All28 pinned files hash correctly. Both v3 and candidate manifest closures are present with matching hashes, including controls, exact kernel source, SASS/cubin proofs, history, reference/checks/baseline and both guards. The actual driver imports only those candidate modules and the pinned reference/checks source from this repository; other imports are its existing standard/PyTorch environment. No missing execution dependency was identified. Completed independent review is already byte-pinned; no incomplete CPU implementation/review task is being silently bypassed.

The review copy `scheduling/reviews/smallm-v3-all5-01.md` is byte-identical to this reviewer's original frozen counter audit, SHA256 `4310156335bd6e6ca6062887238d5ba80f099f4f97a7ec3b3c7808244fc04689`. Its final all-five GO supersedes its historical pending-freeze HOLD. Read-only dispatcher `_validate` passes with29 exact transfers including this review copy. The review token uses the correct absolute `/okc/...` path. Output `/okc/logs/scheduler/smallm-cursor-v3-01` maps to the exact host artifact directory `/root/okc/logs/scheduler/smallm-cursor-v3-01`; the named attempt/container is unique in the store. All five requested shapes are explicit in argv. MAX_JOBS1 and thread bounds add no agent spawning, and profile=false adds no profiling capability.

The score scenario53.2920788100642us matches `sol16-score-opportunity.json`'s exact hypothetical one-microsecond improvement on each of the five affected shapes (53.29207881006414us within rounding). It is labelled a low-confidence conditional prior, with p_gain=.1 and p_valid=.8, never a measured gain or H100-to-H200 forecast. The resulting4.263366304805136us expected scenario value and$0.4166666666666667 expected cost yield priority10.232079131532327 before age. The300s expected full-wall estimate is explicitly distinguished from the1140s inner/1200s outer hard deadline; it is a prior, not a claim all five shapes previously completed in300s. The operations document correctly requires a full outer reservation within the bounded worker budget.

Independently reran all107 scheduler/independent CPU cases: PASS, [log](scheduling-integration-tests.txt). Re-ran frozen v3 `--cpu-check`: PASS, [output](scheduling-integration-v3-cpu.json). Detailed read-only admission/pin/closure/WAL/policy checks: [integration JSON](scheduling-smallm-integration.json). Previously reviewed scheduler source/hash inventory shows no drift. No new hypothetical features or approval gates are required for this job. Remote installed-runner hash/mode, transfer hashes, actual hardware-idle probe and cleanup remain the existing enforced runtime gates. Runtime unsupported selected protocols may still HOLD and any incomplete/invalid attempt admits no performance. This reviewer made no real SSH or GPU call and did not modify the live queue.

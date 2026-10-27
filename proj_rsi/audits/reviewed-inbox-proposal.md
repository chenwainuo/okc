# Reviewed local GPU inbox — unactivated proposal

Status: implementation and isolated CPU evidence only. **No live integration or
activation.** Existing dispatcher, CLI, QueueStore, daemon, WAL, checkpoint, job
states and GPU ownership were not changed or instantiated by this task. Root
must separately accept the independent review before wiring any controller.

Files: `scheduling/_reviewed_inbox_proposal.py` and
`scheduling/test_reviewed_inbox_proposal.py`. Import has no external actions.
There is no CLI, daemon, thread, timer, remote probe, official API or submission
path in this module. All author/independent mutation tests use FakeStore and
TemporaryDirectory. Reading the approved PW43 descriptor is a **synthetic empty
snapshot inspection**, not admission of its real already-RUNNING ID.

The intended future root-worker operation is admission only: newly discovered,
allowlisted local GPU descriptors carrying independent GO are registered with
input state BACKLOG. QueueStore alone decides ROI rebalance/READY capacity.
Existing claim/dispatch remains the only GPU launch authority. This module never
claims, launches, polls hardware, promotes, retries, transitions, or updates jobs.
Every existing ID, including HELD, RUNNING, VALIDATING, DONE and CANCELLED, is a
SKIP. Root must manually review an update or issue a distinct attempt ID for a
repair. No family is discarded because an earlier version lost.

## Root policy and APIs

`inspect_descriptor(repo_root, descriptor_path, snapshot, scheduler_policy,
policy, controller="root-worker")` is pure with respect to queue state. Caller
provides a frozen snapshot; the helper does not construct or read QueueStore.
The existing production `GPUDispatcher._validate` runs against an in-memory
FrozenView, with a transport object that rejects any attempted operation. Exact
production source/review/path/hash/container/argv/guard checks remain authoritative.

`sync_reviewed_new(store, repo_root, descriptor_paths, policy,
root_opt_in=False, controller="root-worker")` is the proposed future integration
hook. Both explicit `root_opt_in=True` and root-authored `policy.enabled=True`
are required. Default-off returns before accessing the store or filesystem.
No actual store was passed to this function during development/review.

Example **disabled** policy for the already reviewed PW43 package:

```json
{
  "enabled": false,
  "controller_id": "root-worker",
  "allowed_ids": ["pw43.stack.fullscope.01"],
  "allowed_prefixes": [],
  "bounds": {
    "normal": {"inner_seconds": 3300, "outer_seconds": 3420},
    "profile": {"inner_seconds": 240, "outer_seconds": 300}
  },
  "admission_lock": "results/proj-rsi/scheduler/reviewed-inbox.proposal.lock",
  "source_review_bindings": {
    "pw43.stack.fullscope.01": {
      "actor": "sol08",
      "decision": "GO",
      "path": "proj_rsi/scheduling/reviews/pw43-source-01.md",
      "sha256": "3a40ff7b5926ac5ecfb5ab8fc5c249960c04535f6ac41c02178473efe0b8ffa4"
    }
  }
}
```

Root-authored prefixes may replace exact IDs for a reviewed workflow, but an
empty/broad implicit wildcard is never inferred. Prefixes must be explicit safe
identifiers of at least three characters. Controller labels are local workflow
identities, not a security boundary against an adversary with root/filesystem
write access. Review GO semantics come from the reviewed descriptor/receipt and
its exact pinned artifact; this helper does not interpret prose, historical HOLD
paragraphs, or a short SHA as authorization.

## Admission conditions

- Only new resource=`gpu`, kind optimization/diagnostic/profile, state BACKLOG.
  CPU work, imports/terminal registrations, direct official/module launch routes,
  and unpinned local drivers are rejected. Python drivers must be pinned `.py`
  files beneath `/okc/proj_rsi/tools/`. Explicit NCU additionally uses the strict
  shared actual-target parser, profile kind/flag and disabled clock control.
- Exact configured ID or prefix, independent GO driver review, all source/driver
  review SHA closure, pinned hardware guard/runner and bounded normal/profile
  guard deadlines must pass. Root caps may narrow but cannot exceed scheduler
  caps. Outer must remain greater than inner+20 seconds.
- If `annotations.source_review` exists, a separate full independent GO receipt
  is mandatory. New descriptors can carry a structured `{actor, decision, path,
  sha256}` receipt in that field. Legacy PW43 text requires the explicit
  job-bound root policy receipt above. Missing/NULL/short/ambiguous source review
  is SKIP, not an inferred GO. Its path must differ from driver review and be
  included at the same SHA in source pins.
- Dependencies must already be DONE with CPU_PASS or VALID_* disposition, and
  satisfy an explicit accepted list if given. INVALID dispositions cannot be
  permitted by an override. RUNNING or VALIDATING is not DONE.
- Existing model validation/ROI must be rankable with explicit finite priors and
  positive full reservation estimates. Rate must equal USD5/GPU-hour. GPU
  seconds and follow-up seconds both remain charged; they are not kernel-only
  timings. No new probabilities, speedups, costs or zero placeholders are
  invented here. Actual wall-time reconciliation remains the existing worker's
  responsibility. Age bonus and deterministic ROI order remain store policy.

Every skip has a fixed reason code and SHA256-bound evidence: descriptor hash,
policy hash when serializable, snapshot sequence, already-known state and
validated pins when available. No exception/log/source contents are emitted as
reasons. Partial/missing/duplicate-key/nonfinite JSON, missing files, HOLD,
unknown costs, hash drift or malformed policy/snapshot/annotations fail closed.
These are skipped inputs, not automatically created HELD jobs or GPU attempts.

## Race and persistence boundaries

The future apply hook takes a **separate** dedicated inbox admission flock. It
never nests the scheduler `.lock`, never touches a hardware flock and never
unlinks its lock. It rereads the descriptor, full closure and fresh dependency/
ID state immediately before registration. Descriptor/pin changes invalidate the
proposal. Atomic store registration provides the final unique-ID race guard:
an intervening registration yields SKIP and is never overwritten or retried.
Independent forked-controller testing verifies exactly one registration from
six controllers. The dedicated inbox lock coordinates inbox controllers; an
external manual controller still must obey normal store/dispatcher invariants.
Any later file drift remains a final dispatch failure, not permission to launch.
No claim is made that filesystem review can prevent every post-validation write.

Store.register receives BACKLOG only. A real store may immediately rebalance it
to READY/HELD according to its existing authoritative checks; the helper neither
forces READY nor preempts any RUNNING/VALIDATING job. The existing dispatcher
rechecks source/review/container/hardware immediately before actual execution.
No official challenge API or submit functionality is added. Independent source
and driver review must continue to enforce the local-experiment scope of the
pinned Python program; generic Python is not a sandbox.

## CPU evidence and follow-up

Author:19 tests pass, including default-off/no-access, read-only inspection,
all seven existing states, partial/duplicate/nonfinite JSON, independent and
dual-review checks, hash/file drift, exact/prefix allowlists, dependency validity,
profile caps, full follow-up cost, explicit official-entrypoint rejection,
source change between approval/registration, atomic stolen-ID race, dedicated
lock checks, malformed policies/snapshots, reason hashes and synthetic PW43
closure. Independent review adds malformed/subcase and six-process FakeStore
race evidence. No GPU, SSH, official API, real queue snapshot or registration ran.

Root also identified a separate performance issue in the live work loop:
`rank("gpu")` followed by `store.get` for each row repeatedly strict-replays an
approximately600MB WAL. An opt-in inbox that scans many files can likewise make
multiple authoritative snapshot reads. This proposal intentionally does **not**
change that live code or activate worker polling. After the separate cache09
proposal review, a bounded follow-up can batch one snapshot for discovery and
readiness, retaining fresh validation/atomic claim as final authorities. Measure
wake-to-claim CPU time and replay counts on synthetic/replayed WAL; no GPU test
or weakened correctness gate is needed. Do not make old-worker/cache coexistence
or activation assumptions before the cache's own critic GO.

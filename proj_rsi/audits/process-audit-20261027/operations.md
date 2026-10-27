# Operations process audit — 2026-10-27

The visible bottleneck is getting finished, reviewed work admitted and keeping an existing worker supplied. Building another coordinator before addressing that repeats the problem. In the strongest retained example, the last managed experiment was collected at15:42 and the next job was not registered until17:46; once registered, it launched in32seconds. Existing queue persistence also performs disproportionate work:436 heartbeat events account for1.555GB of a2.595GB WAL prefix, because each heartbeat appends a complete queue snapshot.

This is a CPU/read-only evidence audit. No scheduler, GPU, SSH, API, queue, lock or Git state was changed. `operations.py` reproduces compact timestamp extracts and per-record offset/length/raw-byte hashes in `operations.json`; it does not import QueueStore. The captured WAL prefix contains771 complete records; the later direct checkpoint view is sequence772, so this is explicitly not one atomic snapshot or a validated WAL replay. The audit hashes raw records as evidence pointers but does not revalidate their inner payload checksums. Runtime files remain the preserved primary evidence.

## The MoESS → RQ32 gap

| Retained receipt | UTC | Meaning |
|---|---|---|
| MoESS launch requested |15:40:41.059|Bounded guard requested; includes JIT/setup, not all GPU compute|
| MoESS host terminal log |15:41:36|Exit1: actual resource assertion rejected static_shared0 versus expected1024|
| First clean terminal probe |15:41:51.887|No guard/container/compute PIDs, zero GPU memory at this observation|
| Raw artifacts collected |15:42:05.681|Collection inventory retained in runtime artifacts|
| RUNNING → VALIDATING, WAL727 |15:42:17.688|Cleanup verified and GPU capacity released; result validation remains separate|
| RQ32 descriptor created_at |15:53:00.912|Author-supplied descriptor timestamp; not proof that review/dependencies were ready|
| RQ32 register, WAL728 |17:46:20.812|First recorded admission of this exact job|
| RQ32 launch requested |17:46:53.086|32.274s after registration|
| RQ32 host terminal log |17:49:10|Exit0; numerical/performance interpretation separate|
| RQ32 collection |17:49:44.628|Raw artifacts retained|

MoESS collection → RQ32 launch is124m47.404s. MoESS terminal → RQ32 launch is125m17.086s. WAL727 → WAL728 has no intervening scheduler events for124m03.124s. RQ32 descriptor creation → registration is113m19.901s. These receipts show a long period without managed scheduler progress, followed by fast dispatch. They do **not** prove continuous physical GPU idleness: the first clean probe is a point observation and this audit has no remote/external utilization history. Nor do they prove that all113minutes were waiting for root; the descriptor timestamp is not a review-ready receipt.

The RQ32 lab GO currently has filesystem mtime16:02:06 and its admission-validation JSON16:03:58; root admission was17:46:20. These are observations of current local file mtimes, not authenticated historical review completion. They suggest a roughly102minute ready-to-root delay worth explaining, but cannot establish that causal number alone.

There is direct evidence of avoidable pin churn: `scheduling/jobs/rq32-predxor-01.root.json` records that independent CPU rerun changed only `author-tests.log`'s unittest-duration output hash while review/source/driver stayed exact. Root created a descriptor revision for that volatile log, whose current mtime is17:45:45; root descriptor mtime17:46:19. Preserve audit outputs append-only or outside launch source closure. A timing string changing should not require a new source admission review when executable/review bytes are unchanged. This is a concrete narrow removal of process work, not a proposal to accept stale executable pins.

Primary receipt pointers are the MoESS/RQ32 rows and WAL sequences712–746 in `operations.json`; each WAL event has a source offset, byte length and raw SHA256. MoESS collection inventory is `results/proj-rsi/scheduler/artifacts/research.moe.ss.stage.screen.v2.01/collection-1793115722457840089/inventory.json`. RQ32 inventory is `results/proj-rsi/scheduler/artifacts/research.rq32.predxor.gpu.01/collection-1793123380495441155/inventory.json`.

## Repeated gaps, not just one incident

| Previous managed terminal → next managed launch | Gap | Next registration → launch |
|---|---:|---:|
| PW45 14:23:02 → PW44attempt02 15:04:59 |41m57s|91.84s|
| PW44attempt02 15:08:33 → trace01 15:21:33 |13m00s|19.49s|
| trace01 15:22:46 → trace02 15:29:12 |6m26s|22.77s|
| trace02 15:30:33 → MoESSv2 15:40:41 |10m08s|28.99s|
| MoESSv2 15:41:36 → RQ32 17:46:53 |125m17s|32.27s|
| RQ32 17:49:10 → MoESSv3 17:50:21 |1m12s|21.53s|

From PW45 launch14:19:30 through RQ32 terminal17:49:10, the six managed launch-to-terminal envelopes total770.267s (12m50s) across209m39s elapsed, about6.1%. This includes compilation, preparation and validation inside a guard, so it is not a GPU kernel utilization metric. Remaining time lacks a managed execution envelope; external work and unobserved hardware activity are unknown. The13minute PW44→trace01 gap contains scheduler snapshot optimization deployment/restart work (activation at15:11), but receipts do not allocate its exact minutes to each human/CPU activity. Do not relabel the entire gap as measured review time.

The later RQ32→MoESSv3 transition is an observed counterexample: existing dispatch can turn over in about72seconds, without waiting for completed numerical analysis to become DONE. This is the desired operating pattern for unrelated admitted jobs.

## Persistence work is itself expensive

The captured WAL prefix is2,594,849,174bytes; checkpoint is roughly8MB. Of771 events,436 are heartbeats and120 are attachments. Heartbeats alone contribute1,555,317,566 WAL bytes (~59.9%); attachments another455,021,956bytes (~17.5%). This is repeated full-state persistence, not useful GPU work. Many observations repeat the complete source closure and host-log tail. This audit's compact metadata scan plus raw-record hashing took4.36s; that is an audit scan, not a live scheduler operation benchmark.

Historical measured baseline at589,393,887bytes/402events: strict replay4.140s, prefix hash0.348s; the read-only last-event model took0.380s. Sources: `audits/scheduler-wal-cache/live-readonly-profile.json` and `live-readonly-bound-model.json`. The activated cache removes much repeated JSON replay, but still SHA256-hashes every prefix byte on each read (`scheduler-wal-cache-independent.md`). A queue with no changed state therefore gets slower as history grows. The reviewed worker patch removed1001 redundant get calls on1000 ranked jobs (retaining two coherent advisory reads), a real algorithmic repair; its review explicitly does not claim measured live wall-time gain. No live operation timing was measured in this audit.

Repeated source hashing during a changed executable's admission/launch serves a different purpose from replaying all old queue snapshots on every poll. Preserve executable/review verification and atomic claim; eliminate or amortize redundant history work. A reviewed append-event persistence change or trusted incremental validation could address this later, but it should not delay supplying the already working dispatcher now. No WAL compaction, truncation, lock deletion or source-check bypass is proposed as an operational quick fix.

## Which gates to remove, automate or defer

1. **Remove volatile test stdout from launch source pins.** Keep exact executable, driver, review and deterministic proof pins. Retain rerun logs in distinct append-only audit outputs. Concrete cost: RQ32 required a root descriptor revision solely for a duration-string hash change. Risk: provenance weakens if audit results cease to identify their tested closure; require the log to name the immutable input manifest rather than pin its nondeterministic bytes as an executable prerequisite.
2. **Automate deterministic admission of the already independently reviewed closure, once authorized, and keep review of changed code.** Existing monitor is admissions-only, but its helper imports `GPUDispatcher` and calls `_validate` on a frozen queue view (`_reviewed_inbox_proposal.py:156–157`), so pinning dispatch is required by the current design. The unrelated `work` ranking optimization nevertheless invalidated the entire monitor closure; activation receipt explicitly says monitor stopped and manual admissions resumed. A future narrow refactor can separate common validation from worker execution so a ranking-loop change does not require an admissions-monitor restart/review. Cost: the present broad source coupling disabled the admission path and created another root task for every completed agent. Risk: removing the current dependency without extracting the actually used validator would bypass a real source check. For now review the changed worker method once and refresh the unchanged admission closure mechanically under root, rather than repeatedly re-reviewing the already unchanged validator. Do not restart the current stale manifest or auto-admit unreviewed work.
3. **Defer numerical interpretation, learning prose and promotion reviews until after cleanup and next unrelated dispatch.** Existing worker already supports this: VALIDATING releases the physical GPU, while DONE remains explicit. MoESSv2 remains VALIDATING and RQ32 still launched; no new gate needed. Cost of serializing all analysis is potentially the whole recorded inter-experiment gap, but exact attribution is unmeasured. Risk: jobs whose explicit dependencies need that result must still wait; unrelated work need not.
4. **Use one review per changed concern and reuse exact unchanged closure proofs.** RQ32 source review establishes PTX/register/codec semantics; its lab review establishes harness timing/storage/fixtures. Those are distinct concerns and are not shown redundant by this audit. The duplicate author-test rerun mutating pinned logs is shown redundant operational work. Preserve one independent source/harness decision for changed material; do not require rereading inherited source or regenerating all receipts because a timestamp/log changed. Cost of all reviews is not recorded, so do not invent reviewer-hour totals from file mtimes.
5. **Keep a bounded existing worker alive while admitted work is being prepared, and make expiry visible in its retained log.** Existing work supports finite idle wait; default0 exits immediately, and job/wall/reservation bounds intentionally stop it. A stopped worker does not restart across a chat/session transition. Actual worker exit/PID receipts for the125minute gap are absent from this audit, so a timeout is a possible mechanism, not proven attribution. Use the installed command/root handoff now; do not make deployment of another supervisor a prerequisite to the next admitted job. Risk: unknown execution must still stop/reconcile; never blindly restart/launch concurrently.

The minimal operational change is to preserve the working review/source/atomic-claim/cleanup boundaries while removing serial clerical prerequisites around them. Keep the next experiment supplied, collect it, and let CPU analysis/reviews of future changes proceed in parallel. More coordination machinery should have a measured cost/benefit case before it competes with that work.

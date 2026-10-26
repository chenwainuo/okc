# rtk-early01 independent preflight

Reviewer `/root/proj_evidence`, 2026-10-26 UTC. Source/CPU checks only; no GPU execution, remote jobs, source edits or commits.

## Decision

**GO for exactly one bounded exclusive diagnostic trial. No automatic port or judge slot.** The stage0 observer wait has an acyclic progress dependency and preserves the parent arithmetic, ring reuse and self-zeroing protocol. Root must treat a timeout, nonzero launch return, occupancy rejection, any correctness failure, incomplete JSON or missing required post-replay validation as failure. Close unless >=0.5us in both raw graph and prepare-F1 comparisons against the fastest shipped baseline. A favorable startup timestamp alone is insufficient.

This is a router-only lab importing hash-pinned pw35, not a self-contained challenge submission. A successful run admits review of one mechanism, not uploading `variant.py` or declaring end-to-end serving progress.

## Identity and scope checked

- Parent `pw35/submission.py` SHA256 `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8` matches. `gate_source()` has exactly one insertion anchor; removing the insertion reproduces raw RWCU exactly.
- Raw CUDA before `c2049af1f050b8f2f2af72a7dae31986a5553f74788bc7de920a0f02e8326daf`; raw after `8d9b5e765d0c62ce118faa5e9c4b0f71f43d42a18c3e74e340e98a7ffd7b47dc`. Generated source including this trial's switch cases and WG helpers hashes `f4ce5627db31635686c60a9caae1069c49dc084a1feaa3c02f6a16da3dd216f2`; all match the manifest/proof. These are different source scopes, not a hash discrepancy.
- Manifest checks all five named files, including lab `931cd73b11cd05664e17c207b044c9d0f4562fe47a279930c1c4a68e2979202f`, variant `55af0a5353b548b0366fd4c6967b881b9bcc4d0c5a581bec4470f7f0c146b345`. Independent AST parsing passes. Pure-source loading and `_rwsrc()` generation performed; `_rwext()` was not called.
- Exactly three production references (200,201,210), two candidates (300,301), and corresponding stamped instances. Candidate300 differs from200 only MODE4->132;301 differs from201 only MODE4->132. Existing MODE4 self-zeroing remains set; multicast/prefetch/stamp bits remain absent from production instances.

|m|shipped baseline IDs|candidate|identical-geometry parent|
|---:|---|---:|---:|
|128|201 KR256/slots3 +210 KR256/slots4|301 KR256/slots3|201|
|256|200 KR512/slots6 +201 KR256/slots3|300 KR512/slots6|200|

Shipped duplicate211 is correctly omitted. The shipped baselines are compiled from the same translation unit with the new MODE128 branch removed at compile time for their constants; no r6/torch comparator or fallback timing enters the decision.

## Barrier and convergence review

The single new operation is `if constexpr (MODE & 128) if (s==0) rt_mbwait(fb,0u)` immediately after stage0's x/hi/lo TMA requests and sole `arrive.expect_tx(SB)`. Full barrier count1 and expected32768bytes are unchanged. Warp4 lane0 alone issues these requests and performs the wait; other producer lanes can reconverge at the existing return. Consumers remain128 threads, with unchanged aligned WGMMA instruction sequence and named barriers. No new consumer convergence requirement or arrival count is introduced.

PTX permits parity waits on the current incomplete or immediately preceding completed phase; a wait observes completion and does not add an arrival. Stage0 begins at phase0, and the producer is the only actor able to issue the later transfer that could recycle full[0]. It therefore cannot advance beyond the observed phase while stopped at this wait. Consumer and producer observations can coexist. This follows the [NVIDIA PTX mbarrier wait specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait), checked during review.

Progress chain: already-issued TMA0 -> full0 complete -> producer issues stages1..; consumer may consume0 and then wait for1; after committing1 it retires0 and signals empty0 for reuse. All tested NSLOT>=3, so producer does not need empty0 to issue1. New wait cannot create a cycle through a future ring release. It can introduce a throughput bubble by delaying later requests; that is the hypothesis being measured, not a correctness defect or a predicted win.

Full/empty addresses, counts1/4, byte expectations, descriptor layout, stage order, WGMMA retirement, zero warp release/acquire, bar2 160-thread rendezvous, red.add epilogue are identical. Launch refusal -30 for a grid exceeding resident CTA capacity and -31 for multicast+self-zeroing remain. Both are critical because zeroing busy-waits across splits. Root's exclusive GPU ownership remains necessary; the standard launch itself is not a cooperative-grid guarantee under foreign GPU work.

## Workspace/counters and validation

- Each variant receives `slot=64+index(sorted(RTV))`; `counter_check()` derives the same index. Ten instances fit below the debug region. Production and stamped counters do not alias; output is intentionally shared only across sequential calls.
- Active tiles are4 at m128/121 and8 at m256/250. Counter entries are monotonically accumulated release counts, checked divisible by G16/32; odd poll entries and unused tile entries remain zero. Transitioning a reused201 slot from128 to256 is valid: each tile has its own integer multiple of G, and new tile counters start0. This check establishes completed groups, not an independently measured exact per-call increment count or overflow proof; the bounded trial is far below signed-counter overflow.
- Debug space starts at int offset16384, separately zeroed; maximum256 CTA records x64bytes fits. Stamped dispatch sets bit16 only and has its own counter slot. Resource rejection of a stamped kernel is a failed attribution run, not a timing win.
- `_validate()` runs all8192 onehot columns at x8 exactly against hi+lo, then random FP64 reference at even/odd m. Trial adds normal/outlier task-checker and stricter FP64 L2<=2e-5/max-row<=1e-4 checks on original and odd shapes. These are actual numerical tests planned for CUDA, not the CPU event model's claim.
- Each production graph used for timing is subsequently replayed after in-place fresh-X mutation and NaN poisoning of output. Both normal/outliers, two seeds, strict FP64 error and exact hi+lo probe rows are checked. This directly tests the **measured graphs** after their repeated replays, so stale output/counter shortcuts cannot be inferred away merely from pre-timing eager checks. All checks must finish before admitting a row.
- Fresh graph checks also cover121/250, normal/outliers, three new seeds. No output/input state is derived from test answers inside the kernel. The prototype does not add input caching or host synchronization to the served function.

Independent CPU proof rerun PASS:1081344 descriptor/index checks,2560 randomized ring traces,72704 events. Negative controls caught wait-before-issue64/64, wrong-phase64/64, skipped-empty46/64; not every skipped-empty schedule exercises the bug. These are sampled scheduling/index evidence and source guards, not exhaustive GPU execution, compiled SASS, numerical or performance proof.

## Measurement and bounded execution

Same X/out, immutable hi/lo, one-kernel graphs, fresh process. `torch.ones(64<<20,float32)` is256MiB read flush. Flush and start event are on the same stream, so the event begins after flush completion despite no separate synchronize at that point. `_time_us()` matches parent's prepare convention; fastest shipped pre/post sample median forms conservative F1. Both candidate timings and graph samples are outside compilation/validation.

Graph order alternates AB/BA and reverses which baseline is paired first.40 samples per baseline,80 for the candidate; raw samples and paired gain medians are retained. Hard programmatic graph bar uses `min(baseline medians)-pooled candidate median`, while F1 uses min of every shipped pre/post median. This is adequate to screen one bounded hypothesis. **Post-run promotion review must also inspect each paired gain distribution/median**: a pooled candidate median can hide baseline-specific drift. A marginal pooled>=0.5 result with weaker or inconsistent paired evidence is inconclusive, not permission to tune another depth/delay.

Stamps run after production timing; timestamps are attribution only. `consumer_ready` is consumer thread0 setup, not producer TMA issue. `first_full` is a consumer completion observation; a reduction in it alone does not show a full-call gain. Do not subtract instrumented times from graph measurements or project a precise H200 speedup from an H100 bandwidth change.

Root contract: one named container, in-container240s timeout, outer300s monitor, same repaired container-held flock, no GPU work alongside it. Command from manifest: `python -u proj_rsi/candidates/rtk_early/lab.py --reps 40 --stamp-reps 12 --json logs/rtk-early01.json`. Record/compare frozen remote source hashes before execution and after, generated CUDA identity, effective env, GPU/container owner, full output, terminal JSON and exit code. The JSON names parent SHA but does not itself record variant/lab/generated source hashes; the surrounding archived manifest is required for provenance.

On timeout/fault/stall, terminate the named CUDA-owning container/process and confirm compute PID disappearance before releasing the GPU. A host Docker-client timeout alone is insufficient. Final JSON must contain both128/256 shapes, all required validation sections, production/stamped counter checks and finite positive timing samples. `FAILED`, partial `STARTED`, occupancy-declined, missing stamps or malformed/nonfinite data are not fallback results. Terminal `CLOSED_GAIN_BAR_FAILED` closes this hypothesis without a new sweep; `PASS_REVIEW_REQUIRED` requires root's correctness/paired-timing/transfer review and minimal integration before any official check.


## Terminal independent review: CLOSED / no port

Root launched `okc-proj-rtk-early01`, launcher PID347556, inner240s/outer300s. Terminal JSON state **CLOSED_GAIN_BAR_FAILED**; elapsed37.783s including35.897s build/setup. Full log prints both RESULT rows and the same terminal state. At20:21:40 reviewer observed empty Docker/compute-app lists; no timeout or live CUDA owner remains.

### Recomputed timing gate

Units us; positive regression means candidate slower. Every graph sample is finite and positive;40 samples per shipped baseline and80 per candidate. F1 recomputed as min of all shipped pre/post medians.

|m|graph shippedA|graph shippedB|graph candidate|regression vs fastest|paired gain vs A / B|prepare F1|prepare candidate|prepare gain|pass>=0.5 both|
|---:|---|---|---|---:|---|---:|---:|---:|---|
|128|201:10.720|210:10.688|301:11.168|+0.480|-0.480 / -0.528|10.656|11.200|-0.544|NO|
|256|200:13.136|201:13.056|300:13.152|+0.096|-0.144 / -0.096|12.960|13.152|-0.192|NO|

Neither pooled nor paired data suggests a win. Close this scheduling hypothesis; no depth/delay variant, port, repeat or judge slot. The small m256 difference is near a tie versus its same-geometry parent200 (candidate+0.016), but it fails against shipped201 and fails the specified kill bar regardless.

### Numerical/replay and state evidence

All6 shape/version groups report all8192 onehot columns exact hi+lo and parent strict normal/odd validation pass. Each contains12 fresh-graph checks (normal/outliers, original/odd m,3 seeds):72/72 recorded. Actual measured graphs then pass4 changed-X/NaN-poisoned-output checks per group:24/24 recorded, exact hi+lo probe checks included. Independent JSON audit verifies numeric error fields finite and within L2<=2e-5/max-row<=1e-4; actual maxima below.

|shape bucket|fresh-graph checks|post-timing actual-graph checks|max relative L2 across recorded checks|max row error|
|---:|---:|---:|---:|---:|
|128|36/36|12/12|2.581079e-6|3.191228e-6|
|256|36/36|12/12|2.793888e-6|3.610923e-6|

Production/stamped counter checks present, active counts complete multiples of G, odd/unused counters validated in executing lab. Occupancies m128201/210/301=2/1/2, m256200/201/300=1/2/1. No declined occupancy, missing stage or correctness failure is converted to fallback timing.

### Mechanism result

Within-run stamped median-CTA observations, same-geometry parent comparison; us from earliest CTA entry:

|m|version|consumer ready|first full|last full|MMA done|zero ready|reductions issued|
|---:|---|---:|---:|---:|---:|---:|---:|
|128|201 parent|0.320|2.752|4.896|5.504|5.696|6.496|
|128|301 gate|0.288|2.432|5.344|5.952|6.144|6.880|
|256|200 parent|0.384|2.272|7.232|7.840|8.000|8.832|
|256|300 gate|0.320|1.888|7.360|7.968|8.160|8.928|

The gate advances first-full observation by0.320/0.384us, while last-full moves later0.448/0.128us and reductions finish later0.384/0.096us. This supports the anticipated fill-versus-throughput tradeoff: earlier stage0 is insufficient, later issue exposes extra pipeline time. Stamps are attribution from instrumented kernels, not production timing or a proof of one exclusive hardware cause. The full production graph/F1 regressions decide closure.

### Frozen evidence

Original remote artifacts: `/root/okc/logs/rtk-early01.{json,log}`; host launcher log `/root/okc/logs/rtk-early01-host.log` was empty at inspection (terminal process/JSON evidence above is therefore recorded separately).

- JSON SHA256: `a5091c2aed307af88b7896aef5c9c99e884654f174531113568f5f37f47a0854`.
- Log SHA256: `89eb599d6ca24a65763cfb43d6f70d3bd1063c53431bca6a645dd6b7fdff3e41`.
- Post-run manifest SHA256: `1f92fb2978fddec339e79a2837e4ab572d47ad1446a731e60af4f2ecbbfcb36f`.
- Post-run variant/lab/CPU proof/design hashes match the preflight manifest, and parent remains `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8`.
- Snapshot of readonly result harvest: `2026-10-26T20:21:57.963571+00:00`.

No accepted gain. Keep source and raw evidence for the closed queue item; parent/integration source is unchanged.


# rtk_early / stage-0 priority gate

State: source design accepted for one isolated prototype; GPU UNRUN. Parent owns launch. Kill after one bounded trial unless >=0.5 us graph AND prepare-style gain over fastest shipped rtk at the same m. No sweep or automatic port.

## Evidence and exclusion

Source parent: `pw35/submission.py`, SHA256 `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8` (frozen). `handover_folder/06-track-router-pw35.md`, `audits/p3-m256.md`; raw H100 `/root/okc/logs/p3m-s3.log`, `p3m-s6.log`, `p3m-r5.log` read-only.

Raw s6 median CTA stamps, us from earliest CTA entry:

| m / shipped geometry | consumer setup ts1 | first full stage ts2 | last stage ts3 | MMA done | zero-ready | reductions issued |
|---|---:|---:|---:|---:|---:|---:|
|128 / KR256, slots3, 2/SM|0.29|2.66|4.80|5.44|5.60|6.43|
|128 / KR256, slots4, 1/SM|0.32|2.69|4.74|5.38|5.54|6.40|
|256 / KR512, slots6, 1/SM|0.38|2.18|7.14|7.78|7.94|8.77|
|256 / KR256, slots3, 2/SM|0.32|3.10|5.70|6.40|6.66|8.19|

Important: inherited bench calls ts1 `issued`, but source records it on consumer thread0 before accumulator setup, NOT on producer128 after issuing TMA. These stamps establish a post-setup wait, not its causal breakdown. Whole-wave queueing is a hypothesis, not a proven exclusive cause.

Closed evidence: tensormap prefetch s6 changes first stage 2.18->2.18 at m256, 2.66->2.72 at m128; graph r5 changes +0.16/-0.06 us. Do not repeat. Zeroing already runs on warp5; its only consumer rendezvous is AFTER last MMA and staging. Removing its first release atomic from consumers previously recovered ~2.5 us; that change is already shipped. Initial CTA barrier/setup <0.4 us cannot alone justify this >=0.5 us bar. Cluster-of4 + self-zeroing deadlocked and remains forbidden. Co-residency guard remains mandatory.

## One scheduling change

Current producer128 issues all NSLOT stages before waiting for the ring. Every stage is x 16KiB + hi 8KiB + lo 8KiB =32KiB. Before consumer stage0 can run, later stages therefore compete in the wave's requests:

| m / geometry | CTAs | first-ring requested bytes | stage0-only wave |
|---|---:|---:|---:|
|128 / KR256, slots3|128|12MiB|4MiB|
|128 / KR256, slots4|128|16MiB|4MiB|
|256 / KR512, slots6|128|24MiB|4MiB|
|256 / KR256, slots3|256|24MiB|8MiB|

MODE bit128: immediately after issuing stage0's three TMA boxes and expect_tx(SB), producer waits for full[0] phase0 to complete; then issues stages1.. exactly as shipped. Consumer also waits on full[0], independently. No new barrier, arrival, pointer, load, tensor map, byte, accumulator operation or output-zero handshake. This removes later-stage TMA requests from the first-stage completion interval. It adds the stage0 completion dependency to later issue; the possible resulting bandwidth bubble is the main failure risk.

Implement only same-policy derivatives: v300 = shipped v200 (KR512/6, m256); v301 = shipped v201 (KR256/3, m128). Compare m128 against BOTH shipped v201 and v210 (KR256/4); m256 against BOTH v200 and v201. v211 is an identical tuple to v201 and adds no baseline. No larger/smaller geometry sweep.

Cost model: at H100 2.8TB/s, 4MiB takes >=1.50us, 8MiB >=3.00us if all bytes miss HBM. Existing first-stage stamps can be smaller due x reuse / L2 / overlap, so this is not a latency predictor. The available measured wait is 1.8–2.8us, but zero of it is yet proven recoverable. One wait instruction loop can reduce queue interference; it can also expose a first-stage round trip and lose. Expected win 0..~1us, low confidence. A faster first timestamp alone is insufficient. The H200 shortens bandwidth terms, so a marginal H100 stream-only win does not justify a judge slot.

## Legality / invariant proof obligations

PTX barrier rule: `.try_wait.parity` tests completion of current/immediately preceding phase; waits do not arrive or consume completion. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait). Producer's new wait is phase0, immediately after sole expect_tx arrival and all three stage0 requests; full[0] cannot be reused until producer itself issues stage NSLOT. Consumer and producer may both observe completion. No phase can overtake the new waiter.

Wait graph: TMA0 completion requires already-issued transfers, never consumer work or producer future issue. Once full0 completes, producer can issue stage1 because NSLOT>=3; consumer may then retire stage0 after committing stage1, releasing the empty0 barrier. This is the existing ring dependency. No cycle is added. All 4 consumer warps, full/empty counts1/4, warp5 zero coverage/release/acquire, bar2 160-thread rendezvous, final fp32 red.add, one-wave occupancy rejection (-30), MC rejection (-31) remain identical.

CPU proof: exact source-diff guard; TMA SW128 and descriptor/index equivalence, ring generations/phase waits under randomized completion schedules, complete output-fragment and zero partition coverage, repeated call counter divisibility. Negative controls must fail for wrong-phase startup wait, wait-before-issue, skipped empty wait. This is scheduling/index proof only; it cannot prove GPU convergence, compiled instructions, numerical tolerance, or speed.

## Named trial contract (root only)

Budget: outer300s / inner240s, exclusive container-held flock, fresh `rtk-early01` log/json. One build containing 3 shipped, 2 new and their stamped instances. Validate ALL8192 onehot columns exactly vs hi+lo; FP64 normal and outlier input at m128/256 plus odd121/250 at shipped strict f32 tol2e-5 (stronger than task1e-4), row bound, probes, repeated graph zero/monotonic-counter invariants. Trap/timeout/declined occupancy is failure, not fallback timing.

Timing: same x/out, cold256MiB flush, single-node graphs, paired alternating order. Prepare-style `_time_us` baseline before/new/baseline after; baseline F1 is fastest shipped pre/post. Graph win must exceed fastest shipped graph by >=0.5us AND prepare F1 by >=0.5us. Instrumented stamps report first/last MMA/ready/reductions; do not subtract debug times from production graph. If first advances but total does not, close without another depth/delay variant. Never compare success to r6/torch. No official slot unless root reviews full evidence and transfer risk.

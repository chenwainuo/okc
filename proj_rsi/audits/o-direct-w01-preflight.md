# o-direct-w01 independent preflight and runtime review

Reviewer `/root/proj_evidence`, 2026-10-26 UTC. Current target is official score<3000us; this is a bounded o-layer mechanism screen, not a score claim. Reviewer ran local source/CPU/device-compile checks only; no GPU launch, remote build/job, source edit or commit.

## Decision: GO for one bounded screening trial

No blocking device synchronization, lifetime, bounds, source-identity or DCE defect identified. Root may run exactly the frozen NT32/NT128 Q4/X8/KS1 policy under the exclusive container-held GPU lock. Numerical/retained-graph checks and >=0.5us gain in **both** incumbent F1 and median paired graph difference are mandatory per shape. Failed or missing gates close the fixed policy; a survivor requires independent runtime evidence review and stronger actual baseline-output verification before integration/promotion. No geometry/flag sweep, incumbent mutation, direct submission or automatic judge slot.

The local device cubin is certified for the checked source/flags. Full Torch host extension is **not** locally certified: local CPU-Torch/CUDA headers lack `cusparse.h` pulled by inherited ATen CUDAContext. Root's real-image `load_inline` build is a separate gate; a build/setup failure is not a timing result. No header stubs were accepted.

## Identity and independent checks

|artifact|SHA256|
|---|---|
|generated extension source|`2f571e925469ea90476aa1ec87d72d86997d460876a7250603113c1653d3feb1`|
|device-only source|`8466df061cca77cb7fc378a20771ddff679712e708da804d6f86c36037314aa0`|
|prototype.py|`b97af657bbfbcfebc0ba75ba1feaa310dae94128890ac304e02cca03755be534`|
|kernel_body.cuh|`fe3daab6e0eb479099d64cb4827d572ec35842481d0e25dd709c27ae090ef7d3`|
|lab.py|`ee2f10f139d99dd00086da5dc3d7bf617bb8f262e4ce08028a5824b9d3009c94`|
|rs32 helper parent|`a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`|
|pw36 comparator|`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`|

All manifest file hashes independently match. `P.source()` and `P.device_source()` reproduce the archived CUDA files exactly. Python AST checks pass. Reviewer invoked only CPU `bits()`/`schedule()` functions rather than preflight's file-writing main:131072 BF16 values bit-exact,122691 escapes, cap15392 (multiple32-entry reload windows); Q4/A2/X8 schedule model passes. Arbitrary Inf/NaN/subnormal bit coverage is a reconstruction proof, not a claim that GPU matmul arithmetic preserves such values.

Reviewer independently compiled device source using inherited `compute_90a/sm_90a`,O3,DNDEBUG,relaxed-constexpr flags and inspected SASS: both232registers,0stack/spills/no warnings/C75/no local LDL/STL instructions. Each NT instance has16HGMMA,4WARPGROUP.ARRIVE,24LDG.128,0LDS.128,16X-UTMALDG and1output-UTMASTG across four unrolled rounds. All NT accumulator fragments flow into the inherited stmatrix/TMA output epilogue; no D[0]-only sink or mainloop-only output is accepted.

## Direct W, async A and X-ring correctness

- Each CTA computes64 distinct channels, full K1024, no split-K. Grid128 covers N8192 once, without overlapping output tiles. `nb=blockIdx/2`, half=blockIdx&1 selects exactly four16-row warp tiles from the parent's128-row natural-K pack; stage stride12288 and lane offsets0/512/1024 match the CPU pointer/escape proof.
- Four packed Q slots total48registers/thread; A0/A1 total32; D16/64. `rq_unit` decodes Q into an independent A buffer before Q's s+4 prefetch overwrites that Q slot. A0 used at s is only decoded again at s+2, after s+1 wait_group1 has retired group s. The post-retirement `rq_fa(old)` anchors prevent compiler reuse of an asynchronously live A register by intervening Q prefetch. Final wait_group0 drains both A/D and B reads before epilogue stores.
- All128 consumer threads issue the same aligned WGMMA sequence. Escape processing diverges only inside a warp and reconverges before WGMMA; the unchanged sorted-list/shuffle helper and32-entry reload semantics are exhaustively exercised by CPU reconstruction. Actual GPU numerical tests remain mandatory.
- Producer initializes full/empty counts1/4 and issues eight X-only TMA boxes, then initial256-thread CTA barrier. Consumer prefetches four W stages before that barrier. Afterward the complete producer WG (warps4..7) uniformly reduces registers to40; the consumer WG uniformly requests232. With static232x256=59392registers, compile resource allocation fits a single CTA on an SM. No cross-CTA co-residency wait is required; runtime full extension must report zero local bytes and execute the real launch successfully.
- Producer refills X slot s%8 only after all four consumer-warp arrivals for s-8. Release follows the next group's commit/wait_group1, so the old X is retired. For s8 the producer waits for stage0 release; consumers can reach stage1 independently from the already issued initial eight boxes, so no ring cycle is introduced. Phase0/1 waits match first fill/refill. Producer may exit after issuing final fills, but consumers keep CTA/shared storage alive and wait for every full barrier before use.
- At the final wait_group0 every X fill/read is complete; consumer-only128-thread barrier precedes stmatrix overwriting slot0. Producer has no remaining writes to that storage. Fence.proxy.async plus consumer rendezvous orders generic writes before the TMA output read. Thread0 commits and waits for TMA source reads before it can leave; this retains the parent's epilogue contract. `.read` waits source/tensormap reads, not an independently measured global-store completion timer; no wall-time claim follows merely from this instruction. See [PTX bulk wait semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
- C::SMEM is33920/132224bytes for NT32/128, including1024-byte alignment and128bytes of barrier metadata. Output tile is4096/16384bytes and fits slot0. Metadata lies after all eight X tiles. Parent map helper preserves SW128 alignment and fixed tile width64. Host opts in to dynamic shared memory before launch (unlike the earlier port01 query bug).
- Exact host guards reject anything except N8192,K1024,KS1,M==NT32/128. No odd-M serving/fallback is introduced. Count storage is allocated for the inherited host ABI; this kernel has no counter/atomic/reduction path and does not use it as scratch.

## Runtime validation and measurement admission

Prototype checks before performance: all1024 onehot x8 columns exact against BF16 W; normal/outlier FP64 L2<=4e-3 and row<=2e-2; task-checker normal/outlier inputs; retained graph output on original/zero/fresh-outlier X mutations, X unchanged, exact zero output. The zero/fresh sequence is important for the new plain-store/no-counter path. All are fail-fast, not warning-only. Missing a VALID group invalidates that shape/run.

The comparator is freshly prepared frozen pw36's **actual** exact-M `kernel_fn` dispatch, including rq/KS or r6 Lt/gv/sel selection; it is not torch or an imagined parent path. Preparation is outside timed replays and must finish without relevant skipped phases. Prototype weight pack is independently unpacked and bit-compared before launch.

F1 uses discarded warmups, incumbent before/candidate/incumbent after, lower incumbent retained. Twenty-one cold256MiB read-flush graph pairs alternate AB/BA; same prepared baseline dispatch is held throughout both comparisons. Candidate writes caller-owned output; incumbent allocates its serving output inside capture, whose tensor is explicitly retained to keep captured storage live. CPU allocations/preparation are outside CUDA-event timing. Record actual baseline path, every raw pair and both F1 baselines; admit only finite positive samples and recomputed>=0.5us in both gates.

**Measurement limitation:** separately captured mutation validation is strong, but current lab's *actual timed* incumbent graph is checked only for finite output after replays. Its actual timed prototype graph gets the task oracle on the original timing X; changed-X/zero/outlier mutations run on the earlier retained graph. This is sufficient for a bounded screen against already validated frozen pw36, not standalone promotion proof. Any survivor must verify actual baseline output against the task oracle and actual timed graphs on changed/zero/fresh-outlier X before accepting a production claim. A finite baseline tensor alone is not correctness evidence.

Root owns a named bounded process/container; use explicit inner timeout and outer kill monitor (240s/300s is an appropriate existing contract), with all GPU preparation/build under the exclusive lock. Record launch handle, source hashes, full real host/device compiler output, runtime attributes, prepare phase timestamps, all checks, raw pairs, terminal JSON and exit. Confirm compute/container termination after success/fault/timeout. Partial JSON, missing LAB_DONE, launch refusal, compiler/CUDA error, nonfinite sample or missing gate is not an admitted performance result. No timing from a failed setup may be relabeled a loss or win.

## Runtime review status

COMPLETE: independent raw review below closes both fixed shapes. No survivor or promotion.

## o-direct-w01 terminal review: CLOSE both shapes / no port or sweep

Root PID351735, named container, inner240s/outer300s, terminal2026-10-26T21:14:50UTC. Archived host log contains LAB_DONE and LAB_EXIT0. Root reports container gone/GPU0MiB. Reviewer independently parsed local copied full log/JSON, checked source/baseline identities and every numerical/replay/timing field, and recomputed gates from all42 raw pairs. No CUDA/compile/C75/error or skipped-budget line appears. Full real-node Torch extension build succeeded; runtime attributes match232registers/local0, shared33920/132224bytes. All rq/gx/rq32/router prepare phases complete, baseline preparation45.927s (78s from module clock including prototype build).

This is **two exact-M lab validation groups**, not a65-case official harness. Each group covers all1024 onehot columns exactly, parent FP64 normal/outlier L2<=4e-3 and row<=2e-2, explicit task normal/outlier inputs with exact probes, and retained original/zero/fresh-outlier graph checks with input unchanged. Both groups pass. Separate timed prototype graph original-input task checks also passed; actual timed baseline finite-only caveat remains, but no favorable result is being promoted.

|m|actual prepared pw36 pick|F1 before/after us|prototype F1 us|F1 gain us|paired median gain us|prototype slower pairs|gate|
|---:|---|---|---:|---:|---:|---:|---|
|32|rq:t32s5hze,KS2|9.696/9.696|11.936|-2.240|-2.208|21/21|CLOSE|
|128|r6 tunedLt code200014; gvNone|10.656/10.656|14.720|-4.064|-3.936|21/21|CLOSE|

Paired graph marginal medians: baseline/prototype9.760/11.968us at32 and10.688/14.592us at128. Admission uses median **paired differences**, not a fabricated marginal-ratio score. All42 pairs are finite positive and every pair loses. Independently recomputed F1/min-bracket and paired medians match JSON to<1e-9us. Both stored decisions are `CLOSE_FIXED_POLICY_NO_SWEEP`; neither approaches the>=0.5us gain target.

**Disposition:** close the NT32/NT128 fixed Q4/X8/direct-W-GPR/full-K/HALF/plain-store policy. Preserve source and CPU/device/runtime evidence. No retry, depth/geometry tuning, integration, official check or score claim. This is a measured H100 loss for these two policies, not a universal claim that all direct W loads fail. The mechanism's precise stall cause is not established by this timing table and does not authorize an automatic profiling/sweep follow-up.

Root/source owner should mark source/results README queue state CLOSED with this artifact; reviewer changed only this audit because source edits are held. Incumbent remains frozen and the official<3000us objective remains unachieved by this trial. Reviewer stays available for the next independent runtime gate (lt_graph pending).

Raw artifacts: `results/proj-rsi/takeover/20261026/o-direct-w01.json` and `o-direct-w01.host.log`.

- JSON SHA256 `5d772ece0a12737336add57e33113d1ced27aae4f29d0f7d0e80938f59021256`.
- Host log SHA256 `d15f9064b9c86c1dea36acdd1314c09829d05e6c1eeed74fe2bc57822694891f`.
- Runtime generated source SHA256 `2f571e925469ea90476aa1ec87d72d86997d460876a7250603113c1653d3feb1`; baseline `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.

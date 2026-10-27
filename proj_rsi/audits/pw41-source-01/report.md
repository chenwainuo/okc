# PW41 P1 late-ticket independent source/preflight review

**GO for CPU/source readiness of the exact final frozen package.** This is not executable GPU-wrapper, launch, prepare, serving-selection or promotion admission. Final manifest `0dc0e599569568a07d0f43a5ef276c33c62d69e665d538177eebf2be1efde051`, all39 file hashes independently verify. Candidate source `a77f785b06d3b9cdbf9fe3870505bd15f4e68c46807fa61dc6c3b9756dbe199f`, parent PW38 `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`. This decision was issued only after author17 supplied and the reviewer rechecked the final freeze; no partial-source approval is inherited.

The source adds exactly four compile-time bit30 late-ticket types in five existing P1 bake cells: router8/16/32 and unpooler1/8. All original95 GVI entries, GXP prefix, indices and counter-instance identities remain exact; new indices95..98 append. Complete strict four-edit inverse restores PW38 byte for byte. Duplicate anchors reject. Existing named functions, prepare/selection clock/budget/margin/fallback, RQ/EH/router kernels, input dispatch and codecs are unchanged. The only existing AST assignment changed is GVCU; `_P1LT`/P1LATE and the explicit append operations are new. New domain permits only the four declared types; the reviewer independently compiled an unsupported late+PDL combination with real offline NVCC and confirmed the intended static assertion rejects it.

## Ticket, publication, precision and storage semantics

The original ticket is issued by w0/lane0, which is thread0. New types omit only that early issuance, then thread0 obtains the same `atomicAdd(p.cnt+2*(bz*GX+bx),1)` immediately before the original first acquire loop, after HMMA and private shared partial stores. The target uses the identical tile expression and `ticket/S+1`. No ticket is eliminated: exactly S even-word increments and one odd publication occur per completed tile launch. The unused-result flag atomic is distinguished from the returned ticket, including the out-of-line zeroer block.

Both zeroer forms retain all output zero stores, writer fences, warp/CTA rendezvous, odd publication and early-return behavior. Acquire order/cache invalidation, retained2^24 poll trap, CTA barrier and global reductions are unchanged. Private `red[]` shared stores remain before acquire as in the parent; the selected P1 constant branches keep external `p.ws` writes disabled. Old bit masks for codec/cache/PDL/cluster/resource/reduction consume only their original bits; added bit30 neither activates PDL nor changes the original reduction/workspace route. Static domain and compiled negative control bind that statement.

Router remains FP32 output/accumulation with the unchanged BF16 hi+lo weight representation; unpooler remains lossless BF16. No input-derived cache, arithmetic shortcut, precision change or new output reuse appears. Moving the atomic can change global floating reduction arrival order despite unchanged per-CTA computation. GPU one-hot/basis, FP64, task and fresh input/probe checks remain mandatory; CPU state models do not establish numerical GPU results.

`_gvcall` retains per-instance `p1cnt` arrays keyed by GVI index, allocating2048 int32 zeros on first use. Appended types get separate arrays rather than accidentally sharing epochs with another S; old instances keep their original keys. Repeated calls carry completed nonzero epochs, with even=E*S and odd=E. Any permutation of S tickets returns E*S+k,0≤k<S, so every target isE+1 even if the zeroer published before the late atomic. Extra x-zeroers take no ticket. Full-storage counter contracts require exact active even/odd deltas, unchanged dormant/alias/guard bytes and unchanged device/pointer/offset/shape/stride/dtype identity. No harness reset is introduced.

Concurrent launches/graphs/streams sharing a P1 counter/output state remain unsupported; ordered single-owner launches and no signed-int32 wrap are proof preconditions. Nine full-storage cases reject54 active/dormant/guard/alias/offset mutations. This is a source-bound monotonic protocol, not blanket “all counters zero.”

The five bake cells are not an exact-M-only serving restriction: unchanged `_bk` dispatch can serve ragged m5/7/9/17 through them. CPU zero tiling covers every m1..8*MT in112 cases. A future runtime wrapper must validate ragged shapes and actual selected routes, not only the five nominal benchmark m values.

## Independently reproduced machine/resource/protocol evidence

Read-only `verify_frozen` and complete `cpu_proof` pass; all six unit tests pass. The reviewer regenerated exact control/candidate CUDA input strings from the pinned submission/catalog and compared them byte for byte with compiler inputs. Offline `cuobjdump` independently redumped both saved cubins:95 control and99 candidate GV instruction dictionaries exactly equal retained SASS. All95 original PC/predicate/instruction/operand/branch streams and full resource records remain exact, and all95 equal the retained actual current PW38 JIT archive. The15 old dormant/non-target functions with nonzero stack/spill records remain identical; they are not new PW41 failures and are not silently subjected to a zero-spill contract they never had.

All four new types compile without stack/spills. Last HMMA < returned-ticket atomic < first acquire < CTA barrier < first global output reduction holds in every exact new stream. There is still one returned ticket plus one flag atomic and four output reduction sites. Bounded trap and GPU memory fence remain present. This is an exact linear-PC/source-qualified check, not a general divergent-CFG or address interpreter.

| Target | Registers old→new | Rounded regs | Shared B | Grid CTAs | Register/thread capacity upper bound CTAs/SM | Necessary SM count at that bound |
|---|---:|---:|---:|---:|---:|---:|
|router8|56→56|56|4608|272|4|68|
|router16|64→63|64|9216|256|4|64|
|router32|150→149|152|18432|128|1|128|
|unpooler1/8|69→69|72|4608|320 each|3|107|

These4/4/1/3 figures are **static capacity upper bounds**, not actual or guaranteed occupancy. One-register changes do not cross rounding thresholds and supply no occupancy gain. Before BOTH parent and candidate prepare, the future reviewed wrapper must verify actual supported device identity and **at least128 SMs**; it must also query exact-symbol registers/local/shared/actual occupancy and require each complete P1 grid to fit before any P1 launch. A post-selection query is too late because prepare executes variants.

The existing `_gxbake` queries gvocc but its res<grid decline condition excludes P1; it can merely print low residency and still launch P1. This source intentionally preserves that parent helper. Therefore the external pre-prepare guard is a real runtime prerequisite, not something this source has already enforced. `lab-plan.json` now states it explicitly. No executable wrapper currently implements or satisfies that admission contract.

The finite ordered-epoch model independently reexecutes13,377 reachable states for old/new S2..4 z/x models and1,536 random complete S4/8/16 schedules, including repeated/near-limit epochs. Seven negative controls reject missing acquire, early flag publication, misaligned even epoch, odd mismatch, int32 wrap, concurrent ownership and unknown residency/progress. Eventual zeroer/engine progress and real full-grid residency remain hardware/runtime assumptions. No GPU scheduler proof or elapsed-latency inference is made.

## Learning, costs and later boundary

Applied lessons in the frozen package separate source composition from measured additivity and physical counter semantics from counter names. The atomic moves from a potentially overlapped prologue to the final acquire critical path; burst contention or delayed acquire may make it slower. Atomic count remains unchanged. A fixed-arm positive observation is mechanism evidence, while actual full prepared PW38/PW41 policies must be compared separately. Preserve every signed tiny gain/tie/loss and selection/budget effect; no fixed0.3/0.5µs cutoff or family retirement follows.

Author's complete CPU quote is45 implementation/proofs +15 review +15 analysis =75min, within the common90min scenario. Proposed narrow1020s plus cumulative3420s gives4440s conditional local GPU reservation; **1020s exceeds the common300s narrow comparison scenario**. Those numbers are not achieved runtime or hardware authorization. Root must rerank/redesign the execution stage and review/freeze a reusable counter-aware matched-allocation wrapper before any runtime decision. Official H200 cost and attainable gain remain unknown/null.

That future wrapper needs actual old/new JIT and resources before math, pre-prepare P1 capacity gates, real current weights/pack/escapes/input immutability, allKbasis/FP64/task/ragged oracles, NaN/zero/outlier/probe mutations, complete physical active/dormant/alias counter deltas, retained graph lifetimes, strict own-fixture actual final outputs before replay/mutation/profile, balanced raw full-pipeline F1/paired comparisons including actual selected PW38 and named post-exit cleanup. It must not force a favorable route, reset prepared counters or relabel a fixed-arm result as policy gain.

Reviewer performed CPU reads, tests, pure protocol evaluation, offline cubin disassembly and one expected-failing offline domain compile only. No CUDA context/extension load, GPU/prepare, API/remote action, queue/parent/source mutation or promotion occurred. Evidence: `proj_rsi/audits/pw41-source-01/{independent-cpu-proof.json,offline-redump.json,independent-negative-domain.json,review-summary.json}`.

Read-only reproduce: `python3 -m proj_rsi.candidates.pw41.verify_frozen` and `python3 -m unittest proj_rsi.candidates.pw41.test_cpu`. Do not regenerate author build/codegen artifacts to verify their frozen hashes.

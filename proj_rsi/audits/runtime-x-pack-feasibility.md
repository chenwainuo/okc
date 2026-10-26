# Fresh per-call native-A X pack: CPU feasibility only

`state=CPU_MAPPING_AND_SECTOR_MODEL_PASS; implementation=NONE; GPU=UNRUN; attainable_speedup=UNKNOWN; authority=ITERATION.md`

Root-requested novelty/feasibility study; reviewer implemented no CUDA/kernel/consumer/source variant. Current V4/zero preflights took priority. Bounded history search across projection audits/LOOP/current candidates found immutable **W** fragment packs and identity X-staging copies, not an already measured fresh-per-call native-A fragment-pack+reversed-main pipeline. This is a scoped search finding, not proof of all unseen session history.

## Exact compact layout

At exactM128/K4096, choose BF16Xp shape`[stage64][mathwarp8][k16group4][lane32][reg4][pairhalf2]`; size524288BF16=1048576B, no padding/duplication. Let mathwarp=4wg+w, lane=4g+t. Packed registeri in groupj contains the input pair:

```text
row=64wg+16w+g+8(i%2)
col=64s+16j+8(i/2)+2t
Xp[s,mathwarp,j,lane,i,0:2] = X[row,col:col+2] bits
```

Independent CPU enumeration proves all524288original BF16indices exactly once, output positions bounded0..524287, all65536bit-pattern labels preservable by bitwise copying, and no numeric conversion. Consumer bytebase`Xp +16384*s +2048*mathwarp +16*lane`; four16Bvector loads at displacements0/512/1024/1536 directly fill nativeA[4j+i]. Each warp-stage requests2048B/64unique32Bsectors, all16Baligned. Addresses are simpler than strided originalX rows, and no consumer SHFL/SEL is required.

## Coalesced packer requires no shuffle either

Root corrected an initial overly broad gather warning: a flat output-**uint32-word** thread assignment varies i across adjacent threads, whereas consumerV2 held i fixed per instruction. Reviewer independently verified every one of8192consecutive32-thread word-warps: each reads EXACT4unique32Bsource sectors for128Brequested and writes4contiguous32Bdestination sectors. Full262144uint32source-word bijection, lastbyte1048575. Thus no2×sector amplification occurs with this mapping, and no packer shuffle is required. A per-consumer-lane thread loading all16registers would be a different access schedule; do not confuse it with the flat-word route. The earlier suggestion to repeat v1's shuffle inside the packer is unnecessary for this mapping.

Logical pack overhead:1MiBread+1MiBwrite,1MiBscratch and one additional GPU node EACH call. Flat-word pack uses8192warp load requests and8192warp store requests, coalesced4sectors each. Main still requests128MiBlogical Xp across128channel CTAs; it does not eliminate repeated activation delivery, W traffic, B-store/read or barriers. It moves a bit permutation out of128repeated consumers while preserving v1-like coalesced main demand. These are requested logical bytes/instructions, not guaranteed HBM traffic or performance forecasts. Fresh pack writes/cache eviction and kernel-boundary overhead must be measured in the complete pipeline.

## Necessary future implementation contract

- Every`kernel_fn`call enqueues pack from its live originalX, then main on the same current CUDA stream; no prepare-time input pack, content/seed shortcut, pointer/data-derived cache or reuse of stale Xp. Shape eligibility remains exact; original X stays immutable and all BF16payload bits survive.
- Actual timed CUDA graph includes BOTH pack+main, including pack every replay. A main-only/pack-only/profile duration is diagnostic, not admission. Compare total against current actual incumbent, best failedV1 and immediate predecessor as appropriate, with conservative before/after F1 and paired replays. One extra node may erase main savings; no speed guarantee from the sector model.
- Workspace needs explicit lifetime/capture/reentrancy ownership. Retain the real Xp used by each timed graph; no dangling temporary or concurrent calls racing through an unowned shared scratch. Preallocation is not an input cache if fully overwritten each call, but safe reuse/concurrency still needs proof. Stream order supplies pack→main dependency; do not add an untimed pack, hidden host copy or global synchronization to manufacture an unfair comparator.
- Direct Xp→native asynchronous A changes register lifetime: use a proved retired-slot scheme such as v2'sA4, or independently prove separate raw/A buffering. Copying the old current-rawQ refill into the current live asynchronous A is invalid. Resource/occupancy/SASS and full D/output mapping remain required; no tensor/producer/plain-store change is implied by this study.
- GPU validation must compare complete pack bits against the independent permutation and exercise fresh/original/zero/outlier/probe X through the SAME retained two-node timed graph, poison/inspect actual outputs and pack scratch, full4096basis/FP64/task numerics, weights/counters immutability, distinct fresh captures and correct profile node identities/cleanup. Fresh-input mutation must demonstrate that every replay repacks; no normal-weight test is replaced by a diagnostic constant input.

Useful next falsifier after a reviewed source exists: can one fresh coalesced1MiBpack+native-A vector consumer improve TOTALgraph latency versus its predecessor, and does it beat the actual incumbent? A partial predecessor gain is learning; no incumbent/official claim before those measurements. The possible benefit is removing per-channel repeated fragment formation, not lowering W bytes or proving the shared-B producer ceases to be limiting. No implementation or GPU execution is authorized by this document alone.

## Prefreeze author finding: identity-copy physical alias requires repair

Author's first CPU-only pack/main build reportedly compiles main122/pack22registers,0spill/noBFLY, but source rawQ→nativeA identity-copy can be physically coalesced by the backend. Author observed stage0HGMMA usingR56/R60, then future raw refillLDG overwriting those registers before the group's remainingHGMMA and covering wait1. Reviewer has not certified that unfrozen SASS; current verdict is HOLD/noGPU under the conservative asynchronous-A lifetime contract. MissingC75warning does not replace physical lifetime proof. The packing layout/sector proof above remains valid; this is a concrete implementation correctness issue before the first admitted version.

Repair must explicitly separate physical A until retirement or adopt a declared direct-nativeA4 retired-slot schedule with new frozen source/SASS/resource proof (the v2A4 sequence is prior validated evidence, not permission to assume its automatic lowering). Do not silently treat a changed staging/retirement schedule as an isolated address change. Preserve invalid CPU compilation evidence, prove exact load/consume/refill register lifetimes, then complete the full two-node graph/numerical review before any GPU trial.

Root subsequently authorized nativeA4/retired-slot/three-ahead as the initial packed-X version's correctness-driven plan revision. Preserve unadmitted rawQ4/A2 copy source/SASS/hash/hazard proof; no opaqueXOR0 workaround. The four-ahead→three-ahead staging difference is an explicit confound in any predecessor comparison. Exact new freeze, physical load/consume/refill retirement proof, source/map/resource gates and full two-node validation/timing remain pending independent GO. No GPU result exists for this new mechanism yet.

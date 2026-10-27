# Learned QKV geometry feasibility: no geometry-only candidate admitted

**Analytic hold**, not a family rejection. The exact split4 pipeline is a valid local regression:40.288001µs paired versus actual selected RS3217.088000µs/PW3617.408000µs. Geometry and launch waves may contribute, but this archive cannot identify that contribution. Per-node durations, device SM count and actual wave telemetry were not retained. A positive geometry-only performance ROI is unknown; no source variant, offline compile or GPU trial is represented as ready.

Evidence is the source-bound `research.qkv.split4.harness.gpu.01` collection and `audits/qkv-runtime-01`. CPU extraction independently verifies all12 inventory hashes and46 dispatch pins. Executed candidate source SHA256 `29f7f11bdf3246e66b9419fce4003cb27c22a4cf263a3d1f15c0f3770371f1a2` matches the retained actual JIT source record. The reported profiles bind the current pack→main→reduce names and actual selected parent RQ route. See `candidates/qkv_geometry_followup/analysis.json` for exact hashed inputs and assertions.

| Item | Exact retained/source-qualified fact | Missing actual observation |
|---|---|---|
| Fresh pack | Explicit launch512 CTAs×256threads; current-X2MiB read/write mapping exhaustively proved | Kernel duration and active-CTA occupancy |
| Candidate main | Explicit dim3(36,4)×512threads;144CTAs; outputtile128×64; SK4;32K64stages/split | Per-node duration, SM count, waves/tail scheduling |
| Candidate resource | Actual122regs/0local; postoptin occupancy1; source dynamicSMEM103600B | A new version's actual resource result |
| Ordered FP32 reduce | Explicit1152CTAs×256threads;4partial planes,4.5MiB read | Duration and occupancy |
| Actual selected controls | Frozen route s4s5hz NT128/ST5/GS1/FL3/KS7; source launch(NB,KS), NB18,384threads;126CTAs | Runtime launch telemetry and occupancy were not separately recorded |
| Timing attribution | Full graph40.288001µs versus actual RS3217.088000µs | Pack/main/reduce durations are all UNKNOWN, not zero |

The source-bound grid is read from actual executed host launch expressions and exact selected route; it is not guessed by multiplying dimensions. NB18 is bound through the unchanged packed-layout source and validated2304-row weight shape. No captured per-node launch metadata is claimed. `ceil(CTA/(SMs*occupancy))` is only a conditional capacity arithmetic formula: SM count is absent, pack/reduce/control occupancy is absent, and neither actual waves nor elapsed cost is supplied. Hardware-model familiarity is not substituted for missing observations. Kernel profile names do not contain durations, and driver JIT/prepare phase wall time is not node GPU time.

## Partition, CSR and protocol feasibility

Global K has128 K64 stages. Mathematical equal divisors are SK1/2/4/8/16/32/64/128. The exact unchanged kernel unconditionally initializes six W stages and four A stages, then executes two K128 pairs in each loop step: KCS must be at least6 and a multiple4, hence at least8. Only SK1/2/4/8/16 meet these necessary schedule conditions. SK32 hasKCS4, SK64 KCS2 and SK128 KCS1; merely changing launch/split constants permits out-of-slice W/A prefetch or overshoots paired math. Five invalid KCS controls1/2/4/18/19 reject.

For each eligible equal split, the CPU model proves128 global stages covered exactly once, each local A pair consumed once and retired before refill, and a finite publication/retirement dependency DAG. These are necessary source arithmetic checks; they are not compiled physical, numerical, lifetime or performance admission for a new version. The unchanged current-source proof also reexecutes524,288 XP word ownership,294,912 output owners and the canonical physical A/D/fence checker.

Independent CSR coordinate inversion covers144 tiles×128 global chunks for each SK1/2/4/8/16 and ragged3/5/6/7:165,888 tile/chunk cases. Group=`(split*18+tile//8)*8+tile%8`; local seq=`chunk-floor(split*128/SK)`. Every group, reconstructed tile/chunk and extreme lane/register/half/xor field remains in bounds and avoids the sentinel. This establishes address/group arithmetic, not actual escape counts in a new run.

Recorded actual SK4 CAP is32, so each current32-chunk physical list has fewer than32 entries. Every SK8 sixteen-chunk list is a subset of one such list. Its necessary strict CAP therefore remains32. Groups double576→1152 and ESC allocation73,856→147,584B including trailing guard. That exact source-derived bound uses the actual retained CAP, not CPU-reference RNG escape counts. General new versions must still construct/check their real per-list counts, sorted local sequences, strict CAP, sentinel and guard bounds.

SK7 boundaries are0/18/36/54/73/91/109/128 with lengths18/18/18/19/18/18/19. Fixed `k0=32*ks`, fixedKC32 and four-stage pair unroll cannot consume them. Supporting ragged splits needs floor-boundary mapping in main and CSR, per-slice startup bounds, two-stage/single-stage tails and parity-safe A/B retirement/drain, complete SK partial ownership and an ordered reduction covering every plane. A naive KS7 edit is invalid even though the generic stock CSR builder itself understands floor boundaries.

## Why neither suggested knob is a ready minimal performance version

**SK8:** KCS16 has valid necessary full-stage coverage, but it is not one launch edit. The host grid, kernel KC/k0/symbol, G1152 CSR checks/allocation, partial workspace and qcheck extents, ordered reducer's eight planes and strict terminal reconstruction must change coherently. Keeping the four-plane reducer omits half the K contribution or writes out of bounds. Main CTAs double144→288; partial writes/reduce reads double4.5→9MiB; logical main-X payload remains72MiB. Half as much K work per CTA does not establish less elapsed time, especially with unknown node costs/waves and increased auxiliary work. Actual byte traffic, resource/codegen and numerical grouping are unmeasured.

**TT128:** There is no TT symbol in this frozen implementation, and template NT is already128. If TT means N/output-column width128 rather than64, it changes m64n64→m64n128 WGMMA, D32→D64 ownership, decoded-W producer/shared tile, output mapping and resource pressure. It needs new register allocation/occupancy, shared descriptors, A/D retirement and source/JIT proofs; the current512-thread122-register compile cannot be inherited. If TT means NT128, it is already the tested configuration. No invented “TT128 compile” or resource result is provided.

No compiler run is needed for this analytic hold because no changed kernel is emitted. Existing frozen compile/JIT evidence remains attached to the actual losing version. A future coherent source version must get its own offline compile, exact current-X/weight/CSR/output coverage, immutable/workspace and FP32 reduction gates, numerical/basis/probe checks, physical register retirement/source-map proof and actual resource/JIT admission before timing.

## Applied lessons, falsification and next decision

Applied registry lessons: `wring-mechanism-policy-separation@1` keeps matched mechanism controls separate from actual selected policy and warns about identical-main context gaps; `prepare-selection-is-experiment@2` requires actual prepared routes rather than forcing a favorable fixed path; `composition-not-additivity@1` prevents source edits/coverage from being treated as additive time; `rq11-full-residency-ledger@2` requires whole CSR/workspace/allocation accounting and guards. The primary observed-negative anchor is `audits/qkv-runtime-01/lessons.json`: all30 actual pairs lose and coverage144 versus126 CTAs did not imply speed. Old family closure language is not a present decision.

Specific falsifier: a separately authorized frozen diagnostic that records SM count, actual grid/block/dynamic-shared metadata, per-kernel postoptin occupancy and pack/main/reduce GPU durations in the real fresh-input full pipeline, with actual selected PW36/RS32 controls, could show repeatable main-dominated tail exposure. That would support admitting one narrow geometry version for independent review. Large auxiliary contribution, lack of geometry-sensitive main improvement or a context gap explaining a tiny result would refute a wave-only recovery hypothesis. No outcome is predicted from wave counts alone, and no H200/score forecast follows.

This completes the requested CPU feasibility decision while retaining the valid failure and an actionable missing-evidence discriminator. No GPU, prepare, API, queue, original candidate or archive mutation occurred. Reproducer and frozen analytic package: `proj_rsi/candidates/qkv_geometry_followup/`.

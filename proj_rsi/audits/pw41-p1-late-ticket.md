# PW41 late P1 ticket: implemented, CPU proofs pass, GPU unrun

2026-10-27. Owned package `proj_rsi/candidates/pw41/` is frozen:

- Source `a77f785b06d3b9cdbf9fe3870505bd15f4e68c46807fa61dc6c3b9756dbe199f`.
- Manifest `0dc0e599569568a07d0f43a5ef276c33c62d69e665d538177eebf2be1efde051`,39files.
- Exact parent: CURRENT scored PW38 source29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e,
  joba3f0ebf25a754c01,3198.6175079817267us.

This implements the rank1 **conditional-budget** brief
`pw38.next.p1-late-ticket`. That ranking is not a measured gain or live scheduler
priority. No GPU/context/extension/prepare/timing/API/remote/queue/parent action
was performed. The source author reports CPU_PASS, not independent admission;
review18 owns the separate source verdict.

The expected full chain was quoted before implementation and corrected when the
brief's cumulative stage was read:45CPUmin implementation/proofs,15independent
review and15analysis =75CPUmin; provisional1020s narrow full GPU reservation,
plus3420s conditional cumulative follow-up =4440local GPU seconds if both stages
are taken. Official H200 cost remains null. These planning estimates are separate
from the common90CPUmin/300narrow/3420cumulative comparison caps. In particular,
1020 exceeds the hypothetical300s narrow cap and requires reranking/redesign
before execution; proof success is not free follow-up permission.

## Change and current-code evidence

A compile-time Vbit30 defers only the returned P1 ticket atomic from the W/X
prologue until thread0's first epoch/acquire wait. Four optional exact typed
variants append to the95old GVI instances, keeping every old selector/counter
index and every old GXP entry/order. The five affected scored/bake cases are
router8/16/32 and unpooler1/8. Existing bucket dispatch can also serve ragged
m5/7/9/17, so those correctness gates remain mandatory.

Zeroer roles/stores/fences/publication, flag acquire/trap, CTA rendezvous,
arithmetic/order/precision, S, geometry, reduction, fallback, preparation clock/
budget and selection logic remain unchanged. No atomic, graph node, fence or
acquire is removed; no counter reset or host epoch is added. Router retains its
FP32/BF16 hi+lo behavior and weight compression remains exact. No contender is
forced. Exact source inverse restores PW38; duplicate-anchor negatives reject.

The actual archived PW38 router8 JIT has ticketPC0x350 before its vector reads.
All95 offline parent GV streams independently match that current actual JIT
archive, whose SHA is132c222f4a675cabff91dc0188bff30c7d53c5fde5492f9006a69dde47f4470b.
Compiling PW41 gives exact95old PC/predicate/instruction/operand/branch streams
and all95old resource records. Four new functions have zero stack/spill records.
15 inherited old non-target functions already spill; their unchanged resources
are preserved rather than applying an invented blanket-zero requirement.

| Target | Last HMMA | Late ticket | First acquire | Registers old→new |
|---|---:|---:|---:|---:|
|router8|0x4a0|0x570|0x5a0|56→56|
|router16|0x1190|0x1250|0x1270|64→63|
|router32|0x1760|0x1830|0x1850|150→149|
|unpooler1/8|0x12d0|0x13e0|0x1410|69→69|

The two static integer atomics are still one returned ticket and one unused-result
zeroer flag. The latter can be in an out-of-line block. Every new stream retains
strong/acquire loads/cache invalidation, the CTA barrier before all four global
reduction sites, zeroer fence/publication and bounded trap. Register rounding
does not change at the8-register granularity. Instruction position/count and
register differences are not converted to microseconds or occupancy gains.

## Correctness/progress boundary

At completed ordered-launch boundaries even=E*S and odd=E. Any permutation of
exactly S ticket atomics returns E*S+k,0≤k<S, hence every target isE+1. Delaying
the ticket after math changes neither target nor S tickets/tile or one odd
publication, even when a z-zeroer published before its own late ticket. Extra
x-zeroer CTAs take no ticket. All active+dormant storage remains source-bound.

The publication DAG remains global zeros→writer fence→zeroer rendezvous→flag
publication→successful acquire→CTA barrier→first global reduction/output write.
The unchanged pre-acquire shared red[] partial stores are private CTA scratch;
the selected P1 branches never enable external p.ws stores before acquire.

The model checks13377exhaustive states,1536random complete schedules, repeated
and near-int32 epochs, nine full-storage cases with54 active/dormant/guard/alias/
offset corruptions,112ragged zero-tiling cases and seven protocol negatives.
Concurrent streams/graphs sharing P1 state, unknown progress, epoch mismatch and
signed wrap HOLD. It is a source-bound/fair-progress model, not a GPU scheduler,
universal CFG proof or floating reduction-order oracle. Runtime task/FP64/
one-hot/current-input/output/counter checks remain required.

Register/thread-only CAPACITY UPPER BOUNDS are4/4/1/3CTAs per SM, never guaranteed
occupancy. The five total grids are272/256/128/320/320CTAs. Router32 therefore
requires at least128SMs even at the one-CTA upper bound. Before BOTH parent and
candidate prepare, a future wrapper must reject unsupported device identity and
SMcount; actual exact-symbol gvocc/resources must prove full-grid eligibility
before ANY P1 launch, including old P1 types tested during prepare. Existing
production `_gxbake` queries occupancy but does not decline low-capacity P1
variants; there is no new universal production guard here.

## Research decision

The codegen placement discriminator passes, justifying independent source review
and a separately budgeted test-design decision. It does not establish performance:
the early atomic may already overlap useful work, whereas the late atomic may
delay acquire or create a more contended burst. Historical mid-stream prefetch
no-gain evidence is retained, not treated as positive support. Applied immutable
lessons are `counter-contract-not-name@1` and `composition-not-additivity@1`.

`lab-plan.json` provides a CPU-validated concrete five-case design, not an
executable GPU runner. It separates actual PW38 selected policy, original fixed
geometry and same-geometry late ticket under matched counter/workspace/output
roles. A correct reusable monotone-counter/dormant-alias/JIT/terminal wrapper
requires its own frozen review rather than an unreviewed ad-hoc benchmark.
Its source identity, actual capacity, allKbasis/task/FP64/ragged oracles, current
inputs/NaN/zero/outliers/probes, immutability, exact full-storage deltas, own-fixture
last timed outputs before replay/mutation, balanced complete-pipeline raw samples
and physical cleanup are mandatory before timing admission.

All six CPU tests and the read-only frozen verifier pass. Prototype/full compiler
inputs, cubins, logs and SASS are retained; no candidate compile failed or timing
result was discarded. The candidate remains GPU_UNRUN. A later gain, tie,
regression, unselected path, changed pick, budget loss or invalid setup must be
retained against the actual current PW38, with a new diagnosis/version or
cost-based park. There is no fixed tiny-gain kill bar or family-wide rejection.

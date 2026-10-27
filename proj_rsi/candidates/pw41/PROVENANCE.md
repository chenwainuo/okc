# PW41 v1: optional late P1 ticket over the current scored PW38

Source SHA256 `a77f785b06d3b9cdbf9fe3870505bd15f4e68c46807fa61dc6c3b9756dbe199f`.
Parent PW38 SHA `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`,
current scored job `a3f0ebf25a754c01`,3198.6175079817267us. No PW38, queue,
protocol, scored release or historical artifact was edited. No GPU/context,
extension load, prepare, benchmark, API or remote operation was performed.

## Cost before implementation, not a free follow-up

The initial narrow-only estimate was60CPUmin and1020s full GPU reservation.
After reading the complete ROI chain it was explicitly corrected to45CPUmin
implementation/proofs +15independent review +15analysis =75CPUmin, provisional
narrow1020GPU seconds plus conditional cumulative3420seconds =4440local GPU
seconds if both stages are taken. Official H200 cost remains unknown/null.
These are planning estimates, not measured costs or launch authorization.

The opportunity scan's common90CPUmin/300narrow/3420cumulative caps are comparison
scenarios, not this author's expected-cost forecast. The provisional narrow quote
exceeds300s and must be reranked or redesigned before runtime admission. A
source/CPU proof pass does not silently grant either hardware stage.

## Current evidence and one causal change

The brief is `pw38.next.p1-late-ticket`, conditional-budget rank1. The scored
source currently selects P1 for router8/16/32 and unpooler1/8. Actual archived
PW38 router8 JIT has returned-ticket ATOMG at0x350, ahead of its W/X reads.
Its entire95GV-function SASS archive is hash-bound in `current-JIT-proof.json`.
Static position does not identify elapsed-time cost. Historical mid-stream flag
prefetch's no-gain result remains in the P1 audit and is not positive support.

Only a new compile-time Vbit30 changes ticket placement. Old variants retain
their exact prologue ticket; the four new exact types omit it there and allocate
it inside thread0 immediately before the original first flag/acquire wait.
Both zeroer forms, zero stores, fences/rendezvous, odd flag atomic, acquire poll,
2^24 bounded trap, CTA barrier, shared partials, reductions, math/precision,
geometry, S and launch grid remain unchanged. No atomic is eliminated.

The static domain allows only these four selected kernel types:

| Scored/bake target | Old V | New V | S | Old→new registers |
|---|---:|---:|---:|---:|
|router8, FMT2/MT1/ST1/WK8/KU1|536870912|1610612736|16|56→56|
|router16, FMT2/MT2/ST1/WK8/KU1|134217728|1207959552|16|64→63|
|router32, FMT2/MT4/ST2/WK8/KU2|134217728|1207959552|8|150→149|
|unpooler1/8, FMT1/MT1/ST2/WK8/KU2|536870913|1610612737|4|69→69|

All use WR1/RT1/XS0. Router retains FP32 output/accumulation and exact existing
BF16 hi+lo weights; unpooler retains lossless BF16 packing. No input-derived
data/output is cached or reused. The high bit does not change the existing P1,
reduction, decode, ladder, cache or resource masks.

Four optional names append to GXP, giving exactly five new scored/bake cells.
Four unique GVI types append AFTER the original95, preserving every old index,
counter-instance identity and old catalog prefix. The original `_gxbake`
validation, old alternatives/order,0.99 margin, import-clock240s budget,
compile strategy and fallback remain exact. No variant is forced. Those five
cells are NOT an exact-M-only dispatch claim: unchanged `_bk`/kernel_fn can
serve ragged m5/7/9/17 through the same buckets. Their correctness remains a
required runtime gate, and CPU zero-tiling proof covers every m1..8*MT.

`build_patch.py` records four strict edits. Applying them reaches exact PW41;
the inverse restores the entire PW38 file. Duplicate/missing anchors reject.
All existing named AST entries except GVCU remain exact; only new `_P1LT` and
P1LATE definitions plus the explicit append operations are added. All original
functions—including prepare, kernel_fn, `_gvcall`, `_gxbake`, `_gvext` and the
RQ/router kernels—are unchanged. No PW39/PW40 selection/W-ring feature is merged.

## Applied learning

Read and queried the registry before implementation; retained query and record
bindings are `learning-query.json` and `learning-proof.json`.

- `counter-contract-not-name@1`: classify actual selected physical P1 storage,
  preserve active monotone deltas and nonzero dormant baselines, include aliases
  and offsets; do not reset counters or identify an old unnamed failure by guess.
- `composition-not-additivity@1`: isolate the placement change and compare
  actual current PW38 separately from fixed geometry. Source identity and parent
  progress cannot establish additive, robust or cross-hardware performance.

Negative and invalid observations remain linked, and no historical microsecond
delta is transferred into a PW41 forecast. Attainable benefit stays null.

## Epoch, publication, full-storage and progress proof

At a completed ordered launch boundary, each tile has even=E*S and odd=E. Any
arbitrary ordering of its S math-CTA ticket atomics returns E*S+k,0≤k<S, so every
target is floor(ticket/S)+1=E+1. This remains true if ticket issue is delayed
until after math, including a z-zeroer that already published E+1. Extra x-zeroer
CTAs take no ticket. Every completed launch still increments each active even
word byS and odd word by1. No host replay epoch or reset is added.

The unchanged order is all global zero stores → writer fences → zeroer
rendezvous → odd publication → successful acquire → CTA barrier → first GLOBAL
reduction/output write. Private CTA `red[]` shared partial stores precede acquire
in the original source too; those are not external p.ws writes. The selected P1
constant branches never enable the legacy external workspace store path.

`protocol_proof.py` checks13377exhaustive states for old/new z/x models at S2..4,
1536random complete schedules at S4/8/16, repeated epochs including near-int32
limits, exact per-launch deltas and112ragged zero-tiling cases. The count in its
JSON is authoritative if this prose count is corrected later. Seven negative
controls reject removed acquire, publication before zero release, epoch
misalignment, odd mismatch, int32 wrap, concurrent shared-state streams and
unknown residency/progress. Nine full-storage cases reject54 active/dormant/
guard/alias/offset corruptions. No validity assumption is replaced by a zero.

This is an ordered-stream/source-bound model, not a GPU scheduler or arbitrary
compiled-CFG proof. Concurrent launches/graphs/streams sharing prepared P1
counter/output state are unsupported and HOLD. Signed int32 wrap is excluded.
Fair finite zeroer/engine progress and the retained bounded spin trap require
actual runtime validation. Moving the atomic can alter cross-CTA arrival order;
the source retains the same per-CTA partial arithmetic, but fresh task/FP64/
one-hot gates must validate floating reduction behavior.

## Offline codegen and the actual current JIT

All95 old GV functions were compiled from the unchanged parent device body and
the PW41 device body; all PC/predicate/instruction/operand/branch streams AND
all resource records match. Independently, all95 old streams equal the retained
actual current PW38 JIT archive, not just a new local control. Source-only
identity is therefore backed by current machine-code controls.

The four new functions compile on CUDA12.8 sm90a/O3/lineinfo with no stack/spills.
Every returned-ticket atomic remains after the last HMMA and before the first
strong/acquire load; acquire-loop loads/cache invalidation, CTA barrier, four
output reduction sites, zeroer fence/publication and bounded trap remain present.

| Target | Last HMMA | Late ticket | First acquire | CTA barrier before reductions |
|---|---:|---:|---:|---:|
|router8|0x4a0|0x570|0x5a0|0x6c0|
|router16|0x1190|0x1250|0x1270|0x13a0|
|router32|0x1760|0x1830|0x1850|0x1980|
|unpooler1/8|0x12d0|0x13e0|0x1410|0x1530|

The unused-result integer atomic is the odd zeroer flag; it may be in an
out-of-line block and is distinguished from the returned-ticket atomic. PC
order/source-qualified gates are not a general divergent-CFG proof. New
register allocation and scheduling can differ; no instruction-count-to-time
conversion is made. Reduced register counts do not cross an8-register rounding
threshold here.15 inherited old non-target functions already have nonzero
stack/spill records; those records remain exact. They are not new PW41 spills,
and requiring all dormant/unselected controls to have zero spills would invent
an invalid parent contract.

Static register/thread CAPACITY UPPER BOUNDS are4/4/1/3blocks per SM, not actual
or guaranteed occupancy. The rounded register pools are14336/16384/38912/18432
per256-thread CTA. Shared bytes are4608/9216/18432/4608. The five grids have
272/256/128/320/320CTAs. Router32 consequently needs at least128SMs even at its
one-CTA upper bound. **Before BOTH parent and candidate prepare, a future GPU
wrapper must reject unsupported device identity/SMcount; before any P1 launch it
must query actual exact-symbol occupancy and require the entire grid to fit.**
A post-selection check is too late because preparation launches these variants.
The production helper remains unchanged and supplies no new universal device
gate. Do not run this candidate directly on an unverified smaller device.

Prototype and full codegen folders retain compiler inputs/cubins/logs/SASS/raw
results. No candidate compile failed or performance result was discarded.
The prototype's initial collector groups the two integer atomic PCs; the final
collector additionally distinguishes returned ticket from flag and compares
the full95control resource records. CUDA input generation and candidate source
are unchanged between those collectors. Current full-codegen proof is binding.

## Runtime test boundary

The bounded CPU work includes a concrete validated five-case discriminator in
`lab-plan.json`, generated by `lab_plan.py`. **It is not an executable GPU
runner or runtime admission.** A counter-aware, exact-JIT, matched-allocation
wrapper is intentionally separate rather than shipping an unreviewed ad-hoc
benchmark under this source GO. Root must review/rerank its full costs and
freeze/review that wrapper before execution.

The proposed narrow comparison separates actual full prepared PW38 policy,
fixed original geometry and same-geometry late ticket with matched counter/
workspace/output roles. Single ordered stream, full active+dormant storage
contracts, allKbasis/FP64/task/ragged oracles, actual terminal own-fixture outputs
before any replay/mutation, fresh original inputs, NaN/zero/outlier/probe
mutations, W/pack/CSR immutability, balanced complete-pipeline F1/paired calls,
every signed raw result and physical cleanup are mandatory. No forced selection
or counter reset is allowed. A fixed-arm win is not a cumulative PW41 policy win.

Falsifiers include delayed final acquire, increased late ticket contention,
counter/epoch/resource failure, disappearance under matched pointer roles,
remaining unselected, no actual-policy gain, changed other picks or budget loss.
Retain tiny gains, ties, losses and failed setups with their exact source. A new
version/diagnostic or cost-based park follows; no fixed tiny-gain cutoff or
blanket P1 family rejection is introduced.

Read-only validation after freeze:

```bash
python3 -m proj_rsi.candidates.pw41.verify_frozen
python3 -m unittest proj_rsi.candidates.pw41.test_cpu
```

Do not run build_patch/codegen to verify a frozen package: they intentionally
write fresh candidate/compiler artifacts. No official submission, selected
serving policy change, GPU correctness or PW41 speed result is claimed here.

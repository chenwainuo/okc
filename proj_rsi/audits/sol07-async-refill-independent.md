# Native-X async refill independent review

**Kernel GO for a bounded root-only research trial; lab GO pending its new
freeze and separate review.** No GPU/context/extension/prepare/benchmark,
remote import, frozen-file edit or ledger edit performed. Parent's measured
31.856us versus local incumbent23.968us motivates iteration but predicts
nothing for this new source. ITERATION.md applies; retain failures and repair
specific versions, not a family prohibition.

Exact kernel manifest SHA256
`92fbe73d6ed0bbe5d9b6165cfd0e9956494e1e97b48f6f38221739df584d60d2`,
source
`45fbc8b600c6054c43e725b6b5f33b9194b61f5b602c583a7b9bb825ab325170`.
Every listed file hash independently verified, including both history variants.
`all_proofs()` passes without rewriting the frozen proof JSON: exact source
restoration to parent, unchanged complete native-A layout, full DAG and local
physical A4 lifetime. Explicit-warp attempt history is retained, with identical
cubin despite the added source barrier; no claim it emitted a new instruction.

## Non-numerical shared-read / asynchronous-write ordering

Required chain for EACH of8 math warps at each stage:
all32 lane LDSM generic shared reads → per-lane proxy fence → fullmask
source __syncwarp → lane0 Xempty release arrival → elected producer acquire
wait observing ALL8 arrivals → subsequent same-slot TMA write. The source
explicitly synchronizes; lane0 is not assumed to speak for unsynchronized
lanes. The default __syncwarp mask names all32 lanes. Every math lane executes
prefetch in the same stage order; no math early exit or lane-varying branch
skips a handoff. Initial successful Xfull wait orders prior TMA writes before
LDSM, so both reuse directions are covered.

CUDA12.8 documentation gives __syncwarp participant rendezvous and memory
ordering. PTX8.7 defines default mbarrier.arrive as release and successful
try_wait as acquire; prior participating accesses complete before waiter later
accesses. PTX proxy fence orders prior generic accesses against subsequent
async-proxy accesses. Reads are memory accesses covered by this ordering;
there is no dependence on numerical output coincidence or a CPU schedule to
establish it. All inline assembly memory operations have memory clobbers.

Primary sources:
[CUDA12.8 syncwarp](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions),
[PTX8.7 mbarrier wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait),
[PTX8.7 proxy fence](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).

Actual local SASS has eight handoff sites: four LDSM, MEMBAR.ALL.CTA,
FENCE.VIEW.ASYNC.S, then predicated lane0 Xempty arrival, with no intervening
divergent/reconvergence branch. A WARPSYNC exists before initial LDSM after a
wait branch; **the post-fence source syncwarp is elided**. Do not report an
emitted post-fence WARPSYNC or assert MEMBAR alone synchronizes other lanes.
This GO explicitly relies on NVCC preserving the documented source intrinsic
semantics on this converged path, plus exact instruction-stream binding at
runtime. That is a compiler-conformance assumption, not an undocumented SASS
ISA proof. Source contains the needed synchronization and no compiler defect
has been demonstrated. If that assumption is unacceptable, HOLD this exact
version and redesign the handoff in a new retained version; merely requiring
another redundant source bar.warp.sync produced the same binary in history.

Local physical gate positive passes. Scoped mutations of a math LDSM to LDS,
MEMBAR to NOP and proxy fence to NOP each reject. No new raw GPU all65536
BF16 mover proof is claimed: that remains UNPROBED, with prior root acceptance
of source/ISA map plus mandatory numerical runtime gates for bounded research.

## Lifetimes, stages and progress

Two Xfull barriers count1 completion each; two new Xempty barriers count8
warp releases each. At stage s, slot=s%2, full/empty parity=floor(s/2)%2.
An owner cannot issue s+2 until all8 warps have read/released s. Its next phase
cannot complete until that new TMA and new reads, so no parity skip/ABA or
unread overwrite occurs. Exactly64 X issues (2 initial+62 refills),512 warp
arrivals and62 acquire-before-reuse refills. Tail62/63 has release but no
future X issue/wait; it cannot strand producer progress.

Initial0/1 copies are issued by tid256 before CTA init barrier. Math C0/C1
releases enable parity producers0/1 to issue X2/X3 before decoding W0/W1.
C2/C3 then creates all four A banks before first MMA; no WGMMA retirement is
required for those startup releases. Steady C(s>=4) follows retirement at
math(s-3). Existing wait1 retires previous group and preserves current A.
Producer parity owner remains sequential through existing B publication.
B8 reuse depends on math(s-7), since that retires group(s-8); W6 refill follows
B publication(s-6). This corrected dependency is essential.

Frozen DAG578nodes/1495edges topologically passes. Independent per-warp
expansion [sol07_async_model.py](sol07_async_model.py) includes all512 individual
release arrivals and passes1474nodes/4269edges. Injecting a startup read's
B-publication dependency creates a detected cycle; removing one warp's release
is rejected. Thus the progress claim is not based only on counting events or
ignoring producer head-of-line dependencies.

Shared Xempty metadata [136208,136224) is disjoint from Xfull/Xslabs/B/W.
137248B allocation includes1024B alignment slack; SW128 slabs stay1024B aligned.
Output reuses old B only after wait0 and math barrier; X slots/metadata never
alias output. Producers reach final return without participating in math-only
bar1/2; producer bars3/4 retain128 participants each; initialization bar0
contains512. Bar5 removed, leaving5 named barrier IDs. No producer-return/math
barrier cycle is introduced.

Four16-word A banks remain disjoint R56..119; local physical checker proves
no pending-group write across2 loop iterations and all8 outlined wait paths.
Source restoration proves W/B decode/publication/A4 math/output otherwise
unchanged. Local122registers/512threads rounds to65536 register allocation,
0spill/0stack; actual JIT resources and queried occupancy remain mandatory.
No occupancy or speed improvement inferred from removal of a named barrier.

## Required lab admission

New driver must compare actual local incumbent, parent native-X and async new
complete one-node graphs, bind each exact frozen source and actual loaded JIT
before candidate math, and enforce122registers/0local, expected dynamic shared
137232 versus137248, max threads and actual opt-in occupancy. Require all4096
basis columns, FP64/task normal/outlier, all-row/fresh/zero/NaN output graph
mutations, immutable input/W/pack/escape and reset counters. Every terminal
F1/paired actual output must be read against its own policy's last recorded
fixture before replay/mutation/profile; keep raw samples, picks and cleanup.
No XP exists in either native arm and no XP pass may be invented.

Removing math bar5 simultaneously couples X refill to parity W producer
lookahead and adds512 mbarrier arrivals; this is not an isolated bar5 latency
experiment. Elected producer acquire may delay current W decode/publication.
Changes can improve, regress or move exposed dependencies. Keep whole-pipeline
ordinary timings distinct from cold per-node diagnostics and official H200
score. Lab author owns the new freeze; kernel GO does not yet approve a driver.

## New frozen lab review: GO

Exact driver `tools/native_x_async_lab.py` SHA256
`48feda58e6b6fdf70d8481d0d2e5fb2dbb23253e74bc67ce4f98ad250af93689`;
trial manifest
`c2618e313986d9e19c809f79077a1623af8f985f2f6164b8281a1b34af17bf9d`;
support
`fac73e0e6dff6567333115ed0da20f964f7d4a718e5bd0ff609f333f082d5532`.
All trial and three kernel manifests/file hashes pass. Generated source equals
frozen extension text for each arm. Exact local canonical gate passes new,
native and DS; a changed HGMMA opcode rejects separately in all3. Version-
specific liveness/proofs import binds the correct checker, avoiding parent/
async cross-version liveness reuse. All9 extracted terminal-gate corruptions
reject against this actual driver. Old native e96/a73 source is unchanged.

Diff against previously admitted driver is confined to new async prototype/
support identity, DP→native-parent comparator, XP only for DS, correct
native/new resource/node/symbol checks, and explicit module binary hash.
Selected actual local pw36 rq guard and rc0 pass-through remain; no forced pick
or silent serving fallback admission. Each native variant uses current
original X and empty ignored XP. DS gets fresh1MiB XP and2-node pack/main,
parent/new each1-node main; actual profile verifies those symbols/counts after
strict terminal output gates. Actual JIT source/instruction/liveness gates and
137232/137248B shared,122register/0local/512thread/occupancy checks precede
candidate math. No new source operation changes the previous immediate output
read ordering, mutation/oracle coverage, fixture recording or immutability/
reset checks. Additional DS control preserves causal comparison to globalXP.

**GO for one root-only normal-cap trial, inner240s/outer300s, of exactly this
lab and kernel freeze.** This supersedes lab-pending status above. Every
runtime numerical, actual graph, source/JIT, memory/counter and cleanup gate
remains mandatory; partial/timeout/failed gate gives no performance admission.
Source-syncwarp compiler-conformance assumption and UNPROBED native hardware
raw-bit mover limits remain explicit. Gains versus parent are lineage progress;
only an independently valid same-run incumbent win supports integration, and
official H200 validation is still required for official performance claims.

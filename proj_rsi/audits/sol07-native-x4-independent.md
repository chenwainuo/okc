# Native-X4 independent review

**Kernel GO for a bounded root-only research trial; new lab pending its freeze
and separate admission.** No GPU context, extension load, prepare, benchmark,
remote operation, frozen-file edit, commit or global ledger edit performed.

Exact manifest SHA256
`d47ed1663eac87f5bcc05b5f17df8fa62e7c9e91780a13d02ce9b95ac0f05c8a`,
extension source
`e8447cbf0c113618462f5cbddc61d7f1e438ada2efa43130531f96c9fde768d3`.
Every listed hash independently verified. Frozen all_proofs() passes without
rewriting its output. Kernel body diff contains only name, dynamic shared
size, Xfull base, slot mask, phase divisor, refill distance and initial loop
bounds. Finite inverse restores parent025f/ffd body byte-for-byte; W6/B8,
A4/retirement, codec, output and mathbar5/elected math issuer are unchanged.
Host current-pointer descriptor stays the same and there is no XP pack/cache.

Ring depth2→4 has four16384B SW128 slabs [103424,168960), metadata
[168960,168992), and allocation170016B including1024B alignment slack. Slabs
remain1024B aligned, row address providers16B aligned, no W/B/output alias.
Extra32784B is32768B data plus16B metadata, not merely a ring-depth change
without resource cost. Local122registers/0stack/spills,512threads and6 named
barriers retain65536 rounded register allocation. One-CTA resource feasibility
is not runtime occupancy proof; actual opt-in shared/occupancy queries required.

Independent CPU enumeration covers every stage/row/column's physical shared
halfword bounds, exact global(row,K) coordinate and slot phase.524288 unique
BF16 coordinates pass. Initial copies0..3 create four independent full phases.
Consumption0..3 launches4..7 before first MMA while filling only the same four
register-A banks. Steady register refill(s+3) follows wait1 retirement(s-1),
reads shared X(s+3), then issues shared X(s+7) when in range. Slot=s%4 and
phase=floor(s/4)%2; all64 issues/64 reads/60 reuses are matched, tail60..63
issues no future copy and leaves all slots consumed. A phase cannot advance
again before current-stage reader completion; no stale phase/overwrite occurs.

Each of8 math warps waits Xfull, performs four LDSM, then per-reader proxy
fence and256-thread mathbar5. Elected tid0 only overwrites AFTER all readers
rendezvous. Cross-proxy shared-read→TMA-write ordering is inherited unchanged
from independently admitted parent; the barrier makes every reader's handoff
participate. X4 does not introduce the async candidate's elided post-fence
warp-sync mechanism. Initial4 TMA copies are issued before CTA barrier. Math
issuer does not depend on producer W decode, so extra ring slots add no new
producer/publication cycle. W6/B8 edges are unchanged; B(s) reuse waits
math(s-7), which retires group(s-8). Final wait0/bar1 precedes output B reuse;
producer return cannot strand mathbar1/2/5, and producerbar3/4 participation
remains128 each. Same primary PTX proxy/barrier/mover rules as parent apply:
[PTX8.7](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).

Local physical gate verifies role branch0xc50→0x2ef0, four initial TMA sites,
math32LDSM/eightbar5/eightmathTMA/zeroLDG. Each LDSM quartet precedes
MEMBAR→FENCE→bar5. Four16-word A banksR56..119 remain disjoint; checker passes
pending-A retirement over2 loop iterations and all8 outlined wait paths.
Scoped mutations of LDSM→LDS, proxyFENCE→NOP and bar5 participation256→128
all reject. These are bounded CPU/fixed-CFG checks, not an actual GPU oracle
or a general SASS address interpreter; actual JIT canonical stream and version
specific physical gates must rerun before first candidate math.

Raw native GPU all65536 BF16 mover remains UNPROBED; no XP gate is claimed.
Mandatory trial gates remain all4096basis/FP64/task/freshzerooutlier/all-row/
NaN actual graph mutations, immutable current X/W/pack/escape, counter reset,
actual terminal timed outputs against each policy's last fixture before any
replay/mutation, graph symbol/node identity and root cleanup. Compare actual
local incumbent, native parent and X4 complete single-main graphs. Additional
frozen controls are acceptable. Do not attribute depth gain independently of
larger shared footprint/changed addressing; no HBM reduction or speed forecast.
Source-qualified failure gets retained narrow diagnosis/new version under
ITERATION.md. Only official H200 validation supports an official score claim.

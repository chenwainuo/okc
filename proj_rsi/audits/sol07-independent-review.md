# Worker07 independent review

CPU/read-only review only; no CUDA context, extension load, prepare, benchmark,
NCU, parent/driver edit, commit, or global ledger edit. ITERATION.md governs:
failures attach to concrete versions/attempts and require narrow repair.

## Native-X kernel: GO for bounded research validity testing

Reviewed manifest SHA256 `025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc`,
extension source `ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377`.
Every manifest file hash independently verified. `proofs.all_proofs()` passes
without rewriting any frozen artifact. Independent enumeration of every stage,
warp, chunk, lane, matrix and packed BF16 half verifies 524288 unique exact
(row,K) coordinates. x4 providers map matrices in TL/BL/TR/BR order; lane g*4+t
receives native A row g (plus8 for BL/BR), K2t (plus8 for TR/BR). SW128 logical
sector XOR row[2:0] and 1024B slab alignment agree with descriptor layout.
CPU coordinate faults (row+8, column+8) are rejected.

Descriptor is reconstructed from current original contiguous BF16 x pointer
on each call: dims(4096,128), stride8192B, box(64,128), element strides1,
SW128. It never dereferences XP, launches a packer, or uses cached X contents.
Source shape check requires M128/N8192/K4096/KC64/KS1. Each CTA consumes the
same full128-row X tile and its own64 output channels.

Two X slots begin with stages0/1; every prefetch waits parity floor(s/2)%2,
performs four LDSM in each of8 math warps, executes per-reader proxy fence,
then bar5 with256 participants before tid0 issues stage s+2 into that slot.
Startup0..3 therefore safely creates four register-A banks from two shared
slots. Refill s+3 follows retirement of group s-1; tail62/63 issues no further
copy. All64 issues/reads,62 reuses and parity transitions are covered. X has
no W-producer dependency cycle. Producer-only bar3/4 each128 participants,
math-only bar1/2/5 each256, CTA initialization bar0 before role divergence:
6 named IDs. Producer return cannot strand a math barrier. Final wait0 and
bar1 precede B-to-output scratch reuse; X/metadata never aliases B/output.

Generic shared reads versus subsequent async TMA writes are explicitly ordered,
not assumed from a CPU schedule. PTX8.7 section9.7.13.4 documents bidirectional
fence.proxy.async ordering of prior/subsequent accesses across generic/async
proxies; all readers fence before rendezvous and elected writer issues after.
TMA mbarrier completion orders async writes before readers. Primary source:
https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
LDSM mapping and WGMMA fragment sections cited in worker08 audit apply.

Local physical checker passes all four disjoint A banks, two loop iterations,
8 outlined wait paths, and no pending-A clobber. Physical X blocks contain32
LDSM,8 bar5 sites and8 math X TMA sites, with0 math LDG. Mutating math bar5
participation256→128, a math LDSM to LDS, or a math proxy fence to NOP is
rejected. An initial negative replaced a nonmath fence and correctly did not
trip the math-only gate; scoped math mutation then rejects. This is a bounded
fixed-CFG proof, not a general SASS dataflow interpreter or GPU oracle.

No resource/latency inference: local122registers/512threads rounds to65536
registers,137232B shared and0 spill; actual JIT stream/resource/occupancy gates
must pass before first candidate math. Native hardware all65536 BF16 mover
roundtrip is UNPROBED. Root explicitly accepts that limit for this bounded
research run with exact CPU map, ISA mover semantics and mandatory GPU full4096
basis/FP64/task/mutation/timed-output gates; no XP or GPU raw-bit pass claimed.

## Native-X lab: HOLD at first freeze pending narrow incumbent-state gate repair

First frozen driver SHA256 `4df7fc9bd2ec87405ba0a810cba554b4ac68a51cef5e672fcf21e71e1bb6854e`,
support `113a6a08fdcbddeefb9fac23cf5b631a68d36366431e531f648693a372b9e4f4`,
manifest `e4f33b6c8cf27dbece5a6493a6ac326b7a08d59ef3ece58bad1187d10c24733e`.
All lab/three-kernel manifest hashes pass. Canonical instruction gates pass
new/DP/DS; a changed HGMMA opcode in each is rejected. Extracted terminal gate
CPU negatives reject output,XP,currentX,weight,packed,escape,candidate-counter,
incumbent-counter and wrong policy fixture. Actual JIT must equal canonical
PC/predicate/instruction streams, then pass liveness/store gates before math.

The four arms are freshly prepared pw36 development kernel_fn, DP, DS and
native-X. Root confirms pw36 is intentional local development comparator;
it is not official rs32 evidence. DP/DS whole graph requires2 nodes and1
fresh pack, native-X1 node/no pack. Full4096 basis/FP64 and task cases are
present. F1 and paired terminal outputs are read against each policy's own
recorded last fixture before any replay/mutation/profile. No native-X XP gate
is invented. Earlier in-progress graph-check arity mismatch was corrected
before this freeze.

Concrete remaining gap: incumbent immutable/reset collection only includes
selected rq lay/esc/cnt. A fresh actual kernel_fn may select r6/Lt/other state,
whose packed tensors/counters then lack those checks. Minimal repair: cover
actual prepared state packed/escape tensors and counters excluding mutable
scratch, or explicitly constrain/assert selected admitted rq serving plan.
Worker09 informed; re-freeze/hash and re-review required before lab GO.

## Smallm increment lab: HOLD at first freeze; source recurrence accepted

Verified every file in trial manifest SHA256
`040a33edafebb4dd2c662332613b542a9bae77f64255d3b29a61572e37379359`.
Driver SHA256 `196956deb52a1378361493a28f0eec3f843cabcdea265a5e0d39d3a16cbb4786`;
candidate `0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737`.
CPU proof passes16512 recurrence events and unchanged LDS/barrier/math counts;
physical40*128+232*256=64512=168*384,0spill required at actual JIT. Source
restoration/event-order proof in sol04 audit supports unchanged lifetime.

Concrete executable fault: selected attrs rqrun call puts loop `info` in
walias position and final info=0, launching math instead of querying attributes.
Affected picks then produce [0,0,0] and fail expected attributes. Minimal repair:
pass KS,tab,0,rq_e,rq_sink,info, then CPU AST argument-route negative proof,
new freeze preserving old attempt identity. Root informed and author awakened.

Prebuilding original RQ32 list and resetting module import clock _T0 is acceptable
as disclosed local warm-cache research. _rqext retains complete-source slot
identity and collision check; original background build hits same source slot.
This changes compilation/budget overlap and therefore may change actual picks;
it cannot substantiate cold prepare budget or official production policy.
Original full prepare/bake/margins remain; retained r6 pick gets no recurrence
credit. Default qkv32 bounded1140s inner/1200s outer is acceptable for3 full
prepares. qkv16,pooler32,unpooler32,o32 still require separate GPU coverage
before global promotion. Full basis, oracle/mutation/immutability/reset and
immediate final actual outputs are required regardless of chosen plan.

Group8/NCU driver is being independently reviewed by root; no worker07 GO is
claimed. All GPU validation and cleanup remain root-only. No speed promise,
family closure, production integration or official submission authorized here.

## Final re-review: both repaired labs GO within stated bounds

Native-X lab v2 driver SHA256
`e96abbcc8da44051f491ae086d5d94b4589d9c082317ceebddb2746eb61ed6cf`,
manifest `a73cff33a3d87ca925f9bd025c70ad1bfeeac1c94799e4fcd23de45a9e8f4821`.
Every new manifest hash verified. First freeze remains in history/lab-v1-e4f33b6c.
The lab now requires selected serving rq plan, matching N8192, and no CPU state;
its pass-through wrapper calls the original actual _rqcall and requires rc0
and exact selected-plan identity. Source kernel_fn reaches that rq branch before
other fallback paths, so this closes packed/escape/counter scope without forcing
selection or adding graph GPU work. Independent extracted-wrapper CPU negatives
reject nonzero launch status and changed serving-plan identity. Terminal gate
nine-negative proof rerun passes against new driver bytes.

**GO: one bounded root-only native-X research trial of exactly v2**, subject to
all declared actual JIT/resources/full4096basis/FP64/task/mutation/real-timed-
outputs/node-count/immutability/reset/cleanup gates. Pw36 remains explicitly
local development comparator; native hardware raw-bit mover remains UNPROBED.
No official rs32 or H200 admission follows from local H100 evidence.

Smallm lab v2 driver SHA256
`5f1534620d3e62d017fbd8a167b7763aed0d8656ac938bcca8411572c778a308`,
manifest `7c5e90b6b6854f89a58affac6c6bdf8fe14b8578e7ca9ae2784764d6fc7e6235`.
Every manifest hash independently verified. Selected-attribute helper now routes
walias0 and final info1/2/3 correctly. CPU contract extracts actual C++ argument
signature/info branch; actual Python helper on fake extension makes3 queries and
0 launches, and rejects the old swapped routing. CPU full recurrence/counts
rerun passes. Old freeze retained under measurement-history/v1-info-routing-hold.

**GO: qkv32-only smallm v2 root research trial,1140s inner/1200s outer**,
with full original freshly prepared pw36/rs32/new policies, disclosed warm build
clock reset, no forced picks, and all frozen mandatory validity/cleanup gates.
Other affected shapes still require their own GPU coverage before promotion.
These GO decisions supersede first-freeze HOLDs only for the exact repaired
identities above; they do not relax any runtime gate or erase failure history.

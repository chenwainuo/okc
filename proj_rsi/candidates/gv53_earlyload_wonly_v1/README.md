# GV53 M8 W-only earlier issue: three-arm falsifier

New source e3931df0, exact501d parent. Frozen W+X source81fd and selected package
17d remain unchanged. The native clone snapshots only current W, refills next W
before decode, reads current X from the original ST1 ring, then refills next X
at the original source location before repair. The X snapshot is removed.
Codec, addresses/cache policies, escape cursor/repair, HMMA/FP32/reduction/output
source order and final-unit no-refill behavior remain.256arbitrary raw-word
traces verify each of8units consumes its exact W/X and reads no unit8.

CPU compile confirms a distinct realized schedule:

| Arm | Slots/non-NOP | Next W LDG PCs | Next X LDG PCs | First PRMT |
|---|---:|---|---|---|
| parent |360/348|0770/0830/0870|08a0/08b0|0630|
| W+X |384/374|06c0/0750/0790|0660/0690|07a0|
| W-only |392/378|0620/0650/0690|07b0/0860|0760|

All three use64registers, zero stack/spills, unchanged LDG11/SHFL2/P2R1/HMMA4
static counts and17408dynamic bytes. Parent and W+X PC/predicate/128bit streams
remain exact to the prior frozen pilot. Removing a source snapshot did **not**
make code cheaper: W-only has8more slots/4more non-NOP than W+X. The compiler
partly hoists next X within decode; it does not preserve the parent's exact X
schedule. Thus this compares realized implementations, not pure fixed-X causal
isolation. Occupancy and numerical/performance results remain GPU-unmeasured.

The bounded thin trial compares parent/W+X/W-only in one fixed M8 cell. It uses
one actual task W(seed5344) and the same stored X fixtures for every arm.44F1
observations share an11fixture bank across parent-before/W+X/W-only/parent-after.
Paired work uses8unique X fixtures and all6triple orders each:48triples,
144calls, not48independent inputs.15warmups give203total rows/188clean calls.
Seven own-terminal sites run immediately after each F1block and each final
triple arm, before another replay or mutation. All signed three-way pair points,
means and F1endpoints remain separate. Reused events/host timestamps capture
enqueue/sync/journal spans. Eight exact basis columns per arm, task+FP64/currentX,
full immutable/scratch/counter backing and15retained mutation checks remain.

Reuse the validated thin ABI/strong graph owners, explicit current stream,
three-arm pointer-only wrapper and exact admitted collected cleanup bridge2638.
Runtime must match all three target native streams to the CPU pilot; query
actual attrs/CTA capacity before math, and retire owners only after graph
reset/drain. No Torch C++ rebuild or full prepare/policy/profiler occurs.
60sinner/120souter is the complete reservation cap. Nine CPUadapter tests and
focused protocol checks pass. Source/native compilation was one small CPU TU;
no GPU/API/SSH/queue/git action was taken by the author.

W-only retaining a positive contrast with less actual copy cost would support a
narrower mover; W-only losing while W+X wins would refute W-only sufficiency in
this context, without uniquely attributing X latency because generated schedules
also differ. A static slot increase is counterevidence, not an automatic rejection.
Preserve tiny gains, ties, losses, order effects and F1conflicts without a minimum
size/significance gate. Current W+X local+.192us paired/+ .160us endpoint-average
and attenuated same-body package+.032us paired/+ .176us F1 are prior scoped
observations, not forecasts or independent weight replication. No production
selection or official gain follows from this fixed experiment.

The proposal/core is frozen on delivery. Later changes require a new version or
named supplement. Independent focused review creates job-root-ready.json rather
than overwriting the original pending proposal.

Loop: profile → measure → hypothesis → optimize → test → validate → stack →
submit measured gains → learn → reprioritize → repeat; idle ≤10s.
[Contract](../../ITERATION.md).

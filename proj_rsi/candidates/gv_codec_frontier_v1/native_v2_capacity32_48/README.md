# Direct C11 two-cell discriminator, CPU complete / GPU unrun

`driver.py` derives from the reviewed `gv53_maskgate_trial_v1/driver.py` protocol:
same F1/24 alternating-pair schedule, own final timed-output/current-input oracle,
guarded input/output/counter/scratch arenas, full immutable tensor checks,
normal/outlier/zero/basis checks, captured-graph fresh-input mutations and drain /
graph reset. `derivation.json` pins that parent; `focused-driver.diff` records
changes. This introduces no broad prepare/tuning/profiling campaign.

Exactly two published selected cells: pooler M1 tuple
`1,1,1,2,16,8,1,1,0,3`; unpooler M16 tuple
`1,2,2,1,4,16,1,1,2,1`. CPU checks bind both tuples to the frozen selected report
and derive geometry by tuple slicing. Native controls remain byte-for-byte
source copies and their PC/predicate/encoding streams exactly match incumbent
`candidate-full.sass`. The live extension repeats that native control gate
before launching. The candidate inverse proof restricts changes to W layout /
decode / repair transport and disables the old escape walk; HMMA, input loading,
accumulation/reduction/output source restores exactly to current predxor.

Actual CUDA statistics support explicit unit capacities32/48. Physical pack
rejects any nonfinite, oversized unit,32-bit offset overflow or whole-format
no-saving condition. The driver checks exact actual fixture SHA, every decoded
weight bit, old C12 reconstruction, complete measured-format bytes and runtime
resources before timing. Admission failure never activates this candidate.

Each warp owns ST slots of descriptor32B plus record64B(pooler) /96B(unpooler).
Record copies and descriptor copies share one cp.async group per prefetch. The
same slot is refilled only after all four repair steps finish; the final ST2
iteration explicitly waits for zero groups, avoiding an under-wait at the tail.
CPU ownership simulation includes both tails. Added dynamic shared is3072B /
1024B, giving20480B /6144B total dynamic. Offline candidates use64 /108 regs
versus64 /110 controls, no stack/local/spills. Actual occupancy and user-static
metadata must pass independently; offline dumped SHARED1024 is not asserted as
Runtime user-static1024.

Counterevidence remains explicit: format stream reduction is only4.9787% /
2.9782%, and about139/176 repair lane-instructions/unit precede other decoder /
mover overhead. The native mover additionally requests two u32 unit-boundary
metadata loads before its descriptor copies (8B/unit,262144B/matrix), which may
reuse cached descriptor sectors; no physical HBM reuse is assumed. These extra
logical requests must not be hidden in the unique-format byte ledger. The trial
records signed timings; neither instruction counts nor bytes predict a gain.

The C11 physical pack/ring check and native inverse/control proof pass. Driver
CPU tests preserve four terminal negative controls, both basis domains and
selected-template/actual-stat geometry checks. Native device code compiled
offline; the Torch host binding is staged for the normal CUDA runner and has
not been loaded or executed here. This is the outstanding runtime validation,
not an assertion that the trial is already correct or fast.

Proposed bound300s inner /360s outer, profiling false, one normal Torch runner.
Output includes exact emitted CUDA source, loaded SO/SASS, raw attributes, all
signed samples, validity evidence and post-scope allocator cleanup. All run
aliases must be released after graph drain/reset; allocated bytes must be0.
The version64 predecessor remains frozen under `native_v1`; it has offline
evidence only, so no prior measured regression is claimed.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../../ITERATION.md)

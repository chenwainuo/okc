# V2: valid regression; family active

`run=operand-reverse-v201; terminal=2026-10-26T23:11:13Z; exit=0`

Same-run F1: actual incumbent24.224us, frozen v147.231998us, v257.023998us.
Paired median gain versus incumbent=-33.151999us; versus v1=-9.567998us;
0/21wins against either comparator. This version is not integrated.

All three paths pass4096basis columns, normal/outlier/FP64/task checks, retained
actual-graph output mutation/NaN/immutability checks and counter reset. V2 runtime
resources162regs,0local,103600Bshared,occupancy1. Source and manifest unchanged.

Removing transpose instructions did not improve latency. Scalar granularity,
cache policy, register allocation and A-ring lifetime changed together, so the
result does not isolate a unique cause. V1/V2 memory counter diagnosis and a V3
coalesced8B/pair-lane loader are the next iterations. Preserve both failed versions
and compare V3 against both plus the actual incumbent.

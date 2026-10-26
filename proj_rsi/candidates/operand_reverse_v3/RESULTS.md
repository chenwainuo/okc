# V3: valid progress versus V2, regression versus V1 and incumbent

`run=operand-reverse-v301; terminal=2026-10-26T23:36:47Z; exit=0`

|same-run comparator|F1(us)|V3 paired median gain(us)|V3 wins/21|
|---|---:|---:|---:|
|actual incumbent rq/s4s5hz/KS2|24.032000|−30.368000|0|
|frozen V1|47.359999|−7.520001|0|
|frozen V2|56.639999|+2.176002|20|
|V3|55.071998|—|—|

Comparator F1 uses the faster before/after median; full samples and all deltas
remain in operand-reverse-v301.json. V3 is not integrated. The loader revision
improved V2 but failed to improve the family's faster V1, so no overall kernel
speedup is claimed.

All four paths pass4096basis/FP64/normal/outlier oracles, all20actual timed-graph
NaN/mutation/input checks, immutable W/packed/escape checks and counter resets.
V3 runtime152regs/0local/103600Bdynamicshared/occupancy1; source/profile identities
match, all four graphs reset, process/container gone and GPU0MiB after completion.

Next causal test: V4 specializes/hoists X addresses for the existing enforced
K4096 guard. V3 SASS has repeated runtime-stride wide multiplies/address chains
around eight loads. Remove that address work while retaining V3 width/cache,
demux, Q4/A2 schedule, geometry, producer and output. Require CPU overflow/map
proof, actual SASS change and independent review. This is a testable source
hypothesis, not proof that address arithmetic is the unique bottleneck. V1/V2/V3
and their results remain frozen for future comparisons.

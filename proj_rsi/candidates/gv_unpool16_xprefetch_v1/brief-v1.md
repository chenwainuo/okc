# Selected unpooler M16: one-slot X prefetch

CPU feasibility **PASS**, optional hardware proposal; no GPU timing, numerical execution, runtime occupancy or official gain. Parent501d901d; exact selected template `(1,2,2,1,4,16,1,1,2,1)`. M1/M8 integration files untouched.

Changed mechanism: prime one X slot at unit0; each unit snapshots both tokens' current two uint4 vectors, then loads unit u+1 into the slot before current decode/repair. Consume the snapshot with identical HMMA order. Final unit has no refill. Original XS2 global/cache policy, W pipeline, packed banks, exponent/repair semantics, accumulator/reduction/output ownership remain. X is immutable throughout invocation. CPU address regression verifies64 vector loads exactly cover16 units×2 tokens×2 vectors without a unit16 read.

Actual local sm90a SASS falsifier passes. Original control encodings/PCs/predicates match the retained actual501d unpooler binary exactly after whitespace normalization. In control, u1 X loads0xab0–0xae0 follow repair SHFL0x9e0. Candidate u1 X loads0x4a0–0x4d0 precede first decode PRMT0x580; after first repair it already loads u2 X0xaf0–0xb20. Intended one-unit motion is realized. Static LDG113/SHFL17/P2R16/HMMA128 unchanged; PRMT262→261, no claim that one removed instruction causes a gain.

Resource cost:110→126 registers, zero stack/spills.128 threads; rounded128 registers/thread yields static register capacity4 CTAs/SM under65536-register model, same upper bound as current parent. Parent actual queried active capacity4; candidate runtime capacity remains UNKNOWN. Static1024+dynamic5120B shared unchanged. More live registers, different dependencies and instruction schedule can still hurt. Root assesses the tradeoff; no automatic128-register ban.

Conditional ranking using **fixed official9916 route context**, not a forecast:

| Route | Score elasticity | Conditional10% case latency cut score value | Decision |
|---|---:|---:|---|
| unpooler M16 |4.8563%|15.521us|First isolated X-prefetch proposal: actual late-X exposure and four-warp block|
| pooler M16 |4.8292%|15.433us|Second candidate only after this result; XS0/ST2 ring differs,512-thread register capacity risk, no compiled transfer yet|
| Other selected GV routes |Route-dependent|UNKNOWN pending exact selected version|Park broad transfer; no assumption every GV path has this exposure|

For this unpooler case, official case reductions0.1/0.2/0.5us would imply approximately1.0723/2.1518/5.4344us whole-score improvements at fixed9916 references. Those are conditional arithmetic, not extrapolations of the H100 M8+.192us result. Current official route/latency must be rechecked before integration.

Prior counterevidence consulted: lane-base realized32 fewer instructions and110→108 registers but paired median0us (10W/4T/10L); same-seed F1 endpoints mixed with1.216us bracket drift. Keyed repair changes and quarter/sparse packet changes are distinct mechanisms: quarter first-HMMA earlier yet paired−.144us; sparse removing most empty headers still−.224us versus parent. Thus earlier work or lower static work alone is insufficient. This proposal keeps their repair transport/quarter changes out and targets a measured late operand issue. Passing CPU scheduling does not rule out all implementation bugs or establish scoreboard improvement.

Smallest hardware proposal: bounded exact-cell current501d versus this binary, same immutable W/escape bank, prospective current X, original task+FP64 checks, matched AB/BA timing, each arm's terminal before next replay/mutation, empty/zero inputs, full scratch/counter guards, drain/reset/cleanup. Query runtime attrs; retain every signed timing estimator. Reuse thin C ABI seam to avoid40s PyBind compile. **No runnable queue descriptor is included yet**; root can authorize harness binding/review if it prices this few-us conditional opportunity above other work. No unchanged profiler gate, broad matrix proof or H100→H200 claim.

Loop: profile → measure → hypothesis → optimize → test → validate → stack → submit → learn → reprioritize → repeat; idle≤10s. [Contract](../../ITERATION.md).

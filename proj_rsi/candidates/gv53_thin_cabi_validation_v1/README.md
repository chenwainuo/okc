# Thin C ABI: bounded GV53 runtime validation proposal

Projection only. No new kernel design, full prepare/policy, profiler or migration.
Compare the exact collected PyBindSO2638c75a against a newly built thin C ABI,
for each known parent/keyed native body at M1/M8. All four interface comparisons
use identical kernels/geometry/packing/current-input fixtures; no kernel speedup
or official-score claim is made from interface timings.

Reuse the original five codec AST functions/two literals and fixed-driver
numerical, basis, full counter/scratch backing, current-X, immutable weights/
packing, own-terminal, mutation and drain checks. Each cell has91schedule rows:
10warmups,33matched-bank blocked observations,48balanced-paired observations.
Total324clean calls/40warmups,16own terminals and40retained-graph mutations.
Both interfaces' paired terminal and every mutation output must agree bitwise.

Each graph captures on an explicit nondefault Torch stream. The C ABI records
its received stream handle and must match that capture stream. Timed replays
occur on another nondefault stream after explicit input/flush dependencies;
mutations also replay from the ordinary current stream. Current-X mutation
oracles make a default-stream escape or missing captured launch observable.
The dedicated lifetime probe drops caller references to X/output/workspace/
counter while its bound object/graph retain pointers, replays on changed streams
with fresh inputs, then resets/drains before releasing owners and checking weak
references expire. No manual dlclose or automatic lifetime guess occurs.

Compile one target-only CUDA TU (same frozen pilot plus C wrapper) and link
separately; record both phase times and dlopen time. Reuse the actual collected
PyBind module, not a new Torch-header build. Runtime cuobjdump must show exact
nonempty PC/instruction/predicate/encoding/control streams for both targets
versus that collectedSO. Both interfaces query all8attributes per native arm;
values must match, including queried CTA capacity. This is not measured
resident scheduling. Immediate invalid-native-arm error propagation is checked
without launching an invalid pointer kernel; CPU tests cover positive CUDA
status propagation. Unsafe asynchronous-fault injection is deliberately absent;
actual async failures still invalidate at required synchronization/oracles.

Cleanup forwards the exact collected admitted clear_blas_cache bridge. Graphs
must drain/reset, all run-scope owners release, the known Torch workspace map
must clear, and finalallocatedbytes must be zero. External root guard checks
remain inherited. No blanket32MiB allowance or alternative cleanup is added.

Bounds120inner/180outer seconds; all compile/subprocess limits and CPU build
phases are explicit. The prior42s Torch load_inline observation is descriptive:
no contemporaneous old cold build is performed, so actual saved build seconds
remain unknown. This proposal validates the seam and records its actual costs;
it does not establish a40s saving. An ordinary later fixed trial may reuse the
seam only after root reviews actual receipts.

Files are frozen for focused review by gv_codec_frontier. Source/transport
manifest pins the actual priorSO/result/CUDA source and all executed closure.
The ready descriptor will bind the independent review. No GPU/API/SSH/queue/git
work is performed by this author.

Loop: profile → measure → hypothesis → optimize → test → validate → stack →
submit measured gains → learn → reprioritize → repeat; idle ≤10s.
[Contract](../../ITERATION.md).

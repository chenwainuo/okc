# Exact-current pooler32 RQ diagnostic — DONE

All12 requested metrics are available across4 replay passes. Exact incumbent SO/control arm0 was profiled once: graph32×4,384threads,99,488shared bytes,168registers, occupancy1 query. No JIT or selector change.

| Metric | Actual |
|---|---:|
| DRAM read | 51.026432MB |
| DRAM throughput | 67.29% |
| Issue active | 21.44% |
| Eligible warps/cycle | 0.28 |
| Long scoreboard | 54.07% |
| Wait | 11.45% |
| Short scoreboard | 3.03% |
| Math-pipe throttle | 1.62% |
| Branch resolving | 0.71% |
| Warp instructions | 3.795397M |

These aggregate counters show dependency pressure but **do not identify its PC, role or removable time**. Math-pipe throttle is not tensor utilization. Instrumented23.552µs is not a clean benchmark or gain.

Current seed91100 retained output was checked before any later replay/mutation. Four initial validations and five retained-graph mutations pass; all20 actual source pins verified. Graphs drained/reset; cleanup helper released32MiB BLAS workspace; allocated/reserved0, external exit0/GPU0MiB. Diagnostic wall9.216s, capture4.738s.

Next is CPU source/native inspection of the exact escape-window refill load-use dependency, followed only by one minimal overlap candidate and offline compile/proof if justified. Existing descriptor/cursor losses remain counterevidence. No further broad profiler gate or speed claim.

Loop: profile → measure → diagnose → optimize/test → validate → learn → reprioritize → repeat. [Contract](../../../ITERATION.md).

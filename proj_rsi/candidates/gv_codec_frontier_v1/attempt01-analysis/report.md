# Completed C11 data retained; terminal cleanup failed

Attempt `research.gv.codec.frontier.two-cell.01`, collection1793134176827697877.
`analyze.py` reproduces84 artifact/source/schedule/numeric/resource/byte checks.
All182 samples, complete two-cell numerics,8own timed-output terminal oracles,
20retained-graph mutations, exact actual-weight roundtrips and native controls
are retained. The result remains **INVALID_CLEANUP**, not a valid performance
gain/regression admission: internal allocated0 asserted against33,554,432B.
The worker exited1. Independent dispatcher cleanup reports no container,
compute PID or GPU memory and cleanup_failed=false. Neither receipt is erased.

| Unadmitted completed timing evidence | Pooler M1 | Unpooler M16 |
|---|---:|---:|
| Paired control median, us |21.152|22.832|
| Paired C11 median, us |26.592|35.520|
| Median paired control minus C11, us |−5.408|−12.656|
| F1 endpoint-average control minus C11, us |−5.520|−12.368|
| C11 paired wins |0/24|0/24|

F1 and paired signs agree. Actual occupancy stays1CTA/SM for pooler and4 for
unpooler; registers64→64 and110→108, local/user-static0. Dynamic shared grows
17408→20480B and5120→6144B. The data do not support an occupancy-loss explanation.
These are fixed H100 cells, not official H200 score or serving-policy gains.

Unique-format bytes fall50,855,936→48,323,952 and50,593,792→49,087,008
(4.9787%/2.9782%). The mover also requests8B/unit of boundary metadata before
descriptor copies:262144B per matrix. Charging those duplicate logical requests
leaves4.4633%/2.4601% saving. Hardware cache reuse and HBM bytes were not measured.
Added repair work is about139/176 lane instructions/unit before other costs.
Source also adds a global-metadata → async-copy → shared-descriptor/record chain;
ST1/ST2 issue lookahead is short. Instruction work and that dependency are
plausible explanations, not measured stall attribution.

All normal/outlier/zero reference checks and explicit FP64 matmuls run before
graphs and timing in each cell; completed validation receipts and sequential
driver control flow prove those oracle calls executed. Therefore a BLAS cache
can already exist throughout the timing blocks. This is not evidence that it
was allocated only after timing or that no framework state affected timings.

The preceding statistics job records Torch2.8.0+cu128. Its version-matched
[PyTorch cuBLAS handle implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cuda/CublasHandlePool.cpp)
uses a32MiB Hopper workspace stored in a persistent allocator-backed map and
provides a cache-clear function. This supports the32MiB-cache hypothesis; the
failed run did not identify its surviving allocation, and matching size is not
direct attribution. The Python clear binding is absent from that version's
Module.cpp; a later-version local stub must not be assumed callable on2.8.

Do not rerun the identical full campaign merely to fix this tail assertion.
Preserve this completed evidence and failure. The next meaningful C11 source
comparison should drain/reset graphs, release aliases, record allocations,
explicitly clear the ATen BLAS cache through its supported C++ declaration,
record the before/after values and still require allocated0. That would test
the cache explanation while validating a changed source; it cannot retroactively
promote this attempt.

The separate v3 high-byte patch reduces offline instruction totals modestly,
but no source-count-to-time conversion establishes that it recovers the raw
losses. A stronger next discriminator is preloading warp descriptor slabs once
before the loop, so record issues consume landed shared offsets instead of
waiting on global metadata each unit. Preserve its additional shared-memory
cost and actual occupancy gate; keep both current incumbent and v2 comparator.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../../ITERATION.md)

# C11 preload comparison: valid partial rescue, incumbent still wins

Job `research.gv.codec.frontier.preload.01`, collection1793135941737689501.
Recommended DONE / VALID_REGRESSION against current C12. The CPU analyzer passes
155 checks, including all source/artifact pins, six native functions and four
matched control streams,262 raw samples, complete numerics,12 own timed-output
terminal sites,30 retained-graph mutations and internal/external cleanup.
All48 signed triples and every F1 endpoint contrast remain in actual-report.json.

| Same-run paired effect, us | Pooler M1 | Unpooler M16 |
|---|---:|---:|
| C12 median |21.152|22.976|
| v2 median |26.560|35.632|
| Preload/highpatch median |24.320|31.344|
| Median C12 minus new |−3.216|−8.384|
| Median v2 minus new |+2.192|+4.208|
| New wins versus C12 / v2 |0/24 /24/24|0/24 /24/24|
| F1 endpoint-average C12 minus new |−3.008|−8.160|

The partial improvement is valid versus the v2 arm measured in this run. It
does not establish a gain versus the incumbent. Both F1 signs agree with the
paired losses. The original v2 attempt remains INVALID_CLEANUP; no old timing
is retroactively admitted.

GPU resource queries show pooler CTA capacity1 for all arms, registers64 and
dynamic shared17408/20480/28160B. Unpooler maximum CTA capacity is4/4/5 with
registers110/108/96 and dynamic shared5120/6144/8000B. These are occupancy API
capacities, not measured resident occupancy or issue/stall counters. Local and
user-static bytes are0. Preload, high-byte repair placement and register capacity
changed together, so the measured rescue cannot be uniquely assigned to one.

Exact unique-format bytes remain4.9787%/2.9782% below C12. Charging the extra
logical boundary words leaves4.9465%/2.9620% in that word ledger; no cache-sector
or physical HBM traffic was measured. Remaining bitmap rank → shared record →
continuation dependencies and additional fixed header expansion are plausible
costs, not proven bottlenecks.

The new run records Torch2.8.0+cu128 with default workspace config. After graph
drain/reset and alias release, exactly32MiB remains allocated. The ATen cache
contains one entry with chosen workspace32MiB; clearing it gives zero entries
and allocated/reserved0. Independent dispatcher cleanup also reports exit0,
no container, no compute PID and GPU memory0. These matching source/runtime
observations account for this run's remaining allocation as the BLAS workspace.
They do not rewrite the historical v2 receipt.

Park this version pending one focused causal observation: does remaining
record-chain latency dominate after descriptor preload, or is fixed C11 decode
instruction work the larger cost? Prefer a narrowly scoped current/new
stall/resource observation before another source rewrite. Do not repeat the
unchanged timing campaign or reopen a broad codec search from this result.

The delivered done-evidence.json and its report/analyzer hashes are frozen.
Later observations belong in separately named supplements.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../../ITERATION.md)

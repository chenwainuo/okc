# Latency b8 independent preflight

Snapshot: 2026-10-26 UTC. Reviewer `/root/proj_evidence`; source review and local AST/lint only. No GPU launch or remote mutation.

## Decision

**GO for one bounded diagnostic A/B under the root scheduler; no approval to port or submit.** Source SHA256 `e4c68752a7734d3a4644a21bef3df649435ece3b8f126959a755bf3f9678cb1b`, `fixedcost.py` SHA256 `c95b9b21ce978704cbbf6259e799d430663cb63c05cc233d6f8a3dbf053773c8`, CPU model SHA256 `6a1f4ca9ffe8d4684e600a77818d2ae193123fbe00cd479be54a6de5799eac51`. Recheck these bytes on the box before execution.

The proposed incremental change is Python default `DBG: 0 -> 512`, which disables the inherited consumer dry iteration. The larger dirty CUDA diff predates this change: producer issue ordering, dry-loop code, exception-loop rewrite, and extension version v12 are present in **both** b8 arms. Thus b8 estimates the dry-loop flag effect on this frozen implementation; it does not independently validate the entire dirty CUDA change against committed v11.

## Correctness and capture

- `_run()` with `LOADER=3` passes `DBG | 256`; controls become 256 versus 768. Bit256 selects `latency_gemv_t`, bit512 solely skips `warm = !(dbg & 512)`. Neither arm enables no-math/no-exception diagnostic bits. `LATENCY_WLAYOUT=1` is unused by LOADER3; it does not select a second layout here.
- Default empty `LATENCY_PLAN`, `LATENCY_SMAX=16` retain the existing geometry. Every default split is one, so this experiment does not introduce cross-CTA cooperative waiting.
- The nowarm path starts at s=0 and retains x cp.async waits, weight mbarrier waits, accumulation, exception handling, and final reduction. No arithmetic/fallback dispatch change follows from bit512 itself.
- `fixedcost.py` sets `mod.DBG` before each capture; the Python launcher consumes the integer while capturing, so subsequent `mod.DBG` changes do not mutate already captured graphs. Both graphs share the same prepared immutable weight state. Outputs/scratch allocated in capture belong to graph pools; graphs replay sequentially.
- Official nine cases are run eagerly on **both** flags with actual task checker; selected timing shapes receive eager checks on both flags. Failures raise immediately. The CPU model checks packing/Python dispatch, not CUDA barriers, indexing, graph replay, or perf. Root reports CPU 9/9 pass; reviewer independently confirms AST parse for all three files and `proj_rsi lint` PASS.
- Dry control s=-1 intentionally reads shared ring before weight completion and clears accumulators afterward. This is inherited experimental behavior, not a production safety argument. If that control faults or stalls, stop the container and record an invalid control; do not infer a nowarm speedup from a timeout. A passing eager check alone also does not prove repeated-graph correctness: graph output is not checked after repeated replays in this tool.

## Measurement assessment

- Weight generation, compilation, packing, synchronization, checks, and graph creation occur outside timed intervals. Each arm gets two eager warmups before capture. Same inputs, same process, same flush, same events.
- A 256MiB **read** flush plus synchronization precedes each replay. Measurement order alternates AB/BA. CUDA events are reused only after synchronization. `paired = warm_sample / nowarm_sample`; ratio >1 favors new default. Median paired ratios are more useful than subtracting the independently measured tiny-kernel floor.
- Primary evidence is raw paired graph time. `net_*` subtracts a context-only single-kernel overhead estimate; do not use those net numbers for score projections. CUPTI GEMV duration is supporting mechanism evidence. Filter excludes flush kernels and requires exactly `profile_reps` target events; profiler failure invalidates only profiling evidence if raw graph rows already completed.
- The unweighted geometric mean over 15 diagnostic shapes is **not** the challenge score. m16/m128/m256 and the complete all-layer odd/probe matrix are absent from the performance comparison. Neither arm is compared with frozen pw36 in this job.

## Run and kill bar

Use the proposed root-owned named container and in-container timeout 240s, with explicit environment `LATENCY_LOADER=3 LATENCY_WLAYOUT=1 LATENCY_PLAN= LATENCY_SMAX=16`; `--official --layers qkv_gate,o,pooler,unpooler,router --ms 1,8,32 --reps 16 --profile-reps 5`. Preserve source hashes, full output, exit code, active GPU PID/container ownership, and monotonic timeout/termination evidence. Root reports corrected container lock plus ordered same-inode C/D test passed (`C_EXIT=15871561587989`, `D_ENTER=15871581462511`); this reviewer did not rerun that test. Initial wallclock-only test remains inconclusive.

Stop on any correctness failure, CUDA fault, no-progress interval or overall 240s limit; confirm the named container and compute PID are gone before releasing the device. A host Docker-client timeout alone is insufficient.

If new default loses or gains less than 0.3us on the relevant shape in paired raw graph medians, close this default-change performance hypothesis for that shape. A consistent gain admits one quiet full `--official` comparison against frozen pw36, with all 65 checks and actual incumbent picks. Port only a shape beating pw36 by >=0.3us in the same run; freeze the winning bytes before official check. No repeated broad loader sweeps based on a torch-only win.

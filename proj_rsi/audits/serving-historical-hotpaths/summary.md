# Retained historical full-expert hotpaths — extracted27October, captured24/25October2026

Actual raw artifacts are present:12 CUDA graph traces +12 eager-stage traces over six batches, and all three down-projection NCU binary/CSV captures. No new GPU work was done. These are historical single-GPU synthetic local EP8 expert calls, not fresh measurements, the projection benchmark, full-model serving or an official3x result. The original14 suite is NOT covered by these six traces.

##24October H200 full-expert capture

Capture hardware is NVIDIA H200/132SM; associated report records Torch2.14.1+cu130/CUDA13.0. Trace headers independently record CUDA runtime13000/driver13020. Candidate graph captures span 2026-10-24T23:30:43.569535+00:00 through 2026-10-24T23:30:49.172575+00:00. UTC values derive from trace baseTimeNanoseconds + event microsecond timestamps.

Source: experiment09329d68c81549760475f827c16498748dbacfc6; adapter13285ce5b3409d6080f46bbd9ba452b63711f906c12640246552b3145b32744a; stock adapter7f06941671fc5b05c4e8cb3c383bb930df78fa2eea94c83c4faacb1a04382567; retained exact serving diff05c13d3f16d080f3411b115ba2ab404ea8819cc56418afe08f32ee79c57843a6. `results/serving-profiled.json` retains that diff, whose SHA was checked. Neither the captured classfile SHA nor equality to current861333f/source05ec9a8a is asserted. Its stock control is original upstreama55, not the user’s frozen861333f anchor.

Five profiled graph replays per shape/arm; three separately instrumented eager calls. Clean benchmark pairs=20 is a DIFFERENT count. Table values are mean exclusive GPU kernel-duration sums per profiled replay; percentages divide that captured kernel sum, excluding graphid0 cache flush. They are not wall latency, clean benchmark numbers, model time shares or production call frequency.

|Batch|Gate/up: calls/5replays; us; share|Down: calls/5replays; us; share|Third component: calls; us; share|All graph kernels: mean sum us|
|---:|---|---|---|---:|
|1|5; 27.2516; 45.319%|5; 18.6114; 30.950%|alignment: 5; 4.4414; 7.386%|60.1334|
|8|5; 55.5656; 54.259%|5; 32.0066; 31.254%|alignment: 5; 4.7616; 4.650%|102.4074|
|16|5; 102.5672; 59.497%|5; 53.7668; 31.189%|alignment: 5; 5.2994; 3.074%|172.3918|
|32|5; 108.5244; 59.094%|5; 56.4352; 30.730%|alignment: 5; 6.4256; 3.499%|183.6468|
|128|5; 298.9792; 63.090%|5; 153.4734; 32.386%|route_map_cast: 10; 6.6242; 1.398%|473.8932|
|256|5; 355.1406; 63.410%|5; 181.5992; 32.424%|route_map_cast: 10; 6.7966; 1.214%|560.0738|


At batch128, both GEMMs together are452.4526us/replay,95.476% of the exclusive profiled GPU-kernel sum. At batch256 they are536.7398us,95.834%. This supports investigating BOTH expert GEMMs in that historical local context; it does not establish a new implementation gain or full-model Amdahl bound.

The top three PHYSICAL kernel-symbol aggregates at batch128 are:

|Rank|Actual captured symbol/family|Calls across5replays|Total captured GPU us|Mean us per replay|
|---:|---|---:|---:|---:|
|1|`sglang::device::marlin_moe::Marlin<…>` (same symbol for both GEMMs)|10|2262.2630|452.4526|
|2|`_post_expert_norm_weighted_sum_kernel`|5|29.9510|5.9902|
|3|`sglang::act_and_mul_kernel<…>` (SwiGLU)|5|18.9770|3.7954|


The two Marlin calls use the SAME C++ template symbol within each shape. Gate/up versus down attribution uses first/second Marlin occurrence inside each actual GPU graph-replay span, with exactly two verified per replay; independent eager okc_stage/gemm_gate_up/down annotations corroborate the callchain. Full symbol strings, all shapes/both arms, capture timestamps, kernel counts/durations, eager spans and39 input hashes are in `report.json`. Do not add duplicated CPU/GPU key-average annotation rows together.

For batch128 the eager capture independently records3 gate/up ranges with total GPU span906.163us (302.0543/call),3 down ranges460.669us (153.5563/call), and3 each alignment/workspace/SwiGLU/finalizer. GPU SPANS include gaps and differ from exclusive kernel sums; these numbers must not be mixed with the five-replay table.

## Repetitive hot loop

Actual expert callchain is local route mapping/alignment → workspace zero → gate/up Marlin → SwiGLU → down Marlin → post-expert norm/router-WEIGHTED-SUM finalizer. The model router, attention, pre-expert norm, scheduler, distributed collectives andTTFT/throughput are not exercised here.

The corresponding pinned checkout Marlin source loop is `serving/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h:1696`: while(slice_iters), repeatedly fetch operands/scales to registers/shared, wait for stage, matmul(k), decrement slice_iters, then reduce/write out. This is source-level loop identification, not per-PC duration attribution. Current file SHA6b43bc72d64a591bc86f62df2db14bdea5adce7b70b00af1a3e601a3c0d24fdb. It explains where to focus the next measured hypothesis, rather than treating finalization or the separate projection trace as the full serving bottleneck.

##25October isolated down-profile supplements

Each raw CSV contains ONE logical M128 down-kernel row with3 NCU replay passes under cache-control-all. Hardware is H200 directly from each CSV. Exact GPU capture timestamp is not in parsed CSV/setup; below are successful reservation start times, not invented GPU timestamps. Torch/CUDA versions are not recorded in these setup/CSV records and are not inherited from the24October capture.

|Version|Successful reservation start UTC|Profiler gpu__time_duration.sum|Long-scoreboard per active warp|Tensor active % of elapsed peak|
|---|---|---|---:|---:|
|wgmma-down-profile-v1|2026-10-25T01:46:30.141704+00:00|2.238816 ms|72.520945%|0.636717%|
|wgmma-coalesced-profile-v3|2026-10-25T02:20:18.021678+00:00|516.48 us|54.513651%|2.763287%|
|wgmma-pipeline-profile-v4|2026-10-25T02:31:30.482225+00:00|402.784 us|41.803104%|3.677100%|


Do NOT divide duration by replay passes, treat these as clean latency, or compare them as causal speedups: weight/input seeds and routing differ, and they are replacement down kernels rather than the24October Marlin pair. Stall and tensor percentages have different denominators; they cannot be converted into time shares or added. Source/bundle/JIT pins and retained binary/CSV hashes are recorded per version in report.json. The failed initial v1 staging lease is separate; successful v1 report lease was used.

The exact v4 source (SHA50120fdc47b2026be404447aa3bba0aab28bff331e6613a919e080f4399b8e24) is retained at `serving/experiments/wgmma_variants/wgmma-pipeline-v4/routed_down.cuh`. Its loop at line66 steps base+=64 overK4096 (64groups), gathers activation into shared, loads/decodes packed weights/scales, issues four K16 WGMMA operations, commit/wait0, then repeats. The v4 capture has39,040CTAs:2,112 useful,1,344 all-padding,35,584 nonlocal/beyond-post. Those are work COUNTS, not runtime percentages.41.803104% long-scoreboard /10.346706% barrier /3.677100% tensor activity motivate operand/dependency/barrier experiments; they do not prove latency, overlap gain or scheduling-only ROI.

Reproduce this extraction CPU-only: `python3 proj_rsi/audits/serving-historical-hotpaths/extract.py`. It validates all graph-event assignments, exactly5 graph replays with2 Marlin calls each,3 eager stage annotations each, report/diff provenance, and all3 successful NCU captures. No GPU/API/remote/queue action or fresh full-model claim is made.

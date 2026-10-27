# PW38 trace02 independent collected-result review

**VALID diagnostic trace; no gain or internal-loop bottleneck claim.** Independently verified52 artifact hashes and143 dispatched source pins, reparsed every collected Chrome file, and reproduced all five archived loaded SOs' SASS/resources with CPU cuobjdump. All19 RQ canonical/resource and applicable EH physical gates pass again. No GPU/context/prepare/NCU/API/queue operation was performed by the reviewer.

Collection: `results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.02/collection-1793115060902266190`. Capture SHA `6621431f187721855daab0668b53d6db2b5e28378df66ac94ed70d17587e79aa`. Source is unchanged PW38 `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`, driver e71a6c13, trial dd02e9e5, review5f5f8af9. This fresh H100 run prepared in57.769s with its original import clock retained.

Validity totals match the frozen schedule:30 cases/30 reparsed traces,330 cold event samples,90 profiled graph replays/90 actual kernel events,120 mutations,180 own-current-case terminal receipts, and30 final retained-output receipts. All actual graphs contain one kernel in this capture; this was observed, not forced. Normal/outlier task checks cover all30 cases. Full physical counter arithmetic was independently recomputed across24 arenas/113644 integers:236 changed entries match source-bound call progress, while504 nonzero unchanged entries remain unchanged. All30 graphs drained/reset; collection observes zero GPU memory, no compute/guard process and no container. Terminal exit0 is corroborated by these checks rather than used alone.

Observed per-case launch ranking below uses the mean of three **instrumented CUDA kernel durations**. The separate cold median uses eleven unprofiled event samples; different cache/setup boundaries explain why these columns are not interchangeable.

| Case | Actual selected kernel | CUDA mean, us | Cold event median, us |
|---|---|---:|---:|
| pooler M256 | LT `nvjet_tst_64x128_64x8_1x1_v_bz_NNN` | 30.752 | 29.568 |
| unpooler M256 | LT `nvjet_tst_128x128_64x6_1x1_v_bz_coopA_NNN` | 29.727 | 28.320 |
| QKV M256 | LT `nvjet_tst_72x64_64x12_2x1_v_bz_TNN` | 22.539 | 21.632 |
| pooler M128 | RQ128/ST5/FL3, KS4 | 22.123 | 24.256 |
| unpooler M128 | RQ128/ST5/FL3, KS2 | 21.899 | 24.384 |
| pooler M32 | RQ32/ST5/FL35, KS4 | 20.449 | 21.952 |
| unpooler M32 | RQ32/ST5/FL35, KS2 | 20.224 | 22.176 |
| pooler M1 | GV `<1,1,2,2,8,16,1,1,0,1>`, variant51 | 20.075 | 21.312 |

The four score-sensitive local GV routes previously prioritized remain visible: pooler M1/M8/M16 CUDA means20.075/19.808/19.956us and unpooler M16 20.021us; corresponding cold medians21.312/21.600/21.600/21.792us. Their exact mangled symbols and complete PC/predicate/instruction streams match the archived PW45 GV module. This establishes common device-code identity for those routes, not identical fresh selection, runtime, official hardware behavior, or a gain. The pinned official H200 scored routes can select different variants; historical score sensitivity is contextual and is not a measured time share in this trace.

The measured stage boundaries are explicit. Prepare/build/selection happens before capture. Each cold sample excludes the preceding X copy/output poison and L2 flush from its CUDA-event interval. Mutation checks then complete before profiling. Inside each Chrome trace, SETUP contains X copy, fixture copy and output poison (60 total GPU copies and30 fill kernels); these are separate from the90 model kernel launches in the three replay scopes. No copy/pack kernel appears inside the observed replay intervals. Host replay scopes include cudaGraphLaunch and synchronization/wait; for example, pooler M256 host spans94.622/67.911/66.512us while its model CUDA durations average30.752us. The difference is not an exclusive CPU overhead measurement.120 correlated GPU annotation records are retained separately and never counted as kernels or CPU scopes.

Each observed graph has a single model kernel, so this trace locates launch boundaries and durations. It does not subdivide weight decode, shared-memory staging, MMA loops, barriers or epilogues inside that kernel. No NCU metrics, stall samples, memory-bandwidth attribution or instruction-level cause was collected. Three samples per trace, uniformly repeated harness cases and fresh local picks also do not establish production frequencies or official H200 priorities. All30 cases, raw values and exact symbols are retained in the adjacent JSON, including the smaller launches omitted from this concise table.

Output/X/W/pack/escape tensors were not archived for independent offline numerical regeneration. Current-output, current-X, mutation and immutable-byte assertions are source-bound executed receipts; complete counter integer snapshots and actual SOs are archived and independently rechecked. This is the material evidence limit, not an additional numerical hardware-bit oracle. Root remains the sole scheduler finalizer.

Reproduce read-only validation with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 proj_rsi/audits/pw38-hotpath-trace-02-independent.py`. Adjacent JSON retains the full30-case ranking, all cold/raw profile values, JIT hashes, counter result, hardware and exact PW45 hot-GV comparisons. The scheduling validation is a proposal to root; no queue mutation accompanies this report.

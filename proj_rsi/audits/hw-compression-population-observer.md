# Population / observer decision after zero control

`state=ZERO_CONTROL_RESPONDS; next_SM_write_control=DEFERRED_NOT_NEEDED; implementation=NONE; speed_claim=NONE`

The proposed flat-result follow-up is superseded by valid `hw-zero-counters01` (2026-10-26 23:49:07–23:49:24 UTC). Existing D2D population, GENERIC allocation, fixed Lt consumer and selected memory counters can expose a large response to zero weights. There is no present evidence requiring an SM rewrite.

## Actual result / validity boundary

Raw evidence: [driver JSON](../../results/proj-rsi/takeover/20261026/hw-zero-counters/hw-zero-counters01.json), [raw CSV](../../results/proj-rsi/takeover/20261026/hw-zero-counters/hw-zero-counters01.raw.csv), [host log](../../results/proj-rsi/takeover/20261026/hw-zero-counters/hw-zero-counters01.host.log), [parser analysis](../../results/proj-rsi/takeover/20261026/hw-zero-counters/hw-zero-counters01.analysis.json).

| Selected field | GENERIC | VMM NONE |
|---|---:|---:|
| DRAM read bytes | 17,842,944 | 68,174,336 |
| DRAM read sectors | 557,592 | 2,130,448 |
| L2 total sectors | 2,513,523 | 4,268,938 |
| L2 device-fill sectors | 557,588 | 2,130,450 |
| ILC input / output sectors | 65,590 / 65,590 | 65,549 / 65,549 |
| ILC evaluated-for-compression sectors | 0 | 0 |

Difference = **50,331,392 observed DRAM-unit bytes**, 73.8274767% of NONE's aggregate observed reads. This is not a whole-W compression factor, a wire-format measurement or latency evidence. X/output/cache/metadata are included. One ordered pair has no confidence interval. L2 counts change too, so the data do not localize the effect solely to HBM or establish a universal codec.

Independent CPU parsing finds one base-units row plus exactly two results, 309 CSV columns, all21selected fields finite, and read/write bytes exactly32×sectors. Case identity is bound by the CSV `Domain:` range columns and host GEN→NONE markers. Both have identical `nvjet_tst_128x64_64x8_1x1_v_bz_coopB_NNN`, grid[1,128,1], block[384,1,1],168registers and214292B NCU total shared memory (=213268B kernel+1024B driver). Same layout2/config[66,18,1,0,0,4,35,0,2],64MiB zero Wt and nonzero X. NONE's~65MiB observed reads remain consistent with reading the full64MiB payload. These21metrics do not include logical TMA/instruction counts; equal shader/resources do not directly prove equal dynamic instruction counts or exact W-only bytes.

Driver records14complete-output gates: pre/post actual graphs on original and fresh nonzero outlier X, two fresh-output captures, and two immediate profiled-replay checks. NaN poison is overwritten with exact zeros; full original/foreign weight bytes and X are immutable. All4096basis probes are explicitly unrun for zero W, where they add no useful coverage. Both granted flags/query results match1/0, graph teardown precedes unmap, checked Region0/undefined tensor passes. Host reports3counter passes per node, process exit0 and no retained-resource warning. Root independently confirmed node idle; author did no GPU work. Existing real-normal results remain flat and valid.

## Exact official sample branch / supported API path

Pinned CUDA12.8 [`saxpy.cu`](https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu), SHA256`a7fe80f9edbeacc6a96d04c2e6da8aa6f844ca369ae8d85b420df8ad15431c02`: lines146–158 select memcpy-only initialization and ordinary output for **SM8.0 or8.6**. Other devices, including SM9.0, use compressible output; lines47–53/79–84 initialize homogeneous float4 values with an SM kernel. Inputs are constant1.0. This establishes an official SM-store path, not a Hopper D2D prohibition. Our zero result independently establishes responsiveness under D2D.

Pinned [`compMalloc.cpp`](https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp), SHA256`0ecf550b3695ee86b5de0e7e5a05d31598a504fc63e7dbcf964bd22bac9ad415`: pinned/device VMM properties, GENERIC flag, granularity rounding, reserve/create/query/map/access. Installed CUDA12.8 header exposes these APIs and both flags. No special device ISA is needed for a plain float4 init kernel; installed nvcc12.8.93 supports `-arch=sm_90`. The sample's [CMake file](https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/cudaCompressibleMemory/CMakeLists.txt) compiles CUDA/C++17 and links `CUDA::cuda_driver`; a direct equivalent uses nvcc with sample Common includes and `-lcuda`. No compile or new kernel was needed for this audit.

## Counter semantics / deferred causal contrast

Installed2025.1.1 [`MemoryL2Compression.py`](../candidates/hw_compression_counters/MemoryL2Compression.py:63) selects LRC ILC counters for CC≥90 and compares L1TEX write bytes with compression-input bytes to diagnose write coherence. It recommends zero/homogeneous data. Base metadata labels ILC input/evaluated/output sectors, with no dedicated decompressed-weight observer. Here a strong read response coexists with evaluated sectors0 and success0: those fields are **insufficient to detect this read benefit**. Do not require positive read-side ILC events as a success condition.

[Hopper12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html#inline-compression) promises eligible-allocation automatic compression when a suitable pattern exists, including a possible none choice. It does not specify local formats/thresholds. Checked DRAM metadata establishes hardware byte/sector counts, not exact GH100 counter placement relative to compression metadata; empirical responsiveness is now established without certifying on-wire bytes.

Had zero stayed flat, the smallest next observer control would have been **two identical coalesced SM zero-write nodes, GENERIC versus NONE**, using the same64MiB destinations/helper; collect supported LRC events plus L1TEX write bytes, exact byte snapshots and checked lifetime. Existing D2D zero observations would remain the anchor rather than automatically adding four population/flag combinations. A positive write-event response would establish an observer for eligible SM writes only; it would not prove D2D was defective or predict the subsequent consumer's benefit. Consumer profiling after SM population would require a separate causal decision. Kernel replay memory save/restore is another compression-state qualification, not an excuse to change profiler policy.

**Decision now:** do not implement/run that follow-up. Investigate which exact reversible representation could expose normal-BF16 redundancy with less total cost than current packed12 kernels. Keep zero, native-normal, packed and ownership versions preserved.

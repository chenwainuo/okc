# Compression next cause: establish observer/path responsiveness

`state=ZERO_CONTROL_CPU_SOURCE_AUTHORIZED_BY_ROOT; GPU=UNRUN_PENDING_REVIEW; prior_versions=PRESERVED; speed_claim=NONE`

Yes: a low-entropy positive-control candidate is warranted before attributing the flat native result to weight entropy, D2D population or metric placement. Root authorized only a new known-zero BF1664MiB Wt GENERIC/NONE comparison, retaining the fixed Lt config, X8851, helper7f2, D2D copy and existing counter method. No serving-weight rerun, SM rewrite, latency search or new kernel is part of it.

## What the current pair establishes

Stored `hw-native-counters01` completes all driver/oracle/identity/Region0 checks. GENERIC/NONE measured `dram__bytes_read.sum`68174080/68174592B:512B or16sectors difference, about0.000751%. LRC ILC input/output65549sectors in BOTH, compressible0, success0. These are valid flat observations under the collected conditions; no useful native speed improvement or effective compression factor follows. Original native v1 and packed v1 remain evidence, not discarded families.

The65549ILC sectors correspond to2097568B if interpreted in32Bsector units, only416B above the2MiB output and about32x fewer sectors than64MiB Wt. That is **a clue**, not proof, that the observed ILC events describe a write/output path rather than every immutable-W read/decompression operation. The installed2025.1.1 `MemoryL2Compression.py` discusses write patterns, and its raw metadata merely says input/evaluated/output sectors; it does not label a dedicated decompressed-weight counter. Unchanged read-side ILC activity cannot locate the missing effect by itself.

## Primary-source boundary

- [Hopper inline compression](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression): eligibility is per allocation, algorithm choice is automatic and may choose no compression. It does not promise success for normal BF16 weights or mandate a D2D-to-SM rewrite.
- [CUDA VMM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory): support/granted-property queries establish allocation eligibility, not effective ratio or a population-engine guarantee.
- [NVIDIA's compressible-memory sample](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu) uses homogeneous constant data, copies on its older8.0/8.6 branch and an init kernel on other architectures. That demonstrates a useful control pattern/allowed paths; it does not establish Hopper SM writes as mandatory. Installed NVIDIA Nsight rule recommends regions with many zero/homogeneous values. All-zero BF16 is therefore a reasonable maximal-redundancy control candidate, not a guarantee that this exact pipeline will expose compression.
- [Nsight's device-memory table](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#device-memory) describes bytes as L2/device-memory transfers. [NVIDIA's sector explanation](https://forums.developer.nvidia.com/t/how-to-compute-dram-bytes-read-sum-dram-bytes-read-sum/307220/2) establishes hardware sectors×32 and real transfer overhead. Neither checked source explicitly specifies GH100 counter placement relative to inline compression/metadata or guarantees postcompression wire-byte accounting. Call the values observed DRAM-unit bytes, not a certified on-wire compression ratio. A responding positive control would supply empirical observability for this path.

## Minimum authorized diagnostic

One zero-weight payload, two allocations, two selected Lt consumer nodes. Source raw BF16[8192,4096] and already prepared contiguous Wt[4096,8192], both64MiB and all bits0; helper clones only Wt byte view into GENERIC1 and NONE0 with the same `cuMemcpyDtoDAsync`+stream sync. Source ordinary zero initialization occurs outside the selected graph; no SM rewrite of either foreign destination. Keep native-v1 exact layout2/config[66,18,1,0,0,4,35,0,2], existing Lt workspace, source hashes/library, precision, geometry and selected NVTX/replay/cache/clock method. No prepare/autotune or normal-weight rerun.

Use original nonzero X8851 for the selected profile nodes. Before/after profiling also mutate the same input to a second fresh nonzero X and capture an additional fresh output, retaining every graph/output/owner. NaN-poison the real output before each replay and require all128x8192elements finite and EXACT0, plus exact X/zero-weight bytes. This proves complete writes and fresh captures under the known-zero task, not useful all-K weight coverage. Do not repeat4096trivial one-hot cases or claim the real-data/full-basis suite was rerun; those prior gates remain separate frozen evidence. Kernel/config/grid/block/register/shared identity and logical read counters must still be checked to detect a different data-dependent path.

Collect only the same21supported DRAM byte/sector, L2, ILC and launch metrics through the existing two-node cold diagnostic, with immediate output checks before any repairing replay. Graph reset→all state/views dropped→checked Region0 teardown stays mandatory; no frame-locals snapshot. Proposed bound remains inner180/outer240 after critic/root handoff. `hw_compression_zero_control/` is a new version, preserving every earlier source/result. No author GPU launch.

## Outcomes that change the next question

| Zero control result | Supported learning | Still unproved / next question |
|---|---|---|
| Same fixed shader/logical read work, GENERIC observed read bytes clearly lower than NONE | This D2D allocation+consumer+observer combination responds to maximal redundancy. Real native data failed to yield a comparable observed effect. | Not a serving speedup or universal/wire-byte ratio; inspect real weight local patterns/recognized formats only if ROI remains credible. |
| Read bytes respond but read-side ILC fields remain identical/zero | Those ILC fields are not sufficient observers of this read benefit. | Keep their event scope qualified; do not require a nonzero event as proof of every read compression path. |
| Both allocations are flat and ILC remains uninformative | This control did not establish path/observer responsiveness. | Entropy, D2D/feature state and metric placement remain unisolated. Do not conclude hardware compression is impossible or that an SM rewrite is necessary. |
| Kernel/logical read/resource identity changes for zero input, or either baseline bypasses substantial reads | Positive control is not comparable to original consumer work. | Diagnose that behavior before interpreting byte differences; it is not evidence for normal serving weights. |
| Allocation/poison/oracle/immutability/cleanup failure | Invalid version/measurement. | Repair the specific fault; preserve all data and continue one hypothesis at a time. |

If the zero pair is still flat, a separately proposed next question could use NVIDIA's known homogeneous-data sample/init path to observe compression-eligible write events, testing whether the installed observer can ever register success. That would be a deliberate population/instrumentation control with its own scope, not an inferred requirement to rewrite production weights, and is not authorized or implemented here. Do not add initialization variants or a codec sweep without the zero-control evidence first.

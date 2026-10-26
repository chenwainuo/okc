# Hardware inline compression of immutable RQ weights

`state=CPU_HOST_BUILD_PASS_GPU_UNRUN; policy=ONE_ALLOCATION_EXPERIMENT; incumbent_edits=NONE`

Source/API feasibility GO; runtime/performance HOLD pending independent review and root's named exclusive GPU handoff. One host-only helper clones the exact selected frozen pw36 RQ uint8 packed W into a Driver VMM allocation requesting `CU_MEM_ALLOCATION_COMP_GENERIC`. The original lossless12-bit layout, escape lists, math, shader binary, KS, counters, input handling and output path stay unchanged. No CUDA device code is added. Targets are pooler128 and unpooler128 only when the actual prepared pw36 state selects RQ. Otherwise keep the original dispatch and report a skipped target; never force RQ for a synthetic win.

## Evidence and limits

Repository search found no earlier `compressionType`, `CU_MEM_ALLOCATION_COMP`, `cuMemCreate` or generic-compression experiment in `proj_rsi/` or handover notes. Existing H100 NCU raw rows0/1 report both generic compression and VMM support1 (`results/proj-rsi/takeover/20261026/operand-reverse01-ncu.raw.csv`). These are device capabilities, not granted allocation properties or compression ratios.

NVIDIA's [Hopper tuning guide, Inline Compression](https://docs.nvidia.com/cuda/archive/12.2.0/hopper-tuning-guide/index.html#inline-compression) describes per-allocation hardware compression with automatic algorithm selection, including no compression when patterns do not qualify. It preserves the allocation footprint; bandwidth benefit is conditional on contents/access. This could remove physical W traffic without adding another software decode, but does not remove existing rq decode work. No2x/4x ratio or speedup is predicted.

The [CUDA VMM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory) requires allocation-time opt-in and an actual `cuMemGetAllocationPropertiesFromHandle` check; allocation can decline the requested attribute despite device support. The [CUDA12.8 allocation-property reference](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/structCUmemAllocationProp__v1.html) agrees with installed `/usr/local/cuda-12.8/targets/x86_64-linux/include/cuda.h`: enum NONE0/GENERIC1, attribute107,32-byte CUmemAllocationProp, allocation flags and VMM APIs are available. The helper queries minimum granularity, rounds safely, creates, checks granted compression, reserves/maps and grants current-device access. It does not infer compression from a successful cuMemCreate alone.

NVIDIA's [allocation sample](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp) uses this VMM API sequence. Its [SAXPY initialization](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu) explicitly uses cudaMemcpy where needed, including older8.0/8.6 restrictions. This helper performs exact Driver D2D copy during prepare, then synchronizes the caller's current Torch stream before exposing the tensor. No packing/copy/allocation enters a measured graph. TMA accesses ordinary global virtual addresses here, but the precise compressed-allocation+RQ bulk-TMA combination is runtime-unverified; full complete-output tests are its gate, not documentation inference.

**Does cudaMalloc already enable it?** Public CUDA documents explicit VMM opt-in; cudaMalloc exposes no compression flag. Current [PyTorch allocator source](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp) delegates its ordinary path to cudaMalloc and zero-initializes expandable-segment CUmemAllocationProp without setting compressionType. That current upstream source is not proof of the remote binary's exact backend. Lab records backend/Torch/CUDA versions and attempts `cuMemRetainAllocationHandle`+property query on the actual ordinary weight pointer. Non-VMM retain failure is reported **UNKNOWN**, never falsely NONE. If query shows ordinary GENERIC, the target closes as already enabled. The explicit VMM NONE control is essential when the ordinary allocation is unqueryable.

NVIDIA's [2025 small-transformer optimization session](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73168/) reports transparent hardware compression was unhelpful for its data patterns. That is a relevant negative prior, not a measurement of this packed W. Here two-thirds of12-bit pack bytes hold low BF16 bits; little redundancy may remain. Generic allocation can succeed with zero effective compression. A missed complete-graph gain bar closes the fixed policy without initialization-method, layout, geometry or hint sweeps.

## One controlled experiment

For each actual selected RQ target prepare frozen pw36 exactly once (SHA`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`). Helper builds before importing pw36 so helper compile time does not consume its prepare clock. Original prepare has measured~163s on the judge; this lab does not port anything into that budget.

Three cases use the exact same `kernel_fn`/selected p/module/KS/escapes/counters. Case state copies replace only selected p['lay']; baseline state remains intact:

| Case | Weight storage | Purpose |
|---|---|---|
| ordinary | Actual prepared Torch allocation | Required incumbent comparator |
| VMM GENERIC | Clone, actual queried compression1 | Only proposed policy |
| VMM NONE | Clone, actual queried compression0 | Controls VMM/page/allocation layout; not another kernel policy |

Targets share the same fixed timing/validation procedure, not an algorithm or geometry sweep. Allocation failure/declined properties stops the probe with the original fallback retained. Non-RQ target is skipped. There is no silent fallback timed as a successful compressed RQ: final captured kernel names/count must match across all3 cases and contain one rq kernel.

Admission before timing: complete source packed bytes→CPU snapshot; both clones independently D2D/D2H roundtrip exactly against that snapshot; query each mapped pointer's retained handle; record bytes/granularity/mapped size/requested and actual flags. Then all K one-hot columns must be exact in every case (8192 pooler/4096 unpooler), FP64/random tolerances and official normal/outlier oracles must pass, with immutable X. Validator-only output copies remain outside timing.

Timing captures the actual `kernel_fn` output in each case, retains that real output and allocation owner, and adds no copy node. Same256MiB read flush/event method as frozen `_time_us`, with5 discarded cold replays,21-sample ordinary-before/GENERIC/NONE/ordinary-after medians, lower ordinary comparator, and21 rotated3-way paired cold graph samples. The same retained graph execs are used for F1 and paired measurements, then validated directly. No clock, cache policy, driver setting, flush size or arithmetic changes occur. Events, profiler and setup remain outside captured kernels.

Post-timing, each actual graph receives original/zero/fresh-normal/fresh-outlier/all-probe X at the same address. NaN-poison its real output, replay, require every element overwritten and complete oracle agreement, and require immutable X. Original and both cloned full W bytes must remain exact; counters must reset. Diagnostic CUPTI kernel identities are checked after timing and do not select the winner.

Per-target survival requires >=0.5us in BOTH conservative ordinary F1 and paired median, plus >=0.3us over explicit VMM NONE in BOTH methods. This keeps an address-layout accident from being called compression. Passing means local evidence for review, not an official win or a measured compression ratio. At fresh pw36 verify16 sum65.5104us, a hypothetical aggregate2.4us improvement would be about20score-us; no such improvement is established. Ratio attribution would require appropriate matched physical-traffic/compression counters, not a support bit, allocation flag, nvidia-smi memory use, or throughput alone.

## Ownership and streams

`at::from_blob` creates a CUDA uint8 tensor with identical shape/strides and a storage deleter retaining the VMM Region. Allocation is allowed only outside stream capture, on the source tensor's current CUDA context/device. Prepare D2D copy is ordered on the explicitly passed current Torch stream and completed before capture; no caching-allocator stream ownership is assumed for external storage.

CUDA graphs retain raw addresses; external from_blob storage does not become graph-owned automatically. Each lab Case explicitly retains clone+state+graph+captured output. Cleanup drains submitted work, calls graph.reset on every exec, drops captured outputs/state aliases, then invokes checked clone.close. C++ rejects remaining TensorImpl/Storage aliases. Region pushes its original context, synchronizes outstanding work, unmaps, releases the allocation handle and frees the reserved address range, with checked errors propagated. On failed drain/unmap it retains remaining resources and invalidates the trial rather than claiming success. Destructor cleanup is a fallback; normal lab cleanup is explicit and checked. Context synchronization cannot protect a future replay from a separately retained graph, so this standalone helper is not a general production graph-lifetime manager.

No data pointer is unregistered with Torch's allocator or freed through cudaFree. No graph/source reference is remapped while live. All captured graph references must be destroyed before using the helper's close method outside this lab. Production integration would need an equivalent persistent owner in prepared state; it is not authorized now.

## Reproduction and first runtime gates

CPU-only full host build/import, no libcuda resolution/device API call:

```bash
python3 proj_rsi/candidates/hw_compression/preflight.py
```

This builds the actual Driver/from_blob/pybind helper against installed CUDA12.8 headers and CPU Torch, imports it, checks enum/struct ABI, AST/source identity, and rejects CPU input before Driver initialization. It is stronger than syntax-only compilation but does not prove CUDA backend allocation, actual compression, TMA compatibility, runtime graph lifetime or benefit. Local Ninja is absent, so preflight uses an explicit g++ shared-module command; the GPU adapter uses the normal Torch extension loader.

After independent critic GO and root's named exclusive GPU handoff, one bounded command (root selects inner/outer timeout; no author GPU launch):

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/hw_compression/lab.py --json /okc/logs/hw-compression01.json
```

First real gates: compile/import against actual CUDA Torch; device support; allocation-granted flags; mapped-pointer query; bit roundtrip; source/module/state equality; complete output and fresh graph checks; no fallback; checked cleanup. Failure closes without automatic retries. A numerical/runtime/cleanup failure invalidates timings. No official check, full submission, precision change, activation cache, speculative hint or incumbent modification is part of this experiment.

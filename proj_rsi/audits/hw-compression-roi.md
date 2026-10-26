# Hardware compression: independent allocation/representation/ROI review

`state=ONE_BOUNDED_ALLOCATION_PROBE_CREDIBLE; GPU_BY_REVIEWER=NONE; author_folder_edits=NONE; effective_compression_ratio=UNKNOWN; speedup=UNMEASURED`.

Scope: author is integrating one CUDA VMM compressed clone of prepared weights, retaining the frozen pw36 RQ kernel and actual selected plan. This review neither implements that integration nor authorizes a GPU launch. Root serializes all GPU work and supplies any named handoff. Frozen pw36 SHA256 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.

## What NVIDIA actually promises

1. H100 supports inline compression on selected allocations; the compressor can choose a suitable algorithm **or leave the data uncompressed**. Access bandwidth may improve; the allocation's reserved memory footprint does not shrink. The public tuning guide does not specify an entropy codec, block thresholds or a guaranteed ratio. [NVIDIA Hopper inline compression](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression).
2. Query device VMM support and `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`; set `CUmemAllocationProp::allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC` before `cuMemCreate`. Device support1 is necessary, not sufficient. A successful creation can still lack the requested attribute; query the resulting handle with `cuMemGetAllocationPropertiesFromHandle`. That returns allocation properties, **not an effective compression ratio**. [CUDA Programming Guide, compressible memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory), [Driver allocation property](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/structCUmemAllocationProp__v1.html).
3. NVIDIA's sample uses the active device, queries granularity, rounds size, reserves VA, creates the handle, checks the granted attribute, maps and sets device access. It releases the handle after mapping; cleanup unmaps and frees the VA reservation. The mapping retains the physical allocation. Do not copy its process-exit failure handling into a submission. [NVIDIA `compMalloc.cpp`](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp).
4. The sample initializes through GPU stores on Hopper; its memcpy-only restriction is explicitly for SM8.0/8.6. Its performance example uses constant1.0 data, which is a capability example rather than evidence for these random BF16 or packed bytes. [NVIDIA `saxpy.cu`](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu).

**Do not claim `cudaMalloc` defaults to this opt-in.** The documentation gives an explicit VMM allocation path; it does not establish whether a particular PyTorch/runtime/driver allocation already has some compression behavior. Ordinary tensor baseline versus VMM GENERIC measures their observed difference. If a later positive result must be attributed specifically to the compression flag, a VMM NONE allocation with identical construction is the appropriate control, not an assumption about the caching allocator.

Source distinction: NVIDIA compute inline compression is not nvCOMP, application Huffman coding, structured sparsity, quantization, L2 residency pinning or activation caching. This integration introduces no software decompressor and should leave RQ device instructions unchanged.

## Representation headroom: native BF16 > packed12, actual ratio unknown

Read frozen `_c12`/`_unc12`, `_pack_nat`/`_rqpack`, `_gxplan` and prepared-state dispatch; historical source data in [codec-study.md](codec-study.md) and `candidates/pw9/hypothesis.json`. The weights here are Gaussian rounded BF16, not a sparse/repeated-value model. Historical empirical whole-BF16 symbol entropy is about10.46bits/weight, exponent about2.54bits, mantissa about6.97bits; signs are near1bit. Those are distribution statistics from generated weights, not measurements of inline compression.

|representation|logical bits/weight, excluding small sidebands|redundancy prior|what changes in this integration|
|---|---:|---|---|
|native BF16 / FMT0 fragments|16|Unused exponent range/common exponent bits leave substantially more statistical redundancy|Nothing: conditional future allocation target only|
|r6/RQ packed12|12|Most exponent redundancy already removed; low-byte plane remains near-uniform|Same bytes cloned into a compression-eligible allocation|
|escape arrays / sentinels|separate|Some repetitive sentinels/control fields, small traffic|Keep unchanged for this one-variable probe|
|router hi+lo|two BF16 values/weight|Different low-component distribution; does not share one guaranteed ratio with ordinary BF16|Outside this probe|

Each `_c12` unit is1536B for1024weights:1024B preserve original low bytes (exponent LSB plus7mantissa bits),512B hold sign+3bit exponent-window offsets. RQ reblocks units without changing their bits. Thus **two thirds of packed payload is already a high-entropy byte plane**. Remaining regularity is mainly in the code plane and small sidebands. Local layouts, compression granularity and supported hardware patterns determine whether any of it can be exploited; global symbol entropy alone cannot answer that question.

Illustrative source-entropy arithmetic, assuming the historical10.46bit source and an ideal coding model:

```
native16 -> 10.46: 34.625% fewer logical bits, ratio16/10.46 ~1.530
packed12 -> 10.46: 12.833% fewer logical bits, ratio12/10.46 ~1.147
pooler4096*8192:
  native payload67,108,864 B
  packed payload50,331,648 B
  ideal10.46bit model43,872,420 B
  residual packed saving~6,459,228 B
```

This is **not a guaranteed lower bound for this finite dataset, a hardware compression ratio, a forecast of ILC effectiveness, or a walltime estimate**. Histograms/correlation can differ on fresh weights; small-block hardware formats and metadata can erase the modeled residual saving entirely. Running zlib/zstd/nvCOMP on CPU/GPU would establish only those codecs' behavior, not ILC's. No CPU compression experiment is needed to justify a larger ratio claim.

Native BF16's larger prior is relevant if the packed probe is flat. Frozen GV already has FMT0 plain-BF16 fragments, no escape repair and no12bit decode (`_gxplan` accepts evenFMT; GVC creates a plain twin for fallback). **A future native-BF16 compression experiment could compare that existing path against actual FMT1**, potentially trading extra logical bytes for zero decode. It would need its own fixed geometry, exact-byte allocation validation and full incumbent comparison. It is not a port to RQ, an admitted prototype or permission to sweep geometries. Raw BF16 compression must beat packed traffic/decode together; it does not win simply by compressing better in percentage terms.

## Failure conditions to reject independently of speed

|area|required invariant / failure consequence|
|---|---|
|Device/context|Use the tensor's CUDA device and active primary-context-compatible device guard, not device0 by assumption. Handle device support, create/query/map/access/descriptor errors with the original selected plan. H100 flag1 does not prove H200/driver/container availability|
|Sizes/alignment|`nbytes` overflow-safe; nonzero logical allocation; size rounded to queried allocation granularity; exact original dtype/shape/strides visible to Torch. Every copied/read byte lies in mapping; initialize any padding that the chosen vector/bulk accesses can legally read|
|Granted property|Record requested and granted compressionType separately. Query failure or grantedNONE is capability NO-GO/fallback, not a successful compression test|
|Exact bytes|Clone with an exact byte copy; no float conversion, normalization, reencoding or lossy precision change. Compare full logical payload bitwise. Record `data_ptr`, storage span and metadata so a later `.clone()`, `.to()` or materializing reshape cannot silently return ordinary allocated storage|
|Foreign tensor ownership|`torch::from_blob` wraps external CPU/CUDA memory; it does not provide the CUDA allocation's lifetime. Keep an owning context in the prepared state and the plan/closure for every graph that can replay. Views alone or a raw integer pointer do not establish mapping ownership. Never call `cudaFree` on the VMM pointer; cleanup must unmap the original full mapping and free its reserved VA once. [PyTorch external data](https://docs.pytorch.org/cppdocs/api/aten/creation.html#using-externally-created-data)|
|Async initialization|Copy on the correct current CUDA stream, with source lifetime and stream dependency established. Finish/hand off initialization before capture or any alternate-stream consumer; descriptor creation is not a device-memory copy fence|
|Lifetime vs streams|The last Python/C++ reference can disappear before GPU work completes. A custom VMM deleter must not unmap an address still used by eager, side-stream or graph work. Ordinary caching-allocator/`record_stream` guarantees cannot simply be assumed for a custom foreign allocation. Retain immutable weights for the full state/graph lifetime, then ensure all users complete before cleanup outside the timed path|
|Graph capture|Allocate/map/copy/validate/choose only in prepare; no VMM operations, host synchronization, data-dependent allocation or compression queries in `kernel_fn`. Graphs reuse addresses; freeing or remapping the same address after capture breaks that invariant even if a tensor object still exists. [PyTorch CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda#cuda-graphs)|
|TMA and pointer routing|Rebuild/replace any descriptor referring to moved storage; check pointer actually passed to kernel. RQ packedW currently uses bulk copies, while X uses the tensor map: changing only packedW should not require a new X descriptor. Any moved tensor-map operand needs the new aligned address through CUDA API, never manual opaque descriptor edits. TMA encode success is not runtime correctness/effective compression proof. [NVIDIA tensor-map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)|
|All fallback paths|Replace only the admitted plan's immutable weight payload; preserve original selected plans/weights for unsupported devices, allocation failure, errors, odd/unregistered shapes and budget exhaustion. Do not change math, workspace/counter reset semantics or router precision|
|Cleanup/errors|Store mapped size, reservation, device and ownership state together; unwind every partial failure in reverse order. Releasing a mapped handle early is legal (sample), but never leak a retained handle or double-release/unmap it. C++ destructor must not throw during teardown|

The public docs checked do not advertise a special TMA-versus-ILC incompatibility. That is absence of a documented restriction, **not a blanket portability proof**. Exact frozen-path eager and graph validation on the target allocation is still required. Preserve all graph replay zero/counter handshakes; compression of immutable weight memory does not justify modifying them.

## Legitimacy under this task

Local authoritative `muse_bf16_proj/TASK.md` explicitly permits lossless prepared weight compression, fixes precision, forbids caching anything derived from fresh `data`, and requires graph-capturable timed calls. [Public rules](https://openkernelchallenge.dev/rules) state the same lossless-weight and graph restrictions. A bit-exact driver-managed clone of immutable weights, prepared once and consumed by the same kernel, fits those rules in this review's interpretation; hardware compression neither changes numerical precision nor caches outputs.

Keep it confined to weights: no X/output cache, previous-call reuse, advance computation of the timed GEMM, harness introspection, changed L2 flush policy, clock/power changes or services. Compression tests may synchronize and print scalar metadata in prepare/lab; not inside timed `kernel_fn`. A driver API allocation is ordinary implementation work; use public APIs via the authorized extension, not environment/harness modification. Prepare stays within300s task budget and existing240s guard; capability failure returns the incumbent rather than using more time to search.

## One bounded probe with the highest information ROI

**Recommendation: one shape, pooler/m128, same frozen selected RQ plan, original packed payload versus GENERIC clone.** It is the large48MiB packed payload and an actual official RQ winner: pw36 job`e22a4b9d693d4902` picked`s4s5z/KS4` and benchmarked18.15us. Unpooler/m128 in that job picked the r6/Nvjet fallback, so using an arbitrary rq there would weaken practical comparison. On the local trial, query and guard the freshly prepared pw36 pick; if it is not rq, stop this RQ integration probe as unsupported rather than forcing a different kernel.

Keep geometry, kernel source/hash, launch flags, escaped weights, table, X, workspace, counter resets and output allocation semantics identical. Only W payload address and allocation eligibility differ. Record current capability flags, requested/granted properties, logical/mapped bytes, baseline path scalar metadata and source hash. Do not dump tensors. One preparation, one admitted clone, one bounded cold-L2 paired graph comparison (e.g.40 alternating pairs after normal warmup), and the usual before/after F1 baseline measurement is enough. No geometry, seed or representation sweep; timeout/any GPU fault closes this integration attempt.

Before and **after the actual timed graphs**, require exact payload/input immutability and output correctness for both paths on original X, zero X and fresh/outlier X, including the same exact one-hot checks and FP64 tolerance oracle. Check capture/replay after source tensors used only for initialization have been released while the clone owner remains alive; this targets ownership rather than cached values. Keep a fresh output allocation check. All mutation/validation occurs outside measured intervals. Do not accept finite-only baseline checks.

Admission after the one trial: >=0.3us gain **both F1 and median paired graph** against the actual parent pick, correctness/capture/ownership all pass. Otherwise HOLD/CLOSE for this packed-RQ policy. A grantedGENERIC flag with flat/negative timing is a valid negative result. A gain measures end-to-end allocation benefit, not an exact hardware compression ratio; tracing bytes would be a separate root decision after a material win, not an automatic extra profile. A VMM NONE control may then separate flag benefit from allocation/TLB/placement effects.

Optimistic byte-only model for orientation, not a speedup promise: the~6.46MB residual packed entropy saving corresponds to1.70us at3.8TB/s or1.35us at4.8TB/s if all those bytes are saved and exposed. To save0.3us through bandwidth alone at those rates requires1.14/1.44MB, about2.26/2.86% of the packed payload. Effective hardware reduction may be0%, and math/X/synchronization or HBM-to-L2 overlap can make the walltime gain much smaller. The inline allocation is worth this bounded check because it avoids new per-weight software instructions, unlike closed codecE; it is not a credible multi-microsecond gain claim yet.

Exact score context stays separate: pw36 official score3212.415452, verify16 load total65.510400us. A **genuine**0.3us total improvement on that same job would project3209.958924us (−2.456528),0.5us would project3208.316002us (−4.099450). These are conditional arithmetic, not predicted gains. Official best from the separate rs32 job remains3200.615008us. One local H100 allocation result cannot be inserted into an H200 score or establish the `<3000us` goal.

Reviewer performed public-primary-doc/source reads and local arithmetic only; no author-folder edits, CPU codec experiment, compile, remote action or GPU launch.

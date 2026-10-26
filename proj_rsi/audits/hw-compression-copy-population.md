# Compressible allocation population: primary-source check

`state=API_COPY_LEGAL_EFFECTIVE_ILC_UNPROVEN; SM_REWRITE_REQUIREMENT=NOT_FOUND; new_write_variant=NONE; GPU=NONE`.

Question: does populating GENERIC VMM memory through `cuMemcpyDtoDAsync` enable compression, or must the bytes be rewritten by SM/L2 stores first?

**Public evidence does not establish a Hopper D2D-versus-SM requirement.** It supports the allocation opt-in and ordinary exact-copy semantics, but does not promise that the particular payload copied by that API becomes effectively compressed. Do not turn this absence into either “CE definitely compresses these lines” or “an SM rewrite is necessary.” Keep the bounded D2D-populated allocation policy unchanged and scope any result to it.

|primary source checked|what it actually establishes|what it does not establish|
|---|---|---|
|[Hopper inline compression guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression)|Per-allocation eligibility; automatic algorithm selection including no compression; conditional bandwidth benefit|No initialization-engine requirement, D2D-specific guarantee, per-write trigger rule or guaranteed pattern/ratio|
|[CUDA VMM compressible-memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory)|Device support query, opt-in allocation flag, actual granted-property query|No post-memcpy SM-store requirement or API giving effective compression state|
|[CUDA12.8 Driver `cuMemcpyDtoDAsync`](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__MEM.html)|Copies the specified bytes between device pointers in the supplied stream; API reports success/error|No compression outcome/engine-selection contract in this function's documented description|
|[NVIDIA `cudaCompressibleMemory/saxpy.cu`](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu)|Lines65..85 use H2D `cudaMemcpy` for compressible input buffers on the SM8.0/8.6 branch; lines140..150 restrict those architectures to that population path and use an initialization kernel on other architectures, including9.0|The Hopper branch demonstrates allowed SM stores. It does not say SM stores are mandatory or that Hopper D2D copy loses compression. The other branch uses H2D, not the exact D2D Driver API in our helper|
|[CUDA guide on explicit copies/copy engines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)|Some CUDA memory-copy operations may use copy engines; API names alone are not a universal engine-execution guarantee|No ILC guarantee for `cuMemcpyDtoDAsync`; do not label it observed CE execution without an actual trace|

The sample's constant1.0 SAXPY data and architecture branch are demonstrations, not a measured comparison of H100 D2D versus SM writes. Old Ampere forum reports or community interpretations of that branch do not supply an H100 contract and were not used as support for one. nvCOMP/Blackwell decompression-engine documentation addresses a different feature and does not resolve inline compression population.

Current packed pilot directly establishes: GENERIC was granted, D2D/D2H byte roundtrip passed, original RQ bulk-TMA consumed the data correctly, and complete outputs/graph replays passed before cleanup failed. These establish API/addressability/correctness for the payload. They do not establish that ILC compressed any data, and invalid cleanup keeps the pilot's timings diagnostic-only.

Native lab should therefore record **`population=cuMemcpyDtoDAsync_then_stream_synchronize`**, `requested/granted` allocation properties and exact-byte readback, without inferring a hardware ratio. One future correctly cleaned-up negative native trial would close that fixed allocation/population policy; it would not be proof about every possible producer path. That practical scope does not justify automatic initialization-method trials.

**Decision:** no new SM rewrite kernel, memcpy variant, cache-policy manipulation, profile or GPU job is admitted from this source check. A different population mechanism would require positive primary documentation or a separately reviewed measurement question, rather than intuition that the sample's init kernel must be essential. Root's ownership-fixed helper gate remains the immediate blocker; this review changes no frozen shader, allocator or native lab source.

# Conditional native-BF16 compression coverage: unpooler128 Lt

`state=CONDITIONAL_HOLD; target=unpooler_m128; policy=NATIVE_ALLOCATION_ONLY; implementation=NONE; GPU=NONE; official_slots=NONE`.

**One credible target exists:** the existing tuned cuBLASLt fallback for unpooler128, using precisely its selected layout/algorithm and changing only immutable BF16 weight allocation. It was the actual official pw36 winner for this shape. A correct but flat packed12 GENERIC pilot closes that packed policy, not this feature's behavior on a different byte representation. Native BF16 has more statistical redundancy; hardware compression effectiveness remains unknown. This document does not admit a second experiment: wait for the frozen packed pilot result, independent critic review and root's named handoff.

## Why this target, rather than another kernel

Frozen pw36 source SHA256 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Official job`e22a4b9d693d4902`:

- `lt unpooler m128 ... best 2:66.18.1.0.0.2.35.0.5`: layout2, config `[66,18,1,0,0,2,35,0,5]` in parent `lt_cfg` order `{ID,TILE,SPLITK,REDUCTION,SWIZZLE,CUSTOM,STAGES,INNER,CLUSTER}`. This is historical evidence; a fresh preparation's code/config must be recorded and frozen within that process.
- Later RQ bake selected **r6**,22.88us versus best RQ22.72us under its sequential margin. Actual measured official call was one `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNN`,18.6us/CUPTI and19.103us benchmark. It is an already successful native-BF16 implementation, not a new no-decode kernel.
- Shape is X128x4096, W8192x4096, output128x8192 BF16. W contains33,554,432 BF16 elements,67,108,864bytes; X1,048,576bytes and output2,097,152bytes. No router precision exception, packing codec, geometry or reduction rewrite is needed.
- m256 in this same job chose Torch in the Lt search and has a different algorithm/kernel context. It is not an automatic alternate target. FMT0 exists in GV, but starting there would change the kernel family/format relative to the incumbent and require another comparison. Do not queue it as part of this plan.

No read source/history found a native-BF16 GENERIC allocation trial. Existing plain-BF16 fallback performance versus packed kernels is not a hardware-compression closure. Conversely, no compression ratio or achievable gain follows from that novelty.

## Exact weight storage and selected pointer

Parent source landmarks: `LTCPP` near2075, `_tune`2229, `_ltsearch`2279, `prepare`2373, `_v12`3863. Device algorithms live in the loaded cuBLASLt library; parent source plus runtime library identity and captured kernel identity are needed, not just a Python SHA.

Prepared storage:

|key|representation/shape|use|
|---|---|---|
|`st['unpooler_raw']`|original BF16 W, contiguous[8192,4096]|raw Lt layouts0/1, Torch fallback, validation, source for other prepared packs|
|`st['unpooler_t']`|`W.t().contiguous()`, BF16[4096,8192]|pretransposed Lt layouts2/3|
|`st['unpooler']`|`(_pack(W),None)`|GV/older fragment kernel, **not Lt**|
|`st['lt'][('unpooler',128)]`|layout*100000 + plan-local algorithm index|existing tuned Lt handle; index is process-local|
|`st['rq'][('unpooler',128)]` when present|lossless12bit RQ selected plan|higher-priority `kernel_fn` dispatch; unchanged by cloning raw BF16|

`lt_mm` computes `lay=code//100000`, `wp=(lay&2) ? wt.data_ptr() : w.data_ptr()`. Layout0/2 describes column-major operands with W as A; layout1/3 uses row-major descriptors with X as A. The `lay&2` mask means W has already been transposed in storage:

|lay|selected storage|native shape/stride in Torch|operation in existing Lt descriptors|
|---:|---|---|---|
|0|`*_raw`|[N,K], [K,1]|column-major W with transpose|
|1|`*_raw`|[N,K], [K,1]|row-major X times transposed W|
|2|`*_t`|[K,N], [N,1]|column-major Wt-as-[N,K], no transpose|
|3|`*_t`|[K,N], [N,1]|row-major X times stored[K,N], no transpose|

**For historical winning layout2, cloning only `*_raw` is a no-op for the measured Lt call.** Clone only the actual selected storage; retain the untouched counterpart. Do not alter `_pack`, RQ `p['lay']`, escape arrays, X, output, Lt workspace or descriptors. The inactive pointer argument can remain ordinary because existing `lt_mm` does not read it under that fixed layout.

`lt_mm` passes the same existing plan/algorithm, `CUBLAS_COMPUTE_32F`, BF16 input/output, alpha1/beta0, existing64MiB workspace, and current Torch CUDA stream. A cloned base pointer remains subject to that algorithm's alignment requirements. Record layout and all nine config fields plus pointer alignment; success is checked at runtime. Do not retune algorithms after allocating or assume that pointer equality of descriptors proves kernel equality. [NVIDIA cuBLASLt API/configuration](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul).

## Reuse WeightClone through a byte view; no native C++ extension rewrite

Read-only allocator snapshot `candidates/hw_compression/allocator.cpp` SHA256 `dc75f7b5524d7e6840a3565c127138073d8f7e51b3e607a57bfd14e90a93b2e2`; adapter `fff0379c21cb8467d73143a9014f26847cd41818da3802daba2376a2ac2acf00`. Source is owned/frozen elsewhere; do not change it here. Its validation checks CUDA/contiguous/uint8/nonempty/no-grad, then copies `src.numel()` bytes. Despite the packed-W wording in its error string, it has no RQ encoding dependency.

Plan-only pseudocode, outside capture, with exact existing selected code:

```python
code = st['lt'][('unpooler', 128)]
assert isinstance(code, int) and code >= 0   # otherwise SKIP this Lt-only policy
lay = code // 100000
key = 'unpooler_t' if lay & 2 else 'unpooler_raw'
src = st[key]                              # already contiguous BF16 prepared storage
src_bytes = src.view(torch.uint8)          # reinterpretation, no precision conversion/copy
owner_none = WeightClone(src_bytes, current_torch_stream_address, False)
owner_generic = WeightClone(src_bytes, current_torch_stream_address, True)
none_bf16 = owner_none.tensor.view(torch.bfloat16)
generic_bf16 = owner_generic.tensor.view(torch.bfloat16)
# Case-specific shallow state copies replace ONLY key, and retain their own owner.
# Original st, selected algorithm, all other pointers and all dispatch tables remain intact.
```

For raw[8192,4096], uint8 view is[8192,8192] with stride[8192,1]; inverse BF16 view restores[8192,4096]/[4096,1]. For transposed-prepared[4096,8192], bytes are[4096,16384]/[16384,1]; inverse restores[4096,8192]/[8192,1]. This is byte reinterpretation, **not `.to(torch.uint8)`**. No post-clone `.clone()`, `.contiguous()` materialization, numerical cast or new transpose is allowed. Assert identical data_ptr between owner tensor and BF16 view, exact original shape/strides/storage offset, full67,108,864byte span, and equal contents. [PyTorch dtype views](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html).

CPU proof performed using installed Torch2.13.0+cpu, no CUDA initialization: all65,536 BF16 bit patterns were reinterpreted as uint8, exact-byte cloned to separate storage and viewed back as BF16, for raw and contiguous pretransposed storage. Bitwise int16 equality and metadata equality passed; inverse view has the clone's pointer and mutations through the byte alias appear in the BF16 byte view. This covers every payload pattern including sign-zero/subnormal/NaN encodings **as a storage proof**, not a claim that GEMM accepts nonfinite weights. It does not validate VMM/CUDA backend runtime or effective compression.

Ownership: each case retains WeightClone, uint8 tensor, BF16 view, state, graph and actual output for every replay. The BF16 view shares the external tensor's Storage/deleter; it does not acquire independent CUDA ownership. The Region storage deleter retains the VMM allocation, but graphs still need an explicit owner. In cleanup, drain outstanding work, reset every graph, drop held output/state/closure/BF16 and byte aliases, collect references, then checked `owner.close()`. Existing close rejects TensorImpl/Storage aliases; deliberately retained aliases must make close fail safely rather than permit unmap. Never unmap while a live graph can replay. Do not use cudaFree or mutate/reuse reserved VA. Capture allocation/copy/query/cleanup stays outside timed calls, as in [hw-compression-roi.md](hw-compression-roi.md).

## Comparators: allocation controls plus actual incumbent

Use one frozen preparation and one existing selected Lt code; never rerun `_ltsearch` independently for the three allocations. Same input/output policy and exact existing `_v12`/`lt_mm` route in all native cases:

|case|callable/storage|purpose|
|---|---|---|
|native ordinary Lt|existing `_v12` with original state and selected non-None Lt code|unaltered algorithm baseline|
|native VMM NONE Lt|same route/code, selected weight pointer replaced by NONE byte clone|allocation/VA/page control|
|native VMM GENERIC Lt|same route/code, selected weight pointer replaced by grantedGENERIC byte clone|only proposed policy|
|actual pw36 incumbent, if different|unchanged `kernel_fn(data,st)`|practical comparator, including fresh RQ pick|

If the fresh actual `kernel_fn` already uses that exact Lt route, ordinary Lt and incumbent may be the same case after identity is proved. If it selects RQ or another route, **keep that incumbent graph separately**. Calling `kernel_fn` on a state that merely changes `*_t` while RQ remains selected would benchmark unchanged RQ, not native compression. Use the existing Lt fallback callable explicitly for the three native controls, and compare its GENERIC result against actual `kernel_fn`; do not delete dispatch entries to present a weakened incumbent.

If `st['lt']` is None, the fixed Lt-only policy is unsupported: stop/retain incumbent. Do not substitute Torch, m256, a seeded algorithm, FMT0 or a geometry search automatically. If the selected code/layout changes from historical layout2 during normal prepare, clone that **one actual selected** layout and freeze its full config before any allocation comparison; this is not retuning. A native Lt win versus itself that still loses to fresh RQ is a failed optimization for this target.

## Runtime admission and one bounded comparison, only after root approval

Preconditions before root can consider a handoff: packed pilot terminates and is harvested; result has correct allocation/graph/lifetime behavior but misses its packed performance bar (or root supplies another evidence-based reason); critic reviews a concrete native lab freeze; parent/helper/runtime identities are fixed. This document alone authorizes no source/lab implementation or GPU job.

If admitted, exactly one target and two new VMM allocations, NONE and GENERIC. One prepare, one fixed algorithm and one bounded comparison. No seed/algorithm/layout/geometry/initialization sweep. Unsupported capability/declined allocation, compilation/runtime/cleanup error or deadline closes this attempt. The earlier packed negative remains preserved.

Required gates:

1. **Storage/API:** real CUDA Torch build/import; exact device/context/stream; supported VMM/GENERIC; requested/granted flags0/1; mapped-pointer query; full native-byte snapshots and roundtrip; original W/X immutable. Ordinary pointer query failure means UNKNOWN, not NONE. A support/grant bit is not an effective ratio. Preserve original state on every failure.
2. **Existing math:** BF16 input/output, existing FP32 accumulation/algorithm metadata, no precision/epilogue/workspace changes. `_ltsearch`'s loose float32 `_close` alone is insufficient. Test complete outputs against FP64 and task oracle, normal/outlier inputs, all4096K one-hot columns, and immutable X for **every** native control and actual incumbent. Require the same exact/probe standard already used by the frozen labs; do not widen tolerance to admit this policy.
3. **Fresh graph correctness:** capture actual returned output, no added copy node; separate preserved case owners. Validate the actual timed graphs after timing using original/zero/fresh-normal/fresh-outlier/all-probe X at stable addresses, NaN-poison output first, require all output elements written. Include fresh-output capture validation and clone-owner/alias lifetime checks. Do not let an unprofiled replay hide a failed timed-graph output.
4. **One-source/kernel profile:** outside timing, capture names/count/grid/block for each native allocation using the same loaded parent extension/cuBLASLt version and identical config. All three must execute the same existing kernel sequence; pointer-dependent fallback, hidden conversion/copy or changed algorithm invalidates allocation-only attribution. Actual incumbent may differ, but record its real sequence separately. Existing TMA descriptors internal to Lt must be created by Lt for the supplied pointer; no opaque descriptor edits or changes to RQ's X map.
5. **Measurement:** preserve256MiB cold-L2 flush, output/workspace policy and existing event/graph method; discard standard warmups. Ordinary and actual incumbent F1 before/after (lower comparator); fixed rotated paired comparison across cases. Use the first pilot's bounded21round procedure rather than a new timing methodology. No profiler collection inside measured comparisons, no clock/cache/harness changes.
6. **Kill bar:** GENERIC must gain>=0.5us over **actual incumbent and ordinary native Lt**, both conservative F1 and median paired graph, plus>=0.3us over VMM NONE in both methods. All storage/numerical/profile/ownership gates pass. Otherwise close this unpooler128 native policy; no automatic retry. A gain over NONE does not excuse loss to incumbent. A local win is review evidence, not an official promotion or compression-ratio measurement.

## ROI: conditional GO for coverage, no claimed speedup

Historical empirical entropy around10.46bits/weight gives native16 much more statistical redundancy than packed12; it is **not NVIDIA hardware's codec or a measured ratio**. Under that illustrative idealized source model, native payload67.11MB could carry the same information in43.87MB, versus50.33MB for packed12 before sidebands. Whether ILC recognizes relevant local patterns, and whether physical traffic is exposed in this exact Lt pipeline, is unknown. Allocation opt-in alone earns no gain. CPU zlib/zstd ratios would not settle it.

For orientation only, a0.5us exposed bandwidth saving at3.8/4.8TB/s corresponds to1.9/2.4MB,2.83/3.58% of native payload. This does not price the actual Nvjet X reuse, tensor limit, layout-specific access or compression overhead. Native compute work is8.590GFLOP; hardware compression cannot eliminate it. No all-floor or multi-shape projection is supported.

The target is credible for **one conditional coverage trial** because an existing tuned native kernel was already official best for it and native bytes retain redundancy the packed probe removed, with no added per-weight instructions. If fresh ordinary Lt trails actual RQ by d, GENERIC must recover d+0.5us; that gap is measured in the same job, not inferred from historical profiler numbers. If the packed pilot instead finds capability/lifetime/correctness failure, this plan remains HOLD until that failure has a separately reviewed explanation; changing representation is not a repair for a broken allocator.

Same-official-job score context: pw36 score3212.415452, verify16 load65.510400us; a genuine0.5us load win there projects3208.316002 (−4.099450). No gain is measured here and none is transplanted from H100 to H200. Best separate rs32 job remains3200.615008; this conditional single-shape coverage does not establish `<3000us`.

Only this audit was added; CPU bit/view proof ran in memory. No implementation, candidate-folder edits, compilation, GPU launch, remote job or official submission was performed.

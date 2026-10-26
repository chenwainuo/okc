# Foreign-weight ownership correction: CPU forensics

`state=CPU213_AND_CPU28_OWNER_AND_LOCALS_PROOFS_PASS; GPU=UNRUN_PENDING_CRITIC; packed_v1=FAILED_VERSION`

Frozen v1 remains at `../hw_compression/`; allocator SHA`dc75f7b5524d7e6840a3565c127138073d8f7e51b3e607a57bfd14e90a93b2e2`, manifest`d2b0bfa79f19315497efe4dcebdff6e827773ed6bbf8e7caae98a9700f5b4713`. Its one pooler trial granted GENERIC1/NONE0, roundtripped~50MB exactly and passed8192-column/all3-case15actual-graph oracles, but terminated at the close alias guard. Unpooler was unrun. Diagnostic pooler gains−0.192us F1/−0.224us paired; there is no performance win and no full packed timing rerun is part of this ownership task. Raw JSON/logs remain unchanged. The failed version supplies evidence for continued hypothesis-driven iterations; it does not reject the whole compression family.

## Reproduced counts, not a guessed alias allowance

`cpu_probe.py` builds `owner_cpu.cpp` against CPU Torch and uses only64 toy bytes per owner. No CUDA API, GPU, data values or weight dump. Snapshots expose TensorImpl/Storage/Region owner counts, release count and Python referrer TYPE names only. Torch2.13 results are in `cpu-torch213.json`:

| State | TensorImpl | Direct Storage | Region | v1 close |
|---|---:|---:|---:|---|
| New owner, no getter |1|1|2|passes|
| `.tensor` getter result held |2|1|2|rejects genuine reference|
| Getter result deleted + gc |2|1|2|rejects despite preserved weakref wrapper only|
| True view held |3|2|2|rejects|
| View deleted + gc |2|1|2|still rejects|

Deleting state/cp/default callback references does not remove Torch2.13's otherwise unreferenced cached Python wrapper while the C++ Tensor handle remains. A Python weakref confirms that wrapper survives until the C++ handle is dropped. `gc.get_referrers` does not expose all C++ owners; an empty type list is not proof of no alias. Direct `Tensor.storage()` count is1; explicitly copying the Storage increments it to2. Thus the direct guard is not accidentally counting a temporary Storage copy in this runtime.

The [Torch Tensor caster](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/utils/pybind.h) ignores pybind return policy/parent in its custom Tensor conversion. [Torch2.8 THPVariable wrapping/preservation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/python_variable.cpp#L258) explicitly caches/restores wrappers and switches to a borrowed cdata representation during preservation. Root's actual `2.8.0+cu128` CPU-only result (preserved in `cpu-torch28-initial.json`) confirms getter-deleted TensorImpl1/Storage1, with wrapper alive, and v1 close SUCCEEDS. View/state cleanup also succeeds. Thus **Torch2.13 wrapper bookkeeping does not explain the actual2.8 GPU failure**. The corrected Region-owner guard's genuine-alias/release/idempotence controls pass on both versions; the real retained owner is identified below, rather than waived.

## Concrete v1 Python alias: locals snapshot

Frozen v1's finally block executes `if 'cp' in locals(): del cp`. That materializes the frame's locals mapping containing cp (the LAST VMM NONE p dictionary); deleting the fast local does not refresh that cached mapping. It can retain the real weight Tensor until the frame returns. `cpu_probe.py` reproduces that exact pattern on local Torch2.13 AND root's actual Torch2.8/Python3.11 (`cpu-torch28-locals.json`): even the corrected Region guard rejects Region2; replacing the pattern with initialized/unconditional `cp=None`, with no locals snapshot, permits Region0 after checked close. This is a genuine Python dictionary alias, not a harmless TensorImpl count. The critic independently reproduced two owners: first GENERIC closes, final NONE is blocked, and returning from the bad frame releases its cached snapshot. This identifies a concrete source cause for the failed GPU cleanup; the non-timing GPU gate still checks for any additional retained owners in capture/profiling.

Never call locals()/frame-locals inspection to diagnose this before close: refreshing the snapshot can erase the evidence. Counts and referrer TYPE snapshots remain bounded helper-function locals, and retain no referrer objects. Future lab cleanup initializes optional references and clears them unconditionally. Fixed allocator ownership logic deliberately continues to reject the stale dictionary if it survives.

## Minimal correction and safety proof

Corrected `allocator.cpp` SHA`7f2dea1ecdcbff20efd1daad1a4250f775bbf9cfb8b9a76a16f257c1268140f8` differs from v1 only in close ordering and a no-Driver `counts()` diagnostic:

```cpp
tensor = at::Tensor();
TORCH_CHECK(!region || region.use_count() == 1, "cannot close: storage owners remain");
if (region) region->release();
region.reset();
```

Drop this object's Tensor first, allowing its unreferenced preserved wrapper to disappear. Any genuine Tensor/Storage alias still keeps the from_blob deleter alive, and that deleter retains the Region. The independent object-owned Region reference must then be the sole owner before explicit VMM release. This is not a TensorImpl count1→2 relaxation and does not guess which count is harmless.

CPU negative controls: live same-Tensor, shape view, BF16 dtype-view, detached view, raw untyped Storage, state dictionary and default-argument callback all leave Region2 after the owner's Tensor is reset. Close rejects, with zero Region releases. After dropping the alias and gc, Region1 allows close and exactly one release; repeated close adds no release. Cached-wrapper-only close succeeds, and its weakref dies. A Python graph-reference model retains a Tensor through a callback, blocks close while live, then permits close after reset drops the reference. That model checks Python reference lifetime only; it does not simulate CUDA graph internals or asynchronous execution.

Failed close consumes the helper's own Tensor handle while retaining Region and any real alias's storage. Drop external aliases and invoke close again; do not try to reuse `.tensor` after a close attempt. Existing graph reset/drain ordering is unchanged: raw graph addresses are not storage references, so this guard cannot prove every graph user is gone. The explicit Case owner must destroy all graph execs before close. No allocation, initialization, kernel, precision, timing, flags, geometry or performance policy changed.

## Build, interface and pending admission

`build_helper.py` performs an actual corrected Driver/from_blob/pybind host-module build/import against local CPU Torch+CUDA12.8 headers, rejects CPU input before Driver initialization, and proves normalized exact equality to frozen v1 outside the close change/counts method. `helper-build.json` records the passing build and zero CUDA API calls. Source hash equality covers all original allocation/copy/context/cleanup operations.

```bash
python3 proj_rsi/candidates/hw_compression_owner/cpu_probe.py --json /tmp/hw-owner-cpu.json
python3 proj_rsi/candidates/hw_compression_owner/build_helper.py
```

For root's actual Torch2.8 image, the first command ran without `--gpus`; its source contains no CUDA API/header/link calls and all tensors are CPU. Both original root records are copied byte-for-byte as `cpu-torch28-initial.json` and `cpu-torch28-locals.json`; the first establishes the version distinction/alias controls and the second confirms the exact locals pattern. Do not claim GPU success from CPU proofs.

Consumer API: `adapter.extension()`; module `WeightClone(uint8_contiguous_cuda_tensor, current_stream_address, compress_bool)` exposes `.tensor`, `.info()`, `.counts()`, `.close()`. Native BF16 callers must drop every byte/dtype view, state/cp alias, captured output/graph/callback reference before close. Constructor behavior remains v1, including stderr-only destructor diagnostics on failed construction; any retained-resource warning invalidates a future trial. Error handling must not silently fall back after CUDA context faults.

`gpu_lifetime.py` is a proposed root-only non-timing gate using fixed helper7f2 and the actual unchanged frozen v1 Case/profile functions. One actual prepared pooler128 RQ pick, GENERIC+NONE; it records getter/drop, bit-copy/query, Case state, capture/replay/profile, graph reset, state/output/cp drop/gc and checked-close counts/referrer TYPES. It validates actual eager/normal/zero/fresh-outlier graph outputs and immutable weights, and has a separate same-weight Tensor/BF16-view/rawStorage negative control that must block release while preserving readable aliases, then release after all drops. No F1/paired/performance samples, new kernels or timing claim. Existing frozen prepare still performs its original bakes to establish the actual selected dispatch.

Proposed root command after critic review: `python3 proj_rsi/candidates/hw_compression_owner/gpu_lifetime.py --json /okc/logs/hw-owner-lifetime01.json`, inner180s/outer240s. No author GPU launch. Actual counts after all resets must prove that no genuine owner remains; any additional alias becomes evidence for the next ownership fix, not permission to weaken the guard. Native allocation work remains behind this gate. No parent edit, official slot or integration is authorized here.

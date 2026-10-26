# Hardware-compression ownership diagnosis / correction review

`state=CPU_LOCALS_CAUSE_AND_REAL_GPU_LIFETIME_GATE_PASS; storage_guard=FAIL_CLOSED; native_v1=VALID_NO_GAIN; packed_v1=INVALID_UNCHANGED; family=CONTINUE_MECHANISM_DRIVEN_ITERATIONS`

Scope: independent read-only diagnosis of frozen v1 cleanup failure and review of `hw_compression_owner` CPU model/helper. Reviewer changed no allocator/lab/source, launched no GPU/remote job, and wrote only this audit. Parent/author own implementation and any later probe. The earlier packed-W benchmark remains INVALID_CLEANUP_FAILURE; no corrected ownership model retroactively validates it or justifies rerunning it for a clean exit.

## Frozen failure and version-specific evidence

V1 allocator SHA`dc75f7b5524d7e6840a3565c127138073d8f7e51b3e607a57bfd14e90a93b2e2` checks TensorImpl count1 and Storage count1 **before** dropping its own `at::Tensor`. Runtime stopped at this guard after graph reset/state/output/cp cleanup. The generic error text did not record exact live counts; the GPU log alone cannot prove which owner survived.

Initial Python Case/measure/profile review found that `Case.close` resets graph exec first, clears out/hold/state; outer finally clears every case, nominally deletes cloned-p alias, then closes clones. The timing function returns only numeric arrays; helper validators retain no weight tensor on successful return; actual `_rqcall` takes tensors as transient pybind arguments. The initial review missed that the deletion itself calls `locals()` and leaves a cached dictionary alias; the later exact source-pattern proof below resolves that miss. A retained raw CUDA graph would not necessarily appear in TensorImpl counts, so correct graph ordering remains independently required.

The CPU from_blob/property model reproduces the same guard error **without CUDA or a graph under local Torch2.13**. Fresh owner has TensorImpl1/Storage1/Region2 and closes. Reading `.tensor`, deleting all ordinary Python references and collecting GC leaves its wrapper alive and TensorImpl2/Storage1/Region2. V1 rejects that cached-wrapper state. State/cp/callback deletion yields the same result. This proves a version-specific overstrict pre-drop check in2.13; **it does not establish the failed Torch2.8 GPU trial's cause**. Exact2.8 CPU results below contradict treating the cached wrapper as that cause. CUDA graph cleanup must not be weakened.

Primary version-specific checks: [Torch2.8 TensorBase](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/core/TensorBase.h) returns Storage by const reference, so the direct `tensor.storage().use_count()` is not accidentally counting a temporary Storage copy. [Torch2.8 Tensor pybind caster](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/utils.cpp) forwards returns through THPVariable_Wrap and ignores the supplied return-value policy/parent. [THPVariable ownership preservation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/python_variable.cpp) can retain/resurrect the Python wrapper and changes ownership direction. Local2.13 internals differ from2.8; its exact counts are not silently asserted for the failed GPU runtime.

## Independent CPU reproduction and alias controls

Author model `hw_compression_owner/owner_cpu.cpp`, SHA`37bd530c08e466e73d17603e1e6c8f05dff614c4b12c15a0827317f3a6461400`; observations `cpu-torch213.json`. Reviewer independently loaded the compiled CPU-only model from `/tmp/okc-owner-cpu-__2jggqb` and reran controls (CUDA calls0). An earlier frozen-model binary `/tmp/okc-owner-cpu-dw5tiigx` independently reproduced the original getter/deletion failure.

|state before original close|TensorImpl owners|Storage owners|Region owners|original guard|
|---|---:|---:|---:|---|
|fresh, no Python getter|1|1|2|closes|
|getter returns live same Tensor|2|1|2|rejects correctly|
|getter deleted+GC, wrapper internally retained|2|1|2|rejects despite no ordinary live alias|
|live view|3|2|2|rejects correctly|
|view deleted, retained base wrapper only|2|1|2|still rejects|

Explicit Storage copy in diagnostic `counts()` raises its count1→2 only inside that local diagnostic; this is distinct from the direct guard expression. A weakref staying alive is evidence of wrapper retention, not proof of a user alias. `gc.get_referrers` omits C++ ownership and cannot by itself certify freeability.

## Proposed correction: release own Tensor, then check actual storage owner

Fixed helper `hw_compression_owner/allocator.cpp`, SHA`7f2dea1ecdcbff20efd1daad1a4250f775bbf9cfb8b9a76a16f257c1268140f8`. Independent diff against frozen v1 contains only a counts diagnostic/binding and removal of the pre-drop TensorImpl/Storage uniqueness check. The substantive close ordering is:

```text
drop owner's at::Tensor
require Region absent OR Region.use_count()==1
checked Region.release(): original-context sync -> unmap -> release handle -> free VA
drop owner's Region
```

This is not allowing count2. From_blob's storage deleter owns a shared Region reference. As long as any standard Tensor/view/Storage alias retains the underlying storage/deleter, dropping the owner's Tensor still leaves Region2; unmap is rejected. Only when the storage/deleter is gone does Region become1, the explicit allocator owner alone. The retained wrapper is released by dropping the own Tensor, rather than special-casing PyTorch internals. Allocation/copy/attribute/device code and checked Region cleanup are unchanged. Author's actual fixed helper CPU build/import/ABI report is `helper-build.json`; its build is real but invokes no Driver API.

Reviewer independently tested fixed-model negative controls: same Tensor, view, raw Storage, callback, plus **NumPy zero-copy view and DLPack capsule**. Every live alias rejects close with own Tensor undefined, Region2 and release count unchanged. After deleting the external owner, Region reaches1 and checked close releases exactly once; repeated close is idempotent. Standard Tensor aliases remain readable/writable after the rejected close, demonstrating their backing storage was retained. Cached-wrapper-only close succeeds and its weakref clears. Author additionally covers BF16 dtype views, detach, state dictionary, default-argument callback and Python graph-reference-model reset; those results match the same mechanism.

Important API semantics: a failed close after owner-Tensor drop is a **closing** state, not rollback to an intact clone. Existing external aliases remain valid; the owner cannot safely expose a new tensor or enter another measurement. A later close may release only after all external owners disappear. A genuine-alias failure still invalidates a lab result; it is not an invitation to force-release or ignore the error.

## Graph and runtime conditions stay separate

Neither Region count nor Tensor reference count tracks arbitrary raw pointers or native CUDA graph exec addresses. A Python callback model is not a CUDA graph lifetime proof. Keep explicit Case ownership and order: drain submitted work -> destroy/reset every graph exec -> drop state/output/view/storage aliases -> close external allocation. Context synchronization drains queued work but cannot make a future replay against unmapped memory safe.

A future bounded lifetime probe should retain an ordinary Tensor alias while checking rejection, confirm alias data still valid, then reset the real graph and drop all owners before requiring a successful close. It must not deliberately unmap a live raw-pointer graph to test failure. Record counts/cleanup success and actual mapped properties. No optimization timing or native-BF16 extension is admitted solely from the CPU model.

Current recommendation: **CPU design/alias-guard PASS in both versions; concrete v1 locals alias reproduced and controlled on exact2.8; GO for the separately reviewed non-timing lifetime probe below**. Native timing stays behind its runtime result. Preserve original invalid sources/results. Do not relax TensorImpl1→2, suppress cleanup errors or bypass remaining Region owners. A failed fixed version is evidence for a reasoned successor, not rejection of the entire compression family; this ownership gate makes no performance claim and is not an old packed-policy timing retry.

## Exact Torch2.8 CPU result: important contradiction

Reviewer read `results/proj-rsi/takeover/20261026/hw-owner-cpu28.json`, SHA256`ae0f7ed6244fd3c50b995238f49d50ba420796b1d737305efeeec93ae6f050c3`, actual Torch2.8.0+cu128 with CUDA calls0, same model source37bd530c…61400. Getter deleted+GC leaves wrapper alive **but TensorImpl1/Storage1**, and original v1 close succeeds. Deleted view and state/cp/callback cleanup also let original close succeed. The old GPU failure therefore has an additional owner/backend-lifetime condition absent from this CPU model. Calling the internal cached wrapper the confirmed GPU cause would be false.

Corrected storage-owner guard passes the same seven negative aliases on exact2.8: same Tensor, ordinary/dtype view, Storage, detach, state dictionary and callback all retain Region2 and block release. After deletion each reaches Region1 and closes exactly once; idempotent close passes. This supports the correction's fail-closed property across both Torch versions without explaining the original GPU owner.

The initial simple CPU model omitted the exact `locals()` deletion sequence. At that stage graph/private-pool retention, profiler/trace lifetime and an unobserved Python/native alias remained hypotheses. The subsequent proof below identifies the missed Python alias. A narrow **non-timing** GPU lifetime gate still records scalar counts after getter, copy/query, state creation, capture/replay, graph reset, profiler creation/destruction, output/state/cp drop, and checked close, to detect any additional runtime owner. The correction must still reject a genuine GPU Storage owner rather than bypass it. Do not launch the full native allocation timing lab before this gate succeeds.

Fixed helper source7f2dea1e…140f8 remains unchanged. Its earlier manifest SHA`78e65959146ef9a157519a716970377cad248bcd20936dad38ef43b41448cb68` preceded exact2.8 results; current proof-package manifest is recorded below. Native lab's structural routing/control review can proceed while runtime admission remains HOLD.

## Confirmed source-pattern cause: cached locals dictionary retains last cp

Frozen v1 finally executes `if 'cp' in locals(): del cp`. In its runtime's Python frame, `locals()` materializes a cached dictionary containing the last cloned-p dictionary, and `del cp` clears the fast local without refreshing that dictionary. The last `cp['lay']` is therefore a genuine retained weight alias. The simple getter/state CPU model lacked this operation and could not reproduce it. The guard correctly rejects this alias; changing allocator close ordering alone cannot remove it.

Independent reviewer CPU reproduction with two actual model owners confirmed the asymmetric failure: after clearing the cases and executing the exact bad pattern, GENERIC closes but last NONE remains Region2 and rejects; returning from the bad frame frees the cached locals dictionary and then permits close. An otherwise identical function using initialized, unconditional `cp=None` closes both. This used local Python3.10/Torch2.13, zero CUDA. No referrer/frame-locals inspection was performed before close: refreshing the locals dictionary can erase the evidence.

Root's exact runtime confirmation is `results/proj-rsi/takeover/20261026/hw-owner-cpu28-locals.json`, SHA256`e488ec9b8aff7fd62192ee031934a6aa8941bf528e9ad13d89446cb5ca641df0`: Torch2.8.0+cu128/Python3.11, CUDA calls0, unchanged CPU model37bd530c…61400. Stage `exact_v1_locals_then_del_reproduces_real_alias` records pre-close TensorImpl2/Storage1/Region2 and fixed-close rejection with Tensor undefined/Region2/releases13; the unconditional-None control closes with Region0/releases15. Thus the concrete v1 source bug is reproduced and controlled in the target Python/Torch environment. The GPU failure's original guard remains justified; no harmless-count allowance is warranted. Real GPU capture/profiler retention is still independently checked by the next gate.

## Frozen lifetime driver preflight: GO, lifetime only

`gpu_lifetime.py` SHA256`5e0dc4534b201fb9eb048fd4b2441bd10d5dff2cec2c689e02e9043a73422dac`; proof manifest SHA256`8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781`; fixed allocator7f2dea1e…140f8. Reviewer verified all11 manifest entries, Python AST, and no actual `locals()` calls. The earlier48d880 driver was superseded before launch by standard sys.modules registration of imported original adapter/lab, allowing actual Case source provenance to resolve; no pipeline/kernel change.

Admission is one root-scheduled ordinary-capability named container under the existing exclusive GPU lock, inner180s/outer240s. No F1/paired/performance samples or new kernel. Baseline prepare still runs its original selection bakes; those are setup, not benchmark evidence. Driver loads the actual frozen v1 Case/profile under an isolated module namespace, verifies frozen files, imports unchanged pw36 source2a067059…bccf, and records the actual selected pooler128 RQ. GENERIC and NONE must both grant/query their requested allocation attribute and preserve every source byte.

Separate same-W GENERIC negative control intentionally holds a real Tensor through rejected close, checks Region2 and exact readable bytes, then holds a BF16 dtype view plus raw Storage through another rejection and checks bytes again. After deleting all owners it requires Region0 and idempotent close. This validates the real GPU storage-owner guard, not just a CPU mock. A rejected close consumes only the helper's own Tensor; held aliases keep their backing Region, and no unmap is forced.

For each actual Case, eager oracle/input immutability, actual graph capture/replay and unchanged profile names are checked. The two profiles must be the same single RQ kernel. Each retained actual graph gets original/zero/fresh-outlier input mutations with NaN-poisoned output, complete oracle, input immutability and final W-byte invariance. Counts trace getter/drop, copy/query, state, capture/replay/profile, reset, state/output drop with cp deliberately live, unconditional cp=None/GC and checked Case.close. Count snapshots precede weakref observer accesses, which retain only referrer type strings. No observer calls frame-locals APIs.

Every real graph exec is drained/reset before state/out/cp references are removed and before any unmap. Both normal and exception cleanup retain the Region guard; real cleanup rejection invalidates the gate and is propagated. Finally cannot convert a rejection into PASS. Final Region0 for both clones plus counter0 is required. Native performance remains HOLD until the actual terminal JSON/log show `LIFETIME_GATE_PASS_NO_PERFORMANCE_CLAIM`, both releases, valid numerical/alias controls, no retained-resource warning and clean bounded termination. Additional failures should be diagnosed and iterated with preserved versions.

## Independent real GPU lifetime terminal review: PASS

Actual root guard365178, named container`okc-proj-hw-owner-lifetime01`, terminal2026-10-26T23:01:55Z, exit0. Frozen driver/helper hashes above unchanged. Archived `results/proj-rsi/takeover/20261026/hw-owner-lifetime01.json`10789B SHA256`8d8a78f24998da0945924152c83f2fead549bd380f6c1c0100fc475b254f92e8`; hostlog17456B SHA256`9b6a18120877bffcc96ffd42f5498c82f7cf591c373591faa2c0adfba8314155`. Actual Torch2.8.0+cu128; status`LIFETIME_GATE_PASS_NO_PERFORMANCE_CLAIM`, timing`NONE`. Independent parsed checks pass all22 traces/eight oracles and terminal consistency. Read-only SSH independently found guard365178 absent, named container absent, empty compute-process query and0MiB GPU memory. No retained-resource warning, Traceback, invalid-state or CUDA-error marker in the hostlog.

The negative control retained actual GPU Tensor then dtype-view/Storage owners through rejected close with Region2 and unchanged readable bytes. After dropping all external aliases Region1 became Region0 on explicit close; repeated close passed. Actual GENERIC/NONE packed weights granted/query-checked their requested attributes and roundtripped exactly. Selected pooler128 i1/KS4/N4096 profiled the same single`rq<128,5,1,0,0,0,3>` for both cases. Both eager outputs and six actual graph normal/zero/fresh-outlier checks pass, including poison overwrite/input invariance. Full W bytes and shared counter0 checks pass in the frozen driver.

Ownership traces discriminate the source cause directly: after graph reset plus state/output drop with final cp intentionally held, GENERIC TensorImpl1/NONE2; after unconditional cp=None both1, with no Python dict referrers. Checked generic close changes its Region to0; checked NONE close changes the other to0. Both tensors undefined/weakrefs dead at the end. Capture/profiler paths introduced no further surviving owner in this exact gate. This validates this reviewed real-GPU lifecycle, not arbitrary future raw-pointer graph users.

**Disposition:** GO to bind the separately reviewed native BF16 lab to corrected helper manifest`8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781`/allocator7f2dea1e…140f8. Native lab reviewed before binding is SHA`d72d01520dbef7e9434ce76a5307d066636bf659557f11df9cb6dc9f82633746`, manifest`6c54cdd1b8ef6fc104e1c4bc4c0725e4530987a1e483bf5c24557bb3614b6740`. Exact native selected raw/_t BF16 bytes, unchanged Lt code/config, explicit actual incumbent comparator, actual timed graph poison/oracles, full launch identity and graph-reset/alias-drop/checked-close gates were independently read. Binding-only manifest changes require final hash/command review; no allocation-performance conclusion follows from the lifetime gate. Frozen packed-v1 timings remain invalid and no original file/result is rewritten.

## Final bound native-lab preflight: GO

Bound `hw_native/lab.py` SHA256`0a4db150debff8dfd09d13d33fe97896a27ba39a272e1b7f00b71f7a803c8047`, native manifest`e4cf36ae4f24c30350ff5fe9d95fa59feb6b42648cce29e7d6057afb4b1fbe4d`; fixed helper remains manifest8f32b91e…c781/allocator7f2dea1e…140f8. Reviewer reverified native3/helper11 manifest hashes, independently diffed the bound source against the reviewed d72d snapshot, and independently AST-compared actual Case/selected_source/validate/graph_checks/fresh_output_checks/profile/profile_identity/measure/main. All are identical. Only exact helper-directory, manifest and allocator constants plus hash-only verification gate were added. Binding cannot silently substitute original failed allocator or another reviewed-looking folder. Author's binding-proof.json agrees with independent checks.

Exact root command reviewed:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/hw_native/lab.py \
  --helper-dir proj_rsi/candidates/hw_compression_owner \
  --helper-manifest-sha 8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781 \
  --json /okc/logs/hw-native01.json
```

GO for one named exclusive normal-capability container, inner300s/outer360s under root's existing host/container deadline and lock. One unpooler128 native BF16 allocation policy only: use actual chosen Lt code and its raw(N,K) or transposed(K,N) contiguous source selected by layout bit2. Copy exact source uint8 bytes through the lifetime-proven Driver helper and reinterpret back to BF16 with shape/stride/ptr equality; no numeric conversion, new matmul, repack, config search or weakened incumbent. GENERIC/NONE/ordinary all call existing `_v12`; fourth comparator calls unchanged actual `kernel_fn` state, including its chosen RQ policy. Original ordinary allocation compression remains UNKNOWN if VMM-retain cannot query it; that uncertainty is explicit and NONE is the controlled comparator. Requested GENERIC is an allocation property, not measured effective compression.

All4 cases must pass complete4096-column basis/FP64 and task normal/outlier oracles with immutable inputs. Actual retained timed graphs receive20post-timing original/zero/fresh-normal/fresh-outlier/all-probe checks with NaN poison, plus separate fresh-output captures. F1 uses21samples/series and the faster of each ordinary/incumbent before/after medians;21rotated four-way cold-L2 pairs use256MiB flush outside timing. GENERIC must gain>=.5us over BOTH ordinary native Lt and actual incumbent in BOTH methods, and>=.3us over NONE in BOTH. Profiled exact kernel name/grid/block/register/shared identities must match all3 native cases; hidden CUDA copies or changed Lt cfg invalidate. Full selected/original/allocation bytes stay exact, RQ counters0. Drain/reset every real graph, drop every alias and clear cases, then require checked idempotent close/Region0/undefined for both; any retained-resource stderr, runtime/validation/profile/cleanup failure invalidates regardless of nominal gain. Any local winner still needs independent terminal evidence review before integration/official validation. A nonwin informs the next version's hypothesis rather than rejecting the family.

## Native allocation v1 terminal evidence: valid experiment, no gain

Archived `results/proj-rsi/takeover/20261026/hw-native01.json`22081B SHA256`98f0747db443be1c8dae72bf578a9544362b3dd49f03376871d4dbb9e6958b5c`; hostlog11473B SHA256`bc1defa82cba1549f488b8ad6b01d5938419530edd41048aa6d4a4459bad99b8`. Actual guard366192/named container`okc-proj-hw-native01` terminal2026-10-26T23:07:54Z, exit0; status`CLOSE_NO_NATIVE_ALLOCATION_WIN`, checked cleanup PASS, no retained-resource/Traceback/invalid/CUDA-error marker. Reviewer independently recomputed every F1 median, conservative baseline, paired median, pair count and gain gate from raw samples, and parsed all numerical/storage/profile/cleanup gate fields. Read-only SSH independently confirmed guard366192/container absent. Parent verified0MiB/idle before launching a separate v2 trial, so current GPU occupancy is not used to infer native resource leakage. Parent owns raw trace archival.

Actual unpooler128 selected Lt code200014/layout2/`unpooler_t`[4096,8192], cfg[66,18,1,0,0,4,35,0,2]; source64MiB SHA`09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120`. GENERIC1/NONE0 both granted and pointer-query confirmed at2MiB granularity. Ordinary VMM-retain fails, so ordinary compression remains UNKNOWN. All3native cases run the identical`nvjet_tst_128x64_64x8_1x1_v_bz_coopB_NNN`, grid[1,128,1],block[384,1,1],168registers/213268sharedB. Actual incumbent selects RQ i1/KS2 and profiles`rq<128,5,1,0,0,0,3>`,grid[64,2,1],block384,168registers/144520sharedB. Instrumented profile durations are not substituted for cold unprofiled graph timings.

Reviewer also read the four actual remote Chrome trace files, not only the lab's derived profile array; one kernel and the identities above match each. Full trace SHA256: ordinary`9418fb37d662145c654943aa94e149d7b1a0145dc7b643e53e7d492cf0f115e9`, GENERIC`4ce50190bf72564dcbd014fd08edf0d98ee07a6cabcb59b6eda05fb62925cc84`, NONE`b61880f9a095c2228a93733c948eba641b1f7d832ae51167de22698947ee64db`, incumbent`f8085d37dac72517a12ab930fbb22c689c684ba8125f8d439198d060cc9eb486`.

|comparator to GENERIC26.623999700us|conservative F1 us|F1 gain us|paired gain us|GENERIC wins/ties/losses|
|---|---:|---:|---:|---:|
|ordinary native Lt|26.528000832|−0.095998868|+0.096000731|12/0/9|
|VMM NONE native Lt|26.559999213|−0.064000487|+0.096000731|13/0/8|
|actual incumbent RQ|24.064000696|−2.559999004|−2.623999491|0/0/21|

All4paths pass4096basis/FP64/normal/outlier;20actual timed graph mutations and4fresh-output captures pass, all selected/original/clone bytes exact and incumbent counter0. Counts show each BF16 view adds Storage2 while case live; dropping aliases returns Storage1, then checked idempotent close reaches Region0/tensor undefined for BOTH. This is a successful transfer of the corrected lifecycle to native BF16 views. It does not retroactively validate packed-v1 cleanup/timings.

**Interpretation:** this exact native allocation version does not improve incumbent serving latency and supplies no measured effective compression ratio. Small opposite F1/paired deltas versus ordinary/NONE do not meet either preregistered bar. No integration/official score claim. The hardware path/ownership is validated; future iterations need a distinct data/consumer/transport hypothesis rather than asserting compression unsupported or rejecting the entire family. Frozen version/results remain available for diagnosis.

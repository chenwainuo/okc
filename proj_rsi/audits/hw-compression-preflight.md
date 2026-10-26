# Immutable packed-W hardware-compression preflight — 2026-10-26

`decision=STOP_INVALID_CLEANUP_FAILURE; GPU=TERMINAL_EXIT1; pooler=CORRECTNESS_PASSED_TIMINGS_NOT_ADMITTED; unpooler=UNRUN; integration=NO; scope=selected_RQ_pooler128/unpooler128_only; source_changes=NONE`

Reviewer read frozen allocator/adapter/lab/preflight/manifest/README, actual `_rqcall`/`kernel_fn`, installed CUDA12.8 declarations and NVIDIA VMM/Hopper references. No GPU job, remote mutation, kernel edit or allocation-policy sweep by reviewer. Parent owns launch/cleanup/ledger; source author remains separate.

## Identity and build evidence

Manifest6 and all Python ASTs pass independent verification. Baseline pw36 SHA256`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf` is pinned before import. Frozen artifacts:

|artifact|SHA256|
|---|---|
|allocator.cpp|dc75f7b5524d7e6840a3565c127138073d8f7e51b3e607a57bfd14e90a93b2e2|
|adapter.py|fff0379c21cb8467d73143a9014f26847cd41818da3802daba2376a2ac2acf00|
|lab.py|a17a4ae1699a1c46284f19a7378861660db6e09ae0f67c0c3d9eb9ea1c2693af|
|manifest.json|d2b0bfa79f19315497efe4dcebdff6e827773ed6bbf8e7caae98a9700f5b4713|

Author's actual C++/pybind/from_blob helper was built/imported CPU-only at `/tmp/okc-hwcomp-cpu-hoya3gqg`; reviewer verified the binary/build-log exist and the manifest binds current source. CPU proof checks CUDA_VERSION12080, CUmemAllocationProp32B, NONE0/GENERIC1 and rejects CPU pointers before Driver initialization. It does **not** prove remote Torch2.8 CUDA ABI/runtime, VMM allocation, compression grant, TMA compatibility or benefit. Real helper build/import is first runtime gate. Adapter builds before importing pw36, preserving its prepare clock; no helper compile is inserted into a scored submission.

## Allocation API and exact-data gates

Only contiguous nonempty non-grad CUDA uint8 packed weights are admitted. Current CUDA context/device must match source; allocation/copy is forbidden in stream capture. `cuMemGetAllocationGranularity` minimum is queried for the actual property, rounded size is overflow-checked, and physical allocation/reserved VA/map/access span agree. Zero-initialized CUmemAllocationProp requests device-pinned memory with no exported handle and compression GENERIC or NONE. Driver functions are dynamically resolved with expanded CUDA ABI symbol names; libcuda remains loaded for deleter lifetime.

Actual `cuMemGetAllocationPropertiesFromHandle` is checked immediately after create; requested GENERIC is not treated as granted merely because the capability bit or create call succeeds. A declined grant/API failure stops, retaining original prepared fallback. [CUDA12.8 allocation-property documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/structCUmemAllocationProp__v1.html) requires querying granted properties. [Hopper inline compression](https://docs.nvidia.com/cuda/archive/12.2.0/hopper-tuning-guide/index.html#inline-compression) is lossless and content-dependent; allocation attributes do not establish a compression ratio or bandwidth gain. Packed c12 bytes may be poorly compressible.

Full D2D copy runs on the caller's current Torch stream, then stream synchronization completes before exposing the external tensor. The lab has already synchronized prepared weights, and retains source state. Tensor has identical size/strides/dtype/device; no arithmetic conversion or initialization kernel is introduced. Both clones undergo full D2H equality against a CPU snapshot, plus retained-handle actual-property query. `cuMemRetainAllocationHandle` reference is released after query. Ordinary Torch pointer query failure is recorded UNKNOWN, never NONE; ordinary GENERIC closes the target as already enabled. Explicit NONE control is independently queried0. No compression ratio is inferred from logical/mapped size or memory usage.

Packed W is a bulk-TMA global pointer in rq, not a stale pre-encoded W tensor map. Actual `_rqcall` passes `p['lay']` on every call; only this field changes in a shallow-copied selected-p dictionary. Module/index/KS/escapes/counters/table and all other values remain identical, with runtime identity assertions. Baseline state is intact. Non-RQ actual dispatch is skipped; no forced rq or substitute Torch comparator.

## Storage, graphs, streams and failure unwind

Region owns context, allocation handle, VA, mapped flag/span and cleanup state. `at::from_blob` storage deleter captures shared Region ownership; WeightClone separately retains Region and Tensor. GPU work does not acquire caching-allocator stream tracking for this external storage. The lab therefore explicitly owns allocation+state+graphs+actual captured outputs until all work is drained and every graph exec is reset. Context synchronization alone does not protect future replay; this helper is not a general production graph owner.

Normal cleanup order is correct: drain+reset all graph execs; drop output/hold/state aliases for all cases; drop local cloned-p alias; collect Python references; checked `clone.close`. C++ requires TensorImpl/Storage uniqueness before discarding its last tensor and Region uniqueness before unmapping. Repeated close is idempotent. It pushes the original context, synchronizes outstanding work, unmaps the full mapped span, releases handle, frees VA and restores context. [CUDA12.8 VMM API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__VA.html) describes these matched mapping/unmapping/release operations. There is no cudaFree or unregister of Torch-owned memory.

Partial construction initializes all ownership fields and transfers them only after successful map/copy/tensor creation. Unwind releases only acquired resources. On synchronization/unmap/release failure, cleanup retains remaining resources and propagates explicit-close failure rather than freeing possibly live work. Bounded container teardown is the final resource boundary. **Important result gate:** constructor unwind relies on Region's noexcept destructor, whose failed cleanup is reported as `HWCOMP cleanup retained resources` on stderr. Python may otherwise label that allocation attempt CLOSE. Any such warning invalidates the trial regardless of exit/status, including failed-allocation paths before timing. Context-destroyed/cleanup warnings are not successful release evidence. This was sent to root before GO.

## Measurement and correctness pairing

One frozen pw36 prepare; three actual `kernel_fn` states per selected target: ordinary, VMM GENERIC, VMM NONE. All share kernel/config/counters and differ only W storage. Each validates all K one-hot columns (pooler8192/unpooler4096), full-output FP64 random/outliers and task normal/outlier probes with input immutability. Untimed validation copies are absent from measured graphs.

Each case captures once and retains its real output. Same three graph execs serve F1 and paired measurements; no recapture or copy node. Common256MiB flush,5discarded cold replays each,21sample medians ordinary-before/GENERIC/NONE/ordinary-after, conservative lower ordinary;21three-way rotated paired rounds with equal flush and GPU-event boundaries. Events/flush/setup are outside graphs. Three distinct W pointers can have different address/page effects; requiring NONE control improvement reduces that confound but does not prove a physical compression ratio from one allocation per policy.

After timing, **all three actual timed graphs** receive original/zero/fresh-normal/fresh-outlier/all-probe X at fixed address, with NaN-poisoned actual outputs. All15checks require finite complete writes/oracle/input immutability; original X is restored. Full packed W bytes in original and both clones must still match the CPU snapshot; counters must return to0. Post-timing profiles must show exactly the same single rq kernel in every case; a fallback or changed shader invalidates timings. Unsupported allocation/TMA errors close or invalidate; no silent ordinary fallback is admitted as a compressed result.

Per-target bar:>=0.5us versus conservative ordinary in BOTH F1 and median paired difference, plus>=0.3us versus VMM NONE in BOTH. A missing control, failed oracle/profile/immutability, source drift or cleanup failure rejects a nominal win. A win is local review evidence only, not an official score or measured compression ratio. Original ordinary comparator remains the optimization target.

## GO scope and stopping rules

Reviewer sent GO for **one** named exclusive unprofiled experiment of frozen source, suggesting inner300s/outer360s under current inside-container flock and host cleanup guard. Parent selects exact bounds and verifies actual wrapper/container termination/GPU ownership. No Nsight/SYS_ADMIN option is needed for this allocation experiment; ordinary runner permissions suffice.

First runtime gates: actual CUDA-Torch helper build/import -> support -> actual grant/map/query -> exact byte copies -> same-source complete output -> same actual timed graphs/control -> post-state/cleanup. Stop on unsupported/declined grant, correctness/runtime/cleanup failure or missed gain bar. No initialization/layout/geometry/cache-policy/allocator-address search, automatic retry, precision change, incumbent edit, production integration or official slot is authorized. Any survivor requires independent raw result and terminal review.

## Independent terminal review — INVALID_CLEANUP_FAILURE, stop

Root ran frozen`hw-compression01`, actual guardPID362824/named`okc-proj-hw-compression01`, normal runner (no profiling capability), inner300s/outer360s. Host timestamps22:35:13Z..22:36:28Z (75s), real C++helper compile/link succeeds against Torch2.8.0+cu128; prepared pw36 completes55s. After pooler observations, checked close rejects a remaining TensorImpl/Storage reference and the process exits1. JSON top-level and target status are **INVALID_CLEANUP_FAILURE**, not a validated allocation win/loss. Reviewer independently found actual PID/container absent after terminal. This is a runtime ownership guard failure that preflight's static review/CPU build did not expose; the guard stopped before unsafe explicit unmap.

H100 device reports VMM1/generic-compression support1; actual GENERIC allocation is granted1, explicit NONE0, both50331648logical/mapped bytes with2097152B granularity. Ordinary native-Torch pointer query returns retain_result1/UNKNOWN, as specified. Both full packed-byte D2D/D2H roundtrips are exact; pooler snapshot digest`ede1975ef05672ae0cd45bfa9c121b8f3ffe3fc0fe61df003b6eb04c10e471ab`. This establishes mapped allocation/copy/attribute compatibility for the tested data, **not** an effective physical compression ratio.

Actual pooler selected variant index1,KS4,N4096. Every case passes all8192one-hot columns, full FP64 random/outlier validation and task normal/outlier oracles. All15actual-timed-graph fixed-pointer mutation/NaN-poison/input-immutability checks pass; all128all-probe rows exact in all3. Post-timing full W bytes remain exact in all3 and counters reset. Each profile is the identical single`rq<128,5,1,0,0,0,3>`kernel; no shader/fallback change. These passed checks do not waive cleanup's rejection.

Pre-failure timing observations are retained **as diagnostics only, not admitted benchmark evidence**:

|pooler quantity|us|
|---|---:|
|ordinary before|24.288000539|
|ordinary after|24.351999164|
|GENERIC F1|24.480000138|
|VMM NONE F1|24.351999164|
|conservative ordinary−GENERIC F1|−0.191999599|
|median paired ordinary−GENERIC|−0.223999843|
|VMM NONE−GENERIC F1|−0.128000975|
|median paired NONE−GENERIC|+0.032000244|

Reviewer independently recomputed medians/gains from all raw sample arrays/pairs; ordinary pairs show7wins/13losses/1tie. Even ignoring cleanup failure, none of the required0.5/0.3us two-method bars is met. Single profiler durations22527/21856/21857ns for ordinary/GENERIC/NONE do not override the paired event observations. **Unpooler was never reached**; do not extrapolate pooler diagnostics to it or claim a valid two-target compression benchmark.

Failure occurs in `Case.close`→`WeightClone.close` uniqueness check: `cannot close: external tensor/alias references remain; destroy graph owners and state copies first`. All graph reset/state/hold/out/cp-drop steps executed before that call, but a runtime alias remains. No `HWCOMP cleanup retained resources` warning appears in the archived log; there is still no successful checked-release record. Static reference reasoning was insufficient to certify exact Torch2.8 ownership counts. Author notified to investigate CPU-only with the original sources frozen; do not weaken the uniqueness guard or run more GPU benchmarks to hide the failure.

Archives under `results/proj-rsi/takeover/20261026/`:

- `hw-compression01.json`, SHA256`d98bf03549c70e8d28e7122c6a56adf3526daa356e13ef43a69a43a148fb267e`.
- `hw-compression01.host.log`, SHA256`bc2d9e249ee7986b7200f4acc4c52d310cb7c61517ac20f4a380f2085236dd2c`.

Disposition: **stop, no automatic GPU retry, integration, parameter/address/layout search or official slot**. Preserve successful API/numerical diagnostics and the cleanup rejection separately from performance admission. There is no favorable pooler signal to justify repeating the complete trial merely to obtain a clean exit; any future ownership repair/new measurement needs an independently reviewed purpose. Original ordinary state/incumbent stays intact.

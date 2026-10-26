# Fixed native Lt compression traffic diagnostic: independent preflight

`decision=VALID_COUNTER_CAPTURE_NEAR_EQUAL_TRAFFIC; GPU=TERMINAL_EXIT0; purpose=AGGREGATE_TRAFFIC_NOT_SPEED; allocator_and_shader=UNCHANGED; effective_W_compression=UNRESOLVED`

Reviewer changed no candidate/allocator/source, launched no GPU or remote job, and writes only this audit. Author's source plan is `audits/hw-compression-counters.md`; independent findings here do not overwrite it. Root owns scheduling, exact profiling capability/lock/deadlines, exports and terminal resources.

## Frozen package / metadata proof

|artifact|SHA256|
|---|---|
|manifest16|e4f66b6d321a08c8c28eec6c2cb171ee1d731e4f499a2c2f9b2a8a7bc4562596|
|driver.py|f3b72ddac3d26a148cb82c59bd86066523cc6879b4645884cc73632a306a5719|
|run.sh|4d40209a4f23c1f6b7d9dee049a1f6b92dd37b120d5d9352d20e80363a8a9ed7|
|parse_report.py|95091dae6d46db4426cff213ab37e656f3c49799fcd5d51e9ee75eb412ae3575|
|metrics.txt|efd404cdc7046b6a8fd055f7b6240dd4bbd4a3cc4d4123d882fd3fbd184da7fb|

Independent checks: all16manifest hashes, Python AST, bash syntax, frozen native0a4db150…8047/helper7f2dea1e…140f8 and bound manifests8f32/e4cf; decompressed full GH100 catalog SHA`0b9a7171c2babc824bf1f86d160ce6aa6521733c307eebccbb024caabcbc5f99`; all21exact requested metric names in the full catalog or installed launch catalog. Author's metadata collection was CPU-only installed2025.1.1/GH100, not a GPU capture. Runtime collection availability remains a gate. No speculative old-Ampere metric name is requested for Hopper.

Physical metrics are DRAM read/write bytes and sectors. L2 controls are total requested sectors/device-fill sectors/hit rate. Six ILC fields are actual GH100 input/compressible/output sectors and advertised ratio/success metrics; eight launch fields bind grid/block XYZ/registers/shared memory. Installed MemoryWorkloadAnalysis and MemoryL2Compression definitions support CC90 ILC naming and discuss compression/write-access behavior. Those definitions do not make ILC a dedicated weight-decompression counter. Metadata query success is not evidence of compression success.

## Fixed API/shader/control correctness

Driver reconstructs exact native-v1 transposed unpooler W: raw[8192,4096]BF16→contiguousWt[4096,8192],67108864bytes/SHA`09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120`, input seed8851. Requires actual Torch2.8.0+cu128/cuBLAS12.8.4.1. It imports frozen parent2a067059…bccf and native Case/oracles/lifetime source0a4db150…8047. Only GENERIC and VMM NONE allocations differ, with exact query/grant1/0 and full byte copies checked; D2D+stream synchronization population is unchanged from the valid native-v1 performance trial.

Reviewed actual parent LTCPP: `lt_add` uses native public algo-init/config/AlgoCheck, FP32-accumulator/noTF32 checks and existing64MiBworkspace. Internal `ltplan` may enumerate metadata heuristics but no timed selection/algorithm grid/prepare occurs. Exact layout2/cfg[66,18,1,0,0,4,35,0,2] is appended and recovered through all9`lt_cfg` fields. Process-local code index can differ; configuration is fixed independently of it. `_v12` takes this selected Lt path, and neither case calls general `kernel_fn` fallback. Actual export must match the native-v1 Nvjet shader/grid/block/resource controls.

Each case validates all4096basis columns/FP64/normal/outliers, input invariance, real graph five-input poison/mutations and a fresh-output capture before profiling. Each selected replay is then NaN-poisoned/flushed outside NVTX, replayed once inside its unique range, synchronized and immediately task-oracled/input-compared before any later replay. Postprofile actual graph checks, exact original/Wt/allocation bytes and unchanged config are retained. This prevents an unprofiled replay from repairing the observed output. Both actual captured outputs remain owned and distinct through profiling.

Cleanup reuses lifetime-proven native Case order: synchronize, reset every real graph, drop all state/view/output/owner references, clear cases and loop variable, collect GC, then checked idempotent owner.close and require Region0/tensor undefined for both. No locals()/frame snapshot. Exceptions persist invalid state; cleanup cannot change a prior failure to diagnostic success. Constructor/destructor retained-resource stderr still invalidates regardless of nominal status. No force-release of live aliases.

## Reviewed runtime and CPU import

GO for one root-owned named exclusive profile-capability container after any active job's named container/context drains, inner180s/outer240s, existing host/container deadline/lock/cleanup. No functional/source dependency requires V3 before this diagnostic: native/helper/config and process-local Lt plan are separate from V3; ordering is scheduling only. Subsequent V3 still uses its own warmup/flush/before-after/paired controls. SYS_ADMIN is scoped to the diagnostic option, not global driver policy. Exact command:

```bash
bash /okc/proj_rsi/candidates/hw_compression_counters/run.sh
```

Script uses kernel replay/graph-node profiling, all-cache control, no profiler clock forcing, target-processes all, two explicit NVTX includes`compression_generic/`/`compression_none/`, launch-count2/kill no and the exact21metrics file. Child compiler processes have neither those ranges nor selected GPU launches; actual export still must contain only the two expected nodes. Setup/copy/flush/oracles/cleanup are outside ranges. Internal replay passes are not bounded by launch-count; record actual pass count. Save/restore/cache instrumentation conditions differ from unprofiled timing, so every profiler duration remains diagnostic.

Raw CSV CPU import **must** use installed supported`--print-units base`, with`--csv --page raw`. Parser requires two raw rows plus unit row, exact case-specific NVTX tags, expected Nvjet name/grid[1,128,1]/block[384,1,1]/168registers, equal per-case launch/shared resources, all physical/L2/launch numbers finite/nonnegative, and DRAMbytes=32×sectors in each direction. It reports signed NONE−GENERICread delta and L2 controls; missing optional ILC values stay unavailable, not invented zeros. ILC raw zero-denominator values do not prove no compression; retain raw counters alongside explicit limit text.

**Parser limitation identified:** strict unit-label checks cover read-byte/read-sector columns only. Write/L2/launch fields are numerically processed without all expected unit labels being enforced, and fixture corruption checks only exercise the read-byte unit. This does not invalidate raw capture; mandatory base-unit import and independent full-unit inspection are required before any reported byte/sector/resource interpretation. Do not state that this parser alone proves every column's unit. Any parser/import repair is CPU-only over the same report, not grounds for a GPU rerun.

## Interpretation gates / next question

GO is execution only. Require `DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED`, two immediate profiled oracles, full graph/byte/config/Region0 gates, clean stderr/exit/termination and a fresh importable report with exactly both tagged expected kernels and finite relevant counters. Missing/unsupported physical data or changed kernel/config/resource identity prevents attribution. Optional unavailable ILC fields can coexist with a valid aggregate DRAM comparison if physical/control gates pass; preserve uncertainty.

NVIDIA defines DRAM byte counts from actual memory-unit sectors×32 and distinguishes L2 compression as a unit. This supports an **aggregate traffic** observation, including X/output/cache/metadata/re-read effects. A positive NONE−GENERICread difference is not a whole-W compression factor, confidence interval, serving speedup or H200 result. Report both read/write totals and L2 request/fill/hit controls. Changed L2 reuse can explain part of a traffic delta, so do not assign every byte to compression.

ILC activity/success plus lower reads supports effective traffic reduction under this fixed pair, while unchanged latency remains an independent performance result. Similar reads/eligible ILC activity with low success supports weak compression for these weights/patterns. Zero ILC activity in a readonly kernel cannot distinguish incompressible data from population or counter-path eligibility. A future known-compressible positive control could test that remaining question through unchanged D2D/API/shader, but it is separate, not authorized by this preflight and not a challenge performance claim. The family stays open to a specific measured successor rather than flag-based promotion or family-wide rejection.

## Independent actual capture / units / resource reconciliation

Actual container`okc-proj-hw-native-counters01`,2026-10-26T23:31:04Z..23:31:19Z, exit0, exactly2tagged Nvjet kernels/3passes each. Driver`DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED`, both immediate profiled oracles pass, all4096basis/FP64/task checks pass,10pre+10postactual graph poison/mutation/input checks plus2fresh-output checks pass. Original/raw/cloned Wt/X remain exact, cleanup checked/idempotent/BOTH Region0 and tensors undefined. No retained-resource/error marker. Reviewer independently confirmed named container absent by read-only SSH; root confirmed0MiB/idle before the separate V3 launch. The latter job's current memory is not evidence of a diagnostic leak.

Archive directory`results/proj-rsi/takeover/20261026/hw-native-counters/`:

|file|bytes|SHA256|
|---|---:|---|
|hw-native-counters01.ncu-rep|70456|3c27d29098f468732d9252af46b763d8f21712255885a0b5a0a062ed09a542d8|
|hw-native-counters01.raw.csv|16923|6419d1abec7e739da168c24681e35b41a090bb573597e3bc04c642776fc53045|
|hw-native-counters01.json|11271|d6882aacd97def3c35bdf4ba97e806b76c9185decd5569da816aee621ca15bb3|
|hw-native-counters01.host.log|769|abdd62034a1e2fb52c8dcab88ec72ab6332e66bb825c571a3ea4e3aaced17819|
|hw-native-counters01.analysis.json|2989|abd1ee71ce076c5e0cc7489b4873efcba1cd7b60f87f4c272a1a6aec9801bedc|

Raw wide CSV has309columns, one units row and exactly two actual rows. Reviewer checked **all physical/L2/resource unit labels**, their finite nonnegative values, both32Bbyte/sector identities, exact NVTX cases and shader/config/grid/block/register/resource identities, all driver numerical/immutability/Region0 fields, and independently reran parser/reconciled its entire result with saved analysis. Read/write are`byte`/`sector`, L2counts`sector`, hits`%`, registers`register/thread`, shared`byte/block`; grid dimensions unitless and block dimensions use the installed`block`unit convention. ILC count/ratio fields have empty unit labels as advertised, success uses`%`. This supplies the independent unit check missing from the parser alone.

**Shared-memory question is resolved by collected fields, not guessed:** BOTH kernels have dynamic213268 + static0 + **driver1024** = total214292B/block, allocated214400B/block. This is driver-reserved shared memory, not1024static kernel bytes. The native-v1 profiler/driver expectation213268 exactly matches the exported **dynamic** field; comparisons against NCU total must include driver overhead. Shader`nvjet_tst_128x64_64x8_1x1_v_bz_coopB_NNN`, grid[1,128,1],block[384,1,1],168registers match both controls and fixed cfg. Actual process-local code200008 differs from native-v1's200014, while all9config fields/layout2, Wt hash, cuBLAS wheel and extension SHA5acd39f2…efb7 are identical; plan index is not configuration drift.

|aggregate counter|GENERIC|NONE|GENERIC−NONE|
|---|---:|---:|---:|
|DRAMread bytes|68174080|68174592|−512|
|DRAMwrite bytes|2435584|2395392|+40192|
|DRAMread sectors|2130440|2130456|−16|
|L2requested sectors|4244812|4286931|−42119|
|L2device-fill sectors|2130454|2130459|−5|
|L2hit rate%|25.87|25.78|+0.09points|
|ILCinput sectors|65549|65549|0|
|ILCevaluated-compressible sectors|0|0|0|
|ILCoutput sectors|65549|65549|0|
|ILCinput rate / success%|1 /0|1 /0|0 /0|

Signed read saving512B/0.0007510129% is a near-equal aggregate observation, not evidence of a meaningful compression effect. Including writes, GENERIC moves39680B MORE total DRAM traffic in this one pair. Do not turn either tiny delta or L2lookup variation into a weight-specific ratio. Reported ILCoutput-compressible-data ratio is raw0 in both, but there are0evaluated-compressible sectors: the denominator has no informative eligible data, so semantic achievement for compressible W remains UNKNOWN despite a finite numeric0. These ILC events do not establish whether original D2D population compressed weights or whether the Lt read path exposes decompression events. No serving speed/official score conclusion. Preserve this valid observation, then choose a specific feature/population positive-control hypothesis if its information value warrants a successor; the family remains active rather than being rejected from this version.

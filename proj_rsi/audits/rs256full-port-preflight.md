# RS256 full-K port probe: independent preflight

Reviewer `/root/proj_evidence`, 2026-10-26 UTC. Source and local CPU proof only; no GPU launch, remote job, source edit or commit.

## Decision and frozen identity

**Initial preflight GO, superseded by port01 setup rejection below.** The runtime guard correctly rejected occupancy before any probe kernel/sample; no port-model result was obtained. Initial review found no DCE, buffer-lifetime or barrier-convergence defect but missed the host opt-in/query ordering. This admits no full matmul, dispatch change, absolute throughput ceiling or score inference.

Source SHA256 `cc46c06739966806e4bd86ca078824e6d768068e8bac9426937b0f2b5fbf1774` matches `preflight.json`. Independent rerun of `preflight.py` without build/JSON output passes codec/address/coverage checks. Existing CUDA12.8 compile/SASS record inspected: all three controls168 static registers, zero stack/spills; math controls32 HGMMA,36 LDS.128,2 WARPGROUP.ARRIVE across unrolled two-round code plus initial load; combined4 UTMALDG/2 UBLKCP sites. Reviewer did not repeat compilation. Root must compile exactly frozen bytes, preserve compiler diagnostics and binary hash, and reject warnings/C75/spill/stack or a source mismatch.

## Work retention and numerical checks

- Both math WGs contribute all32 FP32 accumulator registers/thread to external output:128CTAs x256 math threads x32 values=1048576 checked values per launch. Every value must equal256*iters; producer-only must produce zero. MMA source operands come from runtime global input through shared LDS/decode, not compiler-known constants. The compiler cannot replace the arithmetic from the host's all-ones initialization. Volatile TMA/LDS/WGMMA assembly plus the compiled instruction counts provide useful DCE evidence; source-level instruction count alone would be insufficient.
- There are128 output snapshots of the producer's entire81920byte slot; every byte is compared with the synthetic source. TMA completion waits occur each producer iteration. `writes` counts completed software iterations, not hardware performance-counter transactions; final identical snapshots validate contents and bounds, not distinguish every individual intermediate fill.
- All-ones X deliberately makes raw resident layout and SW128 producer layout byte-equivalent. Thus snapshot equality is valid for this synthetic case but does not validate arbitrary input indexing or a general matmul. CPU proof establishes byte coverage/bijection for the modeled addresses. No real escape path or general input probes are implemented.
- Output and snapshot stores follow the common384-thread rendezvous after both consumer/producer timing windows; they do not contribute bulk output-store traffic to the claimed shared-port measurement. Small stamp/progress instructions are still within the windows and are residual costs.

## Synchronization and bounds

Shared resident slot0 and producer slot1 are disjoint,81920bytes each. Alignment/control overhead fits164928bytes including worst-case alignment. Separate init/pbar objects are8byte aligned; seq/completed words are distinct and inside allocation.

Initialization performs expect+bulk into slot0, all threads wait for phase0, then meet a CTA barrier before role separation. Resident operands are never overwritten afterward. Producer reuses pbar only after its current phase completes, with matching `r&1` parity. Its snapshot readers meet the later CTA barrier after producer completion, so no read races the last asynchronous fill. No TMA barrier storage is reclaimed before all outstanding transfers finish.

Each math WG alternates A0/A1. After committing round r, wait_group1 retires the prior round before loading over its A buffer; final wait_group0 completes every accumulator before end stamps/stores. No early return bypasses the final384-thread rendezvous. All128 threads of producer WG uniformly execute setmaxnreg.dec40; each complete math WG executes inc232. Register redistribution total64512 fits the SM budget, and local compile recorded no spill. Runtime resource/occupancy checks remain decisive.

WG0 release-stores progress; producer acquire-loads it and permits at most a small lead. Consumers do not wait on producer slot1, so an underfed producer can finish later without hanging math; final rendezvous waits for completion. Producer's pace depends only on WG0 progress and iters, so with finite hardware completion the wait graph has no cycle. Production-only mode does not enter pacing, and producer-only mode has no progress dependency.

Arguments128 iterations/9 reps satisfy even-iteration loop bounds; maximum CLI bounds are explicit. Kernel grid and allocations are fixed128CTAs, all pointer extents fit. Runtime must report one resident CTA/SM, no local memory, exactly128 distinct SM IDs for each launch; a different device/placement or any failed byte/value/progress check makes the sample INVALID. Strict distinct-SM rejection may conservatively discard a scheduling outcome; it is not permission to reinterpret that outcome as a single-wave result.

## Timing provenance and interpretation gates

`clock64` differences are formed within a CTA/SM: consumer endpoints per WG and producer endpoints belong to the same resident block. Absolute clock offsets across SMs are not compared. Medians across CTAs/samples use per-round deltas, and combined overlap uses endpoints from that same CTA. The128 distinct-SM check prevents mixing repeated placements as a claimed simultaneous128-SM wave. Do not convert cycles using nvidia-smi1980MHz or compare these deltas directly with graph/CUPTI wall time or a board score. The1024/1536/896 values are conditional repository traffic/issue models, not independently established physical ceilings.

Nine rotated/interleaved samples per control follow two excluded warmups. All allocations/source data are shared across samples, no HBM flush, only80KiB unique source globally, reused by all CTAs and iterations. This intentionally measures hot-source delivery/shared activity. It excludes full-K streaming, cold start, real exceptions, output stores and end-to-end scheduling. Repeated identical B descriptors across two math WGs may receive operand reuse/broadcast; nominal WGMMA read bytes are not a measured physical-bank transaction count. A favorable result can refute the **naive nominal aggregate cost model for this workload**, not prove separate physical read/write ports.

Temporal overlap is span overlap and includes producer pacing idle time. `writes_per_consumer_round` is the minimum completion count observed at either consumer's end divided by iters; it checks cumulative completed fill supply, not continuous instantaneous bandwidth. Together with bounded producer lead this is useful to reject obvious underfeeding, but it does not identify a unique arbitration mechanism.

Apply README gates after all validity checks, inspecting individual samples/coverage rather than only final medians:

1. Any numeric/byte/state/occupancy/compiler/source failure -> INVALID, no port-model inference.
2. Combined overlap or completed writes/consumer round below0.95 -> INCONCLUSIVE on shared-port contention; producer did not supply the intended simultaneous traffic. Producer-only timing describes this limitation.
3. Resident<=1200cycles and combined<=1.10x with adequate supply -> evidence against the naive aggregate128B nominal model for this hot repeated-operand case; no proof of physical port topology or full-K speedup.
4. Combined>=1.35x, adequate supply, and near/above1536cycles -> supports material interference; TMA issue, producer bookkeeping, read scheduling and operand reuse remain confounders. It does not uniquely prove bank-port sharing.
5. Resident>1200cycles, intermediate ratio, inconsistent samples or insufficient supply -> record INCONCLUSIVE/insufficient simple model and stop. No geometry sweep under this task.

## Bounded execution/admission

Root owns one exclusive container-held GPU lock, compiles frozen source inside that ownership window, then runs `timeout30s /tmp/rs256full-port-probe 128 9`, with outer180s monitor/kill grace. No other GPU job. Compiler+binary output, source/binary hashes, actual GPU owner, process exit, all27 SAMPLE lines, three RESULT lines, ratio and terminal synthetic PASS must be archived. Every sample must satisfy value/byte/state/resource checks; accept neither partial output nor exit0 without required records. Runtime checks are fail-fast and return nonzero on invalid data.

Confirm named container/compute process termination after success, fault or timeout before releasing the GPU; host Docker-client termination alone is insufficient. Whatever the result, preserve this fixed probe as evidence and stop. Any future full-K kernel requires a separately reviewed implementation and same-run/same-unit tuned baseline; no source expansion is admitted by this preflight.

## Port01 terminal review: INVALID setup / no measurements

Root launch PID348717, container `okc-proj-rs256-port01`, README compile then inner30s binary128/9, outer180s. At20:29:02 reviewer observed no running container/compute process. Build all3 controls168registers,0stack,0spill, no C75/warning/error. Program printed `ATTR mode=1 regs=168 local=0 dynamic_smem=164928 ctas_per_sm=0` and `REJECT spills or occupancy`. There are zero SAMPLE/RESULT/ratio records. This branch calls exit3; host log is empty, so process-exit metadata remains root's launcher evidence.

**No bandwidth, sharing, full-K feasibility or absolute-ceiling inference is valid.** Rejection happened before any `probe<>` kernel launch, although CUDA allocation/memcpy/setup had already executed.

Source-order inspection identifies a host initialization defect: `main()` invokes `info<1/2/3>()` before `launch()`, while dynamic-shared-memory opt-in exists only inside `launch()`. Therefore occupancy is queried for164928bytes without prior opt-in. This likely explains zero occupancy; runtime before/after attributes are needed to establish it. Reviewer missed this ordering in initial GO and records the correction explicitly. The fail-closed occupancy guard prevented an invalid measurement from being accepted.

Root author is preparing a separately named host-only v2 that opts in before occupancy and prints before/after values. Preserve v1 unchanged as invalid evidence; re-review exact diff/frozen hash and compile gates before one fresh bounded probe. This is a setup-bug revision, not permission to change geometry/device code or tune until a desired result.

Readonly harvest2026-10-26T20:29:45.347562+00:00, source unchanged `cc46c06739966806e4bd86ca078824e6d768068e8bac9426937b0f2b5fbf1774`.

- `/root/okc/logs/rs256-port01.log` SHA256 `32e5a29fb9fc5b98da0dd4c86808bfe0be18f594a13132550be73ad0a92e17d1`.
- Build log SHA256 `6e3095d22b9cbf8414f53dd873b925929540be495aca2e129902269967a50584`.
- Empty host log SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## V2 host-only correction: GO for one fresh bounded probe

Frozen new file `port_probe_v2.cu` SHA256 `fe55f19ba18d6e213ec038f4d068a771aa6190766b5aa72926fe95966ffe2ade`. Original v1 stays unchanged at `cc46c067...` for invalid-run provenance. Independent exact comparison proves all device/launch/geometry source before `info()` and map/main after it are byte-identical. Only build comment and host `info()` differ.

The host helper now prints the pre-opt-in maximum, static shared bytes and device opt-in limit; executes `cudaFuncSetAttribute(MaxDynamicSharedMemorySize,164928)` **before** occupancy; refreshes attributes; queries occupancy; and rejects reported max<requested, local memory, or occupancy!=1. CUDA errors remain checked. This fixes the identified ordering without changing the probe's geometry, measurement, workload, thresholds or device instructions.

Reviewer independently compiled/linked v2 with local NVCC12.8 and ran the compile/SASS gate without GPU API/execution: PASS, all3 controls168registers/0stack/0spill/no warnings/C75. Math modes retain32HGMMA,36LDS.128,2WARPGROUP.ARRIVE; combined retains4UTMALDG/2UBLKCP sites. Original CPU address/codec proof remains valid under exact device-source identity and passes independently.

Author's reviewed `preflight_v2.py` additionally compiles both immutable sources and compares normalized instruction streams. `v2-manifest.json` records equality of all3 kernel streams, with normalized hashes: combined `a516ac7de53f43eecdadac4cbb4aa9f4c30c9806fa9f064707afaf89392393a3`; TMA-only `302c58cb81acaf973b16835927e5a7f43f1a59c519b7016dd77c35b5e4e29216`; consumer `70263293747dc6e2759e4d216de35b96c091b1d99faef9ed850ae79022eb50fe`. Reviewer reran v2 compile/SASS counts; author owns the full two-binary normalized comparison record. Runtime before/after attribute values remain unmeasured until root's fresh run.

**GO exactly one v2 run**, same128CTAs/128iterations/9reps, inner30s binary/outer180s monitor, exclusive named container. This is a reviewed setup-bug revision after zero v1 probe samples, not tuning. All original numerical/byte/overlap/clock/cache/interpretation gates and no-expansion stop apply unchanged. No physical-port, absolute-ceiling or full-matmul performance claim follows merely from the host fix passing.

## Port02 terminal independent review: VALID synthetic / INCONCLUSIVE ports / STOP

Root launched reviewed v2, launcher PID349453, container `okc-proj-rs256-port02`, same30s binary/180s outer bounds. Frozen source remains `fe55f19ba18d6e213ec038f4d068a771aa6190766b5aa72926fe95966ffe2ade`. At2026-10-26T20:33:05.295531+00:00 reviewer independently observed empty Docker/compute-app lists and harvested terminal logs. No other GPU job, source change or implementation was introduced by reviewer.

The host fix is now runtime-confirmed: every mode prints `max_dynamic_smem=49152` before opt-in,164928 after, static_shared0, device_optin_max232448, occupancy1, registers168/local0. The correct recorded transition is **49152 ->164928**, while v1 had reported **occupancy0** (these are distinct fields).

Remote compilation again has all3 controls168registers,0stack,0spill, no warnings/C75. All27 printed samples (9 unique rep IDs/control) have exact full accumulator checks,128 distinct SMs, and exact producer snapshots in TMA/combined. Two earlier warmups/control are excluded from printed timing statistics but still run the same validation checks. Program ends its explicit synthetic PASS. The host log is empty, so root's launcher metadata owns process exit confirmation; complete terminal output and absent CUDA owner are independently observed.

Reviewer recomputed medians from all27 SAMPLE records:

|control|consumer cycles/K256|producer cycles/80KiB|span overlap|completed writes/consumer round|
|---|---:|---:|---:|---:|
|RS_LDS_DEC|1303.680|0.203 diagnostic empty producer|0|0|
|TMA-only|0|1460.758|0|0|
|COMBINED|1308.969|1648.902|0.99995|0.77344|

Combined/resident=1.004057 (recomputed from printed medians1.00405698). This **does not refute** the aggregate model: producer completed only about99 of128 fills by the earlier consumer end. Every combined sample's completion fraction is0.76562..0.78125, below preregistered0.95, despite nearly full span overlap. Spanning time includes producer waits/slow delivery and is not adequate traffic supply. Independent of that failure, resident1303.680 also exceeds the preregistered1200cycle model-sufficiency gate. Both conditions require **INCONCLUSIVE on port sharing, STOP**.

This is valid negative evidence about this probe's ability to generate the intended concurrent load, not proof of independent read/write ports, a universal128B arbitration limit, a full-K ceiling, an end-to-end speedup, or that the proposed kernel must lose. Producer-only1460.758 and combined producer1648.902 identify a synthetic ingress/issue limitation worth recording; they do not authorize tuning the producer or implementing a full matmul under this task. No geometry sweep, further diagnostic iteration, implementation, port or official slot follows. Preserve v1 invalid setup and v2 inconclusive model result separately.

Original remote raw files `/root/okc/logs/rs256-port02{,-build,-host}.log`; root owns durable archive.

- Runtime log SHA256 `7a7c6fcf84b9509e409fa9b7a1715e57f9028c8103849ff6b7645a48b6856540`.
- Build log SHA256 `762a1c4e9ab49f74cb103adf52ac56726c3bba8ae677b2e60589d35db187e741`.
- Empty host log SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

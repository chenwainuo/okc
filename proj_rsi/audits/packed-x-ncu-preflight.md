# Packed-X focused memory/PC diagnostic independent preflight

`decision=GO_ONE_BOUNDED_DIAGNOSTIC; GPU=UNRUN; source=FROZEN; performance_claim=NONE; PC_ROLE_SUPPORT=PENDING_ACTUAL_SAMPLES`

Reviewer read complete root driver/frozen source/liveness and prior actual-JIT evidence; edits no candidate and runs no GPU/remote job. Current ITERATION.md. This diagnoses the new valid39.648us wholepipeline, not oldV1PCcauses or a rerun seeking benchmark credit.

Driver`proj_rsi/tools/packed_x_ncu.py` SHA256`a3e81cc0adece02ffbf7f28250bb2b798642b94ccdfc63762ea90054b9b74d4d`; exact manifest21`0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4`, source`2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a`. Driver verifies all21files/source, dumps exact JIT.so SASS before first math, saves artifact+hash and applies same narrow physical-A4 gate. Runtime exact122main/22pack regs/local0/opt-in occupancy>=1; dynamicshared103600 expected. Failures persist INVALID and propagate; no weakening after silent-copy hazard. Parentprepare/selector absent, so no profiler-induced route change or baseline-selection attribution.

One seeded W/X pair; exact lossless Wpack/unpack and escapedata, retained preprofile full W/packedW/escape snapshots. Explicit fresh1MiBXP/mainoutput/counter buffers; two warm calls and capture actual rqrun API, which always launches pack thenmain on currentstream. XP/out/X distinct, live through profiler. XPpoison0x5a5a5a5a and outputNaN plus256MiBflush happen outside selected NVTXrange. Range contains ONLY actual2nodegraph replay/sync. Immediately afterward, before any further replay: taskoracle on exact observedout, Xoriginal equality, everyXPword vs independent current-X permutation, W/pack/escape snapshots, counter0. Refchecker rejects nonfinite output, supporting poison-overwrite result. Original graph drains/resets in finally. Existing allK/20graph/arbitrarybit/freshworkspace performance-lab evidence remains separate and is not claimed rerun here.

Root-only one exclusive profiling container after byteplane/current context drains, inner180s/outer240s, scopedSYS_ADMINonly/existing lock/deadline. Exact command:

```bash
/usr/local/cuda/bin/ncu --target-processes application-only --kill no \
  --replay-mode kernel --graph-profiling node \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include profile/ --launch-count 2 \
  --section LaunchStats --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables --section SchedulerStats \
  --section WarpStateStats --section SpeedOfLight \
  --export /okc/logs/packed-x-ncu01 -f \
  python -u proj_rsi/tools/packed_x_ncu.py \
  --json /okc/logs/packed-x-ncu01-driver.json
```

Known-working2025.1.1 NVTXpush/pop trailing slash, graphnode/kernelreplay and SourceCounters PC sampling on H100. Exactly2selected **kernel nodes**, rx_pack thenrq_px, not2fullpipeline launches; internal passes may be many and must be recorded. `--kill no` permits immediate oracles after cap. No nested torchprofiler or selector tuning. Fresh report/exportprefix only. Actual unmodified clocks/cache/replay conditions recorded; no profiler duration or stallratio converted to unprofiled microseconds.

**Cache limitation:** kernel/node replay/all-cache control separately profiles MAIN after flushing caches each internal pass. This differs from real wholegraph's main immediately following freshly written/warm XP from pack. XPdata contents must be preserved/restored correctly, verified by immediate pack/output checks, but measured cache/traffic/timing is isolated instrumentation. Do not subtract pack latency or declare its main cache state identical to the39.648us benchmark.

ExecutionGO is not counter validity. Require fresh importable report exactly2correct kernels (pack256×256/22regs, main128×384/122regs/dynamic103600), both actual source/JIT hashes/physical gate, immediate task/fullXP/immutability/counter assertions, successful export+driver+exit and clean guard/container termination. Relevant memory/request metrics must be finite/units known; unsupported optional data staysunavailable, notzero. Require explicit **PCstall samples**, not merely dynamic instruction counts/source listing. Record samples/dropped-data/aggregation settings. Main math andproducer each>=200all-state samples; proposed exposed leading phase>=30 with source/CFGdependency mapping and compatible opposite-role backpressure. Otherwise phase attributionINCONCLUSIVE. Packcopy has no math/producer roles and is not subject to that denominator rule.

Main questions: remaining Xdelivery/address/load readiness versus decoded-Bfull waiting, producerWfull/escapedecode/sharedstores, WGMMA/group-retirement/control. Map new actualsource/PCs—no assumed carryover of oldV1SHFL cause. Longscoreboard at mbarrier/SHFL/HGMMA can reflect distinct dependencies; kernel-wide ratios do not assign PCs/phases. Primary support must combine actual sample locations, exact producer/consumer lifetimes and finite memory/request evidence. Correlated samples/minimum counts are coarse guards, not independent-sample CIs or a causal time partition. New version remains active for a concrete evidence-driven successor; diagnostic earns no serving/official score improvement.

## Actual terminal review: 2026-10-27 00:21:07Z

`capture=VALID; role_sample_gate=PASS; leading_source_phase=DECODED_B_FULL_HANDOFF; exclusive_cause=UNPROVEN; benchmark_credit=NONE; family=ACTIVE`

Root guard377801/container`okc-proj-packed-x-ncu01`, inner180/outer240. Archived host log records00:20:50→00:21:07, two selected kernels,34internal passes each, report export, `PACKED_COUNTER_DRIVER_DONE`, `TRIAL_EXIT ... 0`. Root separately confirmed no remaining context/container and0MiB. This reviewer launched no GPU/remote job. Exact session confirms planned flags, H10080GB/132SM/CC9.0/Nsight2025.1.1. Six optional CTC-link metrics explicitly unsupported; they are unavailable and unused. Relevant finite memory/launch/sampling data are present.

Independent SHA256 of archived artifacts under`results/proj-rsi/takeover/20261027/packed-x-ncu/`:

| Artifact | SHA256 |
|---|---|
| raw.csv | `de3dfe5d5154e405dbc3a604e582b8acf409091ff7dca6be5379e13b0a530176` |
| source.csv | `d054ec7a2d870e2086aeba99fe2384b7a7ead9576e94803ad26e5e1227649c1a` |
| driver.json | `f69b857e5efca33f6978a5041ed66afccedcd4396b64441a3bb844e2bb413b4f` |
| driver.jit.sass | `f954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f` |
| ncu-rep | `7298ba002e55ec2062501fffb226acd67c4b82bc827f453b52b0955d366d8688` |
| host.log | `e3066f0bcb1a91b7cd53941e2bc23703cabbe312ab3788e583bae027f6bacaf5` |
| session.txt | `31499bce8da2282ec38f9a69b0d3b96b43aa91634e39b623d0c1c3dd33a1f643` |

Actual loaded JIT hash equals the unprofiled trial artifact and passes the unchanged physicalA4 retirement checker. Driver source/manifest hashes match freeze. Immediate task oracle relativeL2`1.421e-04`, worstrow`2.633e-04`,12probe rows exact; selected-output poison overwritten, **all1MiB packed bytes exact**, originalX/W/packedW/escape unchanged, counterZERO and graph`RESET_AFTER_DRAIN`. No later repair replay is used to establish these checks.

| Actual export field | pack ID0 | main ID1 |
|---|---:|---:|
| kernel | `rx_pack` | `rq_px<128,8,1,0,0,0,20>` |
| grid / block | 256 /256 |128 /384 |
| registers/thread |22 |122 |
| dynamic/static/driver SMEM bytes |0/0/1024 |103600/0/1024 |
| total / allocated SMEM bytes |1024/1024 |104624/104704 |
| global LDG requests |8192 |262656 |
| global LDG sectors,32B |32768 |4196352 |
| global LDG hit/miss sectors |0/32768 |0/4196352 |
| DRAM read bytes |1051904 |51834112 |
| DRAM write bytes |0 |2362368 |
| L2 sectors |144209 |4818695 |
| eligible warps/cycle |0.08 |0.34 |
| issue active percent |4.13 |27.46 |

Units are raw base units (`byte`, `sector`, `register/thread`, `warp`, percent) and not display-scaled quantities. Main requests include escape loads and are consistent with four vectorX loads/stage; no claim that every request is X. Pack32768×32B=1MiB read sectors matches the complete pack mapping. **Pack DRAM writes0 does not imply absent writes:** selected allpacked-byte equality proves writes occurred; output may stay cached and this counter observes a narrower boundary. Main cold-cache replay is unlike freshly producedXP in the wholepipeline, so no cross-report profiler-latency ranking or summation/subtraction. Main diagnostic35.872us and pack4.448us are instrumentation data only.

### PC sample denominator and source mapping

Actual source page has explicit `# Samples`, all/not-issued and individual stall columns. Pack306/main3580 agree exactly with raw`smsp__pcsamp_sample_count`; aggregatedPCpasses2, interval6=2048cycles, buffer32MiB, droppedbytes0 for each. These are samples, not dynamic instruction counts. Main source reason columns sum3441 rather than3580 because this source-page schema **omits `warpgroup_arrive`**; raw includes139 such samples and3441+139=3580. That known omission must remain visible when discussing per-PC reasons. Missing reason labels are not zero and do not invalidate the explicit sample count.

Map main-relative PCs from actual SASS/source base`0x7225c37ad100`. Math region`0xa70..0x1da0` plus outlined mathwaits`0x2ee0..0x2fd0`; producer`0x1db0..0x2ed0` plus its outlined paths`0x2fe0..0x3150`; shared prologue`<0xa70`. Shared prologue82samples, math2387, producer1111, sum3580. Math/prod each exceed preregistered200. Outlined paths have0samples here despite positive dynamic execution; do not mistake that for unexecuted waits or unbiased sampling of polling instructions.

| Mapped phase / PCs | all samples | reason support | denominator |
|---|---:|---|---|
| math decoded-Bfull TRYWAIT/check branches: `c50,c90,f80,1000,11f0,1260,14c0,1540` |1027 |1012longscoreboard, all at the four result-dependent branches |43.0% of math2387 |
| math WGMMA-entry ARRIVE: `ce0,1040,12a0,15d0` |471 |469longscoreboard |19.7% of math2387 |
| producer Wfull TRYWAIT/branch: `2140,2160` |83 |74longscoreboard |7.5% of producer1111 |
| producer Bempty TRYWAIT/branch: `21f0,2200` |70 |58longscoreboard |6.3% of producer1111 |
| producer publication/fence/barrier: `2c10..2c70` |276 |177barrier +71longscoreboard |24.8% of producer1111 |

Largest branch PCs `1540`391samples/389longSB,`1260`328/326,`c90`161/160,`1000`139/137 read predicates from the decoded-Bfull mbarrier at offset`0x19030`; these are not originalX global-data loads. Producer publication`2c70`187samples has177barrier; `2c20`42(33longSB) and`2c60`40(38longSB) precede leader's Bfull-arrive`2ca0`. This supplies compatible opposite-role handoff evidence. Producer Wfull83 and Bempty70 are both supported but are smaller than publication; neither proves a full bandwidth ceiling. Decode firstLDS→LOP3 at`2280`48samples/47shortSB is a separate smaller supported chain.

ARRIVE at`ce0`244/243longSB and`15d0`227/226longSB exposes operand/fence readiness. The latter follows a future-retired-slot LDG at`15b0` while currentgroup uses a differentAset, so broad fence/scheduling interaction remains a plausible source-level investigation. Samples alone do not prove A-ring expansion beneficial or identify which outstanding transaction made ARRIVE wait. The leading mapped phase is decoded-Bfull check/handoff, **not proof that producer arithmetic alone starves math**: TRYWAIT result scoreboard latency, variable publication timing and synchronization are combined. Minimum30 leading-phase gate passes; exclusive H_P-versus-H_X causality remains unresolved, with measurable support for both handoff and A-readiness exposure.

Per-PC `Instructions Executed` sum8943381 for main versus raw`smsp__inst_executed.sum`7985591; pack94208 matches both. Multiple counter passes and data-dependent polling prevent treating this export pair as an exact fixed-work instruction decomposition. This reviewer does not silently reconcile, subtract or convert the difference to useful instructions/latency. Math and producer roles still have directly counted PC samples independent of that dynamic-instruction discrepancy. All percentages above describe sampled locations within a role, not fractions of runtime or realizable speedup.

**Next evidence use:** inspect the decoded-B publication/TRYWAIT handshake and separate A-prefetch/fence dependencies in frozen source before choosing one successor. Preserve full2node pipeline timing against actual incumbent and predecessors; any A-depth change needs physical register retirement/resource proof and remains a distinct hypothesis, not a result implied by these samples. No automatic profile rerun or wholefamily retirement.

**TRYWAIT interpretation addendum:** [PTX8.7 mbarrier.try_wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait) may block until completion or a system-dependent limit. A successfulfirstreturn can include producerwaiting; fewfailedpredicatebranches do **not** prove initiallyreadybarrier/intrinsiclatency. Predicated-OnThreadInstructions plus exactCFG can quantify observedfailedlanes, but /32intoexactfailedwarps needs provenuniformity. Sameaddress/phase andaggregatecounts alone do not supply that proof. Sourcephaseconclusionabove remainscoarse, notproducerstarvation/time attribution.

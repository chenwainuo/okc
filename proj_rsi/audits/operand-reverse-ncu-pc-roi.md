# Operand reverse: independent final PC-sampling interpretation

`state=VALID_COARSE_PC_EVIDENCE; causal_dominance=INCONCLUSIVE; policy=CLOSED_NO_RETRY; further_profiles=NONE; GPU_BY_REVIEWER=NONE; new_implementation=NONE`.

**Result:** candidate math is substantially exposed in X global-load delivery and register-transpose dependency/issue work. This now has PC association as well as the prior fixed-work proof. The “producer cannot publish B fast enough” explanation is not supported as the leading math-side exposure in this capture: mapped math B-full wait sites have2samples versus2388at transpose PCs. Producer still has material W-full, shared-W-load/decode dependency, store and publish work. Mixed backpressure and lack of a temporal trace prevent a unique global bottleneck or microsecond decomposition. Stop here; no third profile or reverse-policy retry follows.

## Capture identity: separate from the seven-section report

Files in `results/proj-rsi/takeover/20261026/`:

|artifact|SHA256|
|---|---|
|`operand-reverse01-ncu-pc.raw.csv`|`cd3e90d46208b6c7e27ef939ae8e65028e66be9206fe8794189a3d47744c6cc8`|
|`operand-reverse01-ncu-pc.source.csv`|`349e8145313ea1271da6857b25c915af977400e38be057f1a45fc992e47111d1`|
|`operand-reverse01-ncu-pc-driver.json`|`23ba6dd80090ad5bee2c04bc07bf368412f3e4b23dd89c9d9ff1d2ff8eab55e2`|
|`operand-reverse01-ncu-pc.session.txt`|`c998d4ba421bb9dcaa41b103cb208721dfbb137155ec55c3646d8eff006a7225`|

Nsight2025.1.1, CUDA13.0, H10080GB/CC9.0/132SMs, process34/context1/stream7. Command replaces the old seven sections with `--section SourceCounters`, retains kernel/node replay, `--cache-control all`, `--clock-control none`, NVTX`profile/`, launch-count2. Session created22:21:10UTC. Driver is `DIAGNOSTIC_COMPLETE` and `COUNTERS_ONLY_NO_SPEED_CLAIM`; both immediately profiled-graph oracles/input immutability pass. Source SHA`ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`, pw36 SHA`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.

|field|baseline ID0|candidate ID1|
|---|---|---|
|actual kernel|`rq<128,5,1,0,0,0,3>`|`rq<128,8,1,0,0,0,20>`|
|pick|`s4s5hz/KS2`, **FL3**|reverse/KS1, FL20|
|grid/block|64x2/384|128x1/384|
|diagnostic duration|24.160us|45.792us|
|all PC samples|2405|4844|
|not-issued samples|1776|3329|
|replay passes|5|5|
|PC-sampling aggregated passes|2|2|
|sampling interval|2048cycles|2048cycles|
|buffer/dropped bytes|33554432 /0|33554432 /0|

The previous seven-section report had **FL2 `s4s5z`**,25.088/45.856us and18passes. Do not splice its scheduler/throughput values into this sampling pass. The present FL3 matches the original trial's variant name, but profiler replay/sampling clocks/cache behavior still do not make this a benchmark replacement or permit decomposition of the original21.472us profiler call. Original complete-graph loss remains decisive and unchanged.

## Parsing and CFG mapping

Raw CSV: header, units row with blank ID, two data rows;378columns. Source CSV: two concatenated tables,1832/1024PCs,65columns. `Warp Stall Sampling (All Samples)` and `# Samples` agree at every PC. All source per-PC samples sum exactly to raw2405/4844; not-issued sum exactly1776/3329. Not-issued is a subset, never added to all-state totals. Source execution counters are warp-level counts independent of instruction predicates; sample counts are observations of warp state, not counts of executed instructions or elapsed microseconds. [NVIDIA source/PC metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#source-metrics).

The source table's per-reason columns omit `warpgroup_arrive`. Sum of the visible all-state reasons is short by171/47, exactly the separately exported raw `smsp__pcsamp_warps_issue_stalled_warpgroup_arrive` totals; not-issued difference141/23 likewise matches raw. This is a missing displayed reason column, **not dropped samples**. Preserve all-state totals and this residual rather than calling it zero or attributing it to X.

Role map derives from actual SASS branch targets and return targets, not lexical order alone. Candidate role split at`0x7ce3797adc00` branches producer to`0x7ce3797afec0`; math enters`...adc10` with USETMAXREG allocation216, producer deallocates64. Candidate math wait loops are lexically after producer EXIT, but return to math code, so classify them as math.

|role|baseline address ranges, inclusive|candidate address ranges, inclusive|
|---|---|---|
|math|`7ce233d3dc90..7ce233d40cb0`; outlined `7ce233d43af0..7ce233d43ba0`|`7ce3797adc10..7ce3797afeb0`; outlined `7ce3797b0f00..7ce3797b0fb0`|
|producer|`7ce233d40cc0..7ce233d43ae0`; outlined `7ce233d43bb0..7ce233d43c20`|`7ce3797afec0..7ce3797b0ef0`; outlined `7ce3797b0fc0..7ce3797b1010`|
|common/setup/unmapped|all other PCs|all other PCs|

The common bucket includes pre-split setup and initial predicated loads; do not arbitrarily assign those samples to a role. Baseline division helpers reached through CALL from multiple contexts remain common/ambiguous; their sampled count is0. No sampled common helper is relabeled W-full or B-empty. All named wait phases below have explicit distinct call-site/barrier-offset/return mappings in this binary.

## Role denominators and preregistered sample gate

|role, all-state denominator|baseline all /not-issued|candidate all /not-issued|
|---|---:|---:|
|math|2039 /1517|3173 /2154|
|producer|257 /179|1532 /1041|
|common/setup/unmapped|109 /80|139 /134|
|total|2405 /1776|4844 /3329|

Candidate both roles exceed the preregistered>=200all-state gate. Leading X transpose2388, direct X loads244, producer W-full153, B-empty80 and production phases419/326/273 each exceed>=30. Math B-full2 and candidate producer startup1 are **too sparse to rank internally**; their low sampled exposure is not proof of zero cycle cost. No dropped data is reported. Sampling remains periodic/correlated; thresholds are qualitative sufficiency rules, not confidence intervals.

## Candidate math: X delivery and transpose are exposed

|math PC category|all-state samples|% of math3173|not-issued samples|long-SB|short-SB|MIO throttle|LG throttle|
|---|---:|---:|---:|---:|---:|---:|---:|
|SHFL.BFLY transpose|927|29.21|636|321|24|292|0|
|SEL transpose|1461|46.04|1047|619|672|0|0|
|transpose combined|2388|75.26|1683|940|696|292|0|
|direct X LDG sites|244|7.69|193|0|0|41|155|
|HGMMA sites|89|2.80|37|0|0|0|0|
|WARPGROUP sites|29|0.91|17|0|0|0|0|
|mapped B-full wait checks/branches/retries|2|0.063|0|0|0|0|0|

These rows are categories/subsets, not an additive time budget. Math setup before the loop has110samples, mainloop2977, drain/epilogue86; all outlined wait-loop samples0. The mainloop accounts for the transpose and X-load rows. Roughly82.95% of math all-state samples land in those two categories together, but that includes selected/not-selected/other states; it is not82.95% of kernel walltime or removable overhead.

Concrete X register dependencies, using actual generated SASS:

|sampled consumer PC|operation / input origin|associated samples|
|---|---|---|
|`7ce3797adf00`|`SHFL.BFLY R124,R106,...`; R106 lies in X LDG.128 R104..R107 (prologue`...ad3a0`, loop refill`...af160`)|307all,305long-SB|
|`7ce3797ae7d0`|`SEL R140,R66,R117,...`; R66 lies in X LDG.128 R64..R67 (`...ad440`/`...af1c0`); R117 is a shuffle of R67|214all,212long-SB|
|`7ce3797ae1b0`|`SEL R140,R73,R140,...`; R73 lies in X LDG.128 R72..R75 (`...ad420`/`...af180`)|142all,139long-SB|
|`7ce3797af1e0`|X `LDG.E.NA.128 R88,...`|53all,46LG|
|`7ce3797af300`|math `SHFL.BFLY` in transpose|49all,45MIO|
|`7ce3797af470`|math `SEL` in transpose|84all,82short-SB|

Long-scoreboard at a SHFL/SEL consuming X's LDG results is evidence for **X delivery dependency**, not an intrinsic shuffle-latency measurement. Short-scoreboard/MIO at other transpose sites supports additional register rearrangement issue/dependency exposure. The first category therefore favors H_XL; it must not all be renamed H_XT merely because the sampled opcode is SHFL. Conversely, direct LDG-site samples are often LG/MIO issue pressure, not waiting for their returned data. [NVIDIA stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).

The compiler issues B-full TRYWAIT early among transpose instructions, then consumes its predicate at branches`...ae420/aeaa0/af0b0/af7a0`. Include both check and predicate-consumer branch in that wait phase. All four consumer branches and outlined slow loops have0samples, while initial checks have0/0/2/0. Treating only SYNCS opcodes as waits would lose this distinction. The low exposed B-full sampling is consistent with producer publication being mostly hidden behind math X work in this capture; it does not prove the producer always leads at every stage.

## Candidate producer: several costs and mixed backpressure

Producer phase ranges map to distinct code, without reallocating common helper samples:

|producer phase, inclusive address suffixes|all /not-issued|% of producer1532|main associated states|
|---|---:|---:|---|
|setup `afec0..b0160`|1 /0|0.07|sparse|
|W address/index `b0170..b0230`|80 /25|5.22|wait39|
|W-full `b0240..b0250` +outlined`b0fc0..b0fe0`|153 /106|9.99|long-SB138,wait9|
|B-empty index `b0260..b02e0`|67 /26|4.37|wait45|
|B-empty `b02f0..b0300` +outlined`b0ff0..b1010`|80 /52|5.22|long-SB72,wait5|
|packed LDS/decode/escape `b0310..b0a70`|419 /271|27.35|short-SB226,MIO34,wait60|
|store addresses +STS `b0a80..b0cb0`|326 /247|21.28|short-SB92,MIO87,wait82|
|proxy/store fence +bar3/continuation `b0cc0..b0cf0`|273 /231|17.82|long-SB136,barrier113|
|publish/refill `b0d00..b0e90`|46 /21|3.00|wait30|
|loop control `b0ea0..b0ef0`|87 /62|5.68|wait43|

Distinct phases:

- W-full check`...b0240` uses`[UR11+0x19000]`, its predicate branch`...b0250` targets`...b0fc0` and returns`...b0260`:153samples total. B-empty check`...b02f0` uses`[UR5+0x19070]`, branch`...b0300` targets`...b0ff0` and returns`...b0310`:80samples. These mappings do not require a guessed helper label.
- Producer's largest decode-region single PC`...b03a0`, `LOP3 R40,R9,...`, has189samples/188short-SB. R9 is returned by preceding `LDS.128 R8` at`...b0370`. This is packed shared-W-load dependency at a decode consumer; it is **not proof of integer codec throughput saturation**. The region also contains scalar/vector preparation and escape repair.
- `...b0cd0` proxy fence59samples/51long-SB; `...b0ce0` deferred bar3 has89/85long-SB; its following branch`...b0cf0` has119/113barrier. Producer store/publish ordering and convergence are exposed. The consumer PC/state alone does not identify whether those delays originated in scalar-store drain, warp skew or other work.

Most W-full/B-empty samples occur at the branch dependent on TRYWAIT, rather than inside the outlined retry loops. Zero producer retry executions in one SASS-patched pass never established zero waiting; this report positively shows those wait phases. TRYWAIT can suspend in its first invocation. [NVIDIA mbarrier semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait).

B-empty80 exceeds the30sample gate and supplies consumer-backpressure association consistent with H_XL/H_XT. W-full153 is also material and exceeds B-empty; the producer is not a pure “ready and idle” role. Production/decode/store/publish regions remain substantial. Without temporal stage alignment, do not call one ratio the pipeline's global critical-path share or convert it to microseconds.

## Baseline associations, with the same cautious mapping

Baseline math full-ring waits (first TRYWAIT +predicate-dependent branches +outlined slow paths) contribute974all /911not-issued samples, including954long-SB. Baseline math total2039, so47.77% all-state samples occur in that phase. This full barrier gates packedW **and X** in the original pipeline; it is not a W-only or decode-only measurement. Baseline producer empty-ring waits/loops and payload issuance are separate from candidate W-full/B-empty roles.

Thus many baseline long-scoreboard observations arise from an asynchronous barrier predicate, not a direct global-load consumer. Comparing raw “long-SB share” across operands without PC/phase context would misattribute them. New report has no contemporaneous old-run bandwidth/issue counters; this sampling pass does not establish a numerical shift in memory/tensor capacity.

## Preregistered decisions and ROI boundary

|hypothesis|updated assessment|
|---|---|
|H_P producer production is the leading cause|**Not supported as the leading sampled math exposure.** B-full2samples; strong X-associated math work and B-empty80. Producer work remains real and could couple stages; no zero-cost claim|
|H_D specifically decode ALU/escapes|**Not established.** Producer decode-region hotspot is LDS-result dependency; store/publish costs and W waits are also material. No attribution of the full loss to decode|
|H_XL X global/L2 delivery|**Supported exposed association.** X-consuming SHFL/SEL940long-SB plus direct X LDG155LG samples; producer B-empty supports opposing backpressure|
|H_XT register transpose after X arrives|**Supported additional exposed association.** Transpose696short-SB/292MIO samples and prior exact4.194M fixed warp instructions; do not double-count its940X-load-dependent long-SB as intrinsic transpose|
|H_W W refill/coupling|**Material but dominance unresolved.** Producer W-full153, yet math B-full sparse; W delay need not be exposed to math at the sampled times|
|H_C synchronization/round scheduling|**Exposed producer publish work**, math final drain22barrier samples below30phase gate; no unique synchronization-first cause|

All-state sampling makes H_XL/H_XT the strongest **observed math-side** costs in this binary. It does not establish a unique global critical path, precise removable latency or gain versus shipped rq. Mixed producer waiting/production and periodic-sample aliasing preserve `causal_dominance=INCONCLUSIVE`; no additional sampling pass is requested. This is a useful finite diagnostic outcome, not a reason to keep a closed policy alive.

Maintain the closed policy and its21/21paired losses; do not tune KS/stages, bypass scalar stores, rewrite barriers or generalize from this capture. Any independently proposed W multicast/allocation mechanism still needs its own source-level novelty, bytes/issue/lifetime/convergence proof and actual-parent comparison. These samples do not supply a20us optimization forecast. No evidence transfers to actual m16 GV X PCs; that queue remains NO-GO pending relevant evidence.

No speedup or official projection changes: pw36 same-job score3212.415452, official m128 unpooler was Nvjet rather than this local RQ, and separate rs32 best remains3200.615008. Reviewer performed local CSV/CFG arithmetic and wrote this audit only; no GPU/remote jobs or source modifications.

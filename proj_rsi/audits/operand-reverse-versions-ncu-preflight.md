# Frozen reverse v1/v2 counter diagnostic

`decision=VALID_COMPLETED_TWO_VERSION_COUNTER_DIAGNOSTIC; performance_admission=NONE; GPU=TERMINAL_EXIT0; candidate_sources=UNCHANGED; phase_causality=UNRESOLVED`

Independent reviewer read the complete root-owned driver and frozen candidate/helper sources. No source edit, GPU launch, remote job or optimization by reviewer. This diagnostic compares the two exact versions already measured unprofiled; it does not run baseline selection or certify v3.

## Frozen identities / proposed execution

Driver`proj_rsi/tools/operand_reverse_versions_ncu.py` SHA256`ce6d3407101b191bae27150f6670691d0440185dc1586df483ee7fde67b09493`. V1 generated extension SHA`ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`, runtime regs168; v2 SHA`97664bdee7593d46e0493b9599062c37af150262d597002a722016f060548faf`, regs162. Both are exact unpooler128/N8192/K4096/KS1/channel64/384thread kernels,128CTAs,103600Bshared. W producer/output source is unchanged except uniform register allocation; actual compile/resource/numerical proofs are in the v1/v2 audits. Driver checks generated source hash before loading each extension, exact register count/local0 and opt-in-before-occupancy>=1. Result regs fields are verified expected values, not independent profiler launch metadata; export must supply the latter too.

Root alone may run one named exclusive normal-profile-capability container with inner180s/outer240s and existing lock/host cleanup. SYS_ADMIN is limited to the existing explicit profiling option; no global driver policy change. Exact proposed command:

```bash
/usr/local/cuda/bin/ncu --target-processes application-only --kill no \
  --replay-mode kernel --graph-profiling node \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include profile/ --launch-count 2 \
  --section LaunchStats --section MemoryWorkloadAnalysis_Tables \
  --section SchedulerStats --section WarpStateStats --section SpeedOfLight \
  --export /okc/logs/reverse-v1-v2-ncu01 -f \
  python -u proj_rsi/tools/operand_reverse_versions_ncu.py \
  --json /okc/logs/reverse-v1-v2-ncu01-driver.json
```

NVTX push/pop range`profile` uses trailing-slash filter syntax already validated in earlier installed2025.1.1 diagnostics. `--kill no` allows immediate task oracles after both selected launches. `--launch-count2` limits selected application launches, not internal replay passes; record actual passes. Section availability/collection is an actual CLI/export gate, not assumed from the command string. An unsupported flag/section, permission error or missing report is invalid setup/data, not a measured zero.

## Source correctness / lifetime / pairing review

No pw36 import or prepare; there is no profiler-influenced policy-selection drift. One fixed seeded W and X, identical lossless pack/table/escape data to both versions. Pack reconstructs all original W bits before capture. Call closures bind ext/out/cnt/empty/sink by default arguments, avoiding loop late-binding; packed/X/escape/table intentionally share immutable values. Each version has its own output/counter allocation. Both warm twice, then capture the actual call. Retained outputs are distinct and kept alive with their graph execs. Every selected range contains only one replay plus synchronization; poison,256MiB read flush, printing and task oracle occur outside. No nested torch.profiler collector.

After each selected profiled replay, before any additional replay, synchronize then check actual retained output against task oracle, X against original and that version's counter0. Output poison was set before the selected replay; task checker rejects nonfinite output, so the logged NaN-overwritten result is justified by that oracle rather than an extra unprofiled replay. Captured output is not replaced with a fresh validation buffer. Source-read-only W/packed/escape arrays are shared safely; **this driver does not post-snapshot/compare their bytes**, unlike the performance lab. Do not claim it ran that extra immutability gate. Frozen kernels contain no writes to those arrays, both immediate output checks are required, and their prior comprehensive numerical/byte checks remain separate evidence.

Normal completion resets both graph execs. Exception path persists INVALID then exits the bounded container; no external VMM allocation/unmap is involved. Owner/capture tensors live through profiling. Kernel replay may save/restore accessible device memory between metric passes and flush caches; it does not establish equivalence to every unprofiled cache sequence. `--cache-control all` requests cold state per pass, in addition to app's256MiB flush, and clock-control none avoids profiler clock forcing without guaranteeing constant dynamic clocks. All profiler durations remain diagnostic only.

## Measurement question / admissible interpretations

Known static trade: v2 replaces4vector128bit X loads plus32SHFL/32SEL per thread/stage with16scalar32bit loads. **Exact cache policies also differ:** v1`ld.global.nc.L1::no_allocate.v4.u32`, v2`ld.global.ca.u32`(SASS STRONG.SM). Both route128MiBlogical X/launch; v2's warp-stage has128sector instances/64unique32Bsectors, whereas the original vector route covers64instances. V2 removes4.194Mtranspose warp instructions but adds786432scalar-load warp instructions. Its unprofiled57.024us vs v147.232us demonstrates that removing transpose count alone did not improve time. This is a comparison of complete fixed policies, not a single-variable coalescing experiment: cache allocation/coherence, A-array retirement schedule and static register allocation also changed. Exact load qualifiers must stay attached to any counter interpretation.

This report can test aggregate **request/sector/cache/throughput** differences on the exact two kernels. For example: more L1 load requests/sectors with similar L2/DRAM transfer supports local reuse combined with extra frontend work; increased downstream transfers is consistent with ineffective reuse/locality. These remain kernel-wide observations: inherited producer escape loads and TMA traffic are also present, changed issue timing can change their hits, and no PC/source sampling section is requested. A throughput percentage is normalized to that unit/cycle convention, not a cross-unit additive byte count. Raw requests/sectors/bytes and elapsed/active definitions must be retained. No aggregate counter directly proves which PC/role caused the delay, physical compression behavior, a universal memory ceiling, or a speedup for v3.

Require fresh importable nonempty report with exactly two correct demangled kernel identities (`rq<128,8,1,0,0,0,20>` then `rq_v2<128,8,1,0,0,0,20>`), grid128/block384/regs168vs162/shared103600; both immediate driver oracles and counter/input checks; actual successful profiler/export exit; finite collected relevant memory/request/cache/scheduler metrics. Preserve missing/unsupported metrics as unavailable, never zero. Driver status`COUNTER_DRIVER_COMPLETE_EXPORT_REVIEW_REQUIRED` alone is insufficient. Root's actual guard/container must terminate and no resource leak/error remain. Counter ambiguity is an explicit learning result; any new implementation requires its own proof and unprofiled same-run incumbent comparison.

## Independent actual export review: valid and informative, no causal time partition

Guard369783/container`okc-proj-reverse-v1-v2-ncu01`,2026-10-26T23:27:34Z..23:27:50Z, exit0,32passes per selected kernel. Reviewer independently confirmed guard/container absent by read-only SSH. Source/driver hashes unchanged, both immediate profiled-output poison/task/input oracles pass with12exact probe rows, both counter0 assertions executed before the logged per-version PASS. Report imports to wide CSV933columns plus units row/exactly2data rows, with expected kernel/config/grid128×1×1/block384×1×1/registers168and162. Dynamic shared103600+static0+driver1024=total104624B/block for both. Planned103600 is the dynamic field, not total including driver reservation.

Archive`results/proj-rsi/takeover/20261026/reverse-v1-v2/`:

|file|bytes|SHA256|
|---|---:|---|
|reverse-v1-v2-ncu01.ncu-rep|264961|645e05e9b881f33a32c20f330ab100e8e32307fa72c27536591bcbe7e7f5615c|
|reverse-v1-v2-ncu01-raw.csv|67934|05257785c07758fbb9c4854670b0b0b40dace63e62dad6794bccdc74ef898751|
|reverse-v1-v2-ncu01-driver.json|945|b278cbd479d2d1490af27b8707ed05c0a1fc5d5f4cf5e939dec91536377438e6|
|reverse-v1-v2-ncu01.host.log|941|587cc617eb177935b8d1c7d20a0d6ce54fbaa504f7a9b69551ca7a8c2fbbdc0f|

Relevant physical/request/cache/scheduler values are finite and units checked. Host warns six CTC receive/transmit metrics unavailable; these are not used in this diagnosis and must not be called zero. The unmodified-clock warning limits timing interpretation rather than invalidating counters. No missing relevant memory metric or wrong role/kernel was substituted.

|kernel-wide measurement|v1 vector nc/no_allocate|v2 scalar ca|
|---|---:|---:|
|LSUglobal-load requests|262656|1049088|
|LSUglobal-load L1sectors|4196352|8390656|
|L1lookup-hit sectors|0|4194304|
|L1lookup-miss sectors|4196352|4196352|
|global-load L1hit%|0|49.99|
|global-load output wavefronts|524800|2102495|
|L2total requested sectors|6404010|6535628|
|DRAMread bytes|52152320|51472896|
|DRAMwrite bytes|2434560|2362880|
|issue-active% of sustained-active|30.03|25.93|
|eligible warps/active cycle|0.39|0.33|
|LGstall /issue-active ratio|0.34|0.23|
|MIOstall /issue-active ratio|0.96|0.09|
|long-scoreboard /issue-active ratio|2.87|5.70|

Independent arithmetic: requests increase exactly786432, matching the extra scalar-X load model. Sectors increase exactly4194304; **all extra sectors appear as L1hits, with misses identical**. Thus the half-sector reuse occurs in L1 under ca. DRAMreads fall679424B/about1.30%, not2×; this report does not support increased HBM traffic as the explanation for v2's regression. L1request/output-wavefront demand rises greatly even though upstream misses do not. That supports recovering efficient frontend delivery as a useful next design question, without proving that one queue/port caused the time loss.

Do not rename request growth a measured LG-throttle bottleneck: aggregate LG ratio actually falls0.34→0.23, while long-scoreboard rises2.87→5.70 and issue/eligible activity falls. Those per-issue-active ratios are not per-phase percentages, PC samples, independent observations or causal microseconds; reduced MIO can coexist with other longer waits. Producer polling/shared counters can also vary despite unchanged algorithmic W/shared payload, so aggregate wavefront changes are not a new source-byte accounting. Profiling durations46.112/59.872us remain isolated diagnostic observations, not replacements for unprofiled paired57.024vs47.232us evidence. Cache qualifier/staging/register differences remain confounded. V3's separately reviewed8B/no_allocate/pair-lane loader addresses request/conversion tradeoffs; only its own four-way unprofiled result can establish progress or an incumbent win. No official/full-submission claim follows from this diagnostic.

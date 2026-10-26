# operand_reverse — unpooler128 KS1 only

`state=CLOSE_POLICY_NO_SWEEP; trial=operand-reverse01; correctness=PASS; F1_gain_us=-23.520002142; paired_gain_us=-23.519998416; promotion=NO`

Terminal2026-10-26T21:49:17Z: complete-output candidate47.648us versus actual prepared pw36 baseline24.128us; both4096one-hot suites, normal/outlier oracles and all10actual-timed-graph mutation/NaN-poison checks pass, but21/21paired samples lose. Full Torch build/launch/opt-in/occupancy gates passed. **Closed: no integration, automatic rerun/sweep, m256 transfer or official slot.** See `RESULTS.md` and `proj_rsi/audits/operand-reverse-architecture-preflight.md`. Executable files and manifest stay frozen; the manifest's README digest is historical after this documentation-only closure. The remainder records the pre-run design/protocol.

Architecture/ROI: `proj_rsi/audits/m256-next-mechanism.md`. One fixed policy: M128,N8192,K4096;64 channels×all128 tokens/full K,128 CTAs,384threads/3WGs. X becomes register-A; W stays losslessly12bit packed, TMA-staged6slots×6KiB, decoded once by the producer WG into shared-B8slots×8KiB. Two cooperative math WGs process distinct64-token sets. One complete `{64,128}` plain TMA output store; no memset, global split-K, reduce-add, counters, or fallback path. Pooler/m256 are not implemented.

This distinguishes the closed pw19 SS transform path, where X was staged and read from shared. It also avoids reopening the failed o_direct_w GPR-W loader. Shared traffic model falls60→36KiB/k64 at this tile; global/L2 traffic does not fall. Four coalesced X vector loads plus32SHFL/32SELP per thread/stage are real costs, so no speedup is predicted.

Generated extension SHA256 `ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; device SHA256 `59f3b775d28410c23616a749ad58661112f9c6a1a959fec05333068ca9f3800a`. Inherited helper source is pinned to rs32 `a7b6e010…f5a35ad`; pack/dec1/escape repair/descriptors/WGMMA helper are unchanged. Host accepts only the one shape/KS1. X is a const pointer in the parameter struct; the unused X tensor map is not encoded. Host setup exposes opt-in-before-occupancy metadata.

First static gate passed without a flags/geometry sweep: inherited sm90a/O3/DNDEBUG/relaxed-constexpr flags; launch bounds384×1;168static registers (64512 initial CTA pool),0stack/0spills/no C75. Initial64/224 role allocation was invalid (65536>64512) and was caught before any GPU execution. Corrected64producer/216math requests63488, leaving1024 in the initial CTA pool; unused SM registers are not assumed available. Current SASS contains DEALLOC0x40 / TRY_ALLOC0xd8,16HGMMA/4WARPGROUP.ARRIVE/1plainUTMASTG/0XUTMALDG. Independent CFG review reports producermaxR53/mathmaxR158; exact launch/runtime limits still require the real device test.

CPU proof covers524288 arbitrary X BF16 values through the exact four-lane transpose,524288 W values through independent direct fragment decode/escape reconstruction/shared-B SW128 placement, and all8192 D scalars in the native token×channel output slab. Dense W has491209 escapes/cap61504, so data spans many32-entry windows. The bit proof reconstructs CSR entries in a host dictionary; it does not emulate rq_fix's lane-cached reload control. The unchanged judge-proven rq_fix supplies that source evidence. The sequential Q4/A2/W6/B8 tag model does not simulate arbitrary GPU interleavings; independent source review checks per-writer proxy fence+producer128 barrier, B-empty count8, and retirement-before-reuse. GPU arithmetic/escape/control correctness is not yet established.

CPU/device reproduction, no CUDA launch:

```bash
python3 proj_rsi/candidates/operand_reverse/preflight.py
```

Full Torch host-extension build is explicitly runtime-pending: this CPU-only workspace lacks the inherited ATen CUDAContext chain's `cusparse.h`. No header stubs or incumbent changes were used. A successful cubin compile is not a successful host extension build or launch.

After final independent review and root's named exclusive GPU handoff, one bounded trial:

```bash
python3 proj_rsi/candidates/operand_reverse/lab.py --json /okc/logs/operand-reverse01.json
```

Lab first builds the full extension, records static regs/local/SMEM/default max-dynamic, opts into103600B before occupancy, records occupancy/actual max-dynamic and validates the role-pool inequality. It then actually launches and checks all4096 one-hot columns, FP64 normal/outlier error, official normal/outlier probes, original/zero/fresh-outlier replay with fixed pointers, and input immutability. The baseline actual callable separately passes all4096 one-hot columns plus FP64 and official normal/outlier oracles; its validator-only output copy is outside timing. Only after those pass does it time the complete output kernel. The exact timed ga/gb graphs are subsequently checked through their actual retained outputs on original/zero/fresh-normal/fresh-outlier/all-probe input mutations: NaN-poison every output before replay, require all elements overwritten, full oracle agreement and input immutability. Separate early candidate graph tests are supplementary, not a substitute for these actual timed-graph checks.

Comparator is one prepared frozen pw36 `kernel_fn` state (SHA2a067059…813bccf), with its actual rq/Lt/sel choice recorded. No torch baseline. F1 times incumbent before/after candidate and keeps the lower baseline;21 rotated AB/BA cold-L2 graph pairs use fresh measured captures. Survive only if gain≥0.5µs in BOTH comparisons. Diagnostic profiler events are outside selection timing. Any build/runtime/correctness failure invalidates the experiment; a missed gain bar closes it. No automatic flag/parameter/geometry retry, no full submission, no official slot, no m256 extension under this task.

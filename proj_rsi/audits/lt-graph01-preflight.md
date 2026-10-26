# lt-graph01 independent preflight

Reviewer `/root/proj_evidence`, 2026-10-26 UTC. Readonly source/CPU review; no GPU, remote build/job, source edits or commits.

## Decision

**GO for one bounded o32 layout2 first-legal policy trial.** No legality/index/workspace-lifetime blocker identified. Source frozen lab SHA256 `8245f822dac5171aaf8a220899729c9ab5d5eacbb8a9bfae2e8404e4503578b3`, pw36 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. All manifest hashes and AST pass. Independent CPU layout product proof and actual tuner-branch/graph-flush assertions pass without invoking preflight's file-writing main.

This is one untimed-metadata selection followed by one candidate measurement, not another broad Lt/o128 sweep. No legal config is a legitimate closed result. Candidate numerical/config/runtime failure is not fallback timing. Survive only >=0.5us in both lower-bracket F1 and median paired graph difference against actual prepared pw36 dispatch, followed by independent correctness/profile review before promotion.

## API legality and source coverage

Actual `_tm` already graph-captures and uses a read flush; there is no eager-vs-graph mismatch to fix. `_tune` at m<128 only enumerates layout0, whereas large-m `_ltsearch` enumerates0..3 plus seeded refinements. Therefore this narrow o32 layout2 admission hole exists in the frozen source; reopening o128 layout search is unsupported.

`choose_one()` requests at most32 layout2 heuristics, in rank order. It skips any heuristic with splitK>1, forces config index2=1/index3=0, asks existing `lt_add`, and stops at the first returned legal code. No candidate timing occurs during selection. Existing vector field order is `{ID,tile,splitK,reduction,swizzle,custom,stages,inner,cluster}`. Parent initializes ID with AlgoInit; uint32 attributes through stages and uint16 inner/cluster setters match the [cuBLAS12.8 config API](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatmulalgoconfigattributes-t). SplitK1 avoids an explicit split-K reduction; actual configuration is read back and asserted1/0. This does not prove the library uses only one internal GPU node.

Parent `lt_add` rejects config length!=9, initialization failure, nonzero config-set failure, FP32 accumulation/no-TF32 capability failure, AlgoCheck/state failure, workspace>64MiB. A failed setter for requested zero is ignored by the inherited wrapper, so post-set returned splitK/reduction assertions are important. Runtime numerical checks remain necessary; heuristic/API legality alone is not a precision proof.

Layout2 algorithm code=`2*100000+index` matches `lt_cfg`/`lt_mm` decoding and its shape-keyed plan vector. Appending a candidate preserves previous integer indices; baseline state is never modified to select it. Metadata setup completes before graph capture. Same process-owned64MiB workspace is reused only by sequential default-stream replays with synchronization, so baseline/candidate scratch lifetimes do not overlap. No new CUDA source/extension or header shim is introduced; parent extension build and API return codes are real-node gates.

## Matrix/index semantics

Original W is N8192xK1024 row-major. Prepared `o_t=W.T.contiguous()` already exists outside timing. Layout0 interprets raw W as column-major KxN and transposes A. Layout2 interprets Wt's same lossless values as column-major NxK without transpose. B is X interpreted column-major KxM; C/D column-major NxM is the requested row-major MxN output. Both compute identical X@W.T with BF16 inputs/output,32F accumulation,alpha1/beta0. Independent3x16x8 CPU index proof matches both expressions and torch exactly. GPU all-column/probe checks still gate real dimensions/layouts.

The pointer/config/shape and beta0 contract adds no activation-derived cache or precision change. Pretranspose cost is prepare work already paid by both policies, not a hidden timed transpose. Benchmark cold read flush and existing workspace capacity are unchanged.

## Validation and measurement gates

Candidate: parent `_rqval` checks all1024 onehot x8 columns exactly and normal/outlier FP64 L2<=4e-3, row<=2e-2. Retained candidate graph sees original ->zero ->fresh-outlier values in one fixed X allocation; input unchanged, zero output exact, task checker on nonzero inputs. Missing any check rejects the candidate.

Baseline is one prepared pw36 `kernel_fn` dispatch, correctly resolved rq/KS ->sel ->r6 Lt/gv priority, not torch. F1 brackets candidate with the same baseline state and retains the lower21-replay median.21 AB/BA event pairs follow256MiB read flushes, with raw pairs preserved. Allocations/prepare/selection/profiles are outside event timing. Captured incumbent output is kept alive; shared Lt scratch is sequential. Recompute gains independently and reject nonfinite/missing pairs.

Runtime review must enforce README's additional profile condition: no extra reduce/copy/memset node or unsupported family attribution. The script's `SURVIVES_BOUNDED_LAB_ONLY` status currently gates only F1+paired gains; **it does not enforce this profile condition programmatically**. Use actual candidate_profile_events after timing, not kernel-name substrings or profile durations as winner selection. SplitK1 does not certify node count. The actual profiler-event metadata calls are supported in reviewer's local CPU profiler check; CUDA events on the real image remain runtime evidence.

Same correctness limitation as o_direct_w: actual timed incumbent output is only checked finite; actual timed candidate graph is task-checked on the original timing X, while original/zero/fresh-outlier mutations ran on a separately retained graph. Adequate for this bounded screen against already validated frozen pw36; **any survivor needs actual timed baseline task-oracle and mutated actual-timed-graph checks before promotion**. Profile failures occur before final JSON write, so a partial/profiler-failed run is incomplete evidence, not a favorable result.

## Bounded execution and interpretation

Root alone schedules one named container under the repaired exclusive GPU lock, with explicit inner/outer timeouts and confirmed CUDA-owner cleanup. Preserve frozen source/manifest, real parent compile output, selected/requested config metadata, numerical/replay checks, preparation phases, raw21 pairs, profile events, terminal JSON and exit. Admit either complete `CLOSE_NO_LEGAL_CANDIDATE` or a full result plus LAB_DONE; no partial record or API failure is timing evidence. No timing sweep, algorithm retry, new layouts, o128 expansion or official slot under this task.

Both tuned Lt and rq o32 were already single-node in the cited judge profiles. An historical benchmark-minus-profile gap does not establish a recoverable constant; a1x2 kernel-name suffix does not measure cluster cost. A favorable result would concern one o32 policy on this H100 session, not an aggregate>=20score-us gain or proof of reaching<3000us. H200 validation/integration remain separate gates.

Runtime review: COMPLETE; independent review below closes this fixed policy. Reviewer remains assigned for subsequent gates.

## lt-graph01 terminal independent review: CLOSE / no port or sweep

Root PID352695, named exclusive container, terminal2026-10-26T21:16:33UTC with LAB_DONE/LAB_EXIT0. Root reports container absent/GPU0MiB. Reviewer parsed copied full raw log/JSON, verified complete21 finite positive timing pairs, config/selection metadata, numerical/replay entries, actual profile events and F1/paired arithmetic. All prepare phases complete; no C75/CUDA/compile/error/skipped-budget line. Baseline hash matches frozen pw36. This is one o32 lab group, not65 official cases.

Only rank0 metadata config was admitted; no timed search occurred. Original=requested=actual config `[66,13,1,0,0,1,35,0,3]`; layout2 candidate code200008. SplitK1/reduction0 verified. All1024 onehot columns exact, FP64 normal/outliers pass (`ok1.7e-03`), original/zero/fresh-outlier retained-graph checks pass. Metadata describes r6 with Lt_codeNone, gv index45. Its stringified gv plan includes about68KiB of tensor text: poor logging, outside the timed call. It is not a second output/weight cache mechanism. Use actual dispatch/profile, not that bulky string, for attribution.

|actual prepared baseline|F1 before/after us|candidate F1 us|F1 gain us|paired median gain us|candidate slower pairs|gate|
|---|---|---:|---:|---:|---:|---|
|r6 gv45|9.920/9.888|10.752|-0.864|-0.864|20/21|CLOSE|

Marginal graph medians are baseline9.888/candidate10.784us; independent median paired difference is-0.864us. F1 is `min(9.920,9.888)-10.752=-0.864us`. JSON status `CLOSE_POLICY_NO_SWEEP` matches both failed>=0.5us gates. No source changes or retries are justified.

After-timing GPU profiles each contain one kernel: baseline `gv<1,4,2,1,8,2,1,1,0,1>(G)` and candidate `nvjet_tst_64x32_64x16_2x1_v_bz_NNT`. No additional memset/reduce/copy node was observed. One profile sample records7.521us baseline and5.664us candidate, opposite the admitted graph ordering. Those separate profiler samples do **not** override21 cold graph pairs, establish an eager/graph defect, or make their benchmark-minus-profile differences recoverable constants. No profiler duration selected this candidate.

**Disposition:** close this one first-legal o32 layout2/splitK1 policy, preserving config and raw pairs. No broad Lt/cluster/layout sweep, o128 expansion, integration or judge slot. This is not proof that every possible layout2 algorithm loses; the bounded policy failed its preregistered ROI bar and remains closed. Source/results README closure records this result without modifying frozen lab or baseline.

Raw local artifacts `results/proj-rsi/takeover/20261026/lt-graph01.json` and `lt-graph01.host.log`.

- JSON SHA256 `7747ff41b187f99ef0f207d86b51ab1d048a7c431f6d91a16556e7c57823e98f`.
- Host log SHA256 `ffffe95944bc5f67b819dda13840afc191a0a8887441b32d1fe5943776cb4bfa`.
- Frozen lab SHA256 `8245f822dac5171aaf8a220899729c9ab5d5eacbb8a9bfae2e8404e4503578b3`; baseline `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.

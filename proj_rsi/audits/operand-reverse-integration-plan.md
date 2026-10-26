# Operand-reverse integration blueprint — CLOSED / NON-AUTHORIZING

`state=HISTORICAL_ONLY; local_trial=FAILED_PERFORMANCE; port=FORBIDDEN_WITHOUT_NEW_ROOT_AUTHORIZATION; official_slots=NONE_REQUESTED`

This plan was requested while root's trial355459/container`okc-proj-operand-reverse01` ran. It is now historical: the terminal21:49:17UTC result passed CPU/full Torch build/GPU arithmetic and all4096 one-hot columns for both implementations, plus the actual timed graphs' ten poison/fresh-input oracle checks. Performance failed: F1 incumbent24.128µs, candidate47.648µs (gain−23.520002); paired median−23.519998µs,21/21 losses; diagnostic CUPTI candidate45.09 versus actual rq21.472µs. Therefore **no integration, full submission copy, kernel port, prepare modification, official check or scored run is admitted**. Sourceca92a4fb…9ae85f8/labea1d4597…d6c827/manifest remain frozen.

## Counterfactual admission gates

Had the fixed policy gained≥0.5µs in BOTH conservative F1 and paired graph time, with independent raw-result review and all correctness/source-identity gates passing, root could have separately authorized the following port. A local win alone would not establish H200 performance, prepare readiness, or a score improvement. The current measured loser satisfies none of the performance admission conditions.

## Exact single-file hooks that would have been required

- Use frozen pw36 source2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf as the development base. Preserve all existing kernel paths and Python fallback. The board incumbent remains rs32/3200.615µs; fresh pw36 scored3212.415µs within overlapping CI.
- Embed only one backend's CUDA string/host adapter and a tiny compiler/bake/dispatch wrapper in a new candidate's single submission file. No filesystem imports of `prototype.py`, rs32 source, cached cubins, or lab files are available to the judge. Keep full source dependency closure in that one file.
- Give the new extension a distinct module hash and exported adapter name (`rev_run`) and prefix its CUDA helper/type/kernel names to avoid symbol/ABI collision with existing rq extensions. Its parameter struct adds const X; this must never alter the old RqP ABI. Audit the normalized device instruction stream after renaming, including producer64/math216, static168 initial64512 registers and requested63488; retain zero spills/no C75 and the all-D output flow.
- Add `_rev_start()` immediately after `prepare`'s existing CPU return, alongside the rq/rq32 compiler starts. CPU-only compiler thread; no GPU packing/baking in that thread. All compile errors/unsupported-device conditions disable this optional backend.
- Add `_rev_bake(st, rev_thread)` after `PREP phase rq32 done` and BEFORE `st.pop("rq_pk",None)`. At that point original rq/gx/rq32 decisions are complete. Reuse `st["rq_pk"]["unpooler"]` (`lay,tab,esc6`), already bit-checked in `_rqbake`; build only KS1 escape lists. Do not repack/unpack50.3MB again. Keep references only if the new backend is actually picked; otherwise the normal pop frees unused packs.
- Keep the new pick in `st["rev_u128"]`, independent of `st["rq"]`, `sel`, `gv`, `lt`, and all zero/poll counters. The prototype's host API has an unused≈1KiB cnt argument; either keep a private inert tensor to preserve the validated adapter or remove it only with a separately reviewed ABI change. It must not use existing mutable counters.
- Wrap `kernel_fn` with an explicitly captured `_kernel_fn_before_rev` alias. Eligibility is exactly layer`unpooler`,M128,N8192,K4096,BF16 contiguous CUDA input on supported sm90a hardware and a validated `rev_u128` pick. Candidate writes its own BF16 output; native nonzero refusal falls through to the unchanged original callable when CUDA remains usable. Asynchronous faults or invalid numeric output must fail validation, not be hidden as a successful fallback. CPU and every unsupported shape go directly to the original callable.
- During bake, use `_kernel_fn_before_rev` as the incumbent; never `_r6` alone, torch, an opening baseline with a different prepare, or a recursive call to the new dispatcher. Record the actual rq/sel/Lt choice. Keep old rq picks intact until the optional candidate has passed validation and the gain bar.

## Compile/pack/prepare budget

Official pw36`e22a4b9d693d4902` measured: rq/rq32 compilers ready41s; r6 ends121s; rq122s; gx124s; rq32125s; synchronous router compile38s ends163s; all bakes ran. Total budget240s, but router's INTERNAL compile admission is180s (`budget−60`), earlier than its outer220s gate. That inner gate must be preserved.

The optional backend would compile one geometry in its own thread from prepare's start. This might contend with existing compilers, so cold-cache prepared integration would need measurement; the163s historical result is not a guarantee. Do not wait for an unfinished optional compiler at bake time. If it is not ready, skip the backend.

Proposed bounded optional bake admission: current elapsed≤150s AND backend ready AND unpooler rq pack already exists. Local deadline=min(start+10s,160s), checked between validation batches/captures. No unbounded join. This leaves≥20s before the180s router compile gate and≥80s before the overall240s deadline. In the observed125s rq32 phase it would end≤135s, with router's38s bringing total≤173s absent additional compile contention. This is a budget calculation, not a measured integrated prepare result.

If the pack is absent because an earlier phase skipped/failed, skip this optional path; do not create a late50MB pack. Packed unpooler is50,331,648B. KS1 escape data is weight-dependent and fixed-stride with sentinels; never hardcode cap. Input-independent weight packing/metadata belongs in prepare. Actual X-dependent transpose/MMA/output stays in timed calls. Retain no outputs or X-derived state across calls.

Validation, warmup and graph timing must fit the optional deadline. Candidate loses or validation is incomplete/deadline expires → no `rev_u128` pick. Existing router/m32 phases and dispatch stay eligible. No production profiler/instrumentation or timing manipulation is introduced. Pack memory adds about50.3MB only when it was otherwise unretained; review memory ownership/reuse explicitly.

## Precision, nearby shapes, fallback and admission

All W BF16 bits remain exact through12bit pack+escape repair. All X BF16 bits remain exact through register permutations. FP32 accumulation, BF16 output, no TF32/FP8/pruning/approximate probes. No raw-input/output cache or known-seed specialization. Unsupported M127/129, M33/130/256/512/1000, other layers, sliced unpooler≤8, wrong dtype/layout/device, or failed compile/pack/bake use the frozen original path. A new backend must not route by a neighboring bucket or pad a supported M silently.

Single-file candidate acceptance, only after a separately authorized measured win:

1. Source/ABI/dispatch/prepare-diff review, helper closure, normalized device/SASS/resource and all-output checks.
2. `python3 -m proj_rsi lint <new-submission.py>` and `python3 -m proj_rsi cpu-check <new-submission.py> --official`. CPU checks certify fallback semantics only.
3. Root-exclusive full65-check GPU harness/30 timings (`proj_rsi/tools/h100_bench.py <new-submission.py> --official --json <artifact>`), bracketed against frozen pw36 source prepared in the same session. The harness's65 cases are5layers×13M; it does NOT alone cover all official outlier distributions.
4. Separately run exact official normal/outlier cases, all4096 one-hot columns for the new path and the actual baseline, and actual timed ga/gb original/zero/fresh-normal/fresh-outlier/all-probe mutations with NaN output poison, immutable X and task oracle. Include M127/129 as nearby fallback cases; check old qkv/pooler/o/router shapes for dispatch/prepare regressions and zero-counter reset where applicable. These checks need the integrated candidate, not only the standalone lab.
5. Require local target gain≥0.5µs both F1 and paired graph on actual integrated `kernel_fn`; all other timing differences need drift-aware review. Verify cold prepare completes within240s with all existing valuable rq/rq32/router bakes admitted and new optional state selected only after complete validation.
6. Root alone handles official quota/promotion. At this snapshot quota is0checks/1scored until00:00UTC reset. Do not spend the remaining scored slot on this losing mechanism or rerun unchanged pw36. A future winner would need a concrete single-file candidate and budget evidence before root elects an official check after reset (or explicitly chooses a scored slot). Read the official9/9 result, prepare picks and target timings before promotion.

## Post-failure consequence

The above hooks are **not authorized work** for operand-reverse01. Both complete-output kernels are one profiler kernel, and the roughly2× slowdown is in kernel execution. Its cause cannot be split among producer decode/barriers, X register-transpose/LDG, extra stage count or WGMMA retirement from aggregate CUPTI/graph durations alone. Exact source/SASS/byte accounting is being recorded in `operand-reverse-postmortem.md`; no changed pipeline is admitted without new bottleneck evidence. No current speedup, m256 extension, integration or official action follows.

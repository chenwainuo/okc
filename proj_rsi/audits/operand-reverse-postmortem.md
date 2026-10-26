# Operand-reverse01 postmortem — closed, no rerun or integration

`decision=CLOSE_POLICY_NO_SWEEP; source/lab/manifest=FROZEN; original_trial_counters=NONE; separate_Ncu_report=UNREVIEWED; GPU_action_by_author=NONE`

**The expected shared-memory saving was against the already rejected pw19 design, not the actual rq comparator. Against rq, whole-K shared bytes are equal; X requests increase57%, math transpose work is added, and stage/fence/refill counts double.** This is sufficient to explain why a win was not justified. It does not partition the observed23.5µs regression into measured causal components.

## Measured facts and identity

Root trial355459/container`okc-proj-operand-reverse01` ended21:49:17UTC, exit0. Candidate generated source`ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`, device`59f3b775d28410c23616a749ad58661112f9c6a1a959fec05333068ca9f3800a`, lab`ea1d4597418618619438f805fb3bebeeac666bb9ec84f8910dda8d1001d6c827`. Raw local result: `results/proj-rsi/takeover/20261026/operand-reverse01.json`, SHA`aecc7ee25d045d3924cb3969938e6658808049e4b70355cdc6a6fd0696debb65`. Independent closure: `audits/operand-reverse-architecture-preflight.md`.

Both implementations passed complete4096-column exact one-hot tests, random/outlier FP64/task oracles, and actual timed ga/gb five-input mutation checks with NaN-poisoned outputs and immutable inputs (ten checks). Candidate supplementary replay tests also passed. The corrected64/216 allocation launched:168static/local0/103600B shared,49152→103600 dynamic opt-in before occupancy1; requested63488≤initial64512 registers. No register deadlock, numerical corruption, silent fallback, extra profiler kernel, or host-compile failure explains this trial.

| Measurement | actual baseline rq | reverse candidate | result |
|---|---:|---:|---|
| F1 graph before |24.224000052µs|47.648001462µs|—|
| F1 after/lower baseline |24.127999321µs|47.648001462µs|−23.520002142µs gain|
| paired median difference |—|—|−23.519998416µs;21/21losses|
| diagnostic CUPTI single kernel |21.472µs|45.090µs|≈2.10×kernel duration|

Actual baseline selected rq`s4s5hz`,KS2, profile`rq<128,5,1,0,0,0,3>`. It was not the historical Nvjet19.103µs benchmark constant. Both graph and profiler comparisons show the regression; this is not a graph-overhead-only effect. CUPTI is attribution evidence, not the selection timer. No H100→H200 latency/score conversion is made.

## Exact geometry and logical traffic versus that comparator

Both compute the same8.59GFLOP unpooler(M128,N8192,K4096) and launch128CTAs/two math WGs. However:

| Per CTA / operation | baseline rq | operand reverse |
|---|---:|---:|
| output-channel tile |128|64|
| token tile |128|128|
| global K split |2|1|
| K per CTA |2048|4096|
| k64 stages |32|64|
| WGMMA form |m64n128k16|m64n64k16|
| WGMMA instruction executions/CTA (two WGs) |256|512|
| W packed fill/stage |12KiB|6KiB|
| X delivered request/stage |16KiB|16KiB|
| X path |one TMA box, shared B|four vector LDGs/thread, GPR transpose|
| global output payload |4MiB BF16 partials +2MiB zeroing|2MiB plain stores|

The tensor work is equal: each new WGMMA does half as much work, twice as many stages execute. The new launch does not eliminate compute. Its smaller stage gives less tensor time under which to hide a similar fixed control/producer chain.

Whole-operation requested input bytes, excluding small escape/metadata streams:

- W: both50,331,648B. No compressed-weight bandwidth saving over rq.
- X: baseline64rowtiles×128tokens×4096K×2B=67,108,864B; reverse128rowtiles×the same X=134,217,728B.
- Total W+X:117,440,512→184,549,376B, **+57.1429%**. New X ingress grows because each activation tile is reused over64 rather than128 output channels. GPR routing does not remove those loads.

These are logical requested bytes delivered to SM consumers, not measured DRAM bytes or guaranteed L2 misses. The unique X is only1MiB. Baseline FL3 also requests W L2-evict-first/X-evict-last, whereas the reverse source uses ordinary W bulk and X `LDG.nc.L1::no_allocate`; actual cache effects were not measured. Prior hints were small/box-specific, so do not assign the23.5µs loss to hints without counters.

Shared-memory mainloop bytes also cancel over full K:

| Per k64 | rq | reverse |
|---|---:|---:|
| packed W fill+LDS |24KiB|12KiB|
| X shared fill |16KiB|0|
| decoded W shared stores |0|8KiB|
| WGMMA shared operand reads |32KiB X|16KiB W|
| sum/stage |72KiB|36KiB|
| stages/CTA |32|64|
| total/CTA |2304KiB|2304KiB|
| total/128CTA mainloop |288MiB|288MiB|

Output/shared-tail traffic is smaller in the candidate, but the intended mainloop shared-byte improvement over **rq** is zero. The earlier60→36KiB comparison is valid against pw19's64channel SS tile; it cannot be used as a speed prediction against128channel rq. No physical128B/clk aggregate-port model is required for this conclusion.

## Added instruction work and exposed producer chain

Source counts are dynamic instruction-work accounting, not measured cycles:

- Every reverse math thread/stage performs32butterfly SHFL and32selections to form register-A. Over64stages/eight math warps this adds16,384warp-level SHFL +16,384selection instructions/CTA, before addressing. Baseline X TMA has no such register transpose. Current compiled math SASS contains128SHFL.BFLY/128SEL sites in its four-stage unroll; the other16SEL sites belong to producer escape repair.
- X uses4LDG.128/thread/stage:2048warp-level vector-load instructions/CTA, versus32elected-lane X TMA issues in rq. These different hardware operations are not assigned equal cycle cost. They establish additional issue/address/dependency work, alongside the doubled delivered X bytes.
- W dec1 arithmetic is not removed: approximately36source ops/warp/chunk ×4decoder warps×64stages equals the baseline's36×8math warps×32stages. It is moved onto one producer WG's critical publish chain.
- That producer adds16scalar shared stores/thread/stage (4096warp-level STS instructions/CTA), per-writer proxy fences, and64producer-only128-thread barriers. Baseline math decodes directly into A registers and has no equivalent decoded-B publication stage.
- Reverse W refills occur only after LDS→dec1→escape repair→16shared stores→proxy fence→producer128 barrier→Bfull publication. The elected lane then refills the packed slot six stages ahead. In rq, the independent W lane refills as soon as consumers have LDS'd packed W, before their decode/MMA completes. This is a real dependency-chain distinction; whether it is the dominant runtime limiter is unmeasured.
- Math stages/fences/commit/wait/release and Wbulk issue counts double32→64. Tensor issue time per stage halves. Bfull gates both math WGs, so a producer that cannot publish at the smaller tensor period can starve both together.

No bank-conflict claim is needed: the CPU bijection proves addresses, and the decoded-W/native-D scalar store formulas map the32 lanes to32distinct32bit banks for each modeled instruction. Actual shared transaction/replay counters are absent. Register pressure is not a measured spill cause: both static168/local0, source-correct redistribution, and no local memory instructions/C75 warnings. Occupancy1 is expected for both and does not show their stall behavior.

## What the compiled schedule shows

Exact-flags candidate device disassembly from independent review: `/tmp/okc-operand-reverse-review-as1b3qur/review_exact.sass`, SHA`48b2bfc1f13f372f89a888e2f950f678e0c71a4f868ff8c17295d928fa91aec6`. For this postmortem only, the actual baseline template`rq<128,5,1,0,0,0,3>` was extracted from frozen pw36 and CPU-compiled with the same inherited flags, no kernel/source mutation or GPU launch: `/tmp/rq-base-postmortem-arbur9sn/baseline.sass`, SHA`e59d69a9b28e77c28d6d3ecc34325929addfaf1dfd007b3d3767d924e0908499`. It also has168regs/0stack/spill/no C75. Temporary compiler artifacts are references, not a new candidate.

Both disassemblies have four static four-HGMMA groups, four WARPGROUP.ARRIVE and one wait_group1 per group plus final wait_group0. There is **no ptxas per-HGMMA wait_group0 serialization**. This rules out the previously encountered C75 compiler failure, not ordinary exposed dependencies.

| Four-HGMMA static group | rq instruction slots first→fourth HGMMA | reverse slots | reverse interleaving |
|---|---:|---:|---|
|0|10|12|uniform descriptor work|
|1|22|45|17SHFL,14SEL between HGMMA sites|
|2|13|70|20SHFL,15SEL,12LDG.128 between sites|
|3|15|20|3LDG.128 plus address work|

These are static PC-span counts, not executed-instruction totals/cycle gaps; predicates/control paths and asynchronous execution matter. They show that X transpose/prefetch is scheduled among the math instructions and must not be modeled as a free operation completely hidden behind a full queued tensor batch. Moving/merging instructions without measured stalls would be another unproven pipeline experiment.

Useful candidate SASS sites (local exact-device build only; remap against the actual profiled binary): math Bfull waits0xfe0/0x1580/0x1df0/0x2500 with cold retry blocks0x3e00..0x3e90; producer Wfull0x3140/retry0x3ec0; producer Bempty0x31f0/retry0x3ef0; producer proxy fence/barrier0x3bd0/0x3be0. Some math retry blocks occur after producer EXIT in address order; CFG reachability, not lexical PC position, determines the role.

## Highest uncertainty and one potentially useful diagnostic

Facts support a bad trade: equal whole-K shared traffic, more requested X/L2 delivery, added transpose/stores/control, and a narrower tensor batch. They do **not** measure how much of the extra23.5µs is producer-B starvation, X-LDG scoreboard/issue pressure, or ordinary tensor/group-retirement waiting. There are no measured bandwidth, cache-hit, issue-stall, bank-conflict or pipe-utilization counters in the original run. A separately root-run Ncu diagnostic was reported terminal while this document was completed; its identities/metrics have not been reviewed here and are not used below.

If root authorizes one diagnostic after checking Ncu availability/permissions, the highest-value question is **which instruction dependency starves the math WGs?** Profile only the unchanged candidate, normal input, warmed exact graph, outside every score/selection timer. Use one bounded Ncu report with source/SASS-correlated warp-stall sampling, tensor utilization, L2/DRAM delivered/read bytes and shared transactions. Tool metric names/availability depend on the installed Ncu version; do not request a huge blind metric sweep or time instrumented execution as a speedup result.

Predeclare interpretation:

- Math waits at Bfull + producer concentrated in decode/escape/ST/fence/barrier with little Wfull wait ⇒ producer publication chain is the leading hypothesis.
- Math Bfull wait + producer Wfull/TMA wait ⇒ packed-W ingress/refill chain, not evidence that decode ALU is the main cause.
- Math long-scoreboard at X LDG/first SHFL and increased L2/DRAM requests ⇒ X ingress/limited effective prefetch/cache behavior.
- Math active issue dominated by SHFL/selection with low tensor use and little memory/barrier wait ⇒ transpose/issue supply is the leading hypothesis.
- Wait_group stalls with high tensor utilization ⇒ ordinary compute overlap/tensor period; no C75 defect is implied.

A single sample can establish a dominant direction but not a precise additive cycle decomposition. Profiling may replay kernels internally; its durations never replace the admitted graph timings. Do not profile one-hot/zero input accidentally or count capture/prepare time as the measured kernel. No such profiling action is authorized by this document; root owns any named diagnostic.

## Disposition and changed-mechanism threshold

The fixed64channel operand-reverse policy is closed. The source accounting suggests that widening channel reuse back to128 would remove the doubled X requests and halve stage/transpose counts, but that also restores KS2/global reduction and the original whole-K shared-byte budget while retaining extra transpose/decoded-store work. **That is not a justified next implementation from the present evidence.** Only a causal counter result showing the new X issue/ingress work dominates could motivate a separately reviewed changed reuse mechanism. If producer publication dominates, widening alone may not help at all. Do not reopen the implementation by guessing prefetch depth, register caps, barrier placement, TMA stage depth or decoded-W prepacking.

There is no kernel port, integration candidate, pooler/qkv/m256 transfer, official quota spend or score promotion. The historical integration blueprint is non-authorizing. Current best remains rs32/3200.615µs; this trial makes no progress toward3000 through performance, but it closes a materially different hypothesis with strong correctness and comparator evidence.

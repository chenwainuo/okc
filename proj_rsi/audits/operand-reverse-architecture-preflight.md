# Operand reversal architecture preflight — 2026-10-26

`scope=v1 unpooler M128 N8192 K4096 KS1 only; decision=V1_FAILED_GAIN_BAR; GPU=TERMINAL_VALID; correctness=PASS; F1_gain_us=-23.520002142; paired_gain_us=-23.519998416; promotion=NO; family=ITERATE_FROM_DIAGNOSED_MECHANISM`

Later user policy clarification: expected failures should be diagnosed and iterated. The historical no-automatic-rerun/sweep disposition below concerns this frozen v1 only. Its NCU/PC evidence motivates the separately reviewed v2 direct native-A loader; it does not prohibit a material successor with new CPU/lifetime proof and a bounded same-run comparison. V1 evidence stays unchanged.

Independent review of `audits/m256-next-mechanism.md`, current `candidates/operand_reverse/{kernel_body.cuh,prototype.py,device_source.cu,extension_source.cu}`, inherited rs32/pw36 rq helpers, `audits/{pw19-critic,pipeline-diff,wgmma-layout}.md`, and official rs32/pw36 raw JSON. No source edit, GPU launch, remote mutation, score promotion or queue edit by reviewer.

## First falsifier caught before runtime: CTA register pool

The original producer64/math224 allocation is **NO-GO**, even though its sum65536 fits the SM's nominal register capacity. A384-thread static168 launch initially allocates64512 registers. `setmaxnreg` redistributes that CTA pool; it does not add the SM's remaining1024 registers to it. Both math WGs cannot acquire224 while the producer retains64. In this source the producer eventually waits for eight Bempty warp arrivals; a math WG blocked acquiring registers cannot provide them, so producer exit cannot rescue the deficit. Occupancy>=1 alone would not detect this dynamic deadlock.

The author corrected this **before freeze/runtime**, keeping geometry and algorithm fixed: producer64/math216 =>128×64+256×216=63488 <=64512, with1024 registers left in the CTA pool. Existing rq40/232 totals64512 exactly; it was supporting evidence for the redistribution pattern, not for the invalid64/224 split.

[NVIDIA PTX setmaxnreg documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) specifies the per-CTA pool and blocking acquisition. [NVIDIA's register-allocation explanation](https://forums.developer.nvidia.com/t/sm90-setmaxnreg-will-change-occupancy-dynamically/302068/3) confirms total CTA register allocation remains fixed at launch. The arithmetic/deadlock above is this reviewer's application to this source.

Independent NVCC12.8.93 sm90a compile of the revised generated device source passed with the exact inherited flags `-gencode=arch=compute_90a,code=sm_90a -O3 -DNDEBUG --expt-relaxed-constexpr`: static168 registers;0stack;0spill stores/loads; no local-memory instructions or C75 warnings; emitted `USETMAXREG.DEALLOC.CTAPOOL 0x40` and `TRY_ALLOC.CTAPOOL 0xd8`;16HGMMA/4WARPGROUP.ARRIVE/oneUTMASTG. Reachability from the producer entry uses at most R53; math entry uses at most R158, within their respective caps. Out-of-line math wait loops appear after the producer EXIT in address order and use R153/R155; they are not reachable producer accesses. No register-count claim was inferred from lexical PC range alone.

Compile snapshot: device SHA256 `59f3b775d28410c23616a749ad58661112f9c6a1a959fec05333068ca9f3800a`; body `35a1eed6386452f7d88ef4dc74d8f2a1274af94d04cda6b6cbdf9d5965d917d9`; generated extension `ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; prototype `06f25053a58de30b7ed95419d40fcf9352f33ab8d99298eaa605a87d39a3d7e2`. Temporary independent compile directory `/tmp/okc-operand-reverse-review-as1b3qur`; canonical exact-flags `review_exact.sass` SHA256 `48b2bfc1f13f372f89a888e2f950f678e0c71a4f868ff8c17295d928fa91aec6`, byte-identical to the author's recorded disassembly. The earlier independent object-build sanity check had an extra `--use_fast_math`; it is superseded by this exact-flags cubin evidence.

## Geometry, mappings and dependency lifetimes

One CTA computes64channels×all128tokens×K4096. Grid128CTAs; threads384 = two128-thread math WGs plus one128-thread producer/decode WG. WG0/1 own tokens0..63/64..127; no redundant token output or inter-CTA reduction. KS1 means no counters, zeroing, atomic reduction or split tail are used by the kernel. Host still checks/accepts inherited counter storage; that is an API requirement, not a timed counter operation.

Reviewer independently simulated the exact `rev_step<1>/<2>` and register permutation on symbolic `(token,k)` pairs: all2WGs×4warps×32lanes×16registers match the BF16 register-A fragment coordinates. It only routes bits. Decoded W SW128 placement is a4096-BF16 bijection over64channels×64K; native D stores are an8192-BF16 bijection over128tokens×64channels, with the correct new `[token][channel]` orientation. The TMA map `{inner=N8192,outer=M128,box={64,128}}` stores the whole slab at column64×blockIdx.x. The old transposed rq epilogue was not retained.

Reviewer also executed the author's CPU proof functions without their file-writing `main`: X524288values, W524288values with491209escape entries/cap61504, output8192D scalars and sequential ring tags pass. Evidence limitation: W proof applies escape patches from a dictionary; a list spanning32-entry windows does not itself emulate `rq_fix`'s lane-cache/reload control flow. Its support is the unchanged inherited helper plus later GPU exact tests. The sequential lifetime tag check does not model two-WG barrier interleavings; source protocol review below is separate evidence.

Packed-W6×6144B and decoded-B8×8192B are disjoint, each stride1024-aligned. Declared SMEM103600B includes alignment/control slack. Wfull transaction barriers count1; initial six packed chunks are issued before the CTA barrier. Before refilling a packed slot, all decoder warps have completed their LDS/decode/store and crossed producer-only `bar.sync3,128`; every decoder had already observed that slot's prior Wfull phase. Wfull phases cannot lap an observer.

Every decoder writer executes the generic→async proxy fence **before** the producer128-thread barrier; only afterward does elected thread256 publish Bfull. This is not a single-lane fence standing in for other writers. Bfull count1 and Bempty count8 are consistent with one publisher/eight consumer warps. Producer reuses a decoded slot only after its prior Bempty phase, which requires both math WGs. Every math warp contributes once, after a covering `wgmma.wait_group1`; initial slots need no prior empty signal. B ring8 and one retained WGMMA group do not form the one-slot dependency cycle closed in pw19.

Math uses two16-register A fragments. Stage s+1's `wait_group1` retires stage s before the corresponding `rq_fa(old)` and release; stage s+2 can then overwrite that A buffer. Raw X prefetch buffers are distinct from the transposed A buffers. `rq_fa` keeps asynchronous A operands live through the covering wait; inherited accumulator fences remain. All64k64 stages (four k16 WGMMA instructions each) are covered. Final `wait_group0` drains the final A/B reads; then both math WGs synchronize before reusing B storage for output. Producer has published stage63 as its final decoded write and has no subsequent B overwrite. Every epilogue writer fences and crosses the256-thread barrier before one plain TMA store; `.wait_group.read0` protects the shared source through its final read.

No source-level race/barrier/lifetime blocker remains in this fixed geometry after the register-pool correction. CPU/SASS evidence does not establish runtime launchability or numerical correctness. Real Torch/CUDA host compilation, dynamic-SMEM opt-in **before occupancy**, actual occupancy>=1, and actual output/replay tests remain runtime gates.

## X ingress and why pw19 does not prove a speedup

The four128-bit vector loads/thread are coalesced at32B sector granularity. For each warp/instruction, eight rows supply64B each:512requestedB=16×32B sectors across eight128B lines. Separate instructions read the other half of those lines; do not call each single instruction a contiguous512B load. Reviewer checked these address sets from the actual source. All128tokens are loaded once per CTA per64K stage, with no cross-CTA reuse guarantee. Total requested X=128CTAs×128tokens×4096K×2B=134217728B, versus half that for128-channel tiles. Packed W adds50331648B plus escapes. These are logical delivered bytes, not measured HBM traffic or guaranteed cache misses.

Per math thread/k64, the source performs four vector loads plus four4×4 lane transposes:32SHFL and32selection operations plus addressing. Across two WGs that is8192SHFL and8192selection lane-operations per CTA/stage. Q4 prefetch hides latency only if the actual instruction/transaction issue path and register lifetime allow it. It removes X shared fill/read, not X ingress or instruction cost. Current independent SASS has130SHFL sites for the four-stage unroll plus escape logic; source counts are not a GPU cycle model.

The pw19 comparison is a **mechanism comparison**, not a paired latency prediction. `pipeline-diff.md` records tw64 **KS2**,416threads, W8/X4/A4 rings and SS WGMMA: W-only19.9us, W+X30.4, +decode32.5, +WGMMA39.5, full43.8. Nominal shared traffic60KiB/k64 included12KiB packed-W fill+LDS,8KiB decoded-W stores,32KiB X fill+reads,8KiB decoded-W WGMMA reads. New design has12+8+0+16=36KiB, a40% byte reduction **for the same logical64channel×128token tile**. It reads decoded W twice because the two math WGs own different token halves. It also changes orientation, threading and split policy. The failed old pipeline does not certify the new one, nor does the byte reduction predict40% time improvement.

Port02 did not establish an aggregate128B/clk physical ceiling: the intended producer write rate was underfed. Its conditional288-cycle traffic budget versus256tensor cycles cannot determine this kernel's runtime. Resident-only tests omit the newly expensive X ingress and therefore cannot admit the implementation.

## One bounded complete-output falsifier; no m256 admission

The concrete first cheap falsifier was the exact launch/pool proof; it already rejected64/224 without GPU time. The corrected fixed unpooler128 source may proceed to a **single** frozen-lab preflight. Before root launches, review the actual comparator callable/output, timer boundary, flush policy and retained graph checks. Comparator must be actual frozen pw36 prepared `kernel_fn`, with its same-run selected Lt/rq dispatch; neither untuned Torch nor a historical19.103us constant is a valid timing baseline.

Required trial sequence: real host/resource gate -> exact lossless pack -> both implementations' complete4096 one-hot columns -> normal/outlier FP64/probe and finite/shape checks -> fixed-pointer original/zero/fresh retained-graph input mutation plus output poison and input immutability -> complete-output F1 and paired graph measurements -> post-timing validation of both **actual timed** graphs. Every D fragment must be observed. Reject any FAIL, missing check, fallback, timeout, stale input/output or source mismatch even if the process exits0.

Performance admission is the preregistered>=0.5us gain in **both** lower-before/after F1 baseline and median paired cold-L2 graph timing. Otherwise close this fixed mechanism; no automatic flag/geometry/prefetch sweeps or official slot. One named exclusive container under the current lock, bounded inner/outer timeouts and terminal ownership cleanup are root's gates. Architecture review alone is not permission to launch an unfinished lab.

m256/pooler/qkv remain conditional. m256's192core registers before overhead and doubled X/transpose work cannot inherit this m128 launch proof. Pooler also restores split/output-reduction costs; qkv changes CTA count. A measured m128 survivor is required before proposing those transfers.

## Final frozen lab review: GO for one trial

Lab SHA256 `ea1d4597418618619438f805fb3bebeeac666bb9ec84f8910dda8d1001d6c827`; manifest SHA256 `e04584c9054a1e345d1b6def6798bb9bd0fbff19aec8cefa53c218cdd1f7dcc5`. Reviewer verified all8 manifest file digests, Python ASTs and exact equality of regenerated source/device source to the recorded files. Kernel/device/prototype hashes above remain unchanged. Proof-document labels were corrected before freeze to distinguish host escape reconstruction/sequential tags from GPU protocol execution.

The early draft lacked an oracle for the actual baseline output and lacked mutation/poison checks of the actual timed graphs; reviewer held it before runtime. Final lab fixes both. Candidate and actual pw36 callable each run all4096 one-hot columns plus FP64 random/outlier and task normal/outlier oracles. Only the untimed baseline validator copies to a supplied destination. The measured baseline callable invokes actual `B.kernel_fn` directly; it adds no output copy. It receives a separate immediate task oracle/immutability check on the timing input before capture.

F1 uses inherited `_time_us`, whose event interval encloses one graph replay after a268435456B flush reduction; it takes21samples and returns their median. There are5warm samples per implementation, baseline21 -> candidate21 -> baseline21, then independent retained ga/gb captures with21alternating AB/BA pairs, same flush before every event interval. Actual baseline captured output is retained as `keepalive`; candidate output is a disjoint buffer. Diagnostic profiler runs occur after timing and do not determine selection. Both **same ga/gb graph objects** then receive original, zero, fresh normal, fresh outlier and all-probe inputs; their real outputs are filled with NaN before replay, checked fully finite/overwritten, task-oracled (zero checked exactly), and X compared with a pre-replay clone. Original input is restored. These10checks and both eager validation lists are persisted in result JSON.

Source identity is checked against preflight before extension construction. Runtime first records regs/local/shared/default max-dynamic, then info4 sets opt-in before occupancy, then info5 rereads max-dynamic. The lab requires occupancy>=1, no local storage, requested63488<=actual static-reg pool<=65536, actual max-dynamic>=103600, before actual computation. Full Torch host build and device numerical execution remain unproven until this run; an exception/timeout is an invalid trial, not a slow-but-correct observation.

Reviewer sent root **GO_ONE_BOUNDED_EXCLUSIVE_TRIAL**, with all source frozen, no automatic sweep/retry/shape transfer and no promotion. Parent owns exact named container, inner/outer deadline and terminal cleanup verification. Require>=0.5us in both conservative F1 gain and median paired difference, plus every correctness gate; otherwise close. Any nominal survivor still requires raw logs/results/profile and terminal ownership review before further work.

## Official score arithmetic independently reproduced

The architecture doc's corrected exact formula is4830.8/geomean(same-job SGLang load sum/ours load sum). Holding that same SGLang vector fixed, a what-if is `score_before×geomean(ours_after/ours_before)`. Reviewer recomputed directly from official benchmark `median_ns` and `speed_of_light_ns`:

|job|verify32 sum us|three m256 layer sum us|three SOL sum us|all three SOL/.85 score us|all three SOL score us|
|---|---:|---:|---:|---:|---:|
|rs32 fb548b5dc6f04696|86.584848719|66.808121783|44.513058427|3104.756353|3045.674237|
|pw36 e22a4b9d693d4902|85.080609635|66.625005587|44.513058427|3115.706847|3055.254217|

Pw36 verify16−3us gives3187.415724; unpooler128−1us plus unpooler256−2us gives3191.511893. Thus the doc's conditional numbers are correct after its formula wording correction. These are unmeasured scenarios, cannot be added across runs, and do not show that m256 optimization alone reaches3000us. No incumbent change follows.

## Independent terminal review: CLOSED

Root launched only this frozen trial in `okc-proj-operand-reverse01`, actual rootPID355459, inner240s/outer300s under the container-held lock and host guard. Raw host log spans2026-10-26T21:47:49Z..21:49:17Z (88wall seconds), records real Torch/CUDA compile+link success, candidate/baseline identities, `PREP done in77s`, `LAB_DONE`, and `TRIAL_EXIT ... 0`. No timeout, fallback error, C75 warning or skipped-budget stage appears. Reviewer independently checked actual PID355459 absent and named container absent after completion; parent additionally reported GPU0MiB. Reviewer did not launch a GPU job or alter the remote machine.

Runtime resource gate passed: regs168/local0/shared103600B; maxDynamic49152→103600 after opt-in; occupancy1; requested role pool63488<=initial64512. All8 pre-closure manifest entries still matched when reviewed; regenerated source identity matches raw JSON/host stdout. Original invalid64/224 source was never launched.

Both complete4096-column one-hot suites pass exactly; both FP64 random/outlier validators pass. Candidate task normal/outlier relative-L2 errors1.369e-4/1.494e-4 (worst-row3.234e-4/3.328e-4); actual baseline2.558e-3/2.836e-3 (worst-row2.975e-3/3.009e-3), all within required tolerance and all embedded probe rows exact. Candidate's supplementary original/zero/fresh-outlier replay passes. Immediate actual baseline callable oracle passes with12exact probes. All10 **actual timed ga/gb** checks pass after NaN poison and input mutation; all-probe outputs have128/128exact rows for both implementations, both zero outputs are exactly zero, all X immutability checks pass. Candidate's smaller numeric error is not a timing win or new objective.

|measurement|actual prepared pw36 baseline us|candidate us|gain (baseline−candidate) us|
|---|---:|---:|---:|
|F1 before|24.224000052|47.648001462|—|
|F1 after/conservative|24.127999321|47.648001462|−23.520002142|
|paired marginal medians|24.288000539|47.775998712|—|
|median paired differences|—|—|−23.519998416|

Reviewer recomputed both gains from raw fields/pairs, agreeing to<1e-10us. **21/21pairs are slower**, with individual gains−24.191999808..−22.240001708us. Candidate F1 is1.974801× the conservative baseline. The>=0.5us gain bar fails in both methods by a large margin.

Actual prepared baseline picked rq`s4s5hz`,KS2; its single profile kernel is `rq<128,5,1,0,0,0,3>`21472ns. Candidate is one `rq<128,8,1,0,0,0,20>`45090ns. This is the actual same-run comparator, not historical pw36's Nvjet19.103us or an untuned Torch kernel. Single profiler durations establish dispatch/context; they do not replace event-graph measurements. This result supports **failure of the fixed complete-output design**. It does not isolate whether X ingress, transpose issue, producer throughput, shared traffic or synchronization dominates; no counter data or new physical-port ceiling is inferred.

Raw artifacts: `results/proj-rsi/takeover/20261026/operand-reverse01.json`, SHA256 `aecc7ee25d045d3924cb3969938e6658808049e4b70355cdc6a6fd0696debb65`; sibling `.host.log`, SHA256 `1ce385db93f8499afda41ec64a03c0e78589fd73269b718eb34b12020bbf270e`. Local JSON matches stdout result. Source/lab/generated files/proof/manifest remain frozen. Parent authorized source README/RESULTS metadata closure; README changes after measurement intentionally do not match its preflight-only manifest digest, while executable digests remain unchanged.

**Disposition:** CLOSE_POLICY_NO_SWEEP. No integration, automatic rerun, parameter/flag/geometry sweep, pooler/qkv/m256 transfer, score-slot use or incumbent promotion. A separately reviewed future profiling action may diagnose this closed source; it does not reopen implementation tuning automatically. Root owns the queue/ledger and any later task.

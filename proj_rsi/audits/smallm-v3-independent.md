# Independent smallm v3 runtime validation

**PASS for validity of the five requested serving cases across pw36, rs32 and increment.** This judgment uses the collected numerical, graph, counter, JIT, terminal-output and cleanup evidence, not exit0 or the result status alone. No GPU/remote action, source edit or queue transition was performed by this reviewer. Root retains disposition authority. This does not validate the new cumulative PW37 source, an official score, cold-cache prepare feasibility or the full65/30 cumulative gate.

Collection: `results/proj-rsi/scheduler/artifacts/smallm.cursor.v3.all5.01/collection-1793074281123122605/`. Inventory SHA256 `dbae290afc99e124ce4fe52c1e6e8e9c3d65abe8dfdd462d442b2496a3fa36c4`; result SHA256 `0246320f407dbc55b25db6b60fd79157f93359cd3c89d4bcff593c5e7309c2c1`. All12 inventory files match their hashes, missing list is empty, and the live WAL collection evidence binds that same inventory hash/path. All29 dispatched source/review pins still match local bytes. The reviewed driver remains `bdc7cb89f1d0afa1604bc855601ceb6fe46f0c5237de5025f18800ba67062f2f`; candidate source remains `0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737`.

Reproducible CPU-only validator: [smallm_v3_runtime_validate.py](smallm_v3_runtime_validate.py). Full per-case/alias/JIT/nonzero-baseline evidence: [smallm-v3-runtime-validation.json](smallm-v3-runtime-validation.json). It parses only source data and pure source-generation functions; it never imports torch, prepares a module, loads an extension or remeasures a GPU.

## Actual selected paths and case decisions

Every row below has successful full-K one-hot x8 bit-exact checks against W and normal/outlier checks against FP64 at relative-L2 tolerance .004 (row tolerance .02), all five graph mutations (normal, outliers, zero, all probes, fresh normal) with NaN output/fresh workspace protection, and both immediate F1/paired terminal-output checks. Every corresponding source-bound counter check passed. Labels refer to independently prepared states, not a forced common dispatch. The displayed residual is `_rqval`'s rounded final outlier random-case residual; full-K probes are checked bit-exact separately.

| Shape | pw36 selected / check residual | rs32 selected / check residual | increment selected / check residual | Decision |
|---|---|---|---|---|
| qkv32 | RQ t32s6hze KS6 / .0036 | RQ t32s6hze KS6 / .0036 | RQ t32s5hze KS6 / .0037 | PASS all3 |
| qkv16 | GV15 S1 V1 / .0017 | GV15 S1 V1 / .0017 | GV15 S1 V1 / .0017 | PASS all3; cursor dormant |
| pooler32 | RQ t32s5hz KS4 / .0032 | RQ t32s5hze KS4 / .0031 | RQ t32s5hze KS4 / .0032 | PASS all3 |
| unpooler32 | RQ t32s5hze KS2 / .0024 | RQ t32s5hze KS2 / .0024 | RQ t32s6hze KS2 / .0024 | PASS all3 |
| o32 | GV47 S1 V1 / .0017 | GV47 S1 V1 / .0017 | GV46 S1 V0 / .0017 | PASS all3; cursor dormant |

The exact GVI tuples reconstructed from each pinned module agree with runtime IDs: GV15 `(1,2,2,2,8,16,1,1,0,1)`, GV46 `(1,4,1,1,4,4,1,1,2,0)`, GV47 `(1,4,1,1,4,4,1,1,2,1)`, fields `(FMT,MT,ST,WR,WK,KU,S,RT,XS,V)`. All have S1 and no P1 bits. sel is None for these cases, so `_r6` reaches those GV paths and no global counter is written. They are valid actual-serving controls, not evidence of an active M16 cursor improvement. RQ n matches the requested n; the guarded `_rqcall` rejects any decline. All selected RQ variants have PRB0 and FL bit1, so their last-poller two-array reset0 protocol applies. Their selected attributes are168 registers, zero local bytes and83080 shared bytes for ST5 or99488 for ST6.

## Counters and physical aliases

The logged prepare baselines contain73 counter paths over70 physical allocations; all views have int32 dtype, stride1, offset0 and cover their whole recorded storage. Three exact alias groups exist:

- pw36 unpooler m32/m16 share1024 bytes. The active m32 RQ reset0 obligation governs this shared allocation, including its dormant m16 dictionary alias.
- rs32 qkv_gate m128/m256 share144 bytes, dormant in all requested cases.
- increment qkv_gate m128/m256 share144 bytes, dormant in all requested cases.

For qkv32/pooler32/unpooler32, the exact selected RQ storage is active and must be all-zero at entry and checked completion. Every other allocation is compared exactly with its prepare baseline. For qkv16/o32, every counter is dormant and compared exactly. The pinned source executes25 invariant checkpoints per shape: prepared boundary1, post-basis3, mutation15 and immediate terminal6. All125 checkpoints completed, checking all73 paths each time, and no invariant-failure payload exists. The source snapshots the whole physical allocation, so offset-view truncation cannot hide a mutation. Immutable W/pack/escape state is also checked at those boundaries.

**Actual nonzero prepared names are now observed in v3:**39 P1 tensors and3 rw_cnt tensors. The first logged nonzero is `pw36/p1cnt/92`:128 nonzero words, starting `[18864,4716,18864,4716,...]`. Its GVI92 tuple has S4/V134217729, which selects the monotonic P1 ticket/flag protocol; the sample ticket equals4 times the flag. pw36/rs32/increment each have nonzero p1cnt IDs82 through94, and a nonzero dormant rw_cnt. Full names, counts, initial digests and first32 values are retained in the validation JSON. These remain dormant for every actual selected path above and passed exact-baseline comparisons. This demonstrates why valid prepared nonzero P1 state must be preserved; **it does not identify the exact failed tensor in attempt01**, whose ordinary saved payload contains no counter snapshot.

Successful final tensor contents are not independently serialized: this counter PASS rests on the pinned fail-closed runtime assertions and recorded successful post-check checkpoints, not a new physical readback or invented values. The logged initial digests/values are direct observations; preservation/reset conclusions are source-bound execution evidence.

## JIT, SASS and correctness completeness

Independently reconstructed each exact generated rq32 CUDA string from its pinned RQV/RQSBAKE/RQCU and pure generation functions. The three reconstructed strings equal the collected `.rq32.cu` bytes and their result hashes. Pw36 and rs32 share source SHA `3d44b42ce1befa00f67f5a6ef57bfcc1211d9acba86291a9961c5a7011e2ab52`; increment uses `645d9c6b789570114d2948b26f5e344e68a48dabfab6a41e65cf100df655a129`. Their extension identities differ only when source differs. Runtime SASS hashes match result metadata, and pw36/rs32 SASS are byte-identical.

All11 functions in each collected JIT resource report have REG168, STACK0, LOCAL0 on sm90a. All three pre-admitted affected symbols match their frozen before/after opcode multisets exactly. A full eleven-function comparison additionally identifies **t16s6hze** as changed by the same source LDS_CURSOR predicate: affected functions are NT16/32 × ST5/6 with GS1, PRB/WM/INS0, FL35. All four preserve LDS/SYNCS/HGMMA/WARPGROUP opcode counts and remove one UIMAD.WIDE.U32; the seven other functions have identical opcode counts. The fourth body was compiled but did not serve qkv16 here. There is no claim it was timed as an active M16 path. This closes the full-JIT review beyond the original three-symbol frozen SASS gate.

The raw JSONL and result.raw arrays agree exactly:770 finite-positive samples,154 per shape. Each shape has9 warmups,55 F1 calls (two control blocks before/after, one candidate block) and90 paired calls in the specified six-order rotation, using the exact expected seeds. There are15 basis checks,75 graph mutation validations,10 terminal records covering30 strict per-label output oracles. F1 terminal last fixtures are20011 for all labels; paired last fixtures are20069. Source inspection confirms the terminal oracle runs before any further candidate replay/output/workspace mutation, validates unchanged current X, reads every label's last timed output, and invokes the counter/immutability gate. Finite values, zero output, fresh input, probes and independent reference checks all ran through the pinned implementation. No favorable median substitutes for these gates.

## Cleanup and limits

The collected host log records `TRIAL_EXIT okc-proj-smallm-cursor-v3-01 0` at2026-10-27T04:11:07Z. The collector's actual probe records no guard PID, container, compute PID or other container, GPU memory0 and cleanup_failed=false. The collector observed cleanup before and after collection; its inventory hash is bound in the WAL. This is actual recorded cleanup evidence, not timer expiration or the driver's cleanup comment.

The result's generic `VALID_FOCUSED_DEVELOPMENT_ONLY_FULL_SCOPE_REQUIRED` status does not erase the explicit five-shape evidence above and does not grant official promotion. Shape-specific performance attribution also needs care: independent prepare selected different ST variants for qkv32/unpooler32 and different regular GV variants for o32. Metrics/source analysis is separate; these validity gates support that analysis. Root must separately validate cumulative PW37, all65/30 cases, cold prepare/check and official H200 requirements before claiming an official gain.

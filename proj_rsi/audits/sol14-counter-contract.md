# Independent smallm counter contract audit

CPU/static review only. Read ITERATION.md. No CUDA context, torch import, extension load, prepare, benchmark, frozen-source edit, queue edit or commit. Prior failed attempt is INVALID; none of its partial performance is admissible.

## Failure attribution

The frozen v2 driver's line162 tests every tensor reached through a path containing `cnt` against zero except `rw_cnt`, regardless of selected serving path. It does not persist the failing path or its baseline/current values: the generator feeds `all`, which returns only a Boolean, and the exception has a generic message. Thus a saved result/traceback from this code cannot recover the failing tensor. Exact failed tensor and causal attribution are **UNKNOWN** unless a separately captured state payload exists. Inspected archived `results/proj-rsi/takeover/20261027/smallm-increment01/result.json`: status INVALID_NO_PERFORMANCE_ADMISSION, raw sample count 0, only qkv32 reached validation, three full-K basis checks recorded `ok 3.6e-03`; first graph normal invariant then failed. No counter snapshot exists in this result. A valid prepared nonzero P1 counter is a source-supported hypothesis, not an observed cause.

`_gvcall`'s “self-zeroing” comment refers to zeroing output; it does not promise reset of ticket bookkeeping. At lines738/794/799, P1 counters increase. Line1083 consumes a ticket-derived zero flag. Nonzero state after prepare is valid for this protocol. Prepared dormant P1 state must remain exactly unchanged while another path serves.

## Dispatch binding and scope

[CPU proof](sol14-counter-contract-cpu.json) binds exact source hashes and confirms identical ASTs for `kernel_fn`, `_r6`, `_v12`, `_gvcall`, `_bk`, `_cublas`, `_rqcall` across pw36, rs32 and increment. This is source evidence, not runtime picks.

| Requested shape | Lookup key | Exact priority, for each of three labels | Runtime selected path recovered here |
|---|---|---|---|
| qkv32 | (`qkv_gate`,32) | matching RQ -> matching sel -> matching gv -> LT/cuBLAS | all labels record matching RQ `t32s6hze`, KS6; return code not journaled |
| qkv16 | (`qkv_gate`,16) | matching RQ -> matching sel -> matching gv -> LT/cuBLAS | UNKNOWN: attempt did not reach this shape |
| pooler32 | (`pooler`,32) | matching RQ -> matching sel -> matching gv -> LT/cuBLAS | UNKNOWN: attempt did not reach this shape |
| unpooler32 | (`unpooler`,32) | matching RQ -> matching sel -> matching gv -> LT/cuBLAS | UNKNOWN: attempt did not reach this shape |
| o32 | (`o`,32) | matching RQ -> matching sel -> matching gv -> LT/cuBLAS | UNKNOWN: attempt did not reach this shape |

RQ requires `rp['n']==n` and `_rqcall()==0`; a declining RQ may fall through. A repaired driver should HOLD a decline rather than silently change its active-counter model. sel/gv require `p[4]==n`. `_bk(16)==16`, `_bk(32)==32`. `_cublas` is true for all five shapes, so a complete fallback touches no `state['cnt']`. The existence of an rq plan alone is insufficient to prove it served. Runtime records need layer/m/n, rq scalar policy, selected GVI tuple/V bits, sel/gv plan ID, lt index, resolved route and launch return.

## Source-bound protocol table

Line references below are to frozen increment submission.py; identical dispatch ASTs across comparators were checked.

| Selected route/protocol | Physical counter supplied | Required boundary contract | Source |
|---|---|---|---|
| RQ ZI (`FL&2`, `PRB==0`) | exact `rp['cnt']` storage | zero at entry/after completed launch; last KS poller resets both tile arrays | 3900–3926, `_rqcall`4388 |
| RQ non-ZI | same argument, no ZI writes | exact baseline unless another source branch is proven to write it | `if(ZI...)` at3900 |
| GV P1 V bit27 or29 | `p1cnt[p[0]]` | active live-tile ticket += S and flag += 1 per completed launch; all unused words unchanged; or HOLD without exact launch counting | 738,794,799,1083; selector1981 |
| GV regular S>1, V&4, no high reduction bits | `gvcnt` | last arrival resets each used tile; untouched words baseline | 1110–1122 |
| GV cluster S>1, !(V&4), no high reduction bits, sm90+ | argument `gvcnt`, unused | exact baseline | 1124–1141, launch clu1190 |
| GV S==1 or redadd/PDL V26/V28 without P1 | argument `gvcnt`, unused | exact baseline | 1072–1105 |
| RR p0>=1000 | none | every prepared counter dormant unchanged |1988 |
| router p0==-7 / p1>=100 | `rw_cnt` | active monotonic protocol needs separately bound progress; HOLD here | `_rscall`3413, kernel3119 |
| router p0==-7 / p1<100 | `rs_cnt` | RED-dependent reset/no-write; HOLD until exact variant bound | `_rscall`3418,1350–1363,1453–1465 |
| LT/cuBLAS fallback | none | every prepared counter unchanged |3557–3562 |
| v12 run/run2/run3 (outside these five complete fallbacks) | `cnt` | last reducer resets used words; untouched words baseline |230–237,413–422,501–508 |
| Any dormant protocol | original prepared tensor | exact baseline, including nonzero valid state | no serving pointer reaches it |

Do not infer a protocol from an arbitrary substring. Bind known counter objects through actual launch arguments. Read-only packs/escape tables stay immutable even if a path happens to contain `cnt`; names alone are not proof of mutability. Sink/workspace arguments require a separate bounded scratch classification.

## Physical aliases

`_rqbake`/small-M bake allocate `cnt` once per layer and reuse it in plan dicts across m=32/16/128 (4567–4591). A dormant plan can alias active storage: resolve obligations by physical storage, not dictionary path, so the active reset contract governs that allocation. Record device, storage base pointer, storage nbytes, storage_offset, dtype, shape and stride. First-element `data_ptr()` alone does not detect offset aliases. Different views of one allocation must receive a proved byte-range/stride analysis or HOLD; do not drop the second view. Exact identical views may be deduplicated with all alias names retained.

## CPU controls and decision

[Standalone proof](sol14_counter_contract_cpu.py) uses no torch or candidate imports. Fourteen controls pass: dormant nonzero unchanged; dormant mutation rejected; active RQ zero accepted/nonzero rejected; P1 exact increment accepted while unchanged/wrong flag/wrong ticket rejected; unknown protocol HOLD; offset overlap detected; disjoint ranges separated; shared m16/m32 RQ aliases both receive active-reset, unrelated storage remains dormant; three-source dispatch binding identical. P1 is a policy-model test with no overflow and exact launch accounting, not admission of an active GPU path.

**Current decision: HOLD full-scope GPU execution pending frozen v3 review.** Required: exact route resolution before active contracts, storage alias safety, dormant equality, active reset requirement only for source-proven reset modes, active persistent protocol HOLD or exact transition, failure path+full values persisted. No performance conclusion follows from these CPU checks.

Draft v3 review: source SHA `de801e30d91c7d163200d30474b2868417f38644e7602cf0a59f36d5653a9e03` has correct selector priority, RQ-decline guard, no-counter GV distinctions, full-allocation int32 snapshots, source-bound active storage identity, dormant exact equality, immutable alias-view checks and full failing values. Direct `counter_cpu_proof()` passes without torch/candidate imports. Full `--cpu-check` initially fails because v3 manifest is not yet present. Semantics are suitable for a fresh invalidity-diagnosing attempt; final GO requires freeze/manifest verification. Conservatively requiring a full active counter allocation zero may reject valid untouched tail words; that is a disclosed false-invalid limitation, not permission to omit the check or claim speed evidence.

## Frozen v3 final decision

**GO for root-only fresh diagnostic execution over all five shapes**, with runtime HOLD on unreviewed active protocols; no performance admission unless the complete attempt passes. Independent review pins driver `bdc7cb89f1d0afa1604bc855601ceb6fe46f0c5237de5025f18800ba67062f2f` and manifest `e9fa7ba4aaa1cfb5b9075701164fa0d50340044de106f819756ccb377c273c33`. Independently reran frozen `--cpu-check`: PASS, including source pins, manifest file hashes, cursor proof, routing guard proof, dormant nonzero acceptance, dormant mutation rejection, active-RQ nonzero rejection, shared m16/m32 storage and offset-alias mutation rejection, full failure values/digests. Saved output: [frozen CPU check](sol14-v3-frozen-cpu.json).

This supersedes the earlier pending-freeze HOLD. The diagnostic may itself HOLD if actual prepare selects active P1/router/cluster; that abort is intentional and cannot become a performance result. The candidate sources remain unchanged. Neither this GO nor the repaired contract establishes that P1 caused attempt01.

# V4 fixed-K address specialization / independent preflight

`decision=VALID_PROGRESS_OVER_V3_REGRESSION_V1; GPU=TERMINAL_EXIT0; family=ACTIVE; official_or_incumbent_gain=NO`

Authority ITERATION.md. Reviewer read frozen source/lab/proofs/SASS; no candidate/source/manifest edits and no GPU/remote job. Root owns execution, resource cleanup and any integration.

|artifact|SHA256|
|---|---|
|manifest10|90a7a4b8c90287c2982619564f4025f4cf744b1e9b26aa621f7f58770ac6f786|
|generated source|5b6c3fb45f686b07a203ba7fb6feb0bdd5329cb48fa105b2a39b530759f7fee8|
|device source|44142ea73d399444dcd2c81c0dec3d0fa3570da12d980c39554748b2245eaa7d|
|body|133a14958c9a4c87cd2e436bdfb8c1725fc20393385d4cdcca2ac2e6fe8f17c6|
|lab|9419eb00d182ff12d7e11ca83cf4a53b2fbf62daff8bfedbde8adaa62c984315|
|SASS|20059f50dc1e30ac4011219265997f72dd01845bd79486e6eaf987b5ae1d2be0|

Independent executions: all10manifest hashes/generated text exact, `sourceproof()` restoring whole generated V3SHA3f90a6c7…c2b9, `addressproof()` all524288indices/maxbyte1048575/no32bitcomponentoverflow/8Balignment, Xall65536payload labels/sector map, rawQ4/A2lifetime with62negative absent-retirement controls, output8192scalar bijection, archived SASS address gate. All PASS, no files rewritten. Source except two address regions/kernel names is byte-identical to V3; actual lab diff changes version labels/controls from incumbent/V1/V2/V3 to incumbent/V1/V3/V4 and result names, retaining the numerical/timing/cleanup method.

Address equivalence for active math threads:

```text
row0=64wg+16w+g, xlane=2(t&2)+8(t&1)
xrow=uint64(X)+(uint32(row0)<<13)+(uint32(xlane)<<1)
addr=xrow+(uint32(s)<<7)
offset[2j+h]=32j+65536h
addr+offset-X =2*(4096*(row0+8h)+64s+16j+xlane)
```

Host exactM128/N8192/**K4096**/KC64/KS1 guard remains. Active starts0..1048568/lastbyte1048575; every vector aligned8B. Producer-role rows outside M use integer address arithmetic only and never dereference/construct an invalid C++array pointer expression. No runtime K multiplication remains; immediate displacements fit actual assembled addressing and retain eight-load order. Same8B nc/L1no_allocate policy, pair-lane demux, W6/B8/plain-output, precision/codec/escapes, Q4/A2four-ahead and all barriers.

RawQ is separate from asynchronous A; demux snapshots inputs/predicate before tied outputs. Prefetch can overwrite rawQ after demux while A remains pinned until the covering next wait1/old-A pin; A reuse occurs only two stages later. Final wait0 and both A/D pins precede decoded-B/output reuse. Compiler now schedules some future rawQ loads earlier than current HGMMA; source dependencies/lifetimes permit it. This is a lowering consequence of the address revision, not a separately selected optimization, and any timing delta cannot isolate arithmetic removal from scheduling.

Actual local sm90a150regs/57600logicalCTAregs,0stack/spill/LDL/STL/USETMAXREG/C75,103600dynamicshared.16HGMMA/4WGarrive/oneoutputTMA/noX-TMA. Same64main fixed-mask BFLY+32outlined alternatives as V3; do not sum alternatives as dynamic execution. Every group of8LDG64 sites shares one64bit base and the exact8immediate displacements. Main-CFG static instructions504→350, IMAD.WIDE32→0, LEA.HI33→1, same32refill-LDG sites. These are verified static code changes, not cycle/phase/port measurements. Full runtime Torch build/attrs/opt-in occupancy/local checks remain mandatory.

Four retained actual graphs: freshly prepared pw36 actual`kernel_fn` without forcing its pick, frozen best predecessorV1, immediate predecessorV3, V4; separate real outputs/counters, no timed baseline copy. Every path full4096basis/FP64/normal/outlier/input oracle;20actualtimedgraph poison/mutation checks, original-X restore, W/packed/escape exact, counters0, expected single kernel profile identity, all4graphs reset in finally. F1 controls before+after candidate/min median,21samples/series,21rotated/reversed4way pairs,256MiBflush outside event region and5discarded warmups/path. Report all3deltas; >=.5us vs actual incumbent BOTH F1+paired is review-only survival. Improvement over failed predecessors is learning, not a serving or official-score win. Preserve invalidity/regression data and iterate per contract.

GO root-only exact command:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v4/lab.py \
  --json /okc/logs/operand-reverse-v401.json
```

One normal-capability exclusive named container, inner240s/outer300s after prior context drains. Frozen source unchanged during run. Independent raw timing/validity/resource/termination review required before promotion; no reviewer GPU action.

## Independent terminal review: valid address-policy improvement over V3

Actual guard373129/container`okc-proj-operand-reverse-v401`, terminal2026-10-26T23:48:38Z, exit0. Reviewer independently confirmed guard/container absent by read-only SSH; root reported0MiB/idle before its separate zero diagnostic. Frozen identities match above. Raw`proj_rsi/candidates/operand_reverse_v4/operand-reverse-v401.json`17646B SHA256`5cf4cbc292188fd4ed68ec92255b2f4995a4752835ef26ada628669d3268bb6b`; hostlog`results/proj-rsi/takeover/20261026/operand-reverse-v401.host.log`13896B SHA256`3c047072ff277e7d25b874426aa4cfd492b50336f453912726cd48c3184bad57`. No Traceback/invalid/CUDA-error marker. Status`V4_PROGRESS_DIAGNOSE_NEXT_LIMIT`; no official/full-submission claim.

All4full4096basis/FP64/normal/outlier suites and20actualtimedgraph poison/mutation/input checks pass; W/packed/escape exact/counters0; all4graph execs reset. Runtime V1/V3/V4regs168/152/150, all local0/shared103600/maxDynamic49152→103600/opt-in occupancy1. Actual incumbent selects`s4s5hz`,KS2; all4profiles are the expected single source-specific kernels. Instrumented profile durations are not raw F1/paired samples.

Reviewer independently recomputed every raw median, faster-before-after control and all21paired differences:

|comparator|before F1us|after F1us|conservative F1us|V4F1gain us|V4paired gain us|V4wins/ties/losses|
|---|---:|---:|---:|---:|---:|---:|
|actual incumbent|24.191999808|24.191999808|24.191999808|−27.904001996|−28.416000307|0/0/21|
|v1|47.168001533|46.943999827|46.943999827|−5.152001977|−4.960000515|0/0/21|
|v3|55.135998875|54.783999920|54.783999920|+2.687998116|+4.608001560|21/0/0|

V4median52.096001804us. This supports the complete address-specialization policy improving V3 in both timing methods; it remains slower than best predecessorV1 and actual incumbent. Static arithmetic removal, register reduction and backend scheduling change together, so the measured gain cannot be partitioned into those costs. Preserve this progress and proceed to a specific new mechanism such as per-call fresh fragment packing only after its source/lifetime/full-graph review; do not infer an incumbent win, unique stall cause or whole-family failure.

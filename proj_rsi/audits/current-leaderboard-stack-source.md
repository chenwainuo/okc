# Current leaderboard stack: source and ancestry audit

The local source associated with winning job `fb548b5dc6f04696` is RS32:
`proj_rsi/candidates/rs32/submission.py`, SHA256
`a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`.
It is byte-identical to `git show fbf6371:proj_rsi/candidates/rs32/submission.py`.
The archived official profile reports 3200.6150084373335 score-us, 1.509334920715312x,
9/9 passed. Root separately verified the live board; this worker performed CPU
source/history inspection only, with no GPU, build or submission.

**Uploaded-byte identity has a remaining evidence limit.** The live API supplies
`content_hash=cd14084a19ec749cfaf6868fb3001b17863cad87f2a3745e02ae3476131ba86a`
and a payload_ref using that value. That value differs from local file SHA256;
the hash algorithm/content normalization is not documented in this repository.
The archived normalized report names the RS32 path; the historical tool action
at 2026-10-26T19:09:30.995Z submitted
`python3 -m proj_rsi check proj_rsi/candidates/rs32 --experiment rs32 --scored --against 846e25d9df09433f`,
after commit fbf6371 at19:07:52. The surviving submission shell script explicitly names
that commit and candidate; its full text and SHA are retained in the machine
companion. But runner.py:53–74 records path/job/time, not
an uploaded-byte digest; the fb548 submitted descriptor is absent. Consequently
this audit proves the local associated source's stack and the official profile's
actual kernel choices. It does not claim raw API content_hash equals source SHA256
or independently attest the uploaded payload byte-for-byte. Root's API evidence:
`results/proj-rsi/takeover/20261027/live-rs32-audit.json`.

The machine companion is `current-leaderboard-stack-source.json`: 18 component
groups, all30 official layer/M routes, source and AST-segment fingerprints,
comparison results across15 candidate sources, exclusions and evidence limits.
Every component below has a source line reference; the JSON supplies the full
parent file SHA and exact named-AST equality witnesses. Isolated gains for each
individual mechanism remain unmeasured unless the separate official-gains audit
provides a controlled comparison. Source presence alone is not a speedup.

## What actually ran in this best job

Each cell describes the official profile, not an assumed result of prepare on a
future device. `_bk` maps M to buckets at RS32:1928; dispatch priority is exact
RQ → second-bake selection → r6 GV → v12 fallback at RS32:3525–3577.

| M / task load | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
|1 / decode1|GV short-chain V8193|GV|GV rolled V3|P1 GV selfzero splitK4, n1024|Direct FP32 RR1024|
|8 / verify1|GV short-chain V1|GV|GV rolled V3|P1 GV selfzero splitK4, n1024|P1 GV selfzero splitK16|
|16 / decode16|RQ NT16/ST5/FL35|GV|GV|GV|P1 GV selfzero splitK16|
|32 / decode32|RQ NT32/ST5/FL35|RQ NT32/ST6/FL35|RQ NT32/ST5/FL35|RQ NT32/ST5/FL35|P1 GV selfzero splitK16|
|128 / verify16|RQ NT128/ST5/FL3|cuBLASLt|RQ NT128/ST5/FL2|cuBLASLt|Old RW cp.async/WGMMA, MODE4 selfzero|
|256 / verify32|cuBLASLt|cuBLASLt|cuBLASLt|cuBLASLt|cuBLASLt hi+lo concatenation, splitK reduction, final sum|

Evidence: `proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.txt:43`
through its30 profile sections, and the matching JSON `profiles[index]`0–29.
The JSON companion preserves every exact kernel-report row. Source RQKS at4355
is qkv7/pooler4/unpooler2/o2 for M128/256. Official small-M bake rows in
`results/proj-rsi/jobs/rs32.scored.txt:132` onward select qkv32/16 KS7,
pooler32 KS4, unpooler32 KS2, o32 KS2; profile template widths/stages confirm
those routes. FL35=EARLY32+selfzero2+hints1. FL3=selfzero+hints; FL2=selfzero.

The H100 box NOTES say allM16 and oM32 keep r6; that is **not** what the official
H200 job selected. Similarly, unpooler128 is cuBLASLt here, despite RQ being
eligible. AllM256 BF16 projection RQ alternatives were rejected by this prepare.

## Included stable mechanisms

1. **Lossless 12-bit BF16 weights and exact escapes.** `_c12` chooses an8-value
   high-byte window and packs each value's low8bits, sign and3-bit window delta;
   escaped high bytes are corrected by XOR. `_unc12` independently restores all
   bits. Functions are exact AST matches from r6-c12 through RS32, with source
   fingerprints in JSON. Refs RS32:1863,1889,1908,3743,4308. This is lossless
   compression, not replacing weights with a lossy12-bit approximation.
2. **MMA-oriented immutable weight packing.** GV `_pack` reorganizes weights for
   warp MMA fragments; RQ `_pack_nat` retains naturalK for X TMA and reblocks
   eight16-row tiles into one12288B K64/128-row chunk. `_pack` matches r6-c12;
   RQ helpers match PW34/RS128. Refs RS32:2364,4300,4308,4321. The comment names
   PW10's layout reuse; it does not establish that every PW10 kernel proposal
   became an ancestor.
3. **Register-direct GV streaming.** Per-warp vector W loads use L1 no_allocate;
   a register ring and9-op add/funnel-shift/PRMT decoder feed register-A
   `mma.sync m16n8k16`, with FP32 accumulators and decoded-weight reuse across
   token tiles. Rolled or unrolled loops and X-register/L1 modes are selected
   per geometry. Refs RS32:537,568,708,758,935,1039. GVCU is exact PW27h/PW31
   content; selected vanilla GV shapes appear in the route table.
4. **GV local reduction with retained cluster/global fallbacks.** WK warps sum
   FP32 shared partials and emit directly forS1. Other retained templates use
   cluster DSMEM or global last-arriver cleanup; selected P1 paths instead use
   red-add. Refs RS32:1069,1091,1106,1125. In-CTA reduction is old r6 behavior;
   the presence of dormant cluster code is not a new current gain.
5. **PW33 short QKV dependency chains.** More K-warps and shorter per-warp KU,
   two-row-tile sharing, ST1 and a minBlocks2 candidate extend the GVC bake.
   OfficialM1 chooses V8193 andM8 V1. PW31→PW33 changes only GVC among named
   AST nodes; commit eb4412e and `pw33/hypothesis.json` record the actual change.
   Refs RS32:1660,1664,1665. The currentM16 winner is RQ, not this GV candidate.
6. **P1 one-node selfzero split-K.** RouterM8/16/32 and small unpooler use
   per-instance persistent zero flags/tickets, acquire waits and red-add,
   avoiding the cluster/memset path. Router reducesFP32; unpooler uses packed
   BF16 reduction with additional rounding. Bounded spin traps prevent an
   endless zeroer wait. Refs RS32:581,1081,1793,1800,1806,1978. This is the
   PW27h branch merged into PW31; RQ uses a different reset-to-zero protocol.
7. **Direct FP32 routerM1.** RR reads originalFP32 weights, computesFP32 FMA,
   and reduces within aCTA; rr6's1024-thread variant is selected. It does not
   use the hi+lo approximation. Refs RS32:1213,1971,2036.
8. **BF16 hi+lo router precision path.** For tensor-core paths prepare forms
   `hi=BF16(W)` and `lo=BF16(W-FP32(hi))`; both contributions accumulateFP32.
   Large-M fallback concatenates hi/lo into oneGEMM and adds output halves.
   This approximates originalFP32 weights and is checked against theirFP64
   product; it is not a claim of exactFP32 representation. Refs RS32:2380,
   2948,3423,3564,2477. The12-bit codec is not enabled for the selected ordinary
   hi+lo router GV arm (`_gvprep` at2005).
9. **Old RW shared/shared WGMMA router.** Router128 uses cp.async grouped/ring
   loading of X/hi/lo intoSW128 shared layouts, SS WGMMA with both terms in the
   sameFP32 accumulator, vector red.global.add.v4, and MODE4 selfzero monotonic
   counters. The sm90a architecture-build fix is included. Refs RS32:3020,
   3283,3300,3322,3342,3413. RWCU and five associated router AST nodes are
   exact PW34/PW29h matches. This is not PW35's TMA-producer RTK kernel.
10. **RS128/RQ swap-AB producer/math pipeline.** CompressedW arrives by bulk;
    X arrives by2D TMA SW128. Math warpgroups decode W into register-A while
    shared X is operandB. A merged full mbarrier has two producer arrivals;
    W slots release afterLDS and X slots after WGMMA retirement. The384-thread
    CTA redistributes registers producer40/math232; first MMA scaleD0 avoids
    an accumulator clear. Refs RS32:3581,3720,3790,3862,3932,3959,3982,4058.
11. **Staged escape cursors before the W flood.** RQ preloads a32-entry
    lane-spread escape window and advances by shfl, refreshing every32entries
    rather than a dependent global load per escape. Refs RS32:3752,3829,4330.
    This RS128 R2 mechanism is included; its individual gain is not separately
    established by this source audit.
12. **Fused RQ epilogue and selfzero handshake.** InlineFP32→BF16x2 conversion,
    stmatrix.x4.trans into aSW128 slab and one elected issuer of TMA BF16
    reduce-add produce output without a separate memset/reduction node. Idle
    producer warps zero disjoint1/KS slices, then publish; final pollers reset
    counters. Refs RS32:3781,3903,4122,4139,4154. This is inherited RS128
    machinery adapted toNT, not rs128b's later half-token EH epilogue.
13. **RS32 NT16/32 specialization.** Adds m64n16/32 WGMMA widths, NT/2
    accumulators, smaller X boxes and NT/16 epilogue loops. ST5/6 and KS
    alternatives are selected after GX against the actual incumbent. Refs
    RS32:3644,3792,4275,4511,4516,4532. Package-level controlled improvements
    exist in official bakes; no decomposition into widths/depth/KS is asserted.
14. **Early initial loads and L2 hints.** EARLY bit5 issues firstmin(ST,KCS)
    W+X from the elected W lane before theCTA barrier; hints evict W first and
    retain X last. All selected small-NT kernels use FL35. Refs RS32:3817,
    3839,3868,3887,4516. Hints also run onqkv128; pooler128 explicitly selects
    the no-hint FL2 variant. EARLY/hints have no isolated official gain claim.
15. **HALF/full-K alternatives are included but unselected.** FLbit4=16
    supports64W rows/CTA, one mathWG, doubledrowgrid, and aKS1 plainTMAstore
    variant. Shipped RQSBAKE tries it ono128; the official job retainscuBLAS.
    Refs RS32:3813,3933,4152,4522. OtherHALF experiment names remain defined
    but are absent from the production-build list. No realized HALF gain here.
16. **cuBLASLt search and numerical restrictions.** Prepare compares torch
    default/heuristics and a bounded layout/algorithm search, including valid
    splitK settings. FP32 accumulator/TF32 restrictions and output checks filter
    choices. Selected fallback shapes are listed above. Refs RS32:2128,2158,
    2229,2262,2279,2360,3555. `_tune` and `_ltsearch` exactly match r6-c12.
17. **Layer/M/N dispatch and fallbacks.** ExactRQ requiresmatchingN and a
    successful launch; second-bake/GV plans requirematchingN and use buckets.
    CPU executesplainmath; odd-M/unsupported/declined paths retainbaseline
    dispatch. Refs RS32:1928,2373,3525,3545,3555. Short-unpoolern1024 is the
    task's output shape, not an optimization that omits required channels.
18. **Parallel build and validity-first prepare bake.** RQ/RQ32 builds overlap
    r6 tuning on separate threads; orderisr6→RQ128/256→GX/P1→RQ32→router.
    There is a240s totalbudget, per-phase guards, packroundtrip/allKonehot
    exactness, RQnormal/outlierFP64 andcounterzero checks, bucketodd-M checks,
    a1% margin, and RQ F1 baselines before/after plus discardedwarmup.
    Refs RS32:2377,2403,2418,2433,2451,2477,4365,4388,4422,4532,4570,4610.
    These controls enable safe selection; each is not an independent speedup.

## Proven ancestry and excluded branches

The important lineage is a merge graph, not ascending version numbers:

- r6-c12 supplies the exact codec, old CUDA fallback, packing andcuBLAS tuning.
  PW21→PW22→PW23 router branches are documented by commits5998581,d631b83,
  a114124/6f36312. PW23 is the recorded commonbase ofPW27h/PW29h's merge.
- PW31 commit2dd2a27 merges P1/PW27h plus oldWGMMA/PW29h overPW23.
  PW33 commit eb4412e addsQKVgeometry candidates; named-AST diff isGVC only.
- RS128 commits cca221c/7700384/afe782e/e8d7ab8 developRQ on aPW23base.
  PW34 integrates that branch intoPW33 (bea78c8), then fixes prepare availability
  (9eb8318 and49b22d8). The currentPW34 SHA is
  `958ad2a0197fe1f2bb57206494d9020cdc299b55f8c139a32befc7d76d4c547e`,
  exactly the source at49b22d8.
- RS32 is **PW34@49b22d8 plus small-NT/HALF/EARLY and second-bake plumbing**.
  PW34→RS32 changes namedAST nodes prepare,RQV,RQCU,_rqsrc,_rqext,_rqstart,
  _rqcall,_rqbake; adds rqmod2/RQS/RQSBAKE/RQPROD2/rqthr2/_rqsbake.
  Oldfallbacks/codec/GVC/router remain exactly inherited. The complete source
  diff is248insertions/39deletions; inheritedRQ code is extended, not byte-identical
  as one stalehandover sentence suggests.
- PW35 is a separatePW34branch addingRTK TMA producer/full-empty ring and
  selfzero router. Its SHA is
  `7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8`.
  RS32 has noRTK/RTV. Its hypothesis.json copied PW35's router path list;
  sourceAST andofficial RW kernel disprove that copied inclusion claim.
- PW36 mergesPW35's router withRS32, SHA
  `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.
  RS32→PW36 changes onlyprepare plusRWCU/_rwsrc/_rwext/RSPALL/RSP/_rsbake
  among namedAST nodes. PW35/PW36 router nodes match exactly; RS32/PW36RQCU
  matches exactly. PW36 is a measuredcombined candidate, not the best row.

Two small measured refinements must remain visible as **unmerged** rather than
silently being counted inRS32: rs128bEH/EP token-half epilogue and separateWring
(`proj_rsi/audits/mx-h100.md:216`,248,265). EH usesFLbit16=65536, unlike
RS32HALFbit4=16. RS32 has neitherEHbranch norEvariant names. The citedEH
lab improvement0.13–0.16us is not an included current-stack gain. rtk_early is
also excluded: it requiresPW35RTK and its+0.480/+0.096 graph values are losses
versus the fastest shippedcontrol (`rtk_early/RESULTS.md:11`), not positivegains.

CodecE, bulkGVring, originalwideGX andcachepolicy ladders survive as text but
are disabled byGX11False/GXPclear, unconditionalbulkcontinue, GEFalse,
GLADempty andemptyGCXloop (RS32:1708,1726,1759,1768,1826,1835). Neither
cleanroomlatencyDBG nor subsequentoperand-reverse/nativeX kernels are imported
or generated by this standalone source. These are not currentrealizedfeatures.

See `current-leaderboard-gains-official.md` for controlledofficial gain evidence
and the separate completeness audit for all positive-but-unmerged, marginal,
invalid and unmeasured outcomes. This inventory deliberately does not turn
cross-job score differences or presence of candidatecode into incremental gains.

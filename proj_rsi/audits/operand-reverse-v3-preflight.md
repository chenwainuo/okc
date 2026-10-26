# Reverse v3: coalesced8B/pair-lane demux independent review

`decision=VALID_PROGRESS_OVER_V2_REGRESSION_V1; GPU=TERMINAL_EXIT0; family=ACTIVE_ITERATION; predecessor_versions=IMMUTABLE; incumbent_improvement=NO`

Authority: current `proj_rsi/ITERATION.md`; a valid regression informs a successor and does not retire the family. Reviewer changes no candidate/source/manifest, launches no GPU/remote job, and owns only this audit. Root owns the exclusive named trial and integration/official decisions.

## Frozen identities / independent verification

|artifact|SHA256|
|---|---|
|manifest12|dc460284e95b930e98a214264f66a139dffa8bbe69ee0fbbc2a1af9fddd3db66|
|generated extension|3f90a6c77eeec6494c21019828ef01f93c0451f0d64e567f78a9ce0349abc2b9|
|device source|9895edb59a5924f56afb1147d1f5de36f393f1ee25d4bc3add4d6f9eb56df0f2|
|kernel body|c5cf2f4887bb85198f7c3dbc2387c75bdc6b9ed42b75b3848a3cae68ed455376|
|lab|d5f56b4678c01a2a1e31624e5eeea8af74e94ef77ff6bda20283bc516381eb62|
|SASS|271912eee132a509abd568b8e35c5e71eb91307907467e58a51d353ed703d831|

All12manifest hashes independently pass. Generated source/device text equals archived text. Reviewer independently ran CPU X524288symbolic-coordinate/all65536bit-label mapping,8B alignment/64touched=64unique-sector model, rawQ4/A2lifetime with62negative absent-retirement hazards, output8192scalar bijection, and bytewise producer/output/math-loop source comparison to v1. Pack/escape helpers are pinned rs32 SHAa7b6e010…5ad; v1ca92/v297664 and actual pw362a067059…bccf remain frozen. Inherited arbitrary-W repair/escape/window/SW128 proof is separate from a GPU control-flow proof; actual full-output tests remain mandatory.

## Source/lifetime/resources

One unpooler M128/N8192/K4096/KS1,128CTAs×384threads/channel64, W6/B8/plain-output policy. V3 uses8×8B`ld.global.nc.L1::no_allocate.v2.u32` per thread/stage, matching v1's nc/no_allocate policy. For t=lane%4, g=lane/4, row=wg64+w16+g+8h, k=s64+j16+2(t&2)+8(t&1), every uint2 load is8B-aligned. Two XOR1shuffles plus two selects/group map exactly to native A k=s64+j16+2t and k+8. All inputs and predicate are snapshotted/evaluated before outputs in opaque PTX, so compiler-tied input/output registers do not early-clobber a later input. Warp-wide shuffles use full active mask within warp-uniform math roles; no lane-divergent barrier branch was added.

RawQ4 stays separate from asynchronous A2. Initial Q contains stages0..3; demux currentQ→A(s%2), issue/commit group, refill the now-independent rawQ with s+4, wait_group1, then pin oldA/release B(s−1). A(s%2) is not rewritten until s+2, after s+1 wait retired and pinned its prior group. Final wait0 pins both A and D before final B release and output-slab reuse. W6/B8 producer barrier/escape/TMA logic and output epilogue are inherited byte-identically except removal of dynamic register declaration. Math loop differs from v1 only raw buffer type and demux call. Both WGs complete before the shared decoded-B ring becomes the output slab. CPU ordered lifetime model does not certify arbitrary barrier interleavings; complete runtime tests cover the actual launch.

Local sm90a compile152registers/58368logical registers per384thread CTA,0stack/spill/LDL/STL/USETMAXREG/C75. No dynamic pool requests. Actual runtime opt-in-before-occupancy>=1/local0/register/shared gates remain required; the local device-only build is not a full Torch-host/runtime certification.16HGMMA/4WARPGROUP.ARRIVE/oneoutputTMA/noX-TMA/all8192Dscalars used, so no partial-work/DCE basis for a speed comparison.

**Compiler evidence qualification:**96static BFLY consists of64fixed-mask main-path sites plus32outlined WARPSYNC.COLLECTIVE alternatives reached through two BRA.DIV entries for the first two unrolled demuxes, returning after the replaced code. They are alternate CFG paths, not96additive dynamic shuffles. Source and CPU proof request16SHFL+16SEL/thread/stage versus v1's32+32, but no dynamic execution or lowering speed benefit has yet been measured. Initial C++ and current opaque-PTX builds both152registers/96static BFLY; first source is not archived, while first SASS/resource evidence is. Do not present the change between those two prefreeze spellings as a measured optimization.

## Prediction and frozen four-way comparison

V1/v2 counter evidence reports requests262656→1049088 and L1sectors4196352→8390656, but misses stay4196352; extra4194304sectors are hits, DRAMreads52.152→51.473MB. Thus no doubled-HBM explanation is supported there. V3 predicts v1-like64sector touches/unique perwarp-stage while using2×v1 load instructions and half source conversion instructions; compared with v2, width/cache/staging/demux/register mode all differ. This is a complete loader-policy successor, not single-variable causal isolation or a speed guarantee.

Lab verifies own manifest plus both predecessors' executable/proof identities, builds all3candidate extensions before baseline import/prepare clock, and freshly prepares actual pw36 without forcing a policy. Four actual captured paths have separate retained outputs; candidate counters are distinct. Incumbent validator's copy is untimed; actual timed baseline graph has no added copy. Every path must pass all4096basis/FP64/normal/outlier and input invariance; actual4timed graphs get20original/zero/fresh-normal/fresh-outlier/probe poison/oracle checks, then original X restored. Final W/packed/escape exact/counters0 and profile-specific single rq/rq_v2/rq_v3 identities remain required. All4graph execs synchronize/reset in finally; errors persist invalid status.

F1 uses21cold samples/series with EACH incumbent/v1/v2 control before and after V3, conservatively taking its faster median.21four-way rotated/reversed paired rounds and5discarded warmups/path use256MiB flush outside each timed event region. Report all raw samples and all3V3deltas/win counts. >=.5us over actual incumbent in BOTH conservative F1 and paired median is only local survival pending independent raw review; improvement over failed predecessors is learning without an incumbent speed claim. Official H200 validation is still required for official score claims.

GO exact root-only command after any active named GPU container/context drains:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v3/lab.py \
  --json /okc/logs/operand-reverse-v301.json
```

One normal-capability exclusive named trial, inner240s/outer300s under existing host/container lock/deadline guard. No reviewer GPU launch. Preserve frozen source and all failed/valid results; follow the measured next question per ITERATION.md rather than automatic family closure or unreviewed variants.

## Independent terminal review: valid partial progress

Actual guard371832/container`okc-proj-operand-reverse-v301`, terminal2026-10-26T23:36:47Z, exit0. Reviewer independently confirmed guard/container absent by read-only SSH; root confirmed0MiB/idle. Source hashes match frozen review. Actual raw JSON`proj_rsi/candidates/operand_reverse_v3/operand-reverse-v301.json`17602B SHA256`4701566b96a7af729ce02eb4a041f7291d53520d83d1b63e54ed466b2b3da618`; hostlog`results/proj-rsi/takeover/20261026/operand-reverse-v301.host.log`13882B SHA256`9fecc7a281bcaa7ae4daa69b0593917214cbbb6542374d6d622a11d5f30b4854`. Root owns any further archive copy; raw version result stays attached to its exact source.

All4paths pass4096basis/FP64/normal/outlier; all3candidate supplementary original/zero/fresh-outlier captures pass; all20actual timed graph poison/oracle/input mutation checks pass. W/packed/escape bytes unchanged, all counters reset, all4real graph execs reset. Runtime V1/V2/V3 registers168/162/152, each local0/shared103600/opt-in occupancy1/maxDynamic49152→103600. Actual incumbent is freshly selected RQ`s4s5hz`,KS2. Profiles are the expected single rq/rq_v2/rq_v3 with real incumbent rq<128,5,1,0,0,0,3>; their durations are not admitted timing samples.

Independent reviewer recomputed all raw medians, faster-before-after controls, all21paired differences and win/tie/loss counts:

|comparator|before F1us|after F1us|conservative F1us|V3 F1gain us|V3paired gain us|V3wins/ties/losses|
|---|---:|---:|---:|---:|---:|---:|
|actual incumbent|24.032000452|24.159999564|24.032000452|−31.039997935|−30.368000269|0/0/21|
|v1|47.359999269|47.359999269|47.359999269|−7.711999118|−7.520001382|0/0/21|
|v2|56.671999395|56.639999151|56.639999151|+1.568000764|+2.176001668|20/0/1|

V3median55.071998388us, status`V3_PROGRESS_DIAGNOSE_NEXT_LIMIT`. This confirms progress over V2 for the complete8B/no_allocate/demux/staging/register policy while remaining slower than V1 and actual incumbent. No integration, port or official score claim. Do not count changed static shuffle alternatives as measured dynamic cost or attribute the paired gain to coalescing alone.

Next proposed V4 question: source/SASS shows repeated address-generation work in the8B loader; fixedK4096 is already enforced by the host guard, so specializing/hoisting that address arithmetic can be a valid distinct instruction-reduction hypothesis while preserving cache policy/demux/Q4A2/geometry/precision. The count of45instructions/refill/8IMAD.WIDE+8LEA+8LEA.HI is source/SASS evidence, not a demonstrated stall cause or speed forecast. Any successor must prove exact same indices/bounds/alignment/lifetimes, inspect actual compiled resource/CFG changes, and compare against V3 plus actual incumbent with full validity gates. Family remains active per ITERATION.md.

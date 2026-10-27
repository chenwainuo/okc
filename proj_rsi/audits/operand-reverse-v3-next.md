# Operand reverse V3: conditional next mechanism / terminal harvest

state=VALID_V3_PARTIAL_RECOVERY_NO_INCUMBENT_WIN; V3_RESULT=TERMINAL_2026_10_26_23_36_47Z; new_GPU=0; new_kernel=0; frozen_sources_modified=0.

Iteration policy: failed versions produce diagnosis and a reviewed revision; no family-wide terminal closure. Older reverse audit wording `CLOSED_NO_RETRY` records the policy at that capture and is superseded for the current authorized V2/V3 mechanism iterations. It does not discard the measured losses or authorize an automatic next launch.

## Provenance / result boundary

- V1 generated source ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8.
- V2 generated source97664bdee7593d46e0493b9599062c37af150262d597002a722016f060548faf; its valid unprofiled same-run F1:57.023998us vsV147.231998/incumbent24.224000. V2 is a measured regression, not an indictment of every direct-X policy.
- V3 manifest dc460284e95b930e98a214264f66a139dffa8bbe69ee0fbbc2a1af9fddd3db66; source3f90a6c77eeec6494c21019828ef01f93c0451f0d64e567f78a9ce0349abc2b9; four-waylabd5f56b4678c01a2a1e31624e5eeea8af74e94ef77ff6bda20283bc516381eb62. Critic GO; root launched23:35UTC PID371832/containerokc-proj-operand-reverse-v301, inner240s/outer300s.
- Expected rawlocal result: `results/proj-rsi/takeover/20261026/operand-reverse-v301.json`; root will supply/archive on terminal. Do not inspect partially written timing as a valid result.
- Actual comparator remains frozenpw36 `prepare/kernel_fn`; selected pick/profile must be read from this same job, not assumed identical to older RQ/Nvjet runs. Best official score3200.615008us remains unchanged until a complete official result.

## Evidence already available, with limits

1. V1 PC capture (FL3 actualbaseline): math3173samples;2388at transposePCs (940longSB/696shortSB/292MIO),244directXLDG (155LG), mathBfull2. Exposed X consumer/data dependencies and transpose issue work were strong associations, not a removable-walltime fraction. Producer153Wfull/80Bempty/419LDS-decode/326store/273fence-barrier. Producer decode's largest consumer uses LDS data; not evidence for codec integer throughput dominance. These are V1 PCs, not measured V3 stall ownership.
2. BaselinePC974/2039mathsamples=47.77% at the combinedW+Xfull barrier. It cannot assign the incumbent gap to W-only or decode-only latency.
3. Fresh V1/V2 memory diagnostic: rawCSV SHA05257785c07758fbb9c4854670b0b0b40dace63e62dad6794bccdc74ef898751, driverb278cbd479d2d1490af27b8707ed05c0a1fc5d5f4cf5e939dec91536377438e6. L1requests262656->1049088; sectors4196352->8390656; V2hits4194304; misses and L2-to-L1 sectors4196352unchanged. DRAMreads52152320->51472896B (~flat); L2TEXreadsectors5458382->5554136. Direct scalarV2 increased frontend requests/half-sector handling without increasing DRAM volume. It reduced normalized shortSB2.15->.26/MIO.96->.09 but increased longSB2.87->5.70/dispatch.11->.63; these are ratios across32metric passes, no PCs or time decomposition. Cache policy, prefetch/A-lifetime, and static/dynamic register modes differ.
4. V3 source/resource proof:8coalesced8B `.nc.L1::no_allocate` loads plus16SHFL/16SEL perthread/stage, rawQ4/A2/fourahead; uniform152regs/0spill/103600smem/oneCTAresourcebound. No dynamicallocation/deadlockpool.64mainpathstaticBFLY +32outlinedcollectivealternatives; no assumption that all96execute or that the outlined path is never taken. Firstcompile static-count inference was corrected and evidence retained.

Preregistered fixed-work predictions, not measured V3 counters:

```
warp_stage_instances =128CTA *8mathwarps *64stages =65536
V3 mathX LDG requests=8*65536=524288
V3 globalLDG requests=524800 IF unchanged512metadata requests
V3 X32Bsector instances=64*65536=4194304
V3 allglobalLDG sectors=4196352 IF unchanged2048metadata sectors
V3 SHFL+SEL conversion=32*65536=2097152 logicalwarp instructions
V1 conversion=4194304; V2=0
allV1/V2/V3 logical Xloads=134217728B; V2touchedsectors double despite same usefulbytes
```

A valid V3 timing result can support progress from the complete loader policy. It cannot alone prove the exact request/sector predictions or isolate width versus demux versus resource changes.

## Residual dependency tree

|condition after terminal V3 validation|interpretation permitted|next smallest evidence/action|
|---|---|---|
|JSON nonterminal / timeout / exception|No performance conclusion; identify exact failing stage and hostlog. Compile/launch-resource failure differs from an in-kernel stall.|Use root terminal log and result status first; compare actual attrs/sourcehash to152/0local/103600; inspect last completed oracle/capture; no blind replay or family closure.|
|all oracles pass, actual graph mismatch only|Specific graph/write/lifetime defect in this version remains possible; eager correctness insufficient.|Identify graph+mutation+first failing region, whether zero output/NaNpoison/input/weight/counter changed. Map to exact output or A/rawQ retirement contract. Fix only demonstrated violation in newV4; preserve V3.|
|valid, V3 beatsV1/V2 but not incumbent by>=.5bothmethods|Meaningful recovery of implementation loss; no incumbent winner or official speedup.|Rank remaining gap using current source plus only relevant new phase evidence. Do not stop iteration just because predecessor gap remains.|
|valid, V3~V1 or regresses vsV1 while betterthanV2|Fewer transpose ops did not produce a net win over the wider-load V1 policy; request/address/LSU work may offset it.|First inspect runtime resources and paired distribution; then belowH_XA/H_XD if source/PC evidence supports. Do not conclude transpose work was free.|
|valid, V3 slowerthanboth predecessors|Hybrid tradeoff failed; possible address/request issue, same-Xdata latency, or shifted producer contention.|Check actual kernel/cachewidth/source identity and failure shape; use minimal source-address hypothesis below only after evidence. No automatic16B/8B/4B cache sweep.|
|valid, >=.5us gainvsactualincumbent BOTH F1+paired|Local incumbent gain only; frozen variant becomes a promotion candidate.|Independent terminal audit first, then actual full-submission legality/prepare budget/numerical/full-workload validation. Officialscore cannot be inferred from diagnostic latency.|

### H_XA: repeated X-address instruction work

Concrete frozenV3 SASS first hotrefill `0x1a40..0x1d00` has45instructions:8LDG,8IMAD.WIDE,8LEA,8LEA.HI.X,4SHF.R.S32.HI,4VIADD,2IMAD.SHL,2LOP3,1IMAD. The following rawQ slots/other rounds are independent addresses, not a claim that this block's full45 repeat identically at everyPC. Current C++ recomputes `row*p.K+k` pervector with runtimep.K despite exacthostguardK4096. RegisterA demux source consumes these loads only four stages later, so actual criticality still needs evidence.

Conditional smallestV4: specialize the existing exactshape X stride4096 and hoist the two row pointers; form8byte-aligned offsets with bounded unsigned32-bit arithmetic. Retain8B/.nc/no_allocate/demux/Q4/A2/fourahead/W6/B8/grid/KS/output and everyasync lifetime. Exactrange row<=127,k<=4095 gives BF16index<=524287, byteoffset<=1048574, no32bit overflow. Require CPU equivalence for all524288indices, generatedSASS demonstrates removed repeated wide-multiply/signextension work, no extra loads/local/spills, actualpool<=65536, critic before GPU.

Selection evidence: meaningful hot X-address/dispatch issue PCs or a source/execution accounting showing redundant address work large enough to justify one bounded causal patch. Rejection evidence: relevant PCs unexposed and another named dependency clearly dominates; no credible>=.5us incumbent gain is asserted now. Even if a CPU prototype removes many instructions, its speed must be measured against V3 and incumbent. Do not combine with another mechanism in the same revision.

### H_XD: X-ready / demux placement

Current round converts current rawQ ->A, waitsBfull, fences/commitsfourHGMMA, issues8future rawQ loads, waits_group1, pins oldA/releases oldB. PTXmemory clobbers and async-A pins deliberately constrain scheduling. First hotdemux `0x1560..0x1860` interleaves16SHFL/16SEL with Bfullcheck `0x1830` and consumerbranch `0x1880`. All preceding16SHFL are semantically required pairlane gathers, but some source-selected operations may wait on the current LDG. Source does not establish returned-data readiness or whether payloadpublication is hidden.

If a future authorized V3 phase diagnostic shows longSB at X-consumingSHFL/SEL and low Bfull, test an earlier currentrawQ-to-A handoff only after deriving a separate safe buffer/retirement lifetime; no unsafe movement across wait1 and no reusing currentA aftercommit. If shortSB/MIO dominates ready-data demux, inspect exact two-SHFL/two-SEL dependency chains/outlined convergence execution. Do not erase fullmask reconvergence or replace proven collective semantics just to reduce static count. This branch needs more than olderV1PC percentages; not currently admitted as V4.

### H_P: producer decode / STS / publication

W producer unchanged in V3 (except removed dynamic register declaration):Wfullwait ->Bemptywait ->3LDS.128 ->r6decode+escape ->16scalarSTS/lane ->proxyfence+bar3 ->publishBfull ->refillW(s+6). Fixed across reverseversions:98304packedLDS warpinst;524288producerSTS +16384outputSTS=540672;524288PRMT. Everyproducerwarp performs64stages, not32, with same aggregate useful decode work as the incumbent.

The source has a concrete dependency: packedW slot s can be refilled only after its LDS uses retire, but currentrefill is delayed until decode, allSTS, convergence and publication finish. A proven new early-refill placement would need independent lifetime proof:allfourproducerwarps must finish everyLDS before TMAoverwrites, preserve all Wfullphase/expecttx and Bempty lifetimes. Moving the refill into a singlethread before otherwarps retire would race. Existing separate-ring/earlyrelease/history must be searched before any design; not a licensed ring-depth or producer rewrite.

Select this branch only if freshV3 shows exposed mathBfull with producer decode/store/publication phases or equivalent stage-timestamp evidence; V1Bfull2samples weighs against calling it the current leading cause. Producer shortSB at anLDSconsumer is not ALUcodec evidence, and publishbarrier samples can encode store drain or warp skew. No immediate Wcodec, escapefooter, aligneddecode, cluster, or weightdecomposition revival.

### H_C: doubled round/issue structure

Actualolder RQ comparator (not guaranteed thisjob's selectedpick) had128CTAs,32k64stages,n128instruction; reverse128CTAs,64stages,n64. Fixedreverse mathHGMMA=262144warpinst vs131072 comparator; same usefulMACs. More commit/wait/release, descriptor/index, producerpublication and genericdecodedWSTS remain after loader cleanup. A request to simplyhalve stagecount would require a newtile/operandlayout architecture, not a local unroll change.

A small source-only observation is signedmodulo work in release(s-1) at `0x1d60..1dd0`:ISETP/VIADD/SHF/LEA.HI/LOP3/IMAD.IADD/IMAD beforeARRIVE. Since release calledonlys>=0, unsigned `&7` indexing could remove the signedremainder fixup while preservingbarrieraddress. This is a narrow potentialfollowup if actual issue/control cost is exposed; predictedgain not established and likely smaller than H_XA. Do not add it to the sameV4 or pretend it removes doubledroundcost.

No KS/grid/NT sweep, no polling/ringdepth/cluster/selfzero/globalreduction changes. These costs may explain a residual gap but cannot justify adding their hypothetical savings or predicting an incumbent win.

## Terminal harvest checklist / append-only result section

Actual terminal artifact `proj_rsi/candidates/operand_reverse_v3/operand-reverse-v301.json` SHA4701566b96a7af729ce02eb4a041f7291d53520d83d1b63e54ed466b2b3da618. Root reports exit0/drained/node0MiB at23:36:47Z. Independently parsed terminalstatusV3_PROGRESS_DIAGNOSE_NEXT_LIMIT, all three sourceidentities match freeze, runtimeV1/V2/V3regs168/162/152,all0local/occupancy1/shared103600. Actualincumbent s4s5hz/KS2 confirmed byprofile FL3. All four allK/FP64/officialnormal/outlier suites pass; threecandidatevalidators eachsixchecks +incumbentthree;20actualtimedgraphmutations pass; immutableweights/pack/esc andresetcounters, RESET_ALL_FOUR. Independently recomputed everyF1median/conservativecontrol,21paired differences andwins from rawsamples. A timing saved before latervalidationfail would be invalid; this result completes validation.

Measured complete-graph table (positive gain means V3 faster):

|control|same-job F1controlmin us|V3F1 us|gain us|pairedmedian gain us|V3wins/21|
|---|---:|---:|---:|---:|---:|
|actualpw36 s4s5hz/KS2|24.032000|55.071998|-31.039998|-30.368000|0/21|
|V1|47.359999|55.071998|-7.711999|-7.520001|0/21|
|V2|56.639999|55.071998|+1.568001|+2.176002|20/21|

Interpretation: hybridV3 recovered1.568usF1/2.176uspaired versusV2, but remains7.712/7.520us behindV1 and31.040/30.368us behind actualincumbent. Halving logical transpose instructions fromV1 was insufficient to win; the change also doubled vectorloadrequests and changed staticresources. This does not identify a unique stall cause. Profilerdurations20.992/44.834/54.049/50.017us forinc/V1/V2/V3 are separate diagnosticcalls and are not substituted for completegraphmedians.

Nextselection=H_XA_ADDRESS_SPECIALIZATION_CPU_ADMITTED_BY_ROOT; GPUrequest=NONE. Root authorized isolated `operand_reverse_v4/` address-hoist/constantK/inline-immediate feasibility only, same8Bloader/cache/demux/Q4A2/producer/output. Source/SASS must remove intended repeatedwork without spills before critic/GPU. Four-waynexttrial, if admitted, retainsactualincumbent+bestfailedV1+immediatepredecessorV3+V4. This is a small mechanistic test, not a forecast of the~30us incumbentgap closure. No port, officialslot orfamilywideclosure. V3/v1/v2 remain frozen.


## Historical shared/shared coverage (read-only after V4 freeze)

Question: replaceV3 register-A X withTMA/shared-A; does this merelyrecreatepw19?

Source facts from `pw19/submission.py`:PwT definesNWM=4*NWG,NWD=4,NTH=(NWM+NWD+1)*32,BN=64*NWG,XB=NT*128,AB=BN*128. `t64` tuple(1,128,8,4,4,0,1):BN64channels×128tokens, **288threads**,oneMMAWG+one4warpdecodeWG+oneproducerwarp, W8×6KiB/X4×16KiB/decodedW-A4×8KiB=144KiBpayload plusbarriers/alignment. `t128` tuple(2,128,6,4,3,0,1):BN128channels×128tokens,416threads,twoMMAWGs splitWchannels. Older `pipeline-diff.md` prose that assigns416threads to t64 conflicts with actualsource; do not propagate it.

Oldoperandorientation:decodedW is sharedA(m64), X is sharedB(n128), SS m64n128k16. Decode uses4STSM.x4 intoSW128 followedbyeachwriterproxyfence, every128writerarrivesAfull;4producerwarps releaseWempty. Separateproducer lane pollsWempty/Xempty and issuesbothstreams. Math waitsAfull+Xfull, issuesfourSSMMA,commit/wait1, releasesoldA+X; finalwait0. `_pw_tws` selectsKS<=2 withNB*KS<=132 for t64c2, so poolerNB64->KS2 while unpoolerNB128->KS1. Oldpoolerladder is not a pairedunpooler128 timing.

Historical correctnessreview found no bit/layout/deadlock defect,90registers/0spills; sourceprotocol requiresXST>=2/AST>=2. Poolert64 ladder inprepare:19.9W-only,30.4W+X,32.5decode,39.5SSMMA,43.8full; same-jobpw10w1d4 full34.6 andtorch29.54. These are oldpartialrungs, not additivecurrentcriticalpath or currentcompletegraph times. Evidence for regression:shallow4slotXring with1–2pinnedslots reduces in-flightX, +5us relativeoldRSdelivery rung; addingSSMMA +7us across decode/store/fence/sharedread lifetime, cannot uniquely assign all7 to bandwidth. Wider/2CTA variants also coupledcluster4/SMEM/contention. No universalSS impossibility follows.

PlainV3->sharedX would keepX as sharedA(m64tokens) and decodedW as sharedB(n64channels), withtwoWGs coveringtokenhalves. Thus it is **not byte-identical** t64:SSm64n64twice insteadoneSSm64n128,128tokenXshared tile, decodedWread twice. It recreates the mainthree-ring/decode-shared/SS mechanism while retaining currentdoubledroundstructure.

Nominal useful sharedtraffic perCTA/k64 (not hardwaretransactions/portceiling):

|component|V3 directX-A / sharedW-B|plainsharedX-A / sharedW-B|oldpw19t64 sharedW-A / sharedX-B|
|---|---:|---:|---:|
|packedWfill+LDS|12KiB|12KiB|12KiB|
|decodedWstore|8KiB|8KiB|8KiB|
|decodedWsharedMMAreads|16KiB(twoWGs)|16KiB(twoWGs)|8KiB(oneWG)|
|Xsharedfill|0|16KiB|16KiB|
|XsharedMMAreads|0|16KiB|16KiB|
|total|36KiB|68KiB|60KiB|

At unpooler128 current128CTAs×64stages, these totals are288MiB/544MiB/480MiB respectively, beforebarriers/output. PlainsharedX does not lower globalusefulXdemand:it replaces128MiB directXloads with128MiBTMAXfill (nominal, beforecache/multicast). It removesXregisterLDG/demux/addresses but addsXring/lifetimes/sharedwrite/read and likely anindependentissuingrole; preservingW6/B8cannotjust fitanextra4×16KiBXring without recalculating sharedresource/occupancy. No currentcode/compile/launch admitted.

Ranking: a literalTMA/shared-A swap has no demonstrated newtraffic/reuseadvantage and substantialpriorcoverage; do not queueitjustbecauseV3lost. A future materiallydifferent design could still meritreview if it changesXreuse/delivery in a proved way and explicitlypayssharedport/producer/lifetimecosts. Preserve this mechanistic distinction; no familywideSSrejection. V4 remains the immediateboundedaddress-only causaltest, independentlyreviewedGO onmanifest90a7a4b8c90287c2982619564f4025f4cf744b1e9b26aa621f7f58770ac6f786.


## V4 terminal update / next single mechanism

Root terminal23:48:38Z exit0/nodeidle; raw `candidates/operand_reverse_v4/operand-reverse-v401.json` SHA5cf4cbc292188fd4ed68ec92255b2f4995a4752835ef26ada628669d3268bb6b. Independently recomputed allF1medians/mincontrols/21pairedgains/wins. Sources matchfreeze; V4 actual150regs/0local/occupancy1/shared103600, actualpw36pick s4s5hz/KS2. Allfourfullnumerical suites/20actualgraphmutations, byteimmutability/counterreset/RESET_ALL_FOUR pass.

|control|same-jobF1controlmin us|V4F1 us|F1gain us|pairedmedian gain us|V4wins/21|
|---|---:|---:|---:|---:|---:|
|actualpw36|24.192000|52.096002|-27.904002|-28.416000|0/21|
|V1|46.944000|52.096002|-5.152002|-4.960001|0/21|
|V3|54.784000|52.096002|+2.687998|+4.608002|21/21|

V4 removedsource-addresswork and improved its immediatepredecessor, including lowering/scheduling consequences. It remains slowerthanV1/incumbent; no port/officialslot/familywideclosure. This does not establish all remainingtime as Xload/demux or producerW. Mainloopfixeddescriptor/release/round and productioncosts persist.

Root admitted NEW CPU-only `operand_reverse_packed_x/`: freshbitexact1MiBXlayout transformation **everycall** ->fourcoalesced16Bnative-Aloads/stage withoutSHFL, sameWproducer/geometry/plainoutput/rawQ4/separateA2. Entirepack+main mustbe two timedgraphnodes; preallocatedworkspace pointeronly, never activationvalues cached/prepared. Packerpoison/fullbytecomparison, changedX/freshoutput/zero/outlier replays and separateA lifetime are mandatory. Conditional fullpipelinegain is unmeasured; extra1MiBread+1MiBwrite and launch can offset savings. No GPUrequest byauthor.


## Packed-X CPU gate outcome / explicit schedule revision

Initialfreshpack rawQ4/separateA2 identitycopy **FAILEDphysicalretirementgate beforeGPU**, evenwith122registers/0spill/noC75: HGMMA readsR56/R60, futureLDG overwritesR56/R60 at0xe30/0xe40 beforelastMMA/coveringwait. Preservedfullsource/SASS/resource/hazard snapshotmanifest538d53494d48b01e7b943d372a4eae86050b8f7c4e0e5aca9a5bb460b99eabf9. Thisversionfailure wasdiagnosed; no familyclosure and no invalidtiming.

Rootapproved correctnessrevision: **V2-nativeA4 retired-slotrefill**, initial0..3,wait1retires(s-1),load(s+3) intoretiredslot,finalwait0. Source mathloop byteidenticalV2, whileV4 Wproducer/output/cache/geometry stayfixed; rawQ4/A2fourahead ->nativeA4threeahead isexplicit confound. Both the combinedpacking change and scheduler/resource/compiler consequences belongto the completeversioncomparison.

NewphysicalSASSgate passesfourdistinct16registerbanks/twoiterations/eightgroups/allpredicatewrites conservatively widened4regs; outlinedmathwaitpaths write noA. Samegate rejects preservedbadcopy exactly0xe30. Localmain122/packer22regs,0spill/local/dynamicallocation,32mainLDG128/noBFLY. ActualJIT .so is dumped and rawartifact saved **before** checker/firstmathlaunch; failureJSON INVALID_ACTUAL_JIT_RETIREMENT_GATE retainsbinary/source/artifacthash and no_mathlaunch. Thisis researchvalidation, not eventualproductiontooling.

Frozenmanifest0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4; source2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a; lab39f9c69775e2fc926afe2d4d2ff3c4841f4533622d54887226556e08b4606b86. Proposedrootboundedtrial240/300s, fullfreshpack+main2nodes on everytimedcall; exactpackedpoison overwrite/freshX/storagebits/freshworkspace/output and allnumericalbasis/graphchecks. Criticreviewpending; noGPUbyauthor, noincumbentgainclaim.

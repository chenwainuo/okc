# Fresh packed-X + native-A4 main: one fixed two-kernel pipeline

state=CPU_PROOF_AND_SM90A_PASS_GPU_UNRUN; root_named_handoff+critic_required; geometry=M128/N8192/K4096/KS1,128CTA×384mainthreads,W6/B8,plainoutput. Parenthelpersrs32 SHAa7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad. No fullsubmission/officialslot/promotion.

## Causal brief / corrected lifetime

V4 valid52.096us vsV146.944/inc24.192, with2.688F1/4.608paired gain overV3 butstillnoincumbentwin. NativeperCTA X delivery remains costly. New hypothesis: transform1MiBX **on everycall**, once, then every128mainCTA reads a coalescednativefragment layout via4v4u32loads/thread/stage, noSHFL. Fulltiming includes pack+main; extra1MiBread+1MiBwrite+launch canoffset savings. No storedinputvalues, prepackinprepare, activationcache, weightprecision/codec/arithmetic changes.

InitialrawQ4->separateA2 identitycopy was **rejected beforeGPU**: ptxas coalesced it into nativeA registers, then refilled currentA before required retirement despite0spill/noC75. Preserved `failed-q4a2/` source/generatedCU/proof/resource/SASS/hazard with snapshotmanifest538d53494d48b01e7b943d372a4eae86050b8f7c4e0e5aca9a5bb460b99eabf9. Samephysicalchecker rejects exact0xe30LDGclobberR56..59. NoXOR0/opaqueextraoperation trick.

Root explicitly approved correctnessrevision: V2-identical **nativeA4 retired-slot schedule**, initialstages0..3; commitstage(s),wait_group1 retires(s-1),pin/release(s-1), then load(s+3) intoits retiredslot. CurrentA remainsimmutable; finalwait0/pinall4/releaseB63 beforeepilogue. Thischanges fouraheadrawQ4/A2 ->threeaheadnativeA4. It is a measuredpipeline-version confound, not an isolatedpacking-only experiment. No cache/geometry/Wproducer/output changes. NVIDIA requires a coveringwait before accessing asynchronous A/registeroperands ([PTXISA,wait_group](https://docs.nvidia.com/cuda/archive/12.1.1/pdf/ptx_isa_8.1.pdf)); source assumptions/zero warnings were not accepted as a physicalproof.

## Layout / bit / sector proof

```
packed_word_idx = ((((s*8+warp)*4+chunk)*32+lane)*4+v)
i=4*chunk+v
row=64*(warp//4)+16*(warp%4)+(lane>>2)+8*(i&1)
k=64*s+16*chunk+8*((i>>1)&1)+2*(lane&3)
packed[idx]=raw_u32(X[row,k:k+2])
inverse: v=idx&3,lane=(idx>>2)&31,chunk=(idx>>7)&3,warp=(idx>>9)&7,s=idx>>12
consumer_bytebase=s*16384+warp*2048+lane*16
fourv4loadimmediates=0,512,1024,1536
```

262144u32 bijection=524288BF16/1048576bytes; maxbyte1048575; all65536BF16patternlabels; no arithmetic applied to bits. Everyconsumer vector16Baligned. Perwarp/stage64touched=64unique32Bsectors, noSHFL;128MiBnominalmainXdemand unchanged butaddresses/demux/layout differ. Sourceinvariant: V4 Wproducer/output byteidentical; nativeA4roundloop byteidenticalto frozenV2.

Fixedpacker launch256CTAs×256threads,4u32words/thread. Eachthreadbaseidx<65536, fourpassesidx+65536*u; packedstage increases16, rawXpair increases512perpass. Eachof8192warpdomains writes32consecutiveu32 and reads exactly4unique32Binputsectors=128Buseful, stores4contiguousoutputsectors. NoV2half-sectoramplification, SHFL or speculativecoalescingclaim. CPU inverse/range/roundtrip/bit/coalescing proofs executeeveryindex/warp; stale-workspace negativecontrol failsall262144words afterchangedinput.

## Explicit API / lifetime gates

`rqrun(...,info,xp)` requires contiguousCUDA X with4Balignedpairloads and takes explicitpreallocatedcontiguousCUDAint32workspace of262144words,16Baligned,samedevice. Infoqueries do notpack. Normalcalls validatefixedshape then launch `rx_pack<<<256,256>>>` and `rq_px<<<128,384,103600>>>` on the samecurrentstream, checkingbothlauncherrors. Everycall populatesallworkspace beforemain. No workspace values initialized insetup/prepare; no global/statecache. Freshworkspace/freshoutput oracle is mandatory.

Localnvccsm90a actual: main122regs/pool46848,packer22regs;0stack/spill/local/USETMAXREG/C75. Main16HGMMA/4WGarrive/1outputTMA/0XTMA,32staticLDG128/0BFLY; packer4LDG32+4STG32static fromfourpasses.37staticMOV/IMAD.MOV are metadata/setup count, not evidence of retainedidentitycopies; thatcopypath was removed.

`liveness.py`: CPUphysicalregistergate for this exactkernel. Fourdistinct16registerAsetsR56..119; allwrites conservatively treated4regs,allpredicatedwritesexecuted; two loopiterations/eightgroups withwait1 retirement; outlinedmathwaitpaths checkedfornoAwrite; finalwait0. LocalpositiveA4passes; preservedcopy-aliasnegativefails atknownclobber. It is a fixedCFG verifier, not ageneralSASS analyzer.

Researchlab runs bounded20s `cuobjdump --dump-sass <exact_loaded_extension.__file__>` **beforefirstcandidate mathlaunch / beforeparentprepare**, saves `.jit.sass` andSHA, applies the samephysicalgate toallrq_px binaries. Failurestops beforecandidateGPUwork; artifactretainedforreview. Actualhostbuild/runtimeattrs still checkmain<=168reg/0local/occupancy>=1,shared103600; pack<=64reg/0local/maxthreads>=256. ThisCPUtooling is labvalidationonly and is not proposedforchallengeproduction.

## Frozen proposed trial / reporting

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_packed_x/lab.py --json /okc/logs/operand-reverse-packed-x01.json
```

Rootexclusive; inner240s/outer300s. Fourway actualpw36`prepare/kernel_fn`,bestfailedV1,immediateV4,newPXpipeline. Allcandidateextensionsbuildbeforeparentprepareclock; source/manifestpins; scalarselectedpick only. Complete graphcapturesnewAPI **pack+main everycall**, asserts2actualkernelprofileidentities; controls1kernel. Nevertimeonlymain orsubtractpackerlatency.

Allfourfulloracles:all4096Konehot exact,FP64random,officialnormal/outlier. AdditionalPXstorage-onlyGPUtest packsall65536BF16encodings(includingNaNs/Infs) and comparesall1MiBexact; itsGEMMoutputisnotgivenafiniteoracle, andnormal/outlier suitesseparatelyvalidatearithmetic.20actualtimedgraphmutations(original/zero/freshnormal/freshoutliers/probes),NaNpoisoncompleteoutput,inputimmutable. BeforeeachactualPXreplay poison1MiBworkspace, afterward exactrawint32compareto independentlyindexedcurrentX. Freshworkspace+freshoutputseparatecall tests explicitpointers. AllW/packedW/escapes bitimmutable; countersreset; resetallgraphs. Fixedoracleindexmetadata neverpassedtokernels.

Timing256MiBflush outsideevents,5discardedwarmups,21F1samplescontrols before/after(minmedian),21rotated/reversedpairedrounds. ReportallthreePXdeltas,wins,profiles/resources/SASSgate; >=.5usBOTHF1+pairedvsactualincumbent isreviewonly. A nonwin retains progressversusfailedversions andrequiresdiagnosis; nofamilyclosure. No guaranteedgain or current<3000us result.

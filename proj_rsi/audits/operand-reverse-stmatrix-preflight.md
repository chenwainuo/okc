# Packed-X producer STSM iteration: independent preflight

`state=EXACT_FREEZE_CPU_AND_DEVICE_PROOFS_REVIEWED; decision=GO_ONE_BOUNDED_TRIAL; GPU=UNRUN_AT_REVIEW; family=ACTIVE`

Authorized next hypothesis from fresh packed-X PCs: faster decoded-W stores may advance Bfullpublication. Existing samples1027/2387math at decoded-Bfullchecks and276/1111producer at publication/fence/bar support inspecting the handoff. They do not establish scalarstoreissue as the cause, a sharedportceiling, or expectedmicrosecondgain. Proposed one mechanism changes16scalaru32stores to4warpcollectivem8n8x4stores while retaining bytecount/layout/decoder/W6/B8/mathnativeA4/pack/cache/proxyfence/bar3/refill/epilogue.

Primary [NVIDIA CUDA12.8.1 PTX8.7, stmatrix §9.7.14.5.16](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix): m8n8.x4.b16 is supported onsm90. All32lanes must execute identical instruction/qualifiers without exitedparticipants. Lane groups0–7/8–15/16–23/24–31 supply eightnaturally16Balignedrowaddresses for matrices0–3. Fourconsecutivedatalanes supply one16Brow. It is weak memory; preserve asynchronousproxypublicationfence and CTAproducerbarrier.

Independent symbolic verification of the **proposed mapping**, not yet author-source verification:

```text
old(w,k,L,i):
  r=16*w+8*(i&1)+(L>>2)
  dst=r*128+(((2*k+(i>>1))^(r&7))*16)+4*(L&3)
  value=a[4*k+i] from dataLane L

stmatrix x4 operand i/dataLane L is routed through:
  addressLane=8*i+(L>>2)
  row=16*w+8*((addressLane>>3)&1)+(addressLane&7)
  chunk=2*k+(addressLane>>4)
  rowbase=row*128+((chunk^(row&7))*16)
  dst=rowbase+4*(L&3)
```

Reviewer Python enumeratedw0..3,k0..3,L0..31,i0..3: **2048u32 destinations exactlyequal** with identical(valueidentity) dictionaries, unique fullstageaddresses`0,4,...,8188`, everycollectiverowbase16Baligned. Distinguishing **addresslane from datalane** is essential; a same-threadonly mapping would be inadequate. No arithmetic/floatconversion occurs, so arbitrarypackedBF16payloads inherit identical bit transport if the actual source implements this mapping.

Pending exact frozenpackage: whole-source diff/hostguards; devicecompile/SASScollectiveopcount/resource/actualphysicalAretirement underchangedallocation; fullCPUbyteproof Wdecoder/escape unchanged; completeactual2nodegraph timing vs incumbent/PX/predecessor; retainedactualgraphs/currentinputmutations/poison/fullpackedbytes/counters/inputimmutability/freshworkspace and teardown. NoGPUlaunch/GO based solely on this algebra. Any launch remains root-owned boundedexclusive after finalreview.

## Final frozen package review / GO

Manifest11`2119b95e5789d4cc280f129d2817cc085e0ef0a1134bfcf2f63880e44afdefb7`; generatedsource`cc4f5189bf0018a1e48941377b639b9c0983ffc2eb5cb6e44e6fa105f7d20b38`; device`81152aedc8e7fcc4bf87e869b7f6ea3bc0fc080238f0ff9c1e30b7291552ba1d`; lab`64bc2a0e2a7c9f409452be8b08996309064a9de8ec98ac2ab74d8ac731f07afd`; liveness`dbf68ab0215f97aeebd0f836cbbf9755cfe6406f338918a0e57f4cec24a2ebfa`. Reviewer verified11filehashes, read actualkernel/prototype/generatedsource/lab/checker/proofs/manifest/README. WholegeneratedCU restoringonlyproducerstoreblock+kernelname hashes to exactparentPX`2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a`. Source Wdecode/escape/refill/W6B8/barrierphase/mathA4/Xpack/cache/output/arithmetic unchanged. No dualproducer/A6 mixing.

Independently executed pureCPU`sourceproof/xproof/wproof/outputproof/lifetime` without authorfilewrites orCUDA: full262144Xwords/bijection/alignment/sector/freshness, **524288WBF16values including denseescapes/all65536payloadpatterns**, exactcrosslaneSTSM-versus-scalar **everybyte** perstage andall64stages×2halves, noalias/crossslot/alignment; all8192D scalars/slab; nativeA4retiredslot asyncschedule allPASS. Recheckedall11hashes unchanged afterward. Separate reviewer algebra above already supplied independent2048u32perstageequivalence rather than relying solely on authorproof implementation.

LocalactualCUDA12.8sm90aSASS`02fd73c7709b6781fe17e4999c78739a55373fad88cab91c4ea6c7940b5bc653`:122main/22packregs/0stack-spill-local/noUSETMAXREG;4unpredicatedSTSM.16.M88.4,16remainingoutputSTS (parent32total),16HGMMA/4ARRIVE/1outputTMA/0XTMA/32LDG128/noBFLY. ProducerlaststorePC10672→proxyfence10704→bar3PC10720, MEMBARbetweenstore/fence. Payloadstill8192Bstage. Nominal122×384=46848 differs warp-rounded49152actualregisterallocation; runtimeoccupancyguard remainsrequired. No throughputfactor inferred from4versus16storeinstructions.

Reviewer re-ran unchangednarrowphysicalA4checker onlocalnewSASS andparentstore-formguard. FourdistinctAsets56..71/72..87/88..103/104..119, strictD24, twoiterations/eightgroups/current+pendinglivewritechecks/outlinedwaits/finalwait0 PASS. Newcheckeronlyextends acceptedmainnames plus separate storeformguard; no parserweakening. PreservedbadQ4/A2copy rejectedexact0xe30 prematureR56write. This localproofdoesnotcertifyfullTorchhostcompiler; **labdumps exactBOTHloadedPX/STMXJIT.soSASS beforefirstcandidate math andbeforeparentprepare, savesartifactbeforephysical+storeformchecks**, writesINVALID/no_candidate_math_launch onfailure. BothactualSASSmustpassindependently.

Fouractualtimedcallables incumbentPW36kernel_fn, frozenPX, frozenV1, STMX; incuntimedvalidatorcopy is absentfromactualtimedgraph. Capturedbaselineout retained/live andallfouroutputsdisjoint. PXandSTMXdistinct1MiBworkspaces; actualcaptures include **freshpack+main2nodes percall**. No preparationXcache, subtraction ormain-onlytiming. Runtimeopt-in-beforeoccupancy,0local/<=168regs/shared103600/occupancy>=1;V1poolguard retained. Bothpackersattrs andactualsourcehashes bound. Fullbasis4096/allfourFP64+officialnormal/outliers/fixedpointertests; BOTHpackedvariantsGPUall65536storagebits/whole1MiBXP andfreshworkspace/output;20**actualtimedgraph** inputmutations withNaNpoison,10XPpoison/fullwordcomparisons, inputunchanged,originalrestore, exactW/packedW/escape/counters0, allfourreset.

Timing:256MiBflushoutsideevents,5discardedwarmups,21F1controls before/after conservativeminmedian,21rotated/reversedfourwaypairedrounds. Reportsallthreecomparatordeltas/wins. Actualincumbentminimum0.5usBOTHF1+paired givesreviewonly; predecessorprogressremains usefulbutnoincumbentpromotion. Profilesmustactual1V1/2PX/2STMXnodes; no taskkernel bypass. Unresolvedofficialeligibility/H200validation remainsseparate.

Root-only exactcommand`MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_stmatrix/lab.py --json /okc/logs/operand-reverse-stmatrix01.json`, oneexclusivecontainerinner240/outer300afterpriorcontextdrains. GO concernsboundedexecution ofthisfreeze, notspeedclaim. RequireactualJIT/numeric/immutability/counter/lifetime gates+successfulterminal/nodecleanup andindependentrawtimingarithmeticbeforeanyadmission. Nonwinretainedforcausaliteration, nofamilyretirement. ReviewerlaunchednoGPU/remotejob andeditednocandidate.

## Actual result review: valid predecessor progress, incumbent nonwin

Rootlaunchedguard380790/`okc-proj-operand-reverse-stmatrix01` at00:44:15Z. CopiedJSON`proj_rsi/candidates/operand_reverse_stmatrix/operand-reverse-stmatrix01.json` and`results/proj-rsi/takeover/20261027/operand-reverse-stmatrix01.json` byteidentical, SHA256`346f603f31bfc2379f3f78c45aaef7b67ca591c6c5a941a6354105d386503c7d`. `status=STMX_PROGRESS_DIAGNOSE_NEXT_LIMIT; error=NONE; graph_cleanup=RESET_ALL_FOUR; weights_inputs_counters=UNCHANGED_OR_RESET`. Hostlog/terminalexit/actualcontextcleanup notyetavailable to thisreviewer at thischeckpoint; they remain required for terminaladmission. NoGPUactions byreviewer.

Exactloaded artifacts independentlyhashchecked and **reran bothphysicalA4+storeformcheckers** with results exactlyequal reportedobjects:

| Actual JIT artifact | bytes | SHA256 |
|---|---:|---|
| `operand-reverse-stmatrix01.px.jit.sass` |207663 |`f954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f` |
| `operand-reverse-stmatrix01.smx.jit.sass` |197985 |`ac8e3807c5f0f0665907e0a71fd0e494a02641e2b7582ff84c21ca07912eb713` |

STMXactual4unpredicatedSTSM/16outputSTS,128mathXvectorloadsunchangedform32staticLDG128, proxy/bar3/A4physicalretirementPASS. PX0STSM/32STSguardPASS. ActualPX/STMX122regs,local0,shared103600,optinoccupancy1;V1168regs/dynamic64/216/0local/occupancy1, bothpackersrecorded. Allfourbasis4096/FP64/officialnormaloutlier testsPASS, bothpackedvariants all65536GPUstoragebits/full1MiBXP/freshworkspaceoutPASS. Exactly20actualtimedgraphchecks =4graphs×5mutations, allNaNpoison/inputimmutability; exactly10XPpoison/fullwordchecks forPX+STMX. Source/generativehashes matchfreeze, no fallback.

Actualincumbentselected`rq s4s5hz/KS2`, profile1`rq<128,5,1,0,0,0,3>`. PXprofile2nodes`rx_pack/rq_px`, STMX2nodes`rx_pack/rq_stmx`, V1profile1`rq<128,8,1,0,0,0,20>`. Profiledurations are diagnostic and not added/subtracted to constructwholepipeline scores.

Independentlyrecomputed all7F1samplemedians (21samples each), conservativecontrolmin-before/after andall21pairedroundgains/wincounts; exactfloatequality withJSON:

| Control | before /after median us | conservativecontrol us | STMXF1gain us | pairedgain us | pairedwins |
|---|---:|---:|---:|---:|---:|
| actualincumbent |24.4800001383 /24.4159996510 |24.4159996510 |−13.8560011983 |−14.3999997526 |0/21 |
| frozenPX |39.4559986889 /39.5199991763 |39.4559986889 |+1.1839978397 |+1.4400035143 |21/21 |
| frozenV1 |47.3279990256 /47.3599992692 |47.3279990256 |+9.0559981763 |+9.4720013440 |21/21 |

STMXentirefreshpack+mainF1median`38.27200084924698us`. PairedPXgainrange`1.2479983270..3.0720010400us`, so all21consistentwins atthisfixedversion. Actualincumbentpairedgainrange`−14.9119999260..−13.0880009383us`, clear nonwin. Validpredecessorprogress supports store/address/publicationbeing worthtesting, not proof of solePCstallcause or sharedbandwidthceiling. Noincumbentpromotion/officialscorecredit/<3000usclaim. Pendinghostclosureabove. Keepnextdualproducerbranch scalarstores to isolate mechanism andincludeSTMXasbestfailedcontemporaneouscontrol; combineonlyaftermeasuringseparateversions.

**Terminal follow-up:** full13962Bhostlog nowarchived`results/proj-rsi/takeover/20261027/operand-reverse-stmatrix01.host.log`, SHA256`acfd922f59fc530e90139e7d480f17cb37026651bd3e4635f61341ba238acf44`. ReviewerreadfinalPREP57s/allthreeONEHOT+baselinevalidator, STMX_RESULTmatchingJSON, `TRIAL_EXIT okc-proj-operand-reverse-stmatrix01 0`, end00:46:06Z. Root explicitlyverified380790/containergone/GPU0MiBat00:48:40. Priorpendingterminalcheckpointresolved: **attemptVALID; predecessorprogress; noincumbentpromotion**. Sourcefreezeunchanged.

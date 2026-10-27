# Packed-X two-producer branch: independent source preflight

`state=EXACT_FREEZE_REVIEWED; decision=GO_ONE_BOUNDED_TRIAL; actual_GPU=UNRUN_AT_REVIEW; no_source_blocker_identified`

Root owns this branch; reviewer edits no candidate and launches no GPU. It is independent of STMX: parentPX scalarstores, originalXpack/nativeA4/cache/output preserved. New512threadCTA has256math+128evenproducer+128oddproducer. Currentresourceevidence root reports122main/22pack,0spill/local/USETMAXREG, originalstrictphysicalA4checkerPASS; actual loadedTorchJIT remains mandatory.

At512threads,122regs/thread rounds to128/thread forwarp allocation, **128×512=65536** exactly. Nominal122×512=62464 is not availablephysicalheadroom; no request can exceed the64Kpool. Hardgates actualregs<=128,roundedpool<=65536,0local/spill, maxthreads512, opt-inbeforeactualoccupancy>=1. Shared103600 unchanged. Staticresourceproof is not a CUDAhostlaunchcertification. No dynamicregister redistribution may be silently introduced.

Actualsource currently assignsowner=(warp−8)/4 (0for256..383,1for384..511), s=owner;s<64;s+=2. Initialissuer256 initsexistingW6/B8barriers andfillsW0..5; initial__syncthreads includesall512. W6 andB8 are even, so eachphysicalslot has exactlyoneproducerowner across allreuses. Eachowner issuesfull4warp decoded64channelstage, retainsproxyfence, usesbar3orbar4 count128, publisher/refillonlytid256+128owner. Mathbar1/bar2 remain256; eachmathwarplane0 issuesoneBemptyarrival,8arrivals totalperretiredstage. No crossproducerbarrierparticipants or producerwritealias.

Wfull(s) is initiallyprefetchedfors<6; otherwise TMA(s) is issuedby sameowner afterstage(s−6) decoding/publication. Bempty(s),s>=8, waits completion of allmathreads ofslot(s−8). More precisely wait_group1 releases(s−8) in **mathround(s−7)**, so producing s may depend onotherparity's(s−7). This is still strictlyearlier, not a future-stagecycle. With finiteTMAcompletion/CSR and fairwarpscheduling, stages can be inductivelyproduced/retired. This sourceDAG is not a universalGPUprogressproof. Barrierphase mathwait(s/8)&1, produceremptywait((s/8)&1)^1 and Wwait(s/6)&1 preserveparentepochs because physicalringownershipneverchanges. Finalmathwait0 releasesB63; producers have no laterstagewrites beforeoutputreuse.

Eachproducerwarp has its **owncursor over same readonly CSRlist**. Added `while(nxs<s)` increments ei, reloadsec at32entryboundary via__ldg(el+ei+lane), broadcastsentryei&31, updatesnxs; then inheritedrq_fix appliesexactcurrentstage. This is required: withoutskip, oppositeparity's earlierrecord blocks everylaterrepair. Sentinel0xffffffff yieldsstage32767 so finitevalidlist terminates. Warp-broadcastnx/nxs makes whileuniform; patchlaneconditions do notchange collectiveparticipants. Escapeaddressindex useswarp&3 so group1 duplicates group0'sfourlists correctly. Traversal/metadatawork duplicated is a measurablecost, not extraW decode orinputprecisionchange.

FreshPXsamplesmotivated producerhandoff question; doublingproducers may advanceBfull but canincreaseissue/CSRtraffic, warpcontention andresourcepressure. No predictionfrom43%samplefraction tospeedup. Preserve actualSTMX/PX results and keepbranchesmechanicallyseparate; do not combineuntil eachversion has validmeasurement.

Pending finalmanifest/wholeCUDArestoration/source+protocolproofs andexactlab: BOTHloadedPX/DPphysicalA4SASS artifacts beforefirstmath,512hostlaunch/occupancy/resourcegate, twoactualnodesineverycall, retainedactualtimedgraphoutputs/workspaces, fullbasis/FP64/officialdist/arbitraryBF16storagebits/freshworkspace, original/zero/fresh/outlier/probechanges with NaN+XPpoison/full1MiBcomparison, W/escape/X/countersimmutability andgraphreset. Rootboundedexclusiveinner240/outer300 onlyafterfinalindependentGO. No GO from this preliminaryread.

## Final freeze / exact GO

13filemanifest`281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241`; source`97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b`; device`f633d76f9eea330f4bbc729990c535826ef68981e66f1212909deae03c194af9`; lab`ad7df5e6dd1c1f120c616d8c0265adf4bf19ae5e086d34c26390a8f6572f6196`; exactkernelbody`05c009c93c22404273ec6bb761aee8cfbd0dafc6f787d921e76724352a2645b0`. Reviewerverifiedall13hashes/readwholehostgenerator/kernel/lab/checker/protocol/preflight/manifest/README. Localpureproof execution retainedhashesunchanged/zeroCUDAcalls.

WholeCUDArestoration reversedonlynewproducerblock,512launchbounds/hostlaunch/occupancy, kernelname and **metadata-onlyinfo6maxThreadsPerBlock**; exacthashrestoredparentPX`2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a`. Metadataadditionhasnodeviceimpact andnoalternateworkload. AllXpack/math/output/Wcodecbytesfixed; producerstoresremain **scalar**, noSTMXmixing. Root'searlierV1-comparatorlab wasUNRUN andarchivedwithmanifestbb4d1296 in`lab-history-bb4d1296/`; finalcomparisonusesactualincumbent/PX/**STMXbestfailed**/DP. Preservehistoryanddonotattributeoldunrunlab measurements.

Reviewer re-ran pureprotocolproof7CSRdensities/missing-skipnegative andexactsource/devicegeneratorhashes; independentlyread actualencodedskip/windowreloadsource. Additionalindependent[protocolaudit](dual-producer-protocol.md) supplies encoded32lanecache/sentinel model at10densities and320-node two-mathWG dependencyDAG; clearlyseparate fromroot'stuplemodel and fromGPUproof. Earlier W/output/X/A4puremodels applyto source-provenunchanged sections; no blindassertthatnewparitycursorwascoveredbyoldweightmodel.

LocalCUDAresource122main/22pack/0stack-spill-local; SASS`556039ac78367c19f692204f2d7f26257aa544432e529b9cbb60149f73870934`. Reviewerre-ran exact originalphysicalA4checker andmatchespreflightobject; preservedbadidentitycopyrejected. NoLDL/STL/USETMAXREG; actualSASSbar0 plusbar1/2count256 andbar3/4count128 confirmed. Rounded512pool65536; actualTorchhostbuild/occupancynotclaimedfromlocalcubin. No known sourceprotocol/resourceblocker.

Read exactfrozenlab fromvalidatedSTMXharness diff. **AllthreeactualloadedJIT.so artifacts(PX/STMX/DP)** savedbeforechecker/firstmath/beforeparentprepare, originalstrictD24physicalcheckerforPX/DP andfrozenSTMXchecker+storeformforSTMX. AnyfailurewritesINVALID/no_mathlaunch andstops. RuntimeDPregs<=128/warp-roundedpool<=65536/local0/info6maxThreads>=512/opt-in-beforeoccupancy>=1/shared103600; sourcehostactuallylaunchesdim3(512), unchanged pack256×256. Allthreepackattrschecked. No uncertified512launchfromonlynominalregistersum.

Fouractualtimedoutputsretained/disjoint (baselinecallreturnkept, noaddedcopy). Threepackedvariants eachownfresh1MiBworkspace andactualtimedgraphcontainsfreshpack+main2nodes; no prepareinputcache/main-onlytiming/packer subtraction. Fullbasis4096/FP64/tasknormal/outliers/allfourcontrols; allthreeGPU65536storagebits+fullXP/freshworkspace-output; **20actualgraphinputmutations/NaNpoisons and15XPpoison/full1MiBbytechecks**, freshnormal/outlier/probes/zero/original; restoreoriginal, originalX/weights/packedW/escapeimmutable/countersreset, allfourgraphsdrained/reset. Actualprofilesmust2correctnodesforeachpackedvariant; preserveactualincumbentprofile/pick withoutassumingRQselection.

Same256MiBflushoutsideevents,5warmups,21F1before/aftercontrols conservativeminmedian,21rotated/reversedfourwaypairedrounds; exactrawsamples/allthreegains/wins. Minimum0.5usBOTHF1+pairedagainstactualincumbent ispromotionreviewonly; predecessorprogress/data retainedwhenincumbentnonwin. NoH200/officialscorecreditfromH100researchlab. NewmechanismmayincreaseCSR/issue/warpskewdespitecorrectness; no speedupguarantee.

Rootonlyexactcommand`MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_dual_producer/lab.py --json /okc/logs/operand-reverse-dual-producer01.json`, namedexclusiveinner240/outer300afterpreviousGPUcontextdrains. **GOforoneexecutionofexactfreeze**, runtime/JIT/correctness/wholepipeline/timing/terminalcleanupindependentreviewpending. Reviewereditednocandidate/launchednoGPUorremotejob. Noautonomousscopeexpansion/flagmix/benchmarkpromotion.

## Actual terminal review: DP01 valid, faster predecessor, incumbent nonwin

`attempt=VALID; best_failed_version=DP; official_or_incumbent_promotion=NONE; family=ACTIVE`

Rawarchive`results/proj-rsi/takeover/20261027/operand-reverse-dual-producer01.{json,host.log,px.jit.sass,smx.jit.sass,dp.jit.sass}`. CandidateJSONidenticaltoarchive,31849B SHA256`f6d272761ec59d751ce0e0ed5929bdb69ea669de25ffd4b79219812413dfee7e`. Host13997B SHA256`3ffa2451fae7319b024ffc9f21bfa4df6356670bdfc36ed2b264b5061599bd22`, PREP58s/allthreeONEHOT4096+actualbaselinevalidatorPASS/DP_RESULTmatchingJSON/`TRIAL_EXIT okc-proj-operand-reverse-dual-producer01 0`, terminal00:56:35Z. Rootverifiedactualcontext/containergone/GPU0MiBat00:58:35. ReviewerperformedoneSSHreadonlyhosttailbeforelocalarchive, nojoblaunch/kill/write; validationusesfullarchivedraw.

Independentlyhashedandreran eachactualloadedJITphysicalchecker, results exactlyequalreportedobjects. STMXstoreformalsorechecked viafrozenchecker:

| Variant | JIT bytes | SHA256 |
|---|---:|---|
| PX |207663 |`f954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f` |
| STMX |197985 |`ac8e3807c5f0f0665907e0a71fd0e494a02641e2b7582ff84c21ca07912eb713` |
| DP |219282 |`bebe472634172ceb69c0a8d3df2c3555c70f89f421693b4d12dee847b2e7e746` |

ActualDP122regs/512threads/**65536roundedregisterpool**/0local/maxThreads512/optinoccupancy1;PX/STMX122/384/49152/0local/occ1. Allshared103600 andmaxdynamic49152→103600afteroptin. Allthreepackers22regs/0local/maxthreads1024. No silentdynamicallocationor512launchresourcefailure. ActualoriginalstrictphysicalA4retirementPASS, sameA56..119fourbanks/strictD24, nohazardparserweakening.

Allfour4096Kbasis/FP64/tasknormal+outlier suitesPASS. AllthreepackedvariantsGPUall65536BF16storagebits/full1MiBcurrentXpack/freshworkspace+outputPASS, freshoracleallrelativeL2`1.421e-04`,worstrow`2.633e-04`,12proberowsexact. Exactly20**actualtimedgraph**records =4graphs×5inputmutations, allNaNpoisonoverwrite/inputimmutable;15packedpoison/fullwordequalitychecks. W/packedW/escape unchanged,countersreset,allfourgraphscleaned`RESET_ALL_FOUR`. JSONstatus`DP_PROGRESS_DIAGNOSE_NEXT_LIMIT`, noerror.

Actualbaselinepick`rq s4s5hz KS2`; profile1`rq<128,5,1,0,0,0,3>`. PX/STMX/DPeachactual2nodes`rx_pack`→correspondingmain, no main-onlytiming; profiletimes not used toclaimwholepipeline savings. OriginalPX/STMXsourceandJIT hashesunchanged.

Reviewerindependentlyrecomputedall7F1series (21samples each), allconservativeminbefore/aftermedians, all21pairedrounddifferences/wincounts with exactfloatequality:

| Control | before /after median us | conservative us | DPF1gain us | pairedgain us | wins |
|---|---:|---:|---:|---:|---:|
| actualincumbent |24.1599995643 /24.0959990770 |24.0959990770 |−10.3999990970 |−10.5919986963 |0/21 |
| PX |39.5839996636 /39.5199991763 |39.5199991763 |+5.0240010023 |+5.1520019770 |21/21 |
| STMX |38.2080003619 /38.1760001183 |38.1760001183 |+3.6800019443 |+3.9039999247 |21/21 |

**DPentirefreshpack+mainF1median34.4959981739521us**. PairedPXgainrange2.8159990907..6.4640007913,STMX1.5040002763..5.2159987390; incumbent−12.8000024706..−8.8319983333. Moreproducersimproveearlierarchitecture, supportingthehandoffhypothesis asausefulmechanismtest, but not uniquelyallocatingthegainbetweenpublicationoverlap/metadata/warpscheduling. Noactualincumbentimprovement, noofficialscore/<3000usclaim.

SeparatefutureDP+STMXstorecomposition now has ameasuredparent andmeasuredprimitive, but benefitsare not assumed additive; newsourcefreeze/CPU+actualJIT/fullpipelinevalidationbeforeGPU. FreshDPPC/memorydiagnosticmustmap **8math+8producerwarps**, notold4producerroles. MostproducerPCs are shared byowner0/1 andcannotseparateowners fromPC alone; dedicatedbar3/bar4paths mayprovidelimitedperownerlocations. Newreportrole/phasecountsandcache/clock/replaylimitations required; no automaticreuseofPXPCcause or samplefractions→microseconds.

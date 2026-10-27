# DP + collective stores: independent preflight

`decision=GO_ONE_BOUNDED_TRIAL; GPU=UNRUN_AT_REVIEW; family=ACTIVE; additivity=UNASSUMED`

Manifest17SHA256`6785da2ec9881ce13148951daff2f8802d8e7bbe6c83abf588607092b3f97072`; source`672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1`;device`327f35765720bbb56c630819854d5631f51637d4e73b11d089ad31bb2bd9167e`;lab`5afa5159e9f8986d05150607d658b38e574039028390d6e97f5c74edb9a87a4a`;checker`402eb10d97ab6a186647a498255755d6a4dc733852b0e984c6d75f5c55934dd5`.

Reviewerverifiedall17hashes/readactualbody/wholeCUDArestoration/prototype/labdiff/checker/proofs/manifest. PureCPUexecution sourceproof,ownerproof,W/X/output/lifetime,localSASSphysical+storeform andpreservedbadaliasnegative PASS, authorhashesunchanged,zeroCUDAcalls. No authorfile/GPUedits. ParentDPmanifest281ff/source97dd validwholepipeline34.496us andsingleproducerSTMXmanifest2119/sourcecc4f38.272uswereindependentlyvalidated; thisisameasuredcompositiontest, notassumedadditivegain.

WholegeneratedCUDArestores **exactDP97dd** afteronlyscalarproducerstoreblockandmainnamereversal. Same512threads/evenoddownership/CSRskip/W6B8/proxy/bar3bar4/refill/Xpack/nativeA4/cache/K4096/output/arithmetic. SamecollectivestoremapprovedinSTMX: addressLane8*registerIndex+(dataLane>>2), +4*(dataLane&3) routesexactold2048u32/8192Bstage. Currentlayoutproofrerunall524288WBF16includingall65536payloadpatterns/denseescape/windows, everybyteoldvsnew/noalias/alignment, fullX262144words/inverse/sector/freshness, all8192Doutputvalues andA4retiredslots. Ownerproofcoversall64stagesexactlyonce,32stagesperowner, eachsame4warpmapping; no crossownerstagewriteorchangedbytes.

LocalCUDA122main/22pack/0stack-spill-local/USETMAXREG/C75, exact512roundedpool65536; no claimedactualoccupancyfromcubin. LocalSASS`5e2e65397434ddf14ee844cd9b020b422c290043c8d92e76049d79a739582324`:4unpredicatedSTSM.16.M88.4,16outputSTS,16HGMMA/32XLDG128/noBFLY, payloadstill8192Bstage. OriginalstrictD24A4physicalcheckerPASS andrejectsold0xe30copyhazard. RootDPprotocol/source[lifetimeaudit](operand-reverse-dual-producer-independent.md) unchanged.

**Bothownerbarrierscheckedaftersharedproxyfence.** CollectivelaststorePC11232→proxy11264→`@P0 BAR4,128`11280→`@!P0 BAR3,128`11296 adjacent. SamecomplementaryP0 andnointerveningwriter. Ownerbranchuniformby128threadgroup fromsource. ParentDPactualformalsopassesnewdualstoregate: bar4/bar3sameP0complements, interveningWARPSYNC/NOP/BSSYonlysafecontrolwhitelist, nopredicatewriter. All4STSMmustremainunpredicated/fullwarp.

InitialcheckerincorrectlyrequiredunpredicatedBARandfailedCPUbeforeGPU; preservedexactchecker/preflight/error in`checker-history/`. Fixvalidateslegitimatecomplementarypredicationandpredicate-liveness, notweakenedmembershiporforcedunpredication. Kernel/sourceunchangedacrosscheckerrevision; noGPUattemptfromoldgate.

ExactlabcomparesactualpreparedPW36kernel_fn, frozenbestDP, frozenSTMX, newDS. Allthreepackedvariantsdistinctexplicit1MiBworkspaces/freshpack+mainactual2nodegraphs. All3exactloadedJIT.soSASSsaved **beforefirstmath**, physicalA4forDP/DSandfrozenSTMXcheckerforSTMX; intended4STSM/proxy/bothownerbarriersforDS andscalarparentDPformrequired. RuntimeDP/DS<=128regs/warp-rounded<=65536/maxThreads>=512/0local/opt-inoccupancy>=1/shared103600; packattrschecked. FailbeforemathpersistINVALID/artifacts, no blindtrustoflocalcompiler.

All4basis4096/FP64/tasknormal+outlier/fulloracles, all3arbitrary65536GPUstoragebits/fullXP/freshworkspace+output, **20actualretainedtimedgraphmutations+15XPpoison/currentinputfull1MiBcomparisons**, NaNoutoverwrite/inputimmutable/Wpackedescapeexact/countersreset/allgraphsdrainedreset. Baselineactualreturnretained, alldifferentoutputs; untimedvalidatorcopyabsentfromtimedkernel. Entirefreshpack+mainevenforcontrols; noactivationpreparecache/subtractedpacker/main-onlyclaim. Actualprofilechecks2correctnodesforall3packedvariants; actualincumbentpick/profilepreserved.

21F1before/aftercontrols conservativeminmedian,21rotated/reversedfourwaypairs with256MiBflushoutsideevents/5warmups; reportall3deltas/wins/rawseries. Againstactualincminimum0.5usBOTHF1+paired ispromotionreviewonly, official/H200separate. Progressagainstfailedpredecessorsretained; nonadditiveinteractionisdata, nofamilyclosure.

Rootonly`MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_dual_stmatrix/lab.py --json /okc/logs/operand-reverse-dual-stmatrix01.json`, namedexclusiveinner240/outer300afterpriorcontextdrains. GOexactfreezeonce; actualJIT/resource/numerical/measurement/terminalcleanupreviewstillpending. FreshDPcounterdiagnostic andthistrialhaveschedulingindependence; neitherpermitssourcemutationoftheother.

## Actual combined01: valid numerics, mixed gain versus DP

`attempt=VALID; best_robust_failed_version=DP_UNCHANGED; incumbent/official_promotion=NONE; DS_retained=YES`

CandidateJSON`operand-reverse-dual-stmatrix01.json`,33940B SHA256`e5a745c41db06507879ada25f21726074d97eaccc316b98f8f003a8b68f2f87e`. Host13990B archive`results/proj-rsi/takeover/20261027/operand-reverse-dual-stmatrix01.host.log`,SHA256`72b3aea6094742b60e48830584abb2398ae35cc58cee2c4471d5fa9f7f0e1c74`, TRIAL_EXIT0/end01:15:42Z. Rootconfirmedcontextgone/node0MiB. NoGPUworkbycritic. Exactsource/freezeunchanged.

Actualall3JITartifactsindependentlyhashedandreranphysical+storeform/bothownerbarriercheckers; results exactlyequalreportedobjects. DPbebe4726…e746 andSTMXac8e3807…b713 unchanged. NewDS209608BSASS SHA256`145c56d97b81451c43cb025f8908687c447a64cef54af0b7dfd6ee9ef51b35fb`, strictA4physicalretirement/unpred4STSM/proxy/complementarybar3bar4/noPwriterPASS. ActualDP+DS122regs/512/65536rounded/maxthreads512/0local/occ1;STMX122/384/49152/0local/occ1; shared103600,optinbeforeoccupancy, pack22regs/0local all3.

All4fullbasis4096/FP64/normaloutlieroracles, all3GPUall65536storagebits/currentXPfull1MiB/freshworkspaceout,20actualretainedgraphmutations/NaNpoison+15XPpoison/fullbytes/inputimmutable PASS. W/packedW/escapeunchanged,counterZERO,RESET_ALL_FOUR. Actualincumbent`rq s4s5hz/KS2`1profilekernel;DP/STMX/DSeachactualfreshpack+correspondingmain2nodes. Profiletimesremain diagnostic; entirepipelines timedwithoutsubtraction.

Independentall7×21F1mediansand21pairedrounds exactarithmeticverified:

| Control | before /after median us | conservative us | DSF1gain us | pairedgain us | wins |
|---|---:|---:|---:|---:|---:|
| actualincumbent |24.5120003819 /24.5760008693|24.5120003819|−9.0879984200|−10.5920005590|0/21|
| frozenDP |34.4959981740 /34.4320014119|34.4320014119|+0.8320026100|−0.2559982240|10/21|
| frozenSTMX |38.6560000479 /38.4319983423|38.4319983423|+4.8319995403|+4.2880028486|21/21|

DScandidatewholepipelineF1median`33.59999880194664us`. PairedDSgainagainstDP ranges`−2.3359991610..+3.1040012836us`, againstSTMX`+2.3680031300..+5.7919993997us`, incumbent`−12.4159976840..−7.0720016956us`. F1andpaired disagreeversusDP; **do notselectonlyF1 oraddsingleproducerSTMXgain** toclaimnewfamilybest. JSON`DS_PROGRESS_DIAGNOSE_NEXT_LIMIT` isaccurateonlyasprogressvsSTMX, notbestparentDP. RootcorrectlyretainsscalarDPasrobustpredecessor. NoautomaticDSdiscard/familyretirement; order/raw-seriesanalysiscaninformfuturemeasurementhypothesis, butnoinstructioncache/clock/ordercauseestablishedhere.

FreshDPPCsleadARRIVEplusproducerBemptybackpressure, providingaplausiblecriticalpathshiftcontextfortheweakcombinedstorebenefit. Thisisqualitativeinference,separatefromdirecttiminggates. Nextfence-orientedversionrequiresfreshproof/measurement; noautomaticunchangedcombinedrerun/port/officialslot.

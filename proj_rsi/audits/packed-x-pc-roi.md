# Fresh packed-X PC/memory diagnosis

state=VALID_PHASE_EVIDENCE; immediateA6=HOLD; next=ROOT_ADMITTED_SINGLE_PRODUCER_STORE_TEST; noGPUbyauthor; frozenPXunchanged.

Result: decoded-Bhandoff is now the largestnamedmathphase:1027/2387allmathsamples(43.02%),1012longSB. Producerstore-proxy/bar3/convergence has276/1111producersamples(24.84%). This supports a boundedproducerstore/handoff test ahead of blindlyincreasingAdepth. It doesnot establishuniquewalltimecausality or thatallBfullsamplesaretimewaitingforalateproducer; TRYWAITresultreadiness/hardwarebarrierlatency is included. Registerfences alsoexposed471samples/469longSB, but arenotuniquelyXconsumers. No~15usgainforecast.

## Identity / reproducibility

Artifactdirectory `results/proj-rsi/takeover/20261027/packed-x-ncu/`; regeneratealltables/hashes with `python3 results/proj-rsi/takeover/20261027/packed-x-ncu/analyze.py`, output `analysis.json`. Program verifies everycanonicalsourcePCinstruction againstactualJITSASS afterabsolute-base subtraction andbranchaddressnormalization; all800main/56packerPCsmatch. NoGPUcalls.

Driver freezes source2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a /manifest0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4 /JITSASSf954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f. ExactA4physicalgatePASS; main122/pack22regs,0local,103600dynamicshared. Immediateprofiledoutputoracle1.421e-4relL2/2.633e-4worstrow/12probeexact;packedall1MiBexact,input/W/packedW/escimmutable,cnt0,graphreset. Rootterminal00:21:07Zexit0; noTraceback. Hostlogwarnsunmodifiedclocks andsixunavailableCTCmetrics; thoseareabsent,notzero.

Nsight2025.1.1/CUDA13/H10080GB/132SM; kernel/node replay,cache-controlall,clock-controlnone,NVTXprofile/,launchcount2,34passes/kernel. IDs0packer/1main only; nofreshincumbentcomparisoninthiscapture. Diagnosticdurations4.448/35.872us arenotbenchmarktimes; validwholepipeline39.648us vsinc24.224us remainsin `packed-x-next.md`.

## Actualrole / phase mapping

Mainfunctionbase0x7225c37ad100; allfollowingrangesarebyteoffsetsinactualbinary:

|role|inclusiveoffsetranges|all/notissued|
|---|---|---:|
|math|a70..1da0 andoutlined2ee0..2fd0|2387/1587|
|producer|1db0..2ed0 andoutlined2fe0..3140|1111/723|
|common/setup/unreachable|otheroffsets|82/68|
|totalmain|800PCs|3580/2378|

Rolebranchata60targetsproducer1db0; ordinarymathendsEXIT1da0. MathBfullslowloopsreturntoca0/1010/1270/1550; producerWfull/Bemptyslowloopsreturn2170/2210. Producerescapeconvergenceoutlinesreturn1e00/2990. Programassertsexactbranch+returntargetsandbarrieroffsets; nolexicalclassificationofallpost-EXITcodeasproducer. CommonstartupincludespredicatedXloadsandcannotbeassignedarbitraryrolefraction.

Sourceall-state/#Samples andrawtotalagree. Visibleperreasoncolumnsomitwarpgroup_arrive:mainresidual139all/94notissuedexactlymatchesraw139/94; allresiduallandsatHGMMA PCs. Nodroppedbytes,33554432Bbuffer,2048cycleinterval,2PCaggregatedpasses. Sourceinstructioncount8,943,381vsrawHWsmsp7,985,591isretainedseparately:patchedsourcecounts/replaypolling/predicatesnotanexacttimeorHWdecomposition.

|mathphase|all/notissued|longSB|interpretation|
|---|---:|---:|---|
|Bfullchecks+predicatebranches+outlines|1027/832|1012|largestnamedhandoffexposure|
|registerproxyfenceWARPGROUP.ARRIVE|471/320|469|ordersA/accumulators; notuniquelyX|
|HGMMA|228/121|0|139omittedwarpgroup_arrivereasonsincludedinall|
|directXLDG|55/24|0|6MIO,0LG inthisrole;returneddatawaitcanbeaconsumerelsewhere|
|wait_group|16/8|0|below30phasegate|
|drain/output|71/58|40|includesfinalEXIT/TMA/drain; noleadingclaim|
|otherloopaddress/release/control|517/222|79|mixedopcodework,notasinglecause|

MathBfull uses[...+0x19030], checksatc50/f80/11f0/14c0; dependentbranchesatc90/1000/1260/1540have161/139/328/391samples, with160/137/326/389longSB. IncludingonlySYNCSchecks wouldlosemostexposure. FirstTRYWAITcanblock; retrycountsnotwaittime.

Fencesce0/15d0have244/227samplesand243/226longSB; otherfences1040/12a0zero. WARPGROUP.ARRIVEiswgmma.fence forregisterordering,currentA andaccumregisters. It hasnoexplicitpayloadoperandintheSASS; nearbyfuture-prefetchloadsanddescriptor/controlreadinessarealsoinflight. ThuslongSBatthisfenceisnotdirectproofthatA4prefetchdepthisthedominantlimit. HGMMA/descPCresidualsalso cannotallbecalledX. [NVIDIAwgmma.fencesemantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence).

|producerphase|all/notissued|associatedstates|
|---|---:|---|
|Windex/address|109/35|wait39,selected30|
|Wfullwait|83/68|74longSB|
|Bemptyindex|61/35|wait31|
|Bemptywait|70/58|58longSB|
|LDS/decode/escape|228/139|56shortSB,69wait|
|storeaddress+16scalarSTS|154/78|76wait,9shortSB,5MIO|
|proxyfence/bar3/convergence|276/225|177barrier,71longSB|
|publish/refill|63/37|27wait,7longSB|
|loopcontrol|52/36|29wait|
|setup|15/12|below30phasegate|

Wfull2140/2160targets3050returns2170, uses+0x19000. Bempty21f0/2200targets3090returns2210, uses+0x19070. Publicationphase2c10..2c70 includesMEMBAR/FENCE/WARPSYNC/bar3+predicatebranch. Branch2c70has187all/177barrier,bar3itself40all/38longSB,proxyfence42all/33longSB. Thiscanincludewriterwarpskew/storecompletion; notmeasuredpurebarrierintrinsiclatency. Decodehotspot2280LOP3R45readsR4fromLDS.128at2220,48all/47shortSB:packedshared-loaddependency,notcodecALUsaturation.

Bothrolespass>=200allgate; Bfull/registerfenceandnamedproducerphasespass>=30. Sourceprofilingperiodiccorrelation androleinterleavingpreservecausaluncertainty; thresholdsarenotconfidenceintervals. Noadditionalprofilebyauthor.

## Memorywork / rejectedexplanations

Main262656LDGrequests/4196352sectors=32Buseful/sector, exactlypreregistered4v4nativeA sourceplusmetadata.0L1hits/4196352missesfitsno_allocate. NoV2half-sectoramplification. MainsharedLDG98304/STS540672,DRAMread51834112B,write2362368B. NormalizedlongSB5.40/short.24/LG.01/MIO.04,eligible.34areaggregateinstructions/cycle ratios,notphasecosts. L2TEXreads3866004sectorshave2151748hit/1655024miss;multimetricreplaymeanssum/denominatorcrosschecksarenotforcedintoexactreconciliation.

Packer8192LDGrequests/32768sectorsmatches1MiBreadandfullsectorproof. DRAMread1051904B/write0doesnotmeanmissingwrites:globalcachecanretainstores,anddriververifiedall1MiBexact. Packer306all/294notissued/246longSBsamplesisaseparatekerneldenominator; donotadditintomainrolefractionsoruse4.448usprofiletimeaswholegraphpacklatency.

## Disposition / one next causaltest

A6/fiveahead remainsconditionalCPUplan, notimplemented:6nativearrays,initial0..5,wait1retires(s-1),refill(s+5)intoretiredslot,60stage6waybody+4tail,lastprefetchats58forstage63;~154regsisprioronlyandfuture6slot/tailactualJITphysicalproofrequired. FreshBfull/publicationevidence outranksunprovenAdepthdominance. No oldV1PCsubstitution.

Rootadmittedone NEW `operand_reverse_stmatrix/` producerstoreversion:16scalarSTS->4stmatrix.x4, samedecodedWbytes/SW128/rings/proxyfence+bar3/refill/cache/mathA4/freshpack/output. Priorpw19usedthisprimitive; noveltyisnarrowcurrentPXport,notnewfamily. CPUfullstorebit/address/layoutproofandactual4STSM/0spill/resources+actualJITretirementgatebeforeGPU; full2-nodeactualinc/PX/V1/newcomparisonandallnumerical/workspacechecks. Prediction:lessstore/addressissuecouldadvancehandoff; payloadbytes/physicalsharedportcost/barrierskewarenotassumedreduced. NoimplementationclaimorspeedforecastfromPCfraction.

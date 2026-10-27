# DP fresh counter/PC diagnostic: independent preflight

`decision=GO_ONE_BOUNDED_DIAGNOSTIC; GPU=UNRUN_AT_REVIEW; performance_credit=NONE`

Driver`proj_rsi/tools/dual_producer_ncu.py` SHA256`65e8d9fda91c2b281f7a8b72a693b0f94f37ef1fd4d61fb2160b991a2dc65525`; frozenDPmanifest13`281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241`, source`97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b`. BasePXdriver`a3e81cc0adece02ffbf7f28250bb2b798642b94ccdfc63762ea90054b9b74d4d` bound. Readcompletebase+newdiff and source/actualJIT/benchmarkevidence, no candidateedit/GPUlaunch bycritic. DP34.496usvsactual24.096us remainsnonwinner; thisasksfreshremainingphase, notabenchmarkretry.

Concreteprefreezedefectcaught: Xloadmap searched`LDG.E.128`, whileactualinstructionis`LDG.E.NA.128.CONSTANT`, producingemptylist. Authorcorrectedmatcherandadded**16looprefill/32totalmainvector-loadsite** assertions. Driverfreezeafterfix; noGPUdatafromoldmetadatarevision. Reviewerexecutedread-only`cpu_check()`: all13hashes/source+512hostinfo6,localphysicalA4, fourfastBfullchecks/failed-entry/outlinedretry/readyreturn, fourfence/fourwait1, twoownerbarriers,16refillmap; invertedpredicate negativecaught; zeroCUDAcalls. Reviewerprint initiallyusedwrongoptionalkeyaftercpu_checkpassed, then correctedprint/reverified; no sourcedefect or changedreport.

No parentprepare/selector/autotune. OnefixedactualWpack/escape/X/scratch/output/counterpair, captureexactfreshpack→DPmain graph, allvalueslive. ExactloadedJIT.soSASSsavedbeforephysical/mapgate andfirstmath; failurephase/no_math_launchedrecorded. **DP122regs/512threads/65536roundedpool/maxThreads512/0local**, shared103600, opt-in-beforeoccupancy>=1, pack22regs/0local/maxThreads>=256; noLDL/STL/USETMAXREG. Source/runtimeidentity recorded. Mapsderivedfromactualartifact, notoldPXPCs.

OutsideNVTX: poisonXP/outputNaN,256MiBflush. Onlyselectedactual2nodegraphreplay/sync inside`profile/`. Immediatelyafterselectedreplay, beforeanyfurtherreplay: taskoracle/finiteoutput/full1MiBindependentpermutation equality, X/W/packedW/escape snapshots/counterZERO; finallydrain/resetgraph. Unchangedbaseassertionsbindactualobservedoutputs, norepairreplay. Existingfullbasis/storagebits/20actualgraphbenchmarkcorrectness is separatepreviousvalidresult, notclaimedrerunbycounterdriver.

Rootonlyoneexclusiveprofilecontainerinner180/outer240/scopedSYS_ADMIN, **afterpreviouscontextdrains**. ProposedexternalCLI (no sourcevariant):

```bash
ncu --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none --nvtx --nvtx-include profile/ \
  --launch-count 2 --kill no \
  --section LaunchStats --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables --section SchedulerStats \
  --section WarpStateStats --section SpeedOfLight \
  --export /okc/logs/dual-producer-ncu01 -f \
  python3 /okc/proj_rsi/tools/dual_producer_ncu.py \
    --json /okc/logs/dual-producer-ncu01-driver.json
```

TargetprocessallunchangedfromauthoredCLI; selectedranges/capmustactuallyexportonlyrx_packand`rq_px_dp`. Requirefreshreport/import, exactly2correctlaunches (pack256×256/22regs;main128×512/122regs), knownunits/finitecurrentmemorycounters, actualsource/JIT/map/attrs, immediateoracles/reset, successfulhostexit+actualcontext/containercleanup. UnsupportedoptionalCTCfieldsremainunavailable. Noabsoluteprofilerlatencycomparison/speedgain. Kernel/node/cacheall flushesmain independently, unlike wholegraph immediatelyfollowingfreshXPwrites; isolatedcoldtrafficcanexposeadifferentlimiter.

PCsampleadmission inheritedcoarsegates: explicitallsamplecounts+dropped/interval/aggregationsettings, **mainmath>=200 andproduceraggregate>=200, leadingmappedphase>=30**, otherwisephaseINCONCLUSIVE. DPnow **8mathwarps and8producerwarps**, notoldPX4producers. CommonproducerPCsaggregatebothowners; ownerbar3/bar4locationsarelimitedspecificsites, notfulldistinctperownerprofiles. Allstateversusnotissueddenominatorsmustbelabelled; sourceomitswarpgroup_arrivereasoninpreviousNCUschema, sounavailableper-PCreasonsnotzero. Dynamicinstcountsvariablepolling/multipass cannotblindlyreconcilefixedworkorconverttoµs.

### Predicate and TRYWAIT interpretation

ReadexactcapturedSASSfastTRYWAIT→predicate-dependent@!Pbranch→outlinedTRYWAIT/retry→readyreturn andconvergencebeforeusingPredicated-OnThreadInstructions. `Instructions Executed` countsbranchinstructionevenwhenpredicatefalse. Forfailedbranchreportpredicated-onthreads/totalexecutedthreads whenavailable; any/32 is **warp-equivalentnormalization**, notexactfailedwarpcountunlessper-instanceuniformityproved. Sameaddr/phase, aggregateAvgThreads32, orcountdivisibilityalone do not establishuniformpredicates.

As [primaryPTX](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait) specifies, TRYWAITcanblockuntilcompletionorlimit. Thus successfulfirstreturn/fewoutlinedentries **cannotproveinitialbarrierready**, pureintrinsiclatency, orabsenceofproducerwaiting. HighBfullsitesamplesandfailedlanecountsarephaseevidence, notproducerstarvationtime. CombinewithcurrentproducerWfull/Bempty/CSR/LDS/stores/proxy/bar3/bar4, actualXLDG/ARRIVE/groupwaitPCsanddependencystructure. Noautomaticprofilerepeat/metricexpansion ifinconclusive.

## Actual DP profile: valid fresh data / coarse residual-phase gate PASS

Rootguard383540/`okc-proj-dual-producer-ncu01`,01:10:29→01:10:47Z,34internalpasses eachof2kernels; hostexit0/export/DONE, sixoptionalCTCmetrics explicitlyunsupported. Parentconfirmedcontextdrained beforecombinedtrial; allGPUidle0MiBafterthattrial. Driverimmediatetaskoracle relativeL2`1.421e-04`,worstrow`2.633e-04`,12proberowsexact; full1MiBXP/currentXbitpermutation, X/W/packedW/escimmutable,counterZERO,graphRESET_AFTER_DRAIN. Noextra replayrepairs. Reviewerreadactualsource/driver/host/session/raw/sourceCSV andreran actualJITphysical+PCmap exactlyequaldriverobjects.

Archive`results/proj-rsi/takeover/20261027/dual-producer-ncu/`:

| Artifact | SHA256 |
|---|---|
| raw.csv |`f8cc84a1dfbe03891c2404b757490cf55208577c90f6c39e1d460beb0c68a24f`|
| source.csv |`009911066bf32a19d0b977d8ced399e57ccb74b822726fb2cfa300de299de0bc`|
| driver.json |`679d357894349ddcb99a69856711f240f3ea1063b8af1c0c5ecfeeb3e8650217`|
| driver.jit.sass |`bebe472634172ceb69c0a8d3df2c3555c70f89f421693b4d12dee847b2e7e746`|
| ncu-rep |`041046635d201acf312336398ce0894c1d0c116a08859b59958a418229008c76`|
| host.log |`7a99bd485167c857f3309f0f99a3745014fad42f9a10e108d4536223eaaae69d`|
| session.txt |`d23d08618a8b2fb3225d465ceeded3932561ee1755e64b49d83be1d355cc0a55`|

Session/source/manifest matchesfixed65e8driver/DP97dd/281ff. ActualJITidenticaltoDPbenchmark artifact; physicalA4andactual4fastBfull/4outlined/16Xrefill/ownerbar3bar4map rerunPASS. Rawactualrx_pack grid256/block256/22regs andDPgrid128/block512/122regs. DP0local/103600dynamic/0static/+1024driverSMEM=104624total,512maxthreads/65536rounded/occupancy1. Pack0dynamic/static/+1024driverreserve. Knownbaseunits/requiredmemorymetricsfinite, DRAMbytes=32sectors; optionalunsupporteddataunavailable, notzero.

| Field | pack | DP main |
|---|---:|---:|
| LDG warp requests |8192|263168|
| LDG sectors32B |32768|4198400|
| LDG hit/miss sectors |0/32768|2048/4196352|
| DRAM read B |1051904|51467008|
| DRAM write B |0|2390528|
| L2 sectors |144200|3660351|
| eligiblewarps/cycle |0.08|0.41|
| issueactive% |4.24|29.68|

CPUpreregisteredsanity matched: XwarpLDG262144 +1024initialescwarpLDG=263168, Xsectors4194304+4096esc=4198400. VersuspreviousPXrequests+512/sectors+2048, alladditionalobservedsectorsareL1hits whilemiss4196352unchanged; thisisconsistentwithduplicatedescape-listreads bynewproducers, **not Xrequestamplification or extraWdecode**. Producerstaticbodycoverage remains32768 =8warps×32stages×128CTAs, samefixedwork asold4×64×128. NoaggregateL2/DRAMsumforcedtoequaltheoreticallogicalpayloadacrossboundaries/replaypasses. PackDRAMwrite0again doesnotnegatefullXPwriteoracle. Diagnosticdurations4.608/33.984usnotwholepipelinebenchmark.

### Actual PC support

Explicitpack291/main3395allsamples agreeexactlyrawsamplecount;2aggregatedPCpasses,2048cycleinterval,32MiBbuffer,0droppedbytes. MainrelativePCbase`0x7dfdbb7ad100`. Conservativecommon`<0xaa0`89samples; math`0xaa0..0x1dd0`plusoutlined`0x3110..0x3200` **1750** (1092notissued); producer`0x1de0..0x3100`plusoutlined`0x3210..0x3410` **1556** (986notissued). Bothroles>=200. Mainreasoncolumns3178+rawwarpgroup_arrive217=3395; sourcepagedoesnotexportthat217reasoncolumn, notunclassifiedzero. PerPCdynamicinstsum8481243versusrawsmsp8176336; preserve multipass/pollingdifference ratherthanclaimfixedinstructionreconciliation.

| Mapped phase | allsamples | reason evidence | within-role fraction |
|---|---:|---|---:|
| mathARRIVE PCs`d10,1070,12d0,1600` |833|832longSB|47.6% of1750|
| mathBfullfastTRYWAIT+failedentry PCpairs`c80/cc0,fb0/1030,1220/1290,14f0/1570` |118|109longSB|6.74% of1750|
| producerBempty `2270/2280` |626|620longSB|40.2% of1556|
| producerWfull `21c0/21e0` |103|99longSB|6.62% of1556|
| producerpublication/proxy/ownerbar/convergence `2e30..2ec0` |212|115barrier+57longSB|13.6% of1556|

ARRIVE`d10`463samples/462longSB and`1600`370/370leadmath, other2ARRIVE0samples; nextHGMMA sites include217missingwarpgroup_arrivereasons. ProducerBemptybranch`2280`622/620longSBleadproducer; proxy`2e40`31, owner1bar4`2e70`33, owner0bar3`2eb0`0, predicatedowner0WARPSYNC`2e80`72/64barrier. Thesearelimitedowner-sites; commonproducerPCs combine8warps andcannotassignallwaits tooneowner. Decode`2300`53/51shortSB stillreadsR4fromLDS`22a0`, a smallerdistinctdependency.

**SamplegatePASS**: bothrolesenoughandnamedleadingphase833>=30. Coarseresidualhypothesis: mathregister/fence-entryexposure withproducerdecoded-ringbackpressure; currentBfullphase remainsbutisnotlargestmappedmathphase. NeitherexactXtransaction/specificGPRnorhardwarewaitmaskcausalityhasbeenproved. Per-registerfence/LDGplacementreviewisjustified; numericalperformanceeffect requiresanewvalidtrial. NohigherA-ringdepthinference (naiveA6cannotfit512uniform128regcap), norpercent→µs conversion.

### Predicate counts: failed returns, not initial readiness

All4fastTRYWAITsites16384warpinstructions/524288threadinstructions each, matchingfixedsourcecoverage. Following@!Pentrybranch predicated-onthreadcounts:

| FastTRYWAIT / failedbranch | PredOnfailedthreads /branchthreads | failedlanefraction |
|---|---:|---:|
|`c80/cc0`|137696/524288|26.2634%|
|`fb0/1030`|25824/524288|4.9255%|
|`1220/1290`|453376/524288|86.4746%|
|`14f0/1570`|196832/524288|37.5427%|

Sum813728/2097152=38.8016%observedfailedlanes. OutlinedretryPredOn/threads:286848/424544,7488/33312,1454336/1907712,417024/613856 respectively. Theseprovefailedreturns/retries occurred; they arenot exactwarpcountswithoutuniformityproof, nottimefractions. Outlinedmathblockscollectively1PCsampledespitemanydynamicinstructions, reinforcing sampling/phasevisibilitylimits.

ProducerWfullfailedbranch`21e0`:0/1048576failedthreads despite99longSBsamples; Bempty`2280`:608/917504failedthreads (~0.0663%) despite622samples. SourceoffsetsidentifyWfull`0x19000`andBempty`0x19070`; actualbranchpredicate/target/retry/returnmapped, nointerveningpredicatewrite ininspectedfastchain. SuccessfulfirstTRYWAIT **mayhaveblockeduntilWarrived/Bslotretired**. Thereforelowfailedreturnsdo notproveinitialreadiness/intrinsic-onlycost. ConverselyBfullfailedlanes38.8%don'tquantifyproducerstarvation. Keepbothobservationswithoutforcingasingleexclusivecause.

NextCPUfence-pairhypothesis remainsaseparateversionwithlookahead3→2confound, primaryper-registerordering/actualphysicalcoverage/resourceproof andwholepipelinegate required. NoextraDPprofilerequest fromthisdata; freshreportadequateforcoarsequestion.

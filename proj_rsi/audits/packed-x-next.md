# Fresh packed-X: terminal audit and preregistered residual-main tree

state=VALID_PARTIAL_RECOVERY_NO_INCUMBENT_WIN; rootterminal=2026-10-27T00:13:09Z_EXIT0_NODE0MiB; newcode=NONE; GPU_BY_REVIEWER=0; frozenfiles=UNCHANGED. RootownsonefocusedPXmainPC/memorydiagnosticaftercurrentbyteplanedrains. Do notreuseoldV1PCfractionsascurrentPXstalls.

## Independent terminal audit

Raw `candidates/operand_reverse_packed_x/operand-reverse-packed-x01.json`, SHA205584d56ae306250d6236f0d75c186fe24b998a95e2393b17c1b21cf2789864. Frozenmanifest0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4,source2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a. All21frozenhashesstillmatch.

ActualJITSASS `results/proj-rsi/takeover/20261027/operand-reverse-packed-x01.jit.sass`, SHAf954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f, independentlyhashverifiedandrecheckedwithfrozenphysicalchecker. FullpositiveproofexactlymatchesJSON. Main122regs/0local/occupancy1/shared103600; pack22regs/0local/maxthreads1024. No dynamicpool. Actualincumbent s4s5hz/KS2, confirmedFL3profile.

All4096Konehots/FP64random/tasknormal+outlier passforallfourversions. GPUstorage-only65536BF16encodings packexact1MiB/inputimmutable, includingNaN/Infbitpatterns; separatefinitearithmeticoraclespass.20actualtimedgraphmutationspass, includingfivePXpackedworkspacepoison/fullrawint32comparisons; freshworkspace/freshoutputoraclepass. W/packedW/escbytesunchanged,countersreset,allfourgraphsreset. Newtimedgraphcontainsexactlyfreshpack+main2nodes on everycall. No standalone-main/cherry-pickedtiming.

IndependentlyrecomputedallF1samples/medians/conservativecontrolbefore-aftermin,21pairedgainsandwins:

|control|same-jobF1controlmin us|PXwholegraphF1 us|F1gain us|pairedmedian gain us|PXwins/21|
|---|---:|---:|---:|---:|---:|
|actualpw36|24.224000|39.648000|-15.424000|-15.840001|0/21|
|bestfailedV1|47.168002|39.648000|+7.520001|+7.904001|21/21|
|immediateV4|52.191999|39.648000|+12.543999|+12.511998|21/21|

Meaningfulcustom-kernelimplementationprogress: beatsbothfailedpredecessorsrobustly, yetstillnotactualincumbent. No port/officialscorechange/<3000usclaim/familyclosure. Completeversionchangesinclude freshlayout,request/demux,addresswork,nativeA4threeaheadvsV4Q4A2fourahead,resourcesandcompilerlowering. Do notassignallgainsolelytopacking.

Separateprofilerinvocations: incumbent21.918us,V144.896,V449.315,PXpack2.112+main34.622. These rankmainastheremainingwork; theyarenotanadditiveF1decomposition orpermissiontosubtract2.112from39.648. Cooperativefusionremainsparked: node/barrier-onlysavingdoesnotcrediblyexplain~15uswholegraphloss, andsame-kernelNCcoherenceneedsnewcontract. See `packed-x-cooperative-feasibility.md`.

## Next diagnostic identity / fixed-work predictions

CollectPX **main** rq_px separatelyfromrx_pack; retainexactsource,pick,grid/block,actualJITbinaryidentity, freshpackbeforemain andimmediateoutputoracle. Packer/workspaceinitialization mustbeexplicitinthemetadata, nothiddenasapersistentactivationcache. Diagnosticreplay/cachebehaviorcanchangeexposure; timingsremainunprofiledcomplete2-nodegraphonly.

Preregisteredmain-onlysourcepredictions:

```
128CTA *8mathwarps *64k64stages =65536warp-stageinstances
XLDG128=4*65536=262144warpinst/requests
allglobalLDGrequests=262656 IF same512escape/metadatarequests
X32Bsectorinstances=64*65536=4194304
allglobalLDGsectors=4196352 IF same2048metadata sectors
Xmainusefulbytes=128MiB; noSHFL.BFLY; noextraAidentitycopy
WproducerLDS=98304; WproducerSTS=524288; outputSTS=16384
WdecodePRMT=524288; mathHGMMA=262144 (sameusefulMACs)
```

Compareactualcounterswithpredictionsbeforeusingstallfractions. Unexpectedrequestamplification/extraLDG/localspill/shapeorpolicydrift invalidatessimpleA-depthinterpretation. L2/DRAMbytesarenotdirectlyequalnominalXrequests; samepackedXpiscached/reusedacross128CTAs. NoDRAMexplosion/bandwidthsaturationconclusionwithoutfreshcounters.

PCphaseadequacy: >=200allstatesamplesforbothmath/producerroleand>=30foranamedphasebeforequalitativeranking; otherwiseINCONCLUSIVE, notzeroexposure. Allstate/notissuedareseparatesets,notadditive; normalizedaggregateSB/MIO/LGratiosare notlatencyshares. EveryWfull/Bempty/mathBfull labelneedsactualSASSbarrieroffset+branch/returnmapping. SourceexecutioncountsarenotPCstallcounts; firstTRYWAITcanblockwithoutretries. Stopatinsufficientdatawithinroot'sboundeddiagnostic; noautomaticadditionalprofile.

## Conditional decision tree (no prototype admitted now)

|freshPXmain evidence|nextinterpretation|smallestreviewedfollowupcandidate|
|---|---|---|
|MathXLDGconsumers showmateriallongSB,mathBfullnotleading;request/sectorpredictionsmatch;noestablishedcapacityceiling|NativeA4/threeaheadmayleaveX/L2deliverylatencyexposed despitecoalescing|H_X:oneA6/fiveaheadnativebufferrevision, below; requiresfullresource/physicalproofandrootadmission.|
|MathBfullmaterial; producerWfullmaterial andrefill/publicationcouplingmapped|FasterXmayhaveexposedWproduction/refill|H_W:inspectexactWrefilldependencyandsourcehistory; establishallwarps'packedLDSretirementbeforeearlieroverwrites. No blindearlyTMA.|
|MathBfullmaterial; producerLDS/decode orSTS/fence/bar3material, notWfullalone|PackedWdecode/store/publicationmaybeunderfeedsmath|H_P:selectonephase-localmechanismonlyafterPCmapping; integercodecALUthroughputcannotbeinferredfromshortSBatanLDSconsumer.|
|ProducerBemptystrong whilemathXdependenciesremain|Consumerbackpressurepersists; producerworknotnecessarilycritical|FavorH_Xoverblindproducerrewrite; noallproducerstalltimeassignment.|
|Newmathwait/descriptor/control dominates, XandBfullunexposed|Doubledround/issuebookkeepingremainstructuralcandidatecosts|H_C:sourcecostaccountingfirst; nolicenseforgeometry/KS/ring/polling/cluster/codec sweeps.|
|InsufficientPCsamples/mixedphases/countersconfound|Causeunresolved|Holdimplementations; retainvalidpartialgain, nofamilyrejectionorpromisedtimebudget.|

### H_X: A6/fiveahead, conditional only

InitialnativeA stages0..5. Aftercommitstage(s),wait1retires(s-1); pin/releaseit,thenload(s+5) intoslot(s-1)%6 since(s+5)%6=(s-1)%6. NeverrefillcurrentA. Lastprefetchoccursats58forstage63; finalwait0retiresstage63beforeall6bufferscanbedropped.64stagesarenotdivisibleby6:futureimplementationneedsproved60stagebody+4stagetailorequivalentlyboundedtail, noout-of-range65thstage. Request/bytes/usefulmathcountunchanged; moreprefetchdepthdoesnotreducetraffic.

Registerprioronly: A6adds2×16u32=32registers/thread toA4's122 ->~154, logical154×384=59136<65536, sharedunchanged103600 andstilloneCTA/SM. Actualallocator,mod6/tailaddresswork,livevariables/codegrowthcanbreakthisestimate;0spill/stack/local,actualCTAoccupancy>=1,nomaxnregpool,exactJITphysicalretirementproof+oldaliasnegativecontrolaremandatorybeforemath. Neednew6-slot/tailcheckerreview; donotquietlyweakencurrentfour-slot/d==24gate. Extraresourcesmayharmratherthanhelp. NoA6sourceorcompileexistsnow.

OldV1PCshowsXexposureinthatoldbinaryandmotivatedtheexperiment; itdoesnotproveA6willhelpcurrentPX. AdmissionrequiresfreshmappedPXdatafitandcredible>=.5uswholepipelinegain; anytrialcompareactualincumbent+PXpredecessor,retainingfreshpackandalloracles/actualgraphpoisonchecks. Noautomaticgainestimatefrom32additionalregisters.

### H_W/H_P: what must be distinguished

Currentproducer:waitWfull ->waitBempty ->3packedLDS ->decode/escape ->16scalarSTS/lane ->proxyfence+128threadbar3 ->Bfullpublish ->issueW(s+6). EarlierrefillcannotoverwriteaslotuntilallfourproducerwarpsretireeveryLDS. A leader'sownloadcompletionisnotenough; movingthecopyaboveotherwarps'loadsmakesarace. Anyearlierrefilldesignmustproveanall-warpretirehandoffandnotaddabottleneckbarrierunaccountedfor.

FreshSTS/publishPCexposuremayjustifyreviewingastore-layoutprimitive; current16scalarstoreswriteexact8KiBdecodedW/CTA-stage. Historicalpw19usedSTSM, soanyportmuststateitspriorcoverageandprovecurrentr6fragment/SW128bitmapping ratherthanclaimingtheprimitiveuntried. Noportisadmittednow. LikewiseLDSdependency/exposuredoesnotadmitclosedescapefooter/aligneddecode/weightdecompositionideaswithoutnewcounterevidence.

### H_C: remainingstructuralgap

ActualolderRQ fullchannel128/KS2 has32stagesperCTA; reversechannel64/KS1has64 withsame128CTAs,twiceHGMMAinstructioncount/commit-releaseperiods anddifferentdecodedWsharedstorework. UsefulMACsareequal. RemovingSHFLdoesnotremove thosecosts; rawthroughput/phasecountersmustshowwhatisexposed. Noadditiveall-floor-savingprojection,newkernelarchitectureorofficialscoreforecast isderivedhere.

Nextaction=ROOT_SINGLE_PX_PC_MEMORY_DIAGNOSTIC_THEN_INDEPENDENT_INTERPRETATION; sourcefreeze=UNCHANGED; GPU_BY_AUTHOR=NONE.

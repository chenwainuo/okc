# STMatrix producer: compiled source-cost delta and conditional predictions

state=VALID_PARTIAL_WHOLE_PIPELINE_GAIN_ACTUAL_JIT_MATCH; STMX_RUNTIME_RESULT=COMPLETE_VALIDATION; GPU_BY_AUTHOR=0; frozenfiles=UNCHANGED. Roottrial380790/containerokc-proj-operand-reverse-stmatrix01 launched00:44:15Z,240/300s. Terminalrawresulttakespriority; noDP/STMXmix ornewkernel.

Artifacts: frozenPXlocaldeviceSASS `candidates/operand_reverse_packed_x/sass.txt` SHA26796f01bd72df371701d3f05c32dd08ab6c429295b383c6e85ff8a99cead236; STMX `candidates/operand_reverse_stmatrix/sass.txt` SHA02fd73c7709b6781fe17e4999c78739a55373fad88cab91c4ea6c7940b5bc653. Exactper-phase/opcode/range data in `audits/stmatrix-producer-delta.json`. BothactualloadedJITartifactsnowindependentlymatchthelocalfullinstructionsequencesexactlyandpassphysical/storeformgates; terminalevidencebelow. Theproducer-countcomparisonisthereforealsoactualJITevidence,notonlylocalpreflight.

## ExactproducerCFG scope

Offsetsrelativeeachfunctionbase; rangesinclusive:

|version|producerordinarypath|produceroutlinedpaths|excluded|
|---|---|---|---|
|PX|1db0..2ed0|2fe0..3140|math/outputthroughEXIT1da0,mathwaitoutlines2ee0..2fd0,commonstartup/padding|
|STMX|1d90..2c50|2d60..2ec0|math/outputthroughEXIT1d80,mathwaitoutlines2c60..2d50,commonstartup/padding|

Rolebranchesarea60->1db0(PX),a40->1d90(STMX). Outlinedproducerescapehelpersreturn1e00/2990(PX),1de0/28b0(STMX); producerWfull/Bemptyhelpersreturn2170/2210and20b0/2150respectively. Ordinaryoutput16STSareexplicitlyexcluded. Commonpre-splitpredicatedW/Xmetadataalsoremainexcluded; compilerremoves/moves2commoninstructionsbuttheyarenotcountedasproducerstagewins.

|compiledstaticregion|PX|STMX|delta|
|---|---:|---:|---:|
|allproducerPCs(includingalternatives)|298|260|-38|
|ordinaryper-stagebody|234|206|-28|
|setup+startupescapealternative|48|38|-10|
|alloutlinedproducerPCs|23|23|0|
|LDS/decode/escape(includingloopescapealternative)|133|129|-4|
|store+addresslexicalregion|35|14|-21|
|followingproxy/convergencelexicalregion|7|4|-3|
|publish/refill|34|34|0|

TheordinarybodyisPX2040..2ed0/STMX1f80..2c50. Setup/staticcoldalternativesarenotper-stageruntimework; donotmultiplyall38by64stages. OutlinePCcountsame23includesstartup7/Wfull4/Bempty4/escape8; branchpolling/replaycountsarevariable.

## Whatactuallychanged

Combinedstore/address+proxy/convergence block:42->18staticinstructions(-24).

|componentincombinedblock|PX|STMX|
|---|---:|---:|
|payloadstoreinstructions|16scalarSTS|4STSM.16.M88.4|
|addressarithmetic(excludingconvergence/proxy)|19|7|
|convergence/proxyinstructions(includingmovedones)|7|7|

WARPSYNC/NOP/BSSYmovetobeforethefirstSTSM; **noteliminatedglobally**. STMXstore14PCsinclude3ofthosemovedconvergenceinstructions; hencecomparing35->14alonewouldmisattribute3instructions. Decode/escape loses2BSSY+2BSYNC; core3LDS.128/16decodePRMT/16escapeSELremainunchanged. OtherproduceropcodeaggregatechangesincludeIMAD.IADD18->4,LOP3.LUT55->45,IMAD11->13; do notlabelalllogicalopsasaddressALU,becausecodec/escape/controlalsousethem.

PXstoreblock29e0..2c00,proxy2c10..2c70. STMXstore/address28e0..29b0(withWARPSYNC/NOP/BSSY),proxy29c0..29f0; last4STSMat2970/2990/29a0/29b0,thenMEMBAR29c0,proxyFENCE29d0,bar3(128threads)29e0. ActualSASS/JITgatechecks thisordering/fullwarpstoreformandphysicalA4retirement.

## Fixedusefulwork / testableinstructionmodel

Uniformproducerwarp-stageinstances=128CTA×64stages×4producerwarps=32768. Payloadstorewarp-instructionmodel524288STS->131072STSM(-393216); addressmodel622592->229376(-393216). Combineduniformstore/addressmodel786432fewerwarpissues. Thisdoesnotincludevariableescape/reconvergence/pollingdomainsorpredictmicroseconds; a rawtotalinstructioncounterneednotfallbyexactlythatamount. ActualSTSMmetricclassificationmaydifferfromscalarshared-storetaxonomy; useper-PCopcodesequivalencebeforeassumingrawshared-storecounter147456.

Payloadunchanged:8192decodedWbytes/CTA-stage=64MiB/launch; packedWfill48MiB+packedW-LDS48MiB+decodedWmathreads128MiB, nominalmainsharedtotal288MiBunchanged. MathX128MiBrequests,fullfresh1MiBread+1MiBwritepacker,output16STS/2MiB, W6/B8/TMArefill/barrierphasesandFP32mathremainunchanged. Coreweightdecode/escapevaluesandprimitivebitmappingarenotchanged; onlythecurrentPXstoreversionusespriorpw19'smatrixstoremechanism.

## Preregisteredpossibleoutcomes

- Ifproducerstore/addressissuewasexposed,earlierBpublicationcanreduceactualwholepipelineF1/pairedtimevsPX. A laterauthorizedsame-workloadPCcapturecouldshowlessmathBfullassociationand/orlessproducerstore/publicationexposure. Thisisaprediction,nottheexpectationthat43.02%oldmathsamplesaretimeorsavings.
- EarlierpublicationmayexposemoreproducerWfullorBemptybackpressure; thosearepipelinephasechanges,notnecessarilyregressions. A shorterwholekernelalsochangesPCsamplingdenominators/aliasing, so rawsamplecountdropsalonearenotlatencyproof.
- Ifbar3/proxywaitsaremainlystorecompletion/warpskew/TRYWAITintrinsicdelay,orotherdecode/Xdependencieslead, fewerissuesmayproduceasmalldeltaornochange. StableBfullphasewouldshowthisstoreversioninsufficient,notproveallproducerchangesfutile.
- STSMsamebytescanhavedifferenthardwarewavefront/bankbehavior; do notclaim4storesimplies4×lowerphysicalsharedportcost. NoDRAM/L2byteimprovementisassumed. Packer/matmulregisters/occupancyremainequal122/22andoneCTA; thisdoesnotfreezecompilerinstructionlatency.
- AnycurrentSTMXtrialpassmustincludeactualJITA4+storeformgates,fullbit/numerical/original-zero-fresh-outlier/probeworkspacechecks, completepack+main2nodetimingandactualinc/PX/V1comparators. Counterdiagnosticsarenotperformanceacceptance; noautoprofile/port/officialslot/familyclosure.

Nextaction=HARVEST_ROOT_TERMINAL_STMX_JSON_AND_BOTH_JIT_SASS; newCPU/GPUkernelwork=NONE.


## Validterminalharvest

Raw `candidates/operand_reverse_stmatrix/operand-reverse-stmatrix01.json` SHA346f603f31bfc2379f3f78c45aaef7b67ca591c6c5a941a6354105d386503c7d, statusSTMX_PROGRESS_DIAGNOSE_NEXT_LIMIT. IndependentlyrecomputedallF1medians/conservativecontrolbefore-aftermin,21pairedgains/wins. Actualparentpick=s4s5hz/KS2, confirmedFL3profile. MainPX/STMX122regs/0local/occupancy1/shared103600; bothpackers22regs/0local.

|control|same-jobF1controlmin us|STMXwholeF1 us|F1gain us|pairedmedian gain us|STMXwins/21|
|---|---:|---:|---:|---:|---:|
|actualincumbent|24.416000|38.272001|-13.856001|-14.400000|0/21|
|immediatePX|39.455999|38.272001|+1.183998|+1.440004|21/21|
|V1|47.327999|38.272001|+9.055998|+9.472001|21/21|

AllfourallKonehots/FP64random/tasknormal+outlierspass; bothpackedvariants65536BF16storage-bitchecksandfreshworkspace/freshoutputchecks pass.20actualtimedgraphmutationspass, including10packedpoison/fullbytecomparisons; input/W/packedW/escapeimmutability,counterzeroandRESET_ALL_FOURpass. All32frozenfiles(11STMX+21PX)remainunchanged.

ActualJITSASSlocalarchive `results/proj-rsi/takeover/20261027/operand-reverse-stmatrix01.{px,smx}.jit.sass`: PXf954729bce07478ad46c6b9fd2ab21f0f32d255737559e672b0533b620ebe36f, STMXac8e3807c5f0f0665907e0a71fd0e494a02641e2b7582ff84c21ca07912eb713. Independentlyhashverified,re-ranbothphysicalA4andstoreform/proxybar3gates,comparedALLcanonicalinstructionoffset/textrows tolocalSASSexactly. FourSTSMproducer/16outputSTSretained; originalPX32scalarSTS. Static298->260producerCFGcomparisonaboveisverifiedontheactualJITbinaries.

Validmeasuredprogress: producerstoreversionimproveswholepipelineoverPXby1.184usF1/1.440uspairedwithoutbyte/arithmetic/workspacechanges. Stillnotanincumbentwin; no port/officialslot/scoreclaim. SeparateprofilercallsPXpack1.760/main34.432us,STMXpack2.112/main33.375usarenotanF1decomposition; identicalpacksourcecanvarybetweencalls. ActualSTMXPCphasecapturehasnotoccurred,sodo notclaimmeasuredBfull/barrierphaseimprovementoruniqueaddressALUcausality. Theversiontestdemonstratesapartialgainwithitscompilerlowering/convergenceconsequences.

Nextaction=ROOT_DP_INDEPENDENT_PROTOCOL/TRIAL_OR_FRESHLY_REVIEWED_NEXT_MECHANISM; no mixingwithdualproducerbeforeitsactualvalidresult; no newkernel/GPUbythisauthor. A6/fusionremainconditional; sourcefreezeheld.

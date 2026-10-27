# Conditional packed-X cooperative fusion feasibility

state=CPU_DOC_ONLY_CONDITIONAL_PARKED; packed_result=VALID_PARTIAL_RECOVERY_MAIN_GAP_REMAINS; candidate_source_edits=0; implementation=NONE; GPU_BY_AUTHOR=0. Frozenpackedmanifest0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4 staysunchanged. Rootpackedpilot375811/containerokc-proj-operand-reverse-packed-x01 launched2026-10-27T00:11:19Z,240sinner/300souter; terminalharvesthaspriority.

Decisionnow: **HOLD, no implementation.** Cooperativegraph support and devicecapacity make a128CTA barrier plausible, but there is no actualpackedtiming yet, no fusedcompiledresourceproof, and unchanged`.nc`inputloads are not a documented coherent consumer of same-kernelpackstores. Thisis a conditionalfollowup, not a promisedlaunchsaving or a familyrejection.

## Primary API / graph contract

CUDA12.8 exposescooperative attribute on kernelgraphnodes and ordinarylaunches; CUDAgraphs explicitly modelcooperative nodes (including restriction that graphupdate cannot switchcooperative/noncooperative). Thus acooperative kernel is a supportedgraphnodecategory, not an invented ordinarygridbarrier. [CUDA12.8attributes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__TYPES.html), [graphAPI](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__GRAPH.html).

Grid-wide `cooperative_groups::this_grid().sync()` requires a cooperative launch. Devicecapability and occupancy must be queried; successfulcooperative API admission provides the participatinggrid contract. Totalblocks maynotexceed `cudaOccupancyMaxActiveBlocksPerMultiprocessor(fused,384,smem) * actualSMcount`; ordinarylaunch plusglobalspin isexcluded. CUDA12.8runtime also prohibitsdynamicparallelism inthiskernel. [CUDA12.8executionAPI](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__EXECUTION.html), [CUDAgrid synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#grid-synchronization).

Theattribute/docsestablishgraphsupport, **not a testedPyTorchcapture on thisnode**. Any admittedversion must actuallycapture thecooperativelaunch in thecurrentcapturestream, instantiate/replay it, inspectthecapturedkernelnodecooperative attribute, andvalidate freshdata. No graph-harness rewrite or hostnodecallback; nothinginthisaudit modifiesparentcapturingcode. MPScondition shouldbe recorded; thegraphAPI's explicit noMPS clause in device-launchedgraph requirements isnot silently generalizedtoeveryhostgraph. Currenttask ishost-replayedCUDAgraph.

## Actualnode metadata / conditional residency arithmetic

Localrawcapturereviewonly: `results/proj-rsi/takeover/20261026/reverse-v1-v2/reverse-v1-v2-ncu01-raw.csv`, SHA05257785c07758fbb9c4854670b0b0b40dace63e62dad6794bccdc74ef898751, ID0deviceattributes:

|field|actualcapturedvalue|
|---|---:|
|cooperative_launch|1|
|multiprocessor_count|132|
|max_registers_per_multiprocessor|65536|
|max_shared_memory_per_multiprocessor|233472B|
|max_shared_memory_per_block_optin|232448B|
|reserved_shared_memory_per_block|1024B|
|MPS_enabled|0|

Frozenstandalonepackedmainlocalcompile:122registers/thread×384=46848logicalregisters/CTA,103600dynamicshared; pluscaptured1024driverreserve gives104624Bbeforeallocationgranularity. These fitoneCTAperSM; tworegisterpools93696>65536, so registercapacity prevents2CTAs even thoughsharedcapacityalone couldfit2. Runtimeoccupancy useshardwareallocationgranularity and actualbinary, not thislogicalproduct.128CTAs <=132SM×1CTA/SM; no oversubscribedbarrier if actualfusedoccupancy remains>=1 andcooperativeAPI accepts128. NeverderiveSMcountfrom%nsmid or countotherprocesses asavailableSMs.

This is a **necessary conditionalcapacity proof**, not a fusedkernelresourceproof: addingpackcode/grid-syncstate canchange allocation andspill. Before anyfutureGPU: exactfusedlocalcompile,0stack/spill/local, no dynamicrole-reallocation, maxdynamicsharedoptin, then actualJITbinary/attributes and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for **thatfusedsymbol**, actualdevicecapability/SMcount/MPS, `128<=capacity`, andcheckedcooperativecapturelaunchsuccess. Currentpackedkernel's122registers cannot substitute forfuturefusedoccupancy. No compiledfusedsymbol iscreatedhere.

## Freshuniquepacking phase across all128CTAs

One fixedproposedpartition, notcode: all128CTAs participate; first256threads/CTA pack, remaining128threads jointhesamegridbarrier. Let `base=blockIdx*256+tid`, tid<256; u=0..7; `packed_idx=base+32768*u`. Thiscovers262144u32 exactlyonce (1MiB),8192packedbytes/CTA. Existinginversecoordinates mapidx toXword; stage increments8 and Xwordoffset256perpass. Everywarp stillcovers32consecutivepackedwords,4full32Binputsectors and4contiguousoutputsectors. No extraV1transpose, no redundant128-waypack, no cachedactivationvalues.

All384threads/CTA callgrid.sync exactlyonce **before** roleearlyreturns, initialXPprefetch, W/TMApipeline work or mainmath. Afterthebarrier, runthesame128CTA geometry/W6/B8/losslessW/FP32accumulators/plainoutput/nativeA4retirement logic. No CTA canconsumeanotherCTA's packedfragmentbeforeallglobalstorescomplete underasuitablecoherent memorycontract. Uniqueinput/packedbytebounds and unchangednativefragment/A4proofs mustbe rerun; source-lifetime equality alone doesnotcarryphysicalregisterallocation throughnewcooperativecode.

## Same-kernel coherence is an additional blocker

Frozenmain reads XP with`ld.global.nc.L1::no_allocate.v4.u32`. PTX8.7 identifiesthis asnoncoherentread-onlycache, excludes`ld.global.nc`fromthegenericmemoryconsistencymodel, and sayssame-kerneltexturefetchesofgloballywrittenlocations areundefined. Eviction hintsdo notchangememoryconsistency. Therefore`grid.sync()` isnotprimaryevidence thatthisexistingNCpathisvalidafterpackstoresinthesamekernel. [PTX8.7 scope](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#scope-and-applicability-of-the-model), [NCload](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc).

Currenttwo-kernelpilot isunchanged:packwritesfinish inaseparatekernel; XPisreadonlyduringmain. Futurefusionmusteitheradoptanexplicitlycoherentload/orderingcontract (e.g.reviewedglobalL2loadpolicy) or produceprimaryproofthattheexactNCbehaviorissupported. Suchapolicychange is aconfound requiringnewsource/bit/correctness/physicalretirement/actualgraphchecks; donotquietlycallitpurelaunchfusion. No cache-policyprototypeisimplementedhere.

## ROI model / admission criterion

```
T_two = packwork256CTAs + inter-node scheduling + main
T_coop = packwork128residentCTAs + grid_sync + fusedmain
possible_gain = removednode/scheduling
                - grid_sync
                - packdistribution/resource/cache/main changes
```

Neitherexpressionis an additive profiler-durationdecomposition. Fusion doesnotremove1MiBinputread+1MiBpackedwrite ormain128MiBXdemand. Packergoesfrom256CTA×256×4words to128CTA×256×8words, withmain-sizedresources andanall-gridbarrier; thiscanbe slowerthanlightweightstandalonepacking. Grid-sync latency, launchsaving, scheduleradmission and coherentloadcost areunknown. No microsecondgain isestimated.

Reopenonlyaftervalidfullpackedpilot: mainiscompetitive withactualincumbent, whilecombined2-nodepipeline isheldbackby nontrivialpacker/nodegap. A significantmainloss ofmanymicroseconds isnot plausiblyexplainedbyoneextra launch alone. Diagnosticpacker/maindurations mayrankwhere tolook, butadoptionrequirescompletegraphpaired/F1 gain>=.5us vsactualincumbent andalsoagainstvalidtwo-nodepipeline, allfulloracles, every-callfreshpack, poisonedworkspace/currentXbyteequality/freshoutput, graphreplaychanges/zero/outliers, unchangedW/counterreset, actualcooperativeoccupancyandphysicalA4gate. Ifpackingworkratherthannodeoverhead dominates, ordinaryfusiondoesnotbyitselfsolveit.

Untilthatcondition: HOLD/noqueueimplementation/noGPU/noautomaticfamilyclosure. Terminalpackedresultwilldeterminewhetherthisconditionalopportunity deservesfurtherwork.


## Terminalpackedupdate

Validpilot205584d56ae306250d6236f0d75c186fe24b998a95e2393b17c1b21cf2789864:wholeF1PX39.648usvsactual24.224,V147.168,V452.192; pairedloss15.840usvsincumbent. Separateprofilepack2.112us/main34.622usisnotF1decomposition. Fusiononlyremovinganodecannotcrediblybetheprimaryrepairforthisgap; parkingisnowevidence-basedconditionalranking,notfamilyrejection. Root'snextfocusedmainPC/memorydiagnosticcomesfirst. See `packed-x-next.md` forfullterminalaudit/nexttree. Noimplementation/GPU.

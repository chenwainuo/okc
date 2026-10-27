# Actual GV53 M8 parent/quarter/sparse counters

**Valid diagnostic, no gain admission.** All12 requested metrics were available. Three exact selected launches exported three rows, each reporting4NCU replaypasses. This is not a threephysicalkernelinvocation cap: kernelreplay executes passes/restoration behind each selectedlaunch. Profiledurationranking differs from the valid clean screens; neither profile timings nor counters replace those signed outcomes.

| Metric | Parent | Quarter | Sparse |
|---|---:|---:|---:|
| Profile duration, us |22.400|21.664|21.984|
| DRAM read, bytes |50,998,272|50,999,296|50,998,784|
| DRAM write, bytes |1,885,184|2,070,016|1,806,592|
| DRAM throughput, %elapsed peak |70.58|73.28|71.89|
| Issue active, %active peak |20.32|27.41|23.24|
| Eligible warps /activecycle |.30|.44|.35|
| Executed warp instructions |3,223,396|4,404,829|3,751,027|
| Long-scoreboard samples, % |82.56|79.48|80.20|
| Short-scoreboard samples, % |.43|.90|.55|
| Fixed-latency wait samples, % |2.68|3.82|4.20|
| Branch-resolving samples, % |.18|.40|.46|
| Math-pipe-throttle samples, % |.73|1.15|.78|

Counterpercentages have their own scheduler/active/elapsed denominators. **82.56% longscoreboard does not mean82.56% of kernelwalltime, repairtime, or attainablegain.** No sampledPC, L1sector/request, L2 or per-source-load counters were collected. Sparse lowerinstructioncount versusquarter is demonstrated, not its criticalpath placement. NCUvalues above describe one logical profiledtargetreport; do not multiply/divide them by4passes to manufacture a cleanlaunch estimate. Globalmemorycounters are reportedvalues, not proof that thekernel semantically writes1.8–2.1MB.

## Mechanism evidence and limits

Quarter increases executedwarp instructions **+1,181,433/+36.6518%** versusparent. Sparse removes **653,802/14.8428%** versusquarter but remains **+527,631/+16.3688%** versusparent. DRAMreads vary by only1024bytes acrossarms, approximately51MB each; there is no measured traffic-reduction benefit. Sparse header-skipping thus realizes a dynamic instruction reduction relativequarter but doesnotremove its netoverhead relativeparent. This is a measured tradeoff; how much it costs on thecleancriticalpath remains **UNKNOWN**.

The sameweight bank5344 contains4096warpgroups/4759acceptedrecords, nofull32groups,4421nonemptyunits. Unitoccupancyhistogram0..8:1261/1641/864/274/50/6/0/0/0. Source expectedlogicalescapeSHFLwork: parent8855; quarter32768header+4759data; sparse4421header+4759data. These sourcecounts explain a candidate instruction-work difference; **they are not measured SHFLcounters**, and do not account for all instructions/branch/address/decode work. Aggregatewarpinst demonstrates extraexecutedwork butcannotprove theexactsourcecomponentthatcausedit.

Allthree show high longscoreboard sampling, modest issueactivity and low eligiblewarps alongside70.58–73.28% elapsedpeakDRAM throughput. This supports investigating **memory-dependency exposure/overlap** ratherthanadding anotherper-recordrepairinstructionvariant. It doesnotidentify W versusX versusescapeloads, establish saturatedbandwidth, or prove branchcostcaused thecleanloss. Branchresolving is<.5% under thismetric; no support for a large branch-resolving-only explanation. Fixedwait/shortscoreboard are higherquarter/sparse, still noPCattribution. Instructionpressure can coexist with memorywaiting; neitherpercentages norprofileduration alone locate thecriticalpath.

Six successful read-only clockcontexts show the sameGPUUUID/P0/SM1980MHz/memory2619MHz,35→36C,121.12–142.96W. No endpointclockshift is observed **in this profile**. These snapshots bracket flush/profile/oracles, not individualreplaypasses; they neither prove constant clocks withineachpass nor explain/reject clockcauses in priorunprofiledruns, which lacktelemetry. NCU clockcontrol=none preserved automatic clocks. Fixedarmorder and NCUmultipass/cachecontrol remaincontextlimits.

Clean evidence is unchanged: quarterpaired−.144us versusparent; sparsefollowup−.224us versusparent/−.064us versusquarter. Profiledurations favorquarter/sparse; those instrumenteddurations are **not newgains** or overrides. Preserved static/resource/source compatibility is distinct from dynamicallyselectedexecution andofficialH200score.

## Validity, identity and cost

CPU replay of strictCSVparser equals archivedanalysis byte-semantics. All18collectedfilehashes and24launchsourcepins pass. SparseSO2eb511c0... suppliesexactparentarm0/sparsearm1; quarterSO4b85c059... suppliesexactquarterarm1, originalPyInitnamesloadedwithoutJIT. CU/SO/SASSreceipts match eachactualselectedsymbol. Livegraphqueries: eachonekernel,nodeType0,clusterDim[0,0,0],grid[1,128,1],block[1024,1,1],dynamic17408B;64registers/local0/activeCTAcapacity1allarms. NCU2025.1.1 version/query/arguments/exportsuccess recorded.

SameWseed5344, exactactualbitreconstruction, exactstabletransportbanks, samecurrentnormalXseed91600allarms. Eighteenjournalrows(15warm+3profile),9initialtask+FP64oracles,8boundarycolumns/arm,3ownprofileterminaltask+FP64/currentXchecks beforeanyreplay/poison/repair ofthatarm,15retainedmutationchecks pass. FullS1counter/scratchbacking/input/weight assertions passed. Internalpostscope allocated/reserved/mapafter0; externalmemory0/noPIDs/container/guards;exit0. No observed setup/selection/numeric/lifetime failure; passing this scopedcontract isnot proof of allpossiblebugs absent.

ActualnativeJITbuilds0; exactSOload.003979321s,pack/transport.775888779s,capturechild6.338447220s. Roottrialhostlog spans23:04:58–23:05:14(~16s includingoutersetup/query/exportcontext, not a preciseGPUbusytime). Rootowns DONE transition. Parent/quarter/sparse sources and allprevioussignedresults remainpreserved.

## Highest-value next code discriminator

Prefer the **incumbent parent** and its unchanged planar payload, not anotherquarterformat variant. CurrentST1 rolledstep issues `ld(s,u+ST)` aftercurrentrawWdecode and beforeescaperepair/HMMA. The smallest candidate is a one-target CPUcompile falsifier that stages thecurrentrawW/X values, then issues the nextunit'ssameglobalW/Xloads beforecurrentdecode. Keep pack, addresses, cachepolicy, cursor/SHFL, math, ST1/WR2/WK16/KU8/geometry unchanged. This tests overlap/longscoreboard exposure ratherthaninstruction-count reduction. It is distinct from bulkcopy/sharedrings or a payloadlayoutrewrite; historical bulkcopy losses and oldearlySHFL schedulingfailure remaincounterevidence, notfamilyrejection.

**First discriminator is codegen, noGPU:** actualSASSmustshow the intendednextloadissue movedearlier relativeitsconsumers; ifptxasundoesit, preservefalsifier anddonotbenchmarkidenticalmechanism. Current1024threads×64regs consumes thelegalregisterbudget: require≤64regs,zerospillandunchangedshared beforeproposingaruntimecapture, then queryactualfunctionlegality. ExtraILP canincrease registerlive ranges; failthatresourcecheck andthecandidateisnotrunnableatthisgeometry. No speedforecast fromearlierPCs or instructioncounts. Ifthisexactload-stage change wasalreadytried, reuseitsactualnative/result evidence beforebuilding itagain.

A cleanmatchedscreen ofa realizedchangedloadschedule wouldaskwhetheroverlapimprovesparent, retainmixedsigns, andpiggybackthelowcosthosttimestamps/eventreuse diagnostic fromthefrozenF1audit. IfloadPCattribution isstillneeded, optionaltargetedPCsampling/per-loadcountercapture is moreinformativethananotherunchangedtimingrun. This report doesnotrequestorlaunchanewGPUjob; rootprioritizeslargeractualselectedopportunities andownsqueue.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize→repeat. [Contract](../../ITERATION.md).

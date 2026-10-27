# Native-X X4 depth isolation

CPU_FEASIBILITY_PASS_REVIEW_REQUIRED_GPU_UNRUN. [Candidate](../candidates/reverse_tma_x4/README.md), [proofs](../candidates/reverse_tma_x4/proofs.json), [manifest](../candidates/reverse_tma_x4/manifest.json). Sourcee8447cbf0c113618462f5cbddc61d7f1e438ada2efa43130531f96c9fde768d3. Onlynewdirectory/auditwritten; frozenparent025f/ffd30 andasync92fbe/45fbc unchanged.

Singlemechanism: parentnativeXsharedring2→4,initialTMA0..3,phasefloor(s/4),read(s)→refill(s+4). Keep parentmathbar5/electedmathissuer, A4/W6/B8/codec/nativeoutput/precision. Exactfiniteinverse-source check restoresparentbody byteforebyte afterreversing kernelname,SMEM,Xfullbase,slotmask,phase,refilldistance,initialloopbounds. Currentinputdescriptor encode unchanged eachcall using x.data_ptr/box64×128; nohostinputread orcache added.

CUDA12.8exactflags fit122regs/0stack/spills,512threads/rounded65536registerpool,6barriers,170016Bdynamicshared. Xring65536B[103424,168960),fourmbarriers32B[168960,168992); extra32784B overparent includes32768data+16metadata. Allocationadds1024B foralignment; allslabs1024aligned, eachrow16Baddressaligned, noW/B/output/metadataalias. OneCTAresourcepossibility retained, noGPUoccupancyquery.

Full64stage model: initialcopies0..3; mathprefetch0..3 consumesall4slabs andissues4..7 beforefirstMMA. Existingwait1 retires(s−1), refillsA(s+3), whichconsumesX(s+3) andissuesX(s+7) when<64. Thus noextraAregisters. ForX60..63 nofuturecopy; tail drainslastWGMMA beforeoutput reusesoldB. Eachslotoverwrite followsall8warpLDSMcompletion, perreaderproxyfence andmathbar5; consumerwait paritytracks everyfouruses. Modelproves64copies,64consumptions,60recycles,all524288BF16coordinates exactlyonce. W6/B8sourceunchanged; Bslotreuse waits math(s−7) aftergroup(s−8)wait1 retirement.

Localcubinactualgate verifies postCTArolebranch0xc50→0x2ef0,4initialTMA,math32staticLDSM/8bar5/8TMA/0LDG. EveryLDSMquartet precedes MEMBAR→proxyFENCE→bar5; nocopybeforebarrier. CanonicalA4physicalgate passes4disjoint16wordbanksR56..119, pendingWGMMAretirement,2fullmainloopiterations andall8outlinedwaitpaths. Compilerflags/resources fixed; noC75/serialwarning/localinstructions. DriverJITactualbinary stillrequiresgatesbeforefirstmath; fixedlocalgate isnotageneralSASSaddressdataflowproof.

Parentreal31.856uspaired stilllosesactualinc23.968us. MoreXlookahead might hidelatency; alteredsharedfootprint/copyaddresswork mightoffsetit. Ringdepth/sharedfootprint explicitlyconfounded; noAdepthchange, noPCfractionlatencyconversion/noHBMreductionclaim/noforecast. RawGPUbitmapping remainsUNPROBED. ThisversionhasnoGPUcontext/extensionload/prepare/driver/timing.

Next isindependentreview thenroot-only boundedwholepipeline actualinc/parent/new driverwithstrictframework: actualJITbeforefirstmath, allKbasis/FP64/task/freshzerooutlier/actualterminaltimedoutputs/immutability/counterzero/cleanup. Preservefailurewithsourceidentity; diagnosis notfamilydismissal.

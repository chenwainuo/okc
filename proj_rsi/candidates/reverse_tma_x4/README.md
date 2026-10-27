# Native-X TMA four-slot ring v1

CPU feasibility PASS; GPU UNRUN; independent review required. Sourcee8447cbf0c113618462f5cbddc61d7f1e438ada2efa43130531f96c9fde768d3. Parent025f/sourceffd30 unchanged; async-refill92fbe/45fbc untouched.

Isolate parentXring2→4. Keep electedmathissuer/bar5, sameA4/W6/B8/math/codec/output. InitialTMA stages0..3, eachcompleted Xread(s) recycles slot(s&3) with TMAX(s+4), parityfloor(s/4)&1. OriginalX tensor-map uses currentx.data_ptr eachcall, box64×128/SW128. NoXPpack/allocation required; retainedxp argument ignored. Finite exactinverse transformation proves remaining source equal toparentbody.

122regs/0stack/spills/local,512threads rounded128×512=65536registers,6barriers. Dynamicshared170016B: X[103424,168960),Xfull[168960,168992),1024alignmentallocation slack. +32768data/+16metadata versusparent. Sourcefit allows oneCTA resourcepossibility; noqueriedoccupancy claim. Ringdepth andsharedfootprint are explicit confounds; A4registercapacity unchanged.

CPU exactbitmap covers every8192BF16/stage/524288BF16wholeK once; sameparent RSregisterA. Schedule models4initialcopies,64consumptions/64copies/60recycles acrossall64stages, A4initial0..3 and wait1 refill(s+3), tail noTMA64..67, scratchdisjointness andfinaldrain. All8mathwarp sharedreads→proxyfence→bar5 happen before electedoverwrite. Actualcubin gate:32staticLDSM,8bar5,8mathTMA+4initialTMA,0mathLDG,physicalA4R56..119 disjoint/pendingretirement across2iterations and8outlinedwaitpaths. DriverJITbeforefirstmath regates mandatory; no generalSASSaddressinterpreter claimed.

Hypothesis: more sharedXlookahead may hideTMA latency while keeping sameA4delivery. It may insteadincreaseaddressing/sharedcapacity cost. Parent paired31.856us stilllosesactualinc23.968us; noX4speedforecast/timing/officialclaim. RawGPUbitmapping UNPROBED; numericaloracles later do not proveallpayloadbits.

`python3 preflight.py` performs CPUcompile/proofs only, regenerating files/manifest; preserveversionbeforechanges. NoGPU/context/extension/prepare/driver run. Nextindependentreview thenroot-only separatelyfrozendriver actualinc/parent/new wholepipeline, allKbasis/FP64/task/timedterminaloutput/mutation/immutability/counterzero/cleanup gates. No unchangedrerun orfamilyban inferredfromanyfutureloss.

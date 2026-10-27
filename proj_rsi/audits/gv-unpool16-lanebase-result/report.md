# GV16 lane-base collected DONE evidence

Recommend **VALID_NO_GAIN** on the primary paired point, preserving the positive mean and mixed F1 endpoints. This is a valid implementation result, not a family rejection.

Actual source b1eb9b3c versus501d control, H100 fixedunpoolerM16. All24 launchpins and5collectedhashes pass. ActualCU equals pinnedpilot+binding; actualSO SHA fab0a543cfae62c9fa4522903c27df6fe6034db981402bae9fa25550738ab820. CPU cuobjdump confirms BOTH native functions exactly match their pilot PC/predicate/128bitencodings. Hoist mechanism actually realized: new1960instructions/S2R5 versus713c1992/S2R20, control2080/S2R5. New108regs vscontrol110,local0; actualruntime activeCTAlimit4 forboth,static1024+dynamic5120bytes. SHFL17/P2R16/HMMA128/LDG113 unchanged. No bandwidth/stall cause established.

Signed control−new pairedmedian **0 us**, medianratio1.0;10wins/4ties/10losses,range−1.280000433..+2.272000536us. Pairedmean **+0.014666701us** retained. F1 armmedians controlbefore22.143999115/new22.272000089/controlafter23.360000923us: endpoints **−0.128000975/+1.088000834us**, average **+0.479999930us**. Controlbracket drift **+1.216001809us** prevents treating thataverage as an unmixed stable gain. Same-seed F1 endpoints and each24pair delta are retained in analysis.json. Difference-of-medians and median-of-differences remain distinct estimators.

Validity:91journalrows equalreceipt(81clean+10warm); currentWseed5151 reconstructedexact; samepackedW/escape input;6initialtaskoracles withtask+FP64 paths,16basiscolumns perarm;4timed ownterminalchecks(currentXexact before anyreplay/mutation);10retained mutationoverwrites; S1 fullbacking/input/weight contracts passed. Internalpostscope allocated/reserved/mapafter0,externalmemory0/noPIDs/container/guard,exit0. Passing these does not prove absence of every possible bug; none observed in this contract.

Cost:compile/load42.614424128s,pack.292665960s,total45.879030187s. No profiler/fullprepare/officialrun. This fixedcell H100 result is not an H200score/fullpolicy claim or a simultaneous713ccomparison. Prior713c singleweight−.080us androbust−.064us paired points remain preserved; cross-job differences cannot establish causal improvement from hoisting.

Cause audit: correctsourceandrealizedhoist **PROVEN**; unchangedruntime4CTA/no spill **PROVEN**; onepair-estimator tie and F1bracketdrift **PROVEN**; cause of residual timinganddifferencefromstaticprediction **UNKNOWN**. Removing rematerialization is insufficient to demonstrate a win in this run; neither hardwarebandwidth norSUBlatency is established.

Next falsifier: optional sameSO twoarm narrowNCU sampledPC/issue/scoreboard counters if rootprioritizes, after actualclusterattrs/supportedmetricquery, separate from unprofiledtiming. A repeatedunchangedperformance-only run would not identify the bottleneck. No newGPUgate. Preserve thisversion, parkintegration on primarytie, prioritize larger actualselected projection opportunities; reopen when counters identify exposedrepairwork or a concrete changeddependency mechanism exists.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize→repeat. [Contract](../../ITERATION.md).

# Two K64 stages per async group: independent preflight

`decision=GO_ONE_BOUNDED_TRIAL; GPU=LAUNCHED_BY_ROOT_RUNTIME_PENDING; reviewer_gpu_actions=NONE`

Exact14filemanifestSHA256`2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7`; source`e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`; device`5671af28b02d983aec46d76ba5e463ebbdc518bdb458bea5abf738f02aba6325`; lab`6bf0cd37b563337ffd8cc872ec20121f93c820756dad1432d60f7e57d35b79fb`; checker`7f71eabcb05a358547d22390fd6874e148ca2379ffaa394c9ad45390e18586cc`.

Reviewer read exact kernel/hostgenerator/whole-source restoration/checker/schedule/model/localSASS/resource/manifest/labdiff. Independently verified all14hashes; ran only pureCPUfunctions, neverpreflightmain (writes compileartifacts), and recheckedallhashes unchanged. No candidateedit/build/GPUjob byreviewer. ParentDPmanifest281ff/source97dd andDSmanifest6785/source672d bound inlab.

## Mechanism and source proof

Two original K64 stages become **one8-HGMMA asynchronous group**, with onefence/commit/wait1 perK128pair. Existing512threads, twoalternatingproducerWGs/scalarstores/CSRskip/W6/B8 **oldK64slots**, oldKC64, oldpackedweights, freshXPpack/index/cachepolicy, nativeoutput andSMEM103600 unchanged. This is not the separatewider-Kproducer/ringproposal with206000Bsharedmemory. WholegeneratedCUDA restores exactDP97dd afteronly math-loop/A-array/pack-load-interface/name reversal.

A4×16words becomes **A2×32words**, still64u32GPRpayload words/thread. EachA32 concatenates nativefragments foroldstages2p and2p+1; samefourK16MMAquads peroldstage, sameorder0..63 =>256totalMMAquads. BotholdBfullbarriers awaited beforefence; firstfourMMAuseBdesc(slot2p),nextfourBdesc(slot2p+1), onecommit. Same-shapedD32 accumulator chain/arithmetic; noconventionalDaccesspermittedduringasyncupdates.

InitialA bankscontainoldstages0/1 and2/3. Groupwait1 afterpairp retirespairp−1, nevercurrentA32. Retiredpair's **botholdBslots** released onlythen; bankrefilledwithpairp+1 ifvalid. Finalwait0 retirespair31 beforebothB62/B63release/outputreuse. Symbolicmodel validates32pairs,64oldBemptyreleases,exact256MMAquadorder,all32A-pairloads; prematurecurrentbankrefill negative rejects. Wait-forDAGacyclicunderfiniteTMA/CSR/fairwarp assumptions; it is not an arbitraryGPUinterleavingsimulator. Oldper-slotB8phase/8mathwarp-arrival protocol remains, no mergingbarriers.

Explicitconfounds: steadyAlookahead distance **3×64=192 →1×128=128 K elements**, notKiB; Bretirement iscoarser (two oldstages together) andMMAgroup/commit/waitcountchanges64→32permathWG. Noexclusivefence-latencycausalclaim or predictedmicroseconds. No shared-memorypin workaround, no C7512/C7520 CPUfailedfence_pair source mixedin.

## Independent CPU/device checks

Pure`sourceproof`,scheduleproof andretainedW/X/outputmodels ranPASS:524288WBF16values/denseescape/windows,262144Xwords/inverse/currentpackmap/all65536payloadbits,all8192Doutputvalues. Producer/escape/metadata source unchanged. SASS`d953e0d166bb64c23ee7463a003aa0a906db648fd4aa66c1c391aa005606e9c2`;localCUDA122main/22pack,zero stack/spill/local/C75/USETMAXREG. Warp-rounded128×512=65536; actualTorchJIT/occupancy remainsruntimegate.

Newphysicalchecker independentlyreexecuted andmatchesphysical.json:16staticHGMMA in **2groups×8**,2staticARRIVE/2wait1; disjointA32banksR88..119/R56..87 andD24..55;16initialvectorloadswriteseveryAword, CS2RzeroesallD. Two loopiterations/fourgroups covercyclicretirement;32payloadvectorwriteschecked. Version/fenced-state tracking rejects genericA/Dupdatesafterfenceorbeforecoveringretirement; source-compatiblevectorwidths included. Externalmathslowpaths traversed asCFG, noA/Dgenericaccess/newMMA/fence/exit permitted, finalwait0 beforefirstgenericepilogueA/Daccess. Nohardwarewait-maskdecoder/latencyclaim/genericISAverifierclaim.

Atreviewer'srequest, actualHGMMA/ARRIVE/DEPBAR (includingtailwait0) mustbe **unpredicated**, so`plain()`cannotmaskamissingcollective. CPUnegativesall5rerunPASS: realoldQ4/A2premature0xe30overwrite, removedinitialfence, predicatedalignedfence, newcurrent-bankoverwrite, genericpendingDwrite. RetainedDSstore-formgatealsorunonlocalG8SASS: unchanged32scalarSTS,0STSM,proxyfence/bothcomplementarybar3/bar4/nointerveningPwriter,16HGMMA/32XvectorLDGsPASS. Allsourcehashesunchangedafterreview.

## Exact lab, measurement scope and admission

Fouractualcallables: preparedPW36kernel_fn, frozenDP, frozenDS, newG8. Threepackedextensionscompilebeforebaselineprepare; all3loadedJIT.soSASSsavedbeforegate/firstcandidatemath. G8usesnewA2/fencechecker; controlsusefrozenA4checkers; alluse retainedscalar/DSstore/proxy/**bothownerbarrier**gate. AnyfailurepersistsINVALID/artifacts/no_math_launchandstops. Runtimeall3<=128regs/512threads/roundedpool<=65536/local0/info6maxThreads>=512/opt-in-beforeoccupancy>=1/shared103600; packer0local/supports256. ResourcefailurealsopersistsINVALIDbeforemath.

Threeexplicitdistinct1MiBXPworkspaces andretaineddistinctoutputs; actualgraphs alwaysfreshpack→main2nodes. Baselineactualreturnretained,nountimedvalidatorcopyintheactualtimedgraph. Full4096Kbasis/FP64/tasknormal+outlieroracles all4; GPUall65536storagebit/fullXP/freshworkspace+output all3packedvariants;20**actualtimedgraph** mutations(original/zero/freshnormal/outlier/allprobes) withNaNpoison/immutableinput,15XPpoison/currentXfull1MiBcomparisons. ExactW/packedW/escape unchanged/countersreset; allfourgraphsdrained/reset. Profilesmust2correctnodesforeachpackedvariant, actualincumbentpickrecorded. Noeagermain-onlytiming/packer subtraction/inputpreparecache.

Rootexplicitlyselectedexisting21sampleF1before/afterconservativemincontrols plus21rotated/reversedfourwaypairedrounds/256MiBflush/5warmups forfirstcausaltest. This remains a **localtimingproxy**, notTASK'sadjacentreversed-pairratio/overhead-calibratedofficialestimator. Preserveallrawseries/gains/wins; largeconsistentgain useful, mixed/smalleffectsrequiremeasurementreview/predeclaredbalanceddiagnosticratherthanautomaticversionrejectionorfavorablemetricselection. Actualincumbentminimum0.5usBOTHmethodsisreviewonly; no official/H200/<3000us claimfromthisH100lab.

Exactrootcommand`MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_group8/lab.py --json /okc/logs/operand-reverse-group8-01.json`, namedexclusiveinner240/outer300. **Root launchedguard388599/container`okc-proj-operand-reverse-group8-01`** after14remotehashes,runner755,noothercontainers,GPU0MiB. Remotehost/logprefix`/root/okc/logs/operand-reverse-group8-01.*`. No reviewerGPUlaunch.

Runtime/JIT/correctness/rawtimingarithmetic/terminal/nodecleanup **PENDING**, handed to next independentworker. Do not treatthispreflightGO asaperformancewin. Existing4trialconsistencyvalidator15eb explicitlyexcludesG8schema/newretirement_gatefield; donotforceitthroughthattoolwithoutanexplicitreviewedextension. Allpredecessors/CPUfailedversionsremainpreserved; anyresultdrivesnextcausalquestion, nofamilyban.

**Historical validation scope clarification at handoff:** this frozenG8lab validates the actual retained timedgraphs by20post-timing changed-input/poison replays. It does **not** inspect their finaltimedoutputs immediatelybeforethosemutation/replaychecks. Therefore thisauditdoesnotattestthatmissingimmediatefinal-outputcheck orofficialper-timed-callchecking. Futureharnessrepair mustgetanewmanifest; preserveoldsource/evidenceandreviewitwithinthisknownscope, withoutretroactivefamilyban. NextworkerSol03 ownsactualG8harvest andfutureGScriticalreview.

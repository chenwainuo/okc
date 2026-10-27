# Fixed DP/DS timing-contract diagnostic: independent preflight

`decision=GO_ONE_BOUNDED_MEASUREMENT_DIAGNOSTIC; GPU=UNRUN_AT_REVIEW; kernel_edit=NONE; official_credit=NONE`

Driver`proj_rsi/tools/dual_stmatrix_measure.py` SHA256`4812bbcfa2314db3db9ca481dcfa052daabb54e6d332c02c2866705dbc3d4293`. Plan`results/proj-rsi/takeover/20261027/dual-stmatrix-measure-plan.json` SHA256`2643a6043239623bc6a67cfcf8e59753c2a9faa38afe3a245f071673284eb997`. FixedDPmanifest281ff/source97dd andDSmanifest6785/source672d. Reviewerreadwholefile/localTASKtiming/plan andexistingactualJITSASS; no authorfileedit, CUDAcall/GPU/remotejobbycritic.

Reasonforoneadditionaldiagnostic isconcretemeasurementuncertainty: previousDS+0.832usF1but−0.256uspaired/10of21versusDP under4wayproxy. Existingraw/source/results retained; no unchangedkernelrerunjusttogetpass. TASKrequirescoldL2, takingturns, adjacentoppositeorderpairs/pairedratios andfixedgraphcostcalibration. Thisdrivercorrectsorder/statisticsquestion, **doesnotclaimexactunavailablejudgecalibration**. Noincumbent orofficialscorecomparison, no format/variantflag sweep.

Fixedseed270151 preschedules32blocks of2adjacentcomparisons, ordersswappedwithinblock;16blocksstartDP/DS,16startDS/DP, total32AB/32BA. Eachcomparisonusesfreshseeded1MiBinput sharedexactlybetweenAandB.64comparisons=128replays.6steadyblocks×32replays=192;320timed+30warm=350rawrecords. SteadybeforeDP/DS/DP andafterDS/DP/DS; reportall6mediansandbothbracketedeffects, **neverbest-blockselection**. Pairedprimary; steady-onlygainissensitivity. Bothmedian64rawDP/DSratiosandmedian32localgeometricoppositeorderratios reported; differentanswersmustremainvisible, notfavorableestimatorselection. Noassumed4uscalibration/noarbitrarysubtraction/CIorwinnerautomation.

CPUindependentlyverifieddriver/planhashandexactplanobject; synthetic350records produceexpectedpairgains/64wins/rawratio/geometricratio/steadybrackets. Missing **timedpaired**record negative rejected; warmrecordsareintentionallyexcludedfromsummary. ExistingactualDPbebe/DS145c JITSASS canonicalPC+instructionlistsmatchfrozenlocalmain+packer, physicalA4andstore/proxy/**bothcomplementaryownerbarriers**PASS. CPUtestusednoGPU/build andchangedno sourcehashes.

At runtimebindbothfullmanifests/generatedCUDA, compile/loadonlysameimmutableextensions, dumpactualJIT.soSASS beforemathandretainbeforegate. StrictphysicalA4/storeform/bar3bar4 and canonicalmain+packPC/instructionidentityagainstfrozencubin; recordfullSASShash+modulebinarySHA andverifyloadedbinary/sourceunchangedaftereachmode. Canonicalcomparisondoesnotassertdecodedcontrolwordmeaning; actualfullhashescanalso becomparedpriorbenchmarkartifacts. Expected122regs/512threads/rounded65536/local0/shared103600/maxThreads>=512/opt-inoccupancy>=1;packer22/0local supports256. NoBprepare/selector/incumbent orproductioncode.

ExplicitfixedX/outputA/outputB/XPA/XPB distinctregularCUDAallocations. Bothactualretainedgraphs alwaysfreshpack→main2nodes.256MiBtorch.sumflushbefore**every**replayoutsideGPUevents. Freshfixturecopies/pre-generationoutsideeventsarelabinputs; neitherkernelcachesinputderivedvalues.320timedand30warmeventtimes/order/previouscase/consecutiverunlength/hostmonotonictimestamps retained. Runtimeadmissionrequiresallrawtimesfinitepositive, exactplan/recordcount/ordering, pairedsamefixture andunchangedactualbinaries; noposthocoutlierremoval.

Crucialoutputpairingcheck: eachpolicyrecordslastTIMEDfixtureseed, includingwhensteadyblocksleaveAandBwithdifferentlastinputs. Immediatelyaftereachmeasurementmode **beforeanypoison/inputchange/replay**, validatesactualoutAagainstA'srecordedfixture/actualXPA, outBagainstB'sfixture/XPB; verifiescurrentXagainstfinaltimedrecord, weights/packed/escapes/counters0. ThisisnotrepairedoutputvalidationanddoesnotmistakeB'scurrentinputforA'soldersteadyinput.

Afterthatimmediatecheck, separatelyrunbothactualgraphs onall65536storagepayloadbits/fullXP, original/zero/freshnormal/outlier/128onehotprobeinputswithNaN+XPpoison/fullcurrent1MiB/immutableX; probeedgecolumns0/63/64/4095included, exactexpected128rows. Freshworkspace/outputforeachtwopolicies, fulltaskoracles/counter/weight/source/module/attridentity afterall4modes(initial+3measured). Thisdoesnotclaimrerunfull4096Kbasisorofficialper-timed-calloracle: immutablekernels alreadyhavefullbasispreviousvalidtrials; thisdiagnosticchecksfinalactualtimedoutputsplusbroadmutationstability. No newkernelcorrectnessclaimfromtimingalone.

Torchprofiler2nodeidentityonlyafterallunprofiledtiming; output/XPandoracleverifiedagain. Finallydrain/resetBOTHgraphs evenonfailure, persistINVALIDorcleanupFAILED onexceptions. Rootnormalcap/exclusivecontainer,inner90/outer120, exactcommand`python3 proj_rsi/tools/dual_stmatrix_measure.py --json /okc/logs/dual-stmatrix-measure01.json`; noGPUuntilthisGOandpriorcontextdrain. Requireterminalexit/actualcontainercontextcleanup, allsource/JIT/attrs/immediatechecks/4modevalidation/2profiles andrawrecalculation beforeadmission. Researchmeasurementonly; refs/subprocess/toolingneverportedintoproductionkernel.

## Actual balanced measurement: validation and statistics pass

`capture=VALID; independent_node_cleanup=PASS; persistent_archive=VERIFIED; DS_vs_DP=POSITIVE_LOCAL_DIAGNOSTIC; incumbent/official_credit=NONE`

Root guard386450/container`okc-proj-dual-stmatrix-measure01`, launched01:42:34Z. Reviewer read remote host tail: TRIAL_EXIT0/end**01:42:46Z**, then read JSON into own temporary`/tmp/critic-dual-stmatrix-measure01.json` without remote writes/job launches. This is an independent read-only harvest, not root archive ownership. JSON226197B SHA256`aa9c14d22deb9110a50e36e12b056658d1ba91b305ab7645ec052e0f47311378`. Status`VALID_FIXED_SOURCE_MEASUREMENT_REVIEW_REQUIRED`,phaseCOMPLETE,RESET_BOTH_AFTER_DRAIN. Root's persistent archive/actualcontext cleanup confirmation remains pending at this checkpoint.

Independently verified driver4812/plan exactobject, all350finitepositive rawtimes,320timed/30warm,6×32steady+64×2paired records,32AB/32BA,32adjacent reversed-order blocks, per-pair samefixtureseed and exact preregistered order. Recomputed complete summary from rawrecords with exactequality. For every measuredmode/policy, immediateactual-timed-output check uses its **lasttimedseed**, which independently matches last record for thatpolicy; no wrongshared-current-X comparison or repairedreplay validation.

Direct remote SHA256 of actualartifact files matched prior full-byte-reviewed actualJITs: DP`bebe472634172ceb69c0a8d3df2c3555c70f89f421693b4d12dee847b2e7e746`, DS`145c56d97b81451c43cb025f8908687c447a64cef54af0b7dfd6ee9ef51b35fb`. Driver additionally passes canonicalmain+pack/physical/store/proxy/bothownerbarriers and records unchangedmodulebinaries (DP`7218c66d5f1b6b9a1baf1555ea0561924778fd23cf3e6358a8668c5fa96cd8a1`, DS`59860bd3138327013ee905fcfc544b5983ff24c7a771f4925b43466da5e4fbda`). Both122regs/local0/512threads/65536rounded/maxthreads512/shared103600/occupancy1, pack22/local0/maxthreads1024.

62validationrecords: initial14 + eachof3measuredmodes16. Includes6 immediateactualfinaltimed-output/current-recorded-XP checks,8all65536storagebit/fullXP checks,40actualgraph inputmutation/NaN+XPpoison/fullbytes/immutable-input checks,8freshworkspace+output checks. Allfourmodes source/module/attrs identity checked; weights/packed/escapeimmutable,countersZERO. Bothpost-timingprofiles exactfreshpack+main2nodes andactualoutputs verified. No fullbasisrerunclaimed; prior fixedsources remainfullbasisvalidated.

| Predeclared statistic | Observed |
|---|---:|
| paired median DP−DS |+1.1679995805us|
| paired DS wins /losses |48/16 of64|
| median64rawDP/DS ratios |1.0347440603|
| median32opposite-ordergeometricratios |1.0309834056|
| DP→DS order median gain (32) |+1.2640003115us|
| DS→DP order median gain (32) |+1.1199992150us|
| beforebracketed DP−DS effect |+0.7759984583us|
| afterbracketed DP−DS effect |+1.1919997633us|

Allsixsteadyblock medians retained: beforeDP34.5919989049/DS33.7760001421/DP34.5119982958; afterDS33.5519984365/DP34.6079990268/DS33.2800000906. No best-block selection. Both predeclared paired estimates, both orders and both bracketed effects favor DS in this fixed diagnostic. This strengthens DS as a useful local successor after earlier mixed fourway observation; it does **not** erase that observation or establish its cause (clock/cache/order remain hypotheses).

No incumbent was measured, no calibrated replay-overhead subtraction, no official estimator clone or confidence interval. Therefore no new actualincumbent/H200/officialscore/<3000us claim. Use these data to choose local working parents while preserving both variants and measurement protocols; any future winning implementation still needs actualincumbent/fullpipeline and official validation. No further unchangedDP/DS run requested here.

**Cleanup follow-up 02:00UTC:** independentread-onlySSH ps guard386450 absent; namedDockercontainerabsent; nvidia-smi computePIDsempty/GPUmemory0MiB. No launches/kills/remote writes. Runtime/termination/nodecleanupcheckpointfullyresolved; persistentarchive remainsroot-owned. Parentearlierpendingcleanupwordingishistoricalreviewcheckpoint, notcurrentblocker.

**Finalarchivefollow-up:** rootpersisted4filesunder`results/proj-rsi/takeover/20261027/dual-stmatrix-measure01/` andpublic`proj_rsi/diagnostics/dual-stmatrix-measure01/`. ReviewerreadprimaryarchiveJSON226197Bexactlyequaltotemporarycopy/SHAaa9c14d2…11378; full13730BhostSHA256`4149ee666b307c5d5c2e25ce6cbabbe87f1aebb73ebc39e2fbf334cd4e1d86c0`provesTRIAL_EXIT0/end01:42:46Z. DP219282B/DS209608BactualSASSfullhashesbebe/145c verifiedagain; reranA4/store/proxy/bothownerbarriercheckers andcanonicalPC/instructionbinding withresultsexactlyequaldriverobjects. RootalsoconfirmedterminalGPU0MiB. **Alladmissioncheckpointsresolved; localdiagnosticVALID**, withstatistical/officialscopeunchanged.

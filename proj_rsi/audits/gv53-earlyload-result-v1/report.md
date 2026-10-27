# GV53 M8 earlier-load actual result

Recommend **DONE / VALID_GAIN** for research.gv53.earlyload.fixed.01. This is a valid same-window fixedcell H100 gain over actual501d nativecontrol, with every loss and timing estimator retained. It is not a fullpolicy/stack or officialH200 gain.

| Signed control−earlyload | us |
|---|---:|
|24paired median |+.192000531|
|24paired mean |+.058666570|
|Control-first12pair median |+.128000043|
|New-first12pair median |+.271999277|
|F1control-before−new |+.191999599|
|F1control-after−new |+.128000975|
|F1endpoint-average−new |+.160000287|
|F1same-seed matched before/after |+.032000244/+.223999843|

Pairedmediancontrol/new1.008879027;15wins/0ties/9losses, individualdeltas −1.312000677..+.992000103us. F1medians controlbefore22.399999201/new22.207999602/controlafter22.336000577us; controlbracketdrift−.063998625us. Bothorders andbothF1endpointspositive inthisrun; no minimumsize/significance claim. F1andpairedstilldifferentXbanks; nonlinear estimators remainseparate.

## Actual implementation and validity

All11collectedartifacthashes and18launchpins pass. Fullsubmission81fdc825... andnativepilot806c0dbf... are distinct receipts: inherited result.source_sha256 explicitlyhashes pilot.cu; fullsubmission is separatelypinned, not a drift/failure. ActualCU390793568... equals frozenpilot+bridge; actualSOd7ed7055... is thefreshthinDLL. CPU cuobjdump of actualSO reproduces everyPC/predicate/128bitinstructionword of BOTHfrozenpilottargets. Matchedcontrol is alsoidentical to theactual501dprofileparent.

Earliernextunitissue isrealized: newnextWLDGs0x6c0/750/790 precedefirstcurrentdecodePRMT0x7a0; originalnextW0x770/830/870 followfirstPRMT0x630. NewnextX0x660/690 versusold0x8a0/8b0. CurrentrawoperandsstagedbeforeST1overwrite; identicaldecode/repair/cursor/HMMA/FP32/outputorder. New384slots/374nonNOP versuscontrol360/348, includingextraMOVwork; LDG11/SHFL2/P2R1/HMMA4unchanged. No staticinstruction-savingclaim.

Actualresourcesboth64registers,local0,userstatic0,dynamic17408B,maxthreads1024,device-reserved1024B,queriedCTAcapacity1. Nooccupancyincrease orsharedmemorytradeoff. ThinCABI passesexplicitcurrentstream andstrongBoundBuffersowners; Native.boundsretains allpointers throughcapture/replay/mutation, retiresonlyafterallgraphsdrain/reset. Actualnativecompileandhostlinkpassed; actualSASSmatchespilotbeforeexecution. NoTorchC++extensioncompile.

SameWseed5344 reconstructsexactly; botharms sharetheoriginalpackedW/escapebank, noquarterrebucketing.91journalrows equalreceipt:81clean+10warm. Sixinitialtask+FP64oracles,8boundarycolumns/arm,4timedownterminalchecks withcurrentXbyteequality beforeanyreplay/mutation,10retainedmutationoverwrites pass. FullS1scratch/counterbacking, packedW/inputimmutability andoutputguards passed. Internalpostscope allocated/reserved/mapafter0; externalmemory0/noPIDs/container/guards,exit0. Noobservedsetup/selection/numeric/lifetimefailure; scopedpassingchecks do notprove everypossiblebug absent.

## Build and timing-method evidence

Nativecompile1.522851941s; hostlink.050189994s; phasecompile+link1.573321522s; taskpack.268795625s; fullrun5.120794627s. This realizes thevalidatedthinABIcontract for a freshchangednativebuild. Earlier42sPyBindcompile attempts were otherjobs; theiractualsavedreservation/cost cannot be inferred by subtractionas apairedresult.

Twoevents arepreallocatedandreused aftereachdevicesync forbotharms. HosttimestampsareoutsideGPUeventunits anddonot replacekernelmeasurements. Cleanhostsubmission(start.record→replay→end.record) median24.356us,range22.980..52.433us. Endenqueue→sync median100.234us,range84.529..104.883us, whichincludespendingcopy/flush/GPUwork plushostsync; itisnotkernel-onlytime. Journalopen/write/flush/close median25.938us,range24.551..126.374us. The33F1rows span6.531522ms start→lastsync, givingthepreviouslymissingtimescale.

F1controlbefore/new/controlafter submissionmedians24.169/23.988/24.389us; pairedcontrol/new24.373/24.658us. These bracketsshowhostcostcontext, not proof of CPUgaps exposingGPUeventintervals or causing thegain. CurrentF1drift isfar smaller thanpriorquarter/lane~1.15–1.22us, but events/interface/buildduration/ambientdevicehistory differacrossjobs. No deviceclocktelemetry inthiscleanrun. Thus olddriftcause remains **UNKNOWN**, notprovenfixedbyeventreuse/thinbuild. PriorprofileP0snapshots were anotherattempt.

## Cause and next decision

PROVEN: intendednative loadmotion atunchangedlegalresources; same-windowfixedcellgain onpairedpoint,bothordergroups andbothF1endpoints; actualtask/current-output/cleanupcontractpasses. LIKELY mechanism consistentwithdesign: earlierissue increases overlapwithdecode. Actualmemorywait/eligiblewarpchange inthisnewbinary is **UNMEASURED**, so do notclaim a quantifiedlongscoreboardreduction orassign allgain tothatmechanism. Previous12metricprofile gives motivation/context, not thisbinary'scounterreceipt. Nineindividualpairlosses remainpreserved.

Smallestnextdecision is retainthismeasuredincrement andmeasureanactualcompatiblepackage/freshselection againstitsparent/incumbent. **GV53keyedchoice andearlyload overlapM8**: naiveconcatenatedhostroutes mayoverrideoneanother. Either explicitlycombineknownkeyedrepairwiththeearlyloadbody whilepreservingoptions, orregisterearlyload as a legitimateoptionalcontender in theexistingchooser; then independentlyreview onlythatdelta andmeasureactualselectedfamily/terminal. Do notaddisolatedgains orclaimsourceinclusionmeansexecution. Noadditionalunchangedfixedcellrerun/profilergate isneededbefore thatstep. Rootchoosesqueue/stack/officialbudget; authorperformedCPUanalysisonly.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize→repeat. [Contract](../../ITERATION.md).

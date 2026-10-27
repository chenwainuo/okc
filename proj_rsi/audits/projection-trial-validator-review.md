# Saved four-arm evidence validator: independent draft review

Scope:`proj_rsi/tools/validate_projection_trial.py`, initialunfrozendraftownedroot. ReviewerreadcompletefileandranCPUcorruptionprobesagainstactualDP31849BJSON withcandidate-dirandartifact-dir. No sourceedit/GPU/remotejob. Existing4human-reviewedtrials remainvalid; thisreviews **automationcoverage**, notnewkernelperformance.

BaselineactualDPreturnsVALID_WHOLE_PIPELINE_EVIDENCE. Recomputingall7×21F1series/21pairedgains/wins, finitepositive samples,20unique(graph,input)coverage,currentcandidatefiles+mainnameandactualJITfilehashes areuseful. NarrowPX/STMX/DP/DSschemaonly; noNCU/official/numericalreexecutionclaim iscorrect.

FiveconcretecorruptedDPcopies were incorrectlyaccepted withfullsource/artifactoptions:

| Mutation | Why current draft passes | Required scope fix |
|---|---|---|
| remove`px`fromJIT,bits,fresh,attrs together | setsareonlycheckedagainstoneanother, notexpectedpackedcontrols | knownpackedlabels=`labels & {px,smx,dp,ds}`; requireall4coverage dictionaries matchtheirknownscope |
| `attrs={}` | resourceforloopvacuouslypasses | attrslabels exactly`labels−{incumbent}`; packerattrs exactlypackedlabels |
| `jit.smx.store_form_gate.status='FAIL'` | storeformrecordignored | rejectrecordedfailedrequiredsubgates, includingnestedphysicalbinaryoutcomes |
| firstactualgraph`oracle='FAIL: deliberately bad output'` | anytruthytextcountsaspassingoracle | rejectexplicitfailureprefix orboundclaimtodeclarativeflags; nonemptytextisnotPASS |
| controlSMXprofilemain renamed`wrong_main_shader` | noncandidatepackedprofilesonlyrequire2eventsand1pack | exactknownpackedmainlabelbinding; sourcecandidatebindingaloneinsufficient |

Reproinputsaredeepcopiesof`operand_reverse_dual_producer/operand-reverse-dual-producer01.json`; archives/source/JITfiles unchanged. Thesearemetadata-inconsistencynegatives, notforgedGPUevidencethatshouldbeacceptedbecauseflagselsewhereclaimPASS. Currentdraftcorrectlylimitsitselfto savedconsistency, butstatusVALID_WHOLE_PIPELINE_EVIDENCEoverstatesmissingcontrol/resource/subgatecoverage. Humanreviews explicitlycheckedallthosefieldsandartifacts, so nohistoricaltrialretraction.

Expectedfixedlabelschemaisavailablefromthese4knownlabs: PXhaspacked{px}, STMX{px,smx}, DP{px,smx,dp}, DS{dp,smx,ds}; othercontrolsv1/v4 are1node. Runtimeattrsincludeallcandidatecontrols exceptactualincumbent; currentDS/DP512threads, otherprototypes384. Uniform512roundedpool<=65536/maxthreads>=512/0local/optinoccupancy; oldV1dynamic64/216isnotfullyprovedbyuniformstaticsumandmuststayexplicitlimitationorhaveitsknownpoolgatechecked. No needgeneralprototypeabstraction.

Limitfutureclaimprecisely: currentmanifestfilehashesvsrecordedsourceSHA bindcandidatecode; notallcontrolsources/harnessat-captureunlessrecordedandverified. ActualartifactSHA bindsrecordedSASSbytes; checkingrecordedPASSdoesnotrerunphysicalchecker. Hostexit/actualGPUcleanuprequiresseparatehost/nodeevidence,notonlygraph_cleanupstring. Raworacleresulttextdoesnotreexecutenumerics. Offboardsource/runtimeevidenceisnotanofficialscoreorTASK-ratio/CIrecalculation. Keephelpfulsavedconsistencyautomation andindependentGPUreviewseparate.

## Re-review: original gaps fixed; resource follow-up fixed

Intermediate35aa8a33fixedall5originalnegatives, coverage/nestedstorestatus/controlmainbindings andhonestclaimlimits. Additionalindependentresourceprobes found: V1regs160initialpool61440<requested63488 accepted; V1regs240initialpool92160>65536 accepted; DPthreads384/maxThreads0 bypassed512guard; packerregs1000accepted. Rootfixedtheseunderthenarrowknownschema ratherthanweakeningactualtests.

FinaltoolSHA256`15eb944df69bf4501b450c02ea7f6e04e68bb32cf41965936d753a24180753df`. Reviewerreadchangedknownlabel/main/resource/dynamicpolicy logic andexecutedall4actualPX/STMX/DP/DS withcandidate-dir+artifact-dir: allPASS, noneeligibleforincumbentreview; DS correctlymixedagainstDP. Independentlyreran**9corruptions** (original5, wrongDPthreads, impossiblepackerregs, V1regs160/240): allraiseValueError withspecificgate. Rootalso reports60built-innegativerejections(16eachPX/STMX,14eachDP/DS). NoCUDAcall/build/sourceeditsbyreviewer.

DynamicV1nowusesrequested64×128+216×256=63488 andrequires requested<=warp-roundedinitialCTAreservation<=65536, notmerelyrequested<=SM. KnownmainthreadsDP/DS512,others384; savedwrong/defaultmissing512valuesreject. Packerpositive<=64/local0/maxthreads>=256 matcheslabguard. PackedJIT/bits/fresh/packer/graph/resourceidentitysets mustallmatchknownpackedlabels; allcandidatecontrolruntimeattrsrequired. KnownpackedmainprefixesboundandallrecordednestedstatusesmustPASS. ActualoracletextprefixguardrejectsexplicitFAIL whileacceptingknown4labs'validmessages.

**Scope verdict:** suitable as saved-evidenceconsistency gate for these4legacy21samplefour-armlabs. Doesnotrerunnumerical/physicalchecks/requeryCUDAattrs/attestallcontrolsourceorat-captureharness/verifyhostandnodecleanup. Doesnotimplementnew64comparisonDP/DSdiagnostic/NCU/futuremathgroup8schemasorofficialpaired-ratioestimator/CI. Root'slimitsnowexplicit; human/independentruntimeandkernelreviews remainrequired. NohistoricalGPUnumericalresultwasinvalidatedbythisautomationrepair.

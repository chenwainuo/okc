# QKV small full-X prefetch v1: preserved CPU failure

Actual9916/40d selected QKV M1/M8 tuple(1,1,1,2,16,8,1,1,2,1) is highesteligible GV route:4.14366796%+4.13727661%=8.28094457% scoreelasticity, coverage notforecast. O M1/M8 sums4.83008312%, withST2/KU2 allW/X alreadyloadedprologue; no Orefill exposure. GV53/poolerM16/unpoolerM16 excluded. Actual archived501dCUDA12.8SO/nativePCs retained; officialprofiles show same tuple butnotofficialH200binarydump.

QKV W alreadylooksahead: bothunits0/1 W loadedprologue; unit2WLDGs0xb30/40/60 precedeunit1decode0xbb0. Thus no newWprefetchimplemented. QKV Xunit1 loads0xb00/b20 followfirstHMMA0xaf0, leaving a specific nextXoverlapquestion. One sourcecloneadds two-vectorXring: initialunit0, copycurrent, refillnextbeforecurrentdecode,lastunitnooverwrite. OriginalWloads/decode/escape/HMMA/FP32/outputorder unchanged; globalXreadcount remains16vectors/warp. Fullsourceinverse recovers501d; sourceprototypehostrouteonlyQKVN2304/K8192,M1or8/exacttuple. No runtimevalidation orproductionadmission.

OnefocusedCPUcompile realizes Xunit1prologueissue0x3f0/400 beforedecodefirstPRMT0x4d0. But original64regs/stack0/spill0 becomes63regs/**32Bstack/48Bspillstores/48Bspillloads**, with6STL64+6LDL64 instructions. New1168slots vs1152; WLDG24/XLDG16/HMMA32/SHFL9 unchanged. ControlfullPC/predicate/128bitencoding matchesactualarchived501d. This is a register-liveness/implementation constraint, not a correctness failure ormeasuredperformance regression. No GPUtiming/countergain invented; currentversion PARKED,noGPUproposal. No compileflags/registerlimit/geometry sweep.

Evidence-based reopening: traceactualspilled liveX halves, then atmostone partial-X/snapshot-after-use revision ifitshortens those live ranges while retainingearlierissue atzero stack/spills. The family remainsactiveforCPUdiagnosis; preservethisfailedversionunchanged. RootownsGPU.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→learn→reprioritize. [Contract](../../ITERATION.md).

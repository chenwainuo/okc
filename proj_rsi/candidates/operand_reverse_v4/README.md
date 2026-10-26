# Operand reverse V4: fixed-shape X-address specialization only

state=CPU_PROOF_SM90A_COMPILE_PASS_GPU_UNRUN; GPU=root_named_handoff_after_independent_critic; incumbent_port=NONE.

## Version evidence / hypothesis

V3 valid completegraph result:55.071998us, vsV147.359999/V256.639999/incumbent24.032000 in the samejob. PairedV3gainus:-7.520001/+2.176002/-30.368000 respectively. Allfourbasis/oracles and20actualtimedgraphmutations/immutability/counters/cleanup passed. ThusV3 recovers part ofV2's loss but is not an incumbent orV1 winner. RawV3JSON SHA4701566b96a7af729ce02eb4a041f7291d53520d83d1b63e54ed466b2b3da618. No family closure, officialscore change or port.

V3 firstrefill45SASS instructions include8LDG +8IMAD.WIDE+8LEA+8LEA.HI.X, despite existing exactshape guardK4096. H_XA: repeated vectoraddress formation contributes exposed issue/dependency work. This is a source-cost hypothesis, not unique PC causality or a forecast that one patch recovers the~30us incumbentgap.

## Only source change

ParentfrozenV3 source3f90a6c77eeec6494c21019828ef01f93c0451f0d64e567f78a9ce0349abc2b9 / kernelc5cf2f4887bb85198f7c3dbc2387c75bdc6b9ed42b75b3848a3cae68ed455376. Keep exactM128/N8192/K4096/KS1 guard;128CTA/384threads;W6/B8; rawQ4 / separateasyncA2 / fourahead; same8B `ld.global.nc.L1::no_allocate.v2.u32`, same twoSHFL/twoSEL demux; same FP32 WGMMA, weightcodec/escapes, barriers, output.

```
row0 =64*wg +16*w +g                    # active0..119
xlane =2*(t&2)+8*(t&1)                  #0,8,4,12 BF16 offsets
xrow = uint64(X) +uint32(row0<<13) +uint32(xlane<<1)
addr(s) =xrow +uint32(s<<7)
q[2*j+h] = LDG64[addr(s) + IMM(32*j+65536*h)]
```

Eight immediate displacements in unchangedloadorder:0,65536,32,65568,64,65600,96,65632B. One cached64bitrow+lane base, one boundedstage offset; no runtimeK multiplication or eight pointeroperands. Explicit8templatecalls retain loopunroll loadorder and make PTX displacement an actualimmediate. No arithmetic/cache/staging/geometry knob changed. Source invariant masks exactly the loadhelper/prefetchaddress regions +kernelnames; all other kernel bytes matchV3, and replacing theV3body recreates its whole generatedsource SHA, including exactwrapperguards.

CPUrange:all524288BF16indices exactly equalV3 coordinates; highestvectorstart1048568/highestbyte1048575; offsets<2^20, eachuint32component<2^32, every8Bvectoraligned. Integerrowbases of inactiveproducerwarps are neverdereferenced and do not construct an outofrange C++pointer. Valid device allocation supplies baseaddress; no wrap in the bounded relativeoffset. ArbitraryXbits only loaded/shuffled/selected; no numeric transforms.

## Resource / actual lowering gate

`python3 proj_rsi/candidates/operand_reverse_v4/preflight.py` performs CPU fullindex/range/alignment, same65536bitlabel coverage, independent arbitraryWdecode/CSR/sharedB/outputbits, rawQ4/A2 lifetime +negativecontrol, parent sourceidentity; one localnvccsm90a cubin compile. No CUDA calls/GPU launches.

Actualfirst localcompile:150registers (V3152),0stack/spill/local/USETMAXREG/C75,57600registers/CTA,103600shared.16HGMMA/4WARPGROUP.ARRIVE/1outputTMA/0XTMA,64staticXLDG64 sameasV3;64mainpathBFLY +32outlinedcollectivealternatives retained. No dynamic registerrequests; no oversubscribed pool.

IntendedSASS removal observed and checked:all64XLDG sites grouped8loadsatime share one64bitbaseregister and the exact8 immediateoffsets (prologue folds its stageselector intoimmediates). MainCFG fromfirstfixedmaskdemux tofinalwait0:V3504staticinstructions/32IMAD.WIDE/33LEA.HI ->V4350/0/1, same32refillLDG. Static counts exclude outlinedalternatives and do not equal runtimeexecutions or latency. ArchivedSASS/resource/preflightJSON are the evidence.

Compiler consequence: somefuture rawQloads now interleave/hoist beforecurrentHGMMA, although round/prefetchcall ordering is sourceidentical. They write separate rawQ registers aftercurrent demux; asynccurrentA remains separate and pinned through retirewait1, so no currentAoverwrite is introduced. This is not a separately selected scheduler variant, and timing cannot isolate ALUremoval from its compiler scheduling consequence. Actualhostextension compilation/runtimeattrs/alloracles remain mandatory.

## Frozen proposed trial

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v4/lab.py --json /okc/logs/operand-reverse-v401.json
```

Rootexclusivecontainer; inner240s/outer300s. Fourway actualfrozenpw36 `prepare/kernel_fn`, bestfailedV1, immediatepredecessorV3, V4. No forced incumbentpick or extra baseline timingcopy. Ownmanifest+V1/V3 six executable/prooffiles/source hashes checked. Allthree candidateextensions build beforeparentpreparebudget clock. Scalarselectedpick only; runtimeattrs0local,<=168regs, occupancy>=1 aftersharedoptin; dynamic64/216 pool guard onlyV1.

Everyversion:all4096Konehot exact/randomFP64/tasknormal+outlier;20actualtimedgraphmutation checks across4graphs(original/zero/freshnormal/freshoutliers/probes),NaNpoisonoverwrite,inputimmutable;W/packed/escape exactunchanged,counterreset;4actualprofileidentities and graphreset. Timing:256MiBflush outsideevents,5discardedwarmups,21F1samples controlsbefore/after(minmedian),21rotated/reversed4waypairedrounds. Report allthreeV4deltas including failedpredecessor progress. Incumbentgain>=.5us BOTHF1+paired is promotionreviewonly; nonwin requires diagnosis and reviewednextmechanism, no familywideclosure.

No GPU byauthor. No code modified inV1/V2/V3/pw36/rs32. No current<3000us or officialspeedup claim.

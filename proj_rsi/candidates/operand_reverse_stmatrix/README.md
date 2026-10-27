# Packed-X producer STMatrix store version

state=CPU_PROOF_LOCAL_SM90A_PASS_GPU_UNRUN; root_named_handoff+critic_required. OnefixedunpoolerM128/N8192/K4096/KS1,128CTA×384threads,W6/B8,plainoutput. FrozenPXmanifest0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4/source2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a isparent; no frozenfilesedited.

## Counter-supported hypothesis / priorcoverage

FreshPXmainPCcapture:math2387samples,1027Bfullphase(1012longSB);producer1111samples,276proxy/bar3/convergence,154store+address,228LDS/decode. Largestnamedmathhandoffphase is43.02% ofsamples, **not43% removabletime**. TRYWAITpredicate/result/hardwarebarrierlatency andproducerarrival arecoupled; registerfence471/469longSB isnotuniquelyX. Fullinterpretation/reproducibleanalysis in `audits/packed-x-pc-roi.md` and `results/proj-rsi/takeover/20261027/packed-x-ncu/analysis.json`.

TestONEproducerstorechange:16scalaru32STS/thread/stage ->4 `stmatrix.sync.aligned.m8n8.x4.shared.b16`, no.transpose. Same4096BF16/8192BdecodedWstage/SW128/rings, sameproxyfence+bar3/publication/refill, samefresh1MiBXpack+main/mathnativeA4/threeahead/cache/output/FP32arithmetic/codec/escapes. Source invariant reconstructswholeparentgeneratedCU exactly afterrestoringstoreblock+kernelname; no mixedproducerparallelism/A6/fusion variant.

Thisprimitive/layoutwasusedbyoldpw19: thisis a narrowlymeasuredcurrentPXport, not a novelfamily. Prediction:lessstore/addressissue mayadvanceBpublication. Samebytescanhaveunchanged/differentsharedportcost; fourcollectivestorescanalsoshiftwarpskew. No speedupguarantee ormicrosecondprojectionbeforeavalidwholepipeline result.

## Primary semantics / cross-lane proof

[NVIDIA PTX8.7stmatrix](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix) supports m8n8.x4.b16 onsm90. All32lanesexecuteidentical.sync/.aligned instruction; matrix0..3rowaddressescome fromlane groups0..7/8..15/16..23/24..31, naturally16Baligned. Fourconsecutivedatalanes supplya16Brow. Operationisweakmemory; retainwriterproxyfence and128threadbar3.

Forproducerwarp w, k0..3, address-providerlaneL:

```
provider_row=16*w+8*((L>>3)&1)+(L&7)
provider_chunk=2*k+(L>>4)
provider_dst=bring+(s%8)*8192+provider_row*128
             +((provider_chunk^(provider_row&7))*16)
STMatrix[provider_dst] <- {a[4*k],a[4*k+1],a[4*k+2],a[4*k+3]}
```

Independenthardware-routingemulation separatesDATA laneD fromADDRESS laneL. Registermatrixv ofdatalaneD usesproviderL=8*v+(D>>2), thenoffset4*(D&3). Substitutiongivesexactoldscalar:

```
row=16*w+8*(v&1)+(D>>2)
chunk=2*k+(v>>1)
dst=bring+(s%8)*8192+row*128+((chunk^(row&7))*16)+4*(D&3)
```

Preimplementation symbolicproof passedall2048u32/4096BF16/8192B perstage. Preflightthenreconstructsr6packedweights+CSRrepair(includingdenselistwindows), testsall65536storagepatterns+arbitraryothers, simulatesbotholdscalarandnewcross-lanecollectivestores, andcompareseverybyteacross4warps×64stages×2halves:524288BF16 exact. Allprovideraddresses16Baligned; every4Bdestinationunique; all eight B-ring slots bounded /1024alignedSW128bases; nocrosswarp/crossslotwrite. Samepack/inverse/allXbits/range,outputslab,A4retirementproofandoldaliasnegativecontrol retained. CPUproofs/sourceidentityexecutebeforeanylocalcompile.

## Actual resource / SASS gates

Localnvccsm90a:main122regs,packer22,0stack/spill/local/noUSETMAXREG/C75. Mainnominalregisterpool122×384=46848; hardwarewarp-roundingcanreserve49152, soactualoccupancyqueriesremainrequired.103600dynamicshared, oneCTAresourcebudget; nodynamicreallocation.

ActualSASS:4unpredicated `STSM.16.M88.4`,16scalarSTSremainingonlyintheunchangedoutput,16HGMMA/4WGarrive/1outputTMA/0XTMA,32LDG128/noBFLY. ProxyMEMBAR/FENCE afterlastproducerstore thenbar3 retained. MainphysicalA4checker passes2iterations/eightgroups/fourdistinctbanks/outlinedmathwaitpaths; samecheckerrejectspreservedbadPXidentitycopy at0xe30. No parserweakening; acceptedkernelnamesonlyextendedfornew rq_stmx.

Researchlab dumpsBOTH exactloadedPX/controlandSTMXJIT.so beforefirstmath/beforeparentprepare, savesrawSASSbeforechecker,checksphysicalA4+intendedstoreform/proxybar3, andrecordsSASShashes. GatefailurewritesINVALID_ACTUAL_JIT_RETIREMENT_GATE JSON/variant/error/no_mathlaunch andstops. No cuobjdump toolingproposedforproduction.

## Frozen proposed bounded trial

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_stmatrix/lab.py --json /okc/logs/operand-reverse-stmatrix01.json
```

Rootexclusiveinner240s/outer300s. Fourwayactualpw36prepare/kernel_fn, bestfailedPX, V1, STMX. Same-sourceparentpins,allcandidateextensionscompilebeforeparentprepareclock, scalarselectedpickonly. Runtime0local/<=168regs/sharedoptin/occupancy>=1; V1's64/216poolguard retained. Packerattrscheckedforbothpackedmodules.

PXandSTMXeachhaveexplicitdistinct1MiBworkspaces andtheiractualtimedgraphscontainthefreshpack+main2nodesoneverycall; V1onekernel. Noprepareactivationvalues/cachedinput/prepackoutsidegraph/main-onlytiming/subtractedpackerlatency. Full4096Konehots,FP64random/tasknormal+outliersforallfour;65536BF16storagebitpacktestandfreshworkspace/outputforbothpackedvariants.20actualtimedgraphmutationchecks(original/zero/freshnormal/freshoutliers/probes),10packedpoison/fullint32bytecomparisons,inputimmutable/fulloutputoverwrite. W/packedW/escbytesimmutable,countersreset,allfourgraphsreset.

Timing256MiBflush outsideevents,5discardedwarmups,21F1samplescontrols before/after(minmedian),21rotated/reversed4waypairedrounds. Reportall3STMXdeltas,wins/profiles/resources/gates; >=.5us bothF1+pairedvsactualincumbent ispromotionreviewonly. ProgressvsfailedPX/V1retained; nonwinrequiresdiagnosisandreviewednextstep, no familywideclosure. Noofficialscore/<3000usclaim.

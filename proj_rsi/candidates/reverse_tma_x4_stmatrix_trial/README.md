# Five-arm native-X X4 plus STMatrix research harness

CPU validated; GPU unrun. Driver `tools/native_x4_stmatrix_lab.py`. This trial is
new and does not alter any frozen standalone trial or kernel. Independent07 GO
and root admission are required before runtime execution.

Policies: `incumbent` is actual LOCAL pw36.kernel_fn after complete fresh prepare;
`native` is025f native-X2 scalar-store; `x4` isd47e native-X4 scalar-store; `stmx`
is2c132 native-X2 STMatrix; `new` is11892 native-X4 STMatrix combo. No official
rs32/H200 score inference. Selected incumbent rq at n8192 is mandatory; the
original rqcall return must be0. This aborts fallback without forcing selection.

All four candidates consume CURRENT original row-majorX and encode current
x.data_ptr each hostcall, with ignored empty int64 XP argument. No XP allocation,
packnode or allstoragebit XP gate exists. All profiles must identify exactly one
correct mainnode and zero rx_pack nodes; tokens respectively rq_tma_x,
rq_tma_x4,rq_tma_x_stmx,rq_tma_x4_stmx. Native/stmx shared137232B; x4/combo170016B.
Runtime122registers/0local/512threads/roundedpool65536/optinbudget/oneCTA occupancy
are mandatory. Larger shared footprint is retained as an experimental difference.

Before first candidate math, verify complete four pinned manifests and regenerated
full extension sources. Save each actual loaded .so path/hash and SASSartifact/hash
under `.native.jit.sass`,`.x4.jit.sass`,`.stmx.jit.sass`,`.new.jit.sass`. Full main
PC/predicate/opcode/operand/control stream equality to CPU cubin is required.
Each label binds its own local liveness and proofs modules while loading the
exact candidate callback. Both STMatrix labels run their own actual_binary_gate;
combo additionally reexecutes full-CUDA composition/inverse proof. AllstageXmap,
Xslot phase protocol,A4 retirement,producerowner/proxy/output gates run. No
cross-label generic checker substitute. Wrong-label all12pairs and four changed
canonical instructions are rejected by CPU proof.

All4096Kbasis exact,FP64,normal/outliertask,newinputpointer,freshoutput/currentX,
25 retainedactualgraph NaN/inputmutation oracles,alloriginalweights plus shared
candidatepacked/escape and selectedincumbentpacked/escape immutable,zero counters
andreset/drain gates are required. Hardware raw TMA/LDSM allBF16storagebit
roundtrip remains UNPROBED for every native variant. No CPU map or ordinary
numeric gate is called a hardware raw-bit proof.

F1 uses21samples in nine blocks: fourcontrols before,new,fourcontrols reversed.
Paired40rounds cover ten cyclic/reverse orders eachfourtimes, same fixture per
round. This is the nearest exactly balanced count at least32 for fivearms.
Total414rawcalls=25warm+189F1+200paired; allsamples/fixtureseeds retained with no
filtering. At F1boundary andpairedend, immediately validate ten actual retained
lastoutputs against eachpolicy's ownlastfixture before ANY replay/mutation/
profile, pluscurrentX/immutability/zero counters. No cached inputvalues passed to
kernels. Nine synthetic negative controls reject output,unauthorizedXP,currentX,
W,packed,escape,candidatecounter,incumbentcounter,wrongpolicyseed corruptions.
Allfourcontrol F1/paired gains,medianratios,win counts are reported. A combo win
over a single edit is progress only; actualincumbent comparison remains required.

**Manifest bases:** this trial manifest `files` keys are relative to `proj_rsi/`.
Transfer `/root/okc/proj_rsi/` plus eachkey, preserving relative paths. Kernel
manifests have their own candidate-directory bases. All four kernel directories
must be transferred with hashclosure, including combo composition proof and its
frozen parent dependency manifests. Do not flatten paths.

CPU-only proof from repository root:

```bash
python3 proj_rsi/candidates/reverse_tma_x4_stmatrix_trial/cpu_proof.py
```

Root-only command from `/root/okc/proj_rsi` after07 GO and node admission:

```bash
bash tools/h100_trial.sh okc-proj-native-x4-stmx-01 300 240 bash -lc 'cd /okc/proj_rsi && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 tools/native_x4_stmatrix_lab.py --json /okc/logs/native-x4-stmx-01.json'
```

Normalcap inner240/outer300 retains bounded full prepare/JIT/five validations.
Extra timed work is a small fraction of fullprepare/basis cost. Timeout/setup/
correctness failure retains allraw failure and requires distinct repaired attempt;
no silent cap/source/flag edits. Root alone executes GPU and cleanup. No official
submission, promotion or score claim is authorized by this research harness.

# Native-X STMatrix four-arm research driver

CPU validated, GPU UNRUN; independent agent07 review required before root runs.
Driver: `tools/native_x_stmatrix_lab.py`. This is a new copy of frozen async lab
48feda58/manifestc2618e31. Old drivers/trials and immutable kernel2c132b82 remain
unchanged. All trial support/proofs are in this directory.

Policies are exactly `incumbent`, `ds`, `native`, `new`:

- `incumbent`: actual LOCAL development pw36.kernel_fn after complete prepare.
  The selected serving rq must exist at n8192 and the actual rqcall must return0;
  reject fallback without forcing selection. No official rs32 inference.
- `ds`: frozen6785/672d dual-STMatrix globalXP control; fresh pack+main every
  normal call, two captured nodes,1MiB XP workspace and complete65536 storage-bit
  XP byte test. Only this arm has XP.
- `native`: frozen025f/ffd3 native-X parent, current original input, one main,
  no XP. Its prior paired31.856us vs localincumbent23.968us is inherited evidence.
- `new`: frozen2c132/f6a765 native-X STMatrix composition, current original input,
  one main, no XP. No timing/numerical admission yet.

Native/new use137232 shared bytes,122 registers,512threads,rounded65536 register
pool and exactly1 active block/SM after opt-in. DS uses103600 shared bytes and
its own<=128-register/local0/occupancy>=1 gate. Kernel-specific actual loaded .so
SASS must match frozen cubin at every PC/predicate/instruction/branch target
before candidate math. Correct source/liveness/proofs are loaded per label;
new explicitly calls its frozen `actual_binary_gate.py`, including fourSTSM,
32LDSM/eightbar5/eightmathTMA/zero mathLDG/A4 retirement/proxy/complementary owners.
Native uses its own native gate, DS its own packed-X store/lifetime gate.
Canonical CPU negatives reject all six wrong-label pairings and three changed
HGMMA operand streams. Runtime attributes/actual occupancy remain mandatory.

All4096K basis exact,FP64,task normal/outlier,fresh pointers/outputs/workspace,
20 retained actual graph mutations,immutable all original weights/candidate
packedescape/actual selected incumbent packedescape,zero counters,graph drain
and reset are retained. DS alone runs complete65536 BF16 storage-bit XP oracle.
Both native hardware raw TMA/LDSM bit roundtrips remain UNPROBED, explicitly.

The exact extracted measurement function is CPU exercised with fake events:
147F1 calls (controls21before/after, new21),128paired calls (32rounds,8balanced
cyclic/reverse orders),20warmup calls; every arm occupies every paired position
8times. All raw samples and own input seeds retained, no filtering. Four actual
retained outputs plus DSXP are read immediately at F1boundary andpairedend
(eight terminal output oracles) against eachpolicy's ownlastfixture before any
later replay/mutation/profile. Nine terminal synthetic negatives reject output,
XP,currentX,W,packed,escape,candidatecounter,incumbentcounter,wrongpolicyfixture.
Only after terminal checks do20graph mutations and profiles execute; profiles
require native/new1main node versus DS2nodes freshpack+main.

Manifest file keys are relative to `proj_rsi/`, NOT this directory or repo root.
Frozen kernel manifests retain their own candidate directory bases.

CPU proof (no CUDA context/extension/prepare):

```bash
python3 proj_rsi/candidates/reverse_tma_x_stmatrix_trial/cpu_proof.py
```

Root-only launch after independent07 GO, from `/root/okc/proj_rsi` on iteration
host, normal inner240s/outer300s cap:

```bash
bash tools/h100_trial.sh okc-proj-native-x-stmatrix-01 300 240 bash -lc 'cd /okc/proj_rsi && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 tools/native_x_stmatrix_lab.py --json /okc/logs/native-x-stmatrix-01.json'
```

Prepare/JIT/validation/timing all fit inside this cap or retain an invalid setup
failure; no silent cap/source/flag changes. No performance admission without
actual incumbent win and required independent/official validation. Root alone
executes GPU and cleanup; author ran CPU-local proofs only.

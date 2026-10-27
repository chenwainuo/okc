# Native-X async-refill four-arm research driver

CPU validated, GPU unrun, independent critic GO required. New driver
`tools/native_x_async_lab.py`; all trial support/proofs are in this directory.
Old native-X e96/a73 trial and both native kernels are untouched.

Policies are exactly `incumbent`, `ds`, `native`, `new`:

- `incumbent`: actual LOCAL development pw36.kernel_fn after complete prepare.
  Selected serving rq must exist at n8192 and actual rqcall must return0; abort
  fallback without forcing a different selection. No official rs32 inference.
- `ds`: frozen dual-STMatrix globalXP control, fresh pack+main every call,
  two captured nodes and1MiB XP workspace.
- `native`: frozen025f/ffd native-X parent, original current input, one main,
  no XP. Parent attempt01 paired31.856us versus incumbent23.968us is prior evidence.
- `new`: async producer-refill kernel92fbe73d/source45fbc8b6, original current
  input, one main, no XP. No timing/numerical admission yet.

All native variants accept ignored empty int64 XP argument. Host descriptor
encodes current x.data_ptr every call; no input value host copy/synchronization.
Native dynamic shared137232B; new137248B; DS103600B. Runtime checks require
512threads,122registers for native/new,zero localspill,roundedpool<=65536,
opt-in shared sufficient and occupancy>=1. Kernel-specific canonical SASS gates
check full PC/predicate/opcode/operand/control equality against frozen cubin,
then correct version's A4 retirement/proxy/owner/map/protocol proofs. Actual
loaded .so path/hash and SASS artifacts/hash are saved before candidate math.
Parent physical32LDSM/eightbar5/eightmathTMA/zero mathLDG differs from async
math0TMA/0LDG/0bar5, eightXemptyhandoffs and oneproducerX TMA site. Async's
source syncwarp is compiler-elided: no emitted WARPSYNC claim. Acceptance of
that compiler/protocol proof remains independent critic responsibility.

All4096K basis exact,FP64,task normal/outlier,fresh original pointers, fresh
outputs/workspace,20retainedactualgraph NaN/input mutation oracles, immutable
alloriginalweights/candidatepackedescape/actualselectedincumbentpackedescape,
zero counters and reset/drain gates are retained. Only DS receives complete
65536BF16 storage-bit XP and byte oracles. Both native hardware raw bit mover
roundtrips remain UNPROBED; CPU bit/address proofs and normal numeric gates are
never described as GPU raw-bit tests.

F1 has21samples eachcontrolbefore/after andnewblock.32pairedrounds use eight
balanced cyclic/reverse orders; allrawsamples retained with fixtureseed for
all warm/timed calls. At F1boundary andpairedend, eightactual retained outputs
and DSXP are immediately checked against eachpolicy's ownlastfixture before
any replay/mutation/profile; currentX matches finalactualcall. Nine CPU negative
controls reject output/XP/currentX/W/packed/escape/candidatecounter/incumbentcounter/
wrongpolicyfixture corruptions. No sample filtering; allcontrolgains/ratios/wins
reported, no promotion absent actualincumbent win and official validation.

**Manifest path base:** every `manifest.json` files key is relative to the
`proj_rsi/` directory, NOT this trial directory and NOT repository root.
Copy files using `/root/okc/proj_rsi/` plus the manifest key; preserve relative
paths. Frozen kernel manifests use their own candidate directory bases.

CPU proof (from repository root, no CUDA context or extension):

```bash
python3 proj_rsi/candidates/reverse_tma_x_async_trial/cpu_proof.py
```

Root-only launch after critic GO, from `/root/okc/proj_rsi` on iteration host:

```bash
bash tools/h100_trial.sh okc-proj-native-x-async-01 300 240 bash -lc 'cd /okc/proj_rsi && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 tools/native_x_async_lab.py --json /okc/logs/native-x-async-01.json'
```

Inner240s/outer300s is normalcap and bounds complete prepare/JIT/runtime. Prior
native attempt completed in113s. Cold-cache timeout remains invalid setup with
raw failure retained and a new reviewed attempt; no silent cap/flag/source edits.
Root alone executes GPU/cleanup; this author runs CPU-local proofs only.

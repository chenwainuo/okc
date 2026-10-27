# Native-X four-arm research lab

CPU validated; GPU unrun. Frozen native-X source remains untouched. Driver:
`tools/native_x_lab.py`; support and proofs live here, outside kernel directory.

Policies: actual `pw36.kernel_fn` after full `prepare`, exact dual-producer DP,
dual-producer STMatrix DS, native-X TMA/LDSM. DP/DS capture fresh pack plus main
(two nodes); native-X passes an ignored empty int64 tensor and captures only main
(one node). No XP storage exists for native-X.

Before first candidate math, verify complete pinned manifests and regenerated
extension sources; dump each actual loaded .so SASS to result siblings. Extract
exact main instructions including predicates, PCs and branch targets, compare to
CPU cubin, then run A4 retirement/store/owner/proxy gates and native-X layout,
protocol, 32 LDSM/eight bar5/eight math TMA/zero math LDG proof. Wrapper/hash
mismatch is allowed only when complete main instruction semantics agree. Runtime
checks enforce zero spills, register allocation, shared memory and occupancy.

Each arm runs all4096 K basis probes plus FP64 and normal/outlier oracle checks.
DP/DS additionally run complete65536 BF16 storage-bit XP and full-byte oracles.
Native-X has no XP; its raw hardware TMA/LDSM storage-bit roundtrip remains
**unprobed**. NVIDIA bit-preserving mover semantics, complete CPU index proof,
and basis/FP64 numeric gates do not constitute a hardware all-bit-pattern test.
Root/critic must accept that explicit limit before admission, or commission a
separate verbatim mover probe version without editing the frozen main.

Retain four actual graphs and twenty NaN/output and XP-poison mutation checks,
with fresh normal/outlier, zero and all-row probes. Check weights (all original
weight tensors), selected incumbent packed/escape, shared candidate packed/escape,
input and zero counters. Fresh output/workspace checks precede timing. Every warm
and timed call records fixture seed. F1 uses21 samples per block, controls before
and after native-X. Paired timing uses32 rounds and eight balanced cyclic/reverse
orders. No filtering. Report raw samples, medians, gains, ratios and win counts.
At F1 boundary and paired end, check each actual retained last timed output and
XP (where present) against its own recorded fixture before any replay/mutation/
profile. Final current X must equal final actual call fixture. CPU adversarial
proof rejects nine corruptions, including distinct per-policy seed mismatch.
Graphs reset/drain in finally; shell wrapper enforces container cleanup deadline.

CPU-only proof command (no extension/GPU context):
`python3 proj_rsi/candidates/reverse_tma_x_trial/cpu_proof.py`

Root-only command, after independent critic GO and raw-bit-limit acceptance, from
`/root/okc/proj_rsi` on iteration host:

```bash
bash tools/h100_trial.sh okc-proj-native-x-01 300 240 bash -lc 'cd /okc/proj_rsi && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 tools/native_x_lab.py --json /okc/proj_rsi/candidates/reverse_tma_x_trial/attempt-01.json'
```

The checked-in `h100_container.sh` mounts `/root/okc` at `/okc`. Inner240s and outer300s match the strict four-arm predecessor lab;
full prepare/JIT cold cache can time out. Timeout retains raw failure, gives no
performance admission, and requires a distinct repaired attempt id. No official
submission or score claim.

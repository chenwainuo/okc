# Native BF16 allocation coverage, one unpooler128 existing Lt

`state=CPU_LAB_HELPER_BOUND; GPU=UNRUN; admission=FINAL_CRITIC_AND_ROOT_HANDOFF_PENDING; implementation=EXISTING_LT_ONLY`.

Scope derives from [native plan](../../audits/hw-compression-native-plan.md). Packed pilot `hw-compression01` is INVALID_CLEANUP_FAILURE; its negative timings are diagnostic only. It established H100 grantedGENERIC, full packed-byte roundtrip, all8192K pooler one-hots,15timed graph checks and the same RQ kernel, then failed close. Unpooler was not run. This source does not retry that packed policy or copy its allocator.

One technically different representation is credible for a bounded trial after the ownership issue is fixed: native BF16 has residual common exponent bits removed by packed12, and unpooler128's existing tuned Lt kernel was the actual pw36 official winner (19.103us benchmark; layout2/cfg66.18.1.0.0.2.35.0.5). Statistical entropy is a prior, not an effective hardware ratio. Saving0.5us of exposed bandwidth at3.8/4.8TB/s needs1.9/2.4MB,2.83/3.58% of native67.1MB. This is plausible enough to test once without adding per-weight instructions; there is no predicted win, m256/FMT0/codebook alternative or geometry/algorithm sweep.

Source imports frozen pw36 SHA`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Normal prepare selects one actual existing Lt code, whose layout and all9config fields are frozen before allocation. `lay&2` selects `st['unpooler_t']` (contiguous[K,N]), else `*_raw` ([N,K]). Byte reinterpretation `BF16.view(uint8)->WeightClone->view(BF16)` preserves bytes/shape/strides/pointer with no cast/transpose/codec. Layout2 cloning only`*_raw` would be a no-op and is rejected by routing proof.

Three native cases explicitly call the existing `_v12` fallback with this identical Lt code and state, replacing only the selected weight pointer for NONE/GENERIC. Fourth comparator calls **unchanged actual `kernel_fn`**, retaining an RQ pick when present. No dispatch table entries are deleted. All native kernel names/count/grid/block/register/shared metadata must match under a separate post-timing CUPTI identity capture; actual incumbent's real sequence is reported separately. The loaded parent-extension hash and cuBLAS wheel/version/link identity are recorded. The loose parent Lt search validation is not sufficient: all4cases need full FP64/all4096K exact/normal/outlier/input-immutable tests.

Each graph retains its real returned output. Same graph exec serves F1/paired timing and five post-timing input mutations (original/zero/fresh-normal/fresh-outliers/all-probes) with NaN poisoning. An extra untimed capture checks a fresh output allocation. Full selected/native original weight bytes and clones remain exact; incumbent RQ counters stay zero.256MiB read flush and graph events remain outside captured work;21sample F1 before/after conservative ordinary+incumbent,21rotated4-way paired rounds. Survive only>=0.5us over ordinary Lt **and actual incumbent**, both F1+paired, plus>=0.3us over NONE in both. Wrong route, oracle/identity/cleanup failure invalidates timing regardless of gain.

All graph execs are drained/reset before all case states/views/output aliases are dropped, Python collection runs, and checked allocator close executes. Fresh-output graphs also reset in a `finally` path. Fixed-helper counts() snapshots record live owners, the state after alias drop and zeroRegion/undefinedTensor after checked/idempotent close. Helper `hw_compression_owner` is now bound to manifest`8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781`, allocator`7f2dea1ecdcbff20efd1daad1a4250f775bbf9cfb8b9a76a16f257c1268140f8`. Actual Torch2.8 CPU proofs and root's non-timing `hw-owner-lifetime01` passed; independent terminal review confirmed22traces/8oracles, readable rejected aliases, both Regions0/undefinedTensor, exit0/no warnings at23:01:55Z. The v1 Python `locals()` snapshot retained a real last-NONE dictionary alias; this native cleanup uses no locals/frame inspection. Runtime rejects any other helper path/manifest/source before import. CPU Python-reference tests here remain separate from that actual GPU lifetime proof. Original v1 and corrected helper remain frozen.

Population stays the fixed helper's exact cuMemcpyDtoDAsync then stream synchronization. [Primary-source population check](../../audits/hw-compression-copy-population.md) found neither a documented mandatory Hopper SM rewrite nor a guarantee of effective compression after that D2D API. The result names the population path and leaves effective hardware compression UNMEASURED. No speculative SM-write variant is introduced.

CPU preflight (no helper build/import, Driver initialization or CUDA API):

```bash
python3 proj_rsi/candidates/hw_native/preflight.py
```

This validates all4real-shape routes using meta tensors, missing-Lt STOP, all65536 BF16 payload patterns in raw/non-square prepared-transposed storage, dtype-view alias metadata, case Python-reference drop, profile launch-identity mismatch guard, rejection of invalid v1 before import and all11bound helper file hashes. It freezes this lab's manifest. The separate lifetime gate establishes the helper's tested ownership behavior; this new native BF16-view path's full runtime/graph/precision/identity gates remain unrun.

The exact bound command is ready for final critic hash review and root's named exclusive handoff:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/hw_native/lab.py \
  --helper-dir proj_rsi/candidates/hw_compression_owner \
  --helper-manifest-sha 8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781 \
  --json /okc/logs/hw-native01.json
```

Helper builds before parent import so parent prepare's clock is preserved. Root sets exclusive container/inner/outer deadlines. No GPU launches/remote compile by this author. A failed version supplies evidence for the next single-hypothesis iteration; no automatic rerun is included in this command. No full submission or official slot. Only helper binding/preflight/documentation changed from reviewed lab`d72d01520dbef7e9434ce76a5307d066636bf659557f11df9cb6dc9f82633746`; every algorithm, measurement, oracle, profile and cleanup function is preserved.

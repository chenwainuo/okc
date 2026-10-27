# Sol03 independent kernel and measurement review

Review 2026-10-27 02:24 UTC. Root remains sole GPU executor. This review ran CPU-only Python and read-only SSH/scp; no extension loads, GPU contexts, launches, prepare, commits, frozen-source changes, or global ledger edits.

Decision: **HOLD** composition runtime and strict performance admission from G8 attempt01. One narrow measurement-validity gap remains: neither frozen lab validates the outputs left by the actual timing runs before overwriting/replaying them. CPU kernel/source gates and the reported runtime checks otherwise pass. This is an attempt/driver issue, not a demonstrated kernel error or family rejection.

## Composition source and CPU gates

Frozen manifest `80dae9122d1f964af99cdf586fbf0d4b5e9ac28db4c7998f253069ae6889f24b`; generated source `8ef15e71424cf35fbb8840bd1000b1f0ca0b355eb0322ff66bacc9d9f83bab77`. Independently verified all **15** manifest file hashes. Invoked sourceproof, schedule proof, actual frozen SASS liveness and mutation controls, store form gate, and the complete DS map on CPU without preflight.main (which would mutate frozen artifacts).

Source restoration removes exactly the DS store fragment and name, restoring the entire G8 generated extension hash `e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`. Inserted store fragment equals frozen DS; flags and layout proof match their retained controls. A2x32, W6/B8, CSR paths, 512 threads, 103600B shared, fresh pack and epilogue remain unchanged.

Actual frozen cubin SASS passes discovered physical A and D register/fence/retirement checks across two loop iterations and all outlined paths. All five negative controls reject: retained premature overwrite, missing fence, predicated fence, new A-bank premature overwrite, pending D generic write. Store gate finds four unpredicated STSM, proxy fence after final producer store, and complementary same-predicate owner barriers3/4 after proxy without predicate clobber. Exact 8192B stage-byte bijection covers all four producer warps, all65536 BF16 storage patterns, escaped weights and no slot alias. Source/model proof checks unchanged K/MMA order and acyclic W6/B8 dependencies. Retained resource record is main122/pack22 registers, zero stack/spill/local, no C75, 16 HGMMA/2 ARRIVE/4 STSM and rounded register pool65536. These are CPU/source gates only.

Lab compares actual pw36-selected kernel_fn incumbent, DS, G8 and composition. Pins all control source/manifests, disassembles all three actual JITs before candidate math and uses the correct per-version physical plus store gates. Runtime resource, full4096 onehot, FP64/task normal+outlier, storage-bit, fresh workspace,20 actual captured graph mutations/15XP poison, fullpack+main2nodes, input/weight/layout/escape/counter and graph-reset gates are present. F1 uses21 each with all controls before/after and the conservative lower control median. Paired32 uses8 rotation/reverse orders exactly4 times each; all raws retained. This is a sound predeclared balanced design.

**Concrete repair required:** in composition `lab.py:295-296`, `measure(graphs)` returns and then `actual_graph_checks` copies x, fills output with NaNs and XP with poison, and replays each graph before checking it. Add a distinct read-only check immediately after measure, before mutation or any new replay: assert x equals the retained timing input; oracle-check all four retained outputs and finite completeness; compare all three retained XP buffers bitwise with the timing-input pack map. Record the checks. Preserve this frozen version and freeze a revised driver/manifest (new attempt identity with unchanged kernel hash is sufficient). Consider checking terminal F1 outputs at the F1/paired boundary too if the intended contract requires both timing methods separately. The final timed outputs cannot be recovered after this attempt terminates.

## G8 attempt01 harvest

Remote exit0 at `2026-10-27T02:21:52Z`; copied JSON, host log and all3 actual JIT SASS into `sol03-kernel-validation-data/`. Generated source identities match frozen DP/DS/G8 expected hashes. Independently matched actual SASS hashes and reran each correct version checker plus scalar/matrix store-form gate; all pass. Did not use the older projection validator with the wrong variant scope.

All reported runtime attrs are122regs/0local/103600B, occupancy1 after opt-in,512threads/pool65536; pack22regs/0local. All three full4096/FP64/task normal/outlier validations pass, all65536 storage bit exact checks pass, fresh workspace/output pass. Twenty captured-graph mutations include15 exact poison-XP overwrites and all128 full probes. Every packed profile has exactly pack+intended-main. Inputs/weights/layout/escapes and counters report pass; all4 graphs reset.

Independently recomputed all147 F1 samples and all21 paired rows without estimator selection. Values exactly match JSON:

| comparison | F1 gain vs G8, us | paired gain vs G8, us | G8 wins/21 |
|---|---:|---:|---:|
| actual incumbent | -10.751998052 | -11.104000732 | 0 |
| DP | -1.439999789 | -0.895999372 | 1 |
| DS | -1.631997526 | -1.407999545 | 2 |

G8 F1 median35.039998591us.21 rotated rows are deliberately slightly unbalanced: first5 of8 orders occur3 times and last3 occur2 times. No alternate subset or estimator was substituted. Profile durations do not establish end-to-end speed or unique stall causality.

The same final-timed-output gap occurs in G8 `lab.py:295-296`. Therefore strict result classification is **retained measurement-validity HOLD / no performance admission** despite the source/runtime gates and diagnostic latency tendency. Frozen source and raws must remain attached to attempt01. A new bounded attempt of unchanged source with repaired, frozen driver is the narrow fix if performance admission is needed. Composition runtime remains conditional on a valid G8 attempt; do not treat exit0 alone as satisfying that prerequisite.

Independent node cleanup at `2026-10-27 02:24:01 UTC`: `ps -p388599` empty, `docker ps -a` exact-name filter empty, compute-app query empty, GPU memory0MiB. Evidence saved in `cleanup-check.txt`. This confirms node cleanup independently of JSON.

Next causal question if a repaired valid G8 run retains this regression: fewer legal math boundaries changed A lookahead192K->128K and B retirement granularity; reduced fences alone did not imply lower end-to-end time. Preserve this version, use a new controlled mechanism/diagnostic, and avoid a family ban. No official H200 score or incumbent change follows from these local numbers.

Machine-readable independent findings and recomputation: `sol03-kernel-validation-data/cpu-review.json`; CPU review script: `sol03-kernel-validation-data/check.py`.

## Strict lab v2 rereview — GO

Manifest `900cf61b1a0b32f419777a3500ed1f9ea58b62532ba5fb09bf2dc57e5453b998`, lab `e90f0014e14c5e6d493d7b27e54eba7f743a5d385ede131ed59b20c8a9ebecd2`: **GO for one root-only bounded four-arm runtime**, with no performance admission until all runtime gates and independent harvest pass. The prior HOLD remains attached to the archived v1 driver and original G8 attempt01; it does not apply to this repaired driver plan.

Independently verified all17 new manifest file hashes and all15 old manifest files under `measurement-history/lab-v1-80dae912`. Source, device source, body, cubin, SASS, prototype and physical checker remain byte-identical. CUDA source remains `8ef15e71424cf35fbb8840bd1000b1f0ca0b355eb0322ff66bacc9d9f83bab77`.

The repair adds read-only immediate gates at both timing-method boundaries. The F1 gate runs before the first paired fixture copy/replay; the final paired gate runs before returning into mutations/profile. Each checks all4 retained outputs against its recorded last input seed (distinct per-policy seeds at F1), all3 retained XP buffers against that same policy's fixture, currentX against the final actual call, immutable W/packed/escape, and all candidate/incumbent counters. No graph replay, fill, copy, reset, capture or output repair occurs inside the gate. All warmup/timed calls and seeds are retained. Pair32 uses the same fresh fixture within each4arm pair and8 balanced orders repeated4 times.

Independently executed the extracted CPU gate proof: valid distinct fixtures pass and all9 corruptions reject (output, XP, currentX, weight, packed, escape, candidate counter, incumbent counter, wrong policy seed). Proof result exactly matches frozen JSON; saved independent copy `sol03-kernel-validation-data/v2-gate-review.json`. Manual source inspection confirms real calling closure binds the actual graph outputs, input, packed workspaces, immutable tensors and counters. No CUDA work was performed by this review.

A single four-arm job can now establish strict measured G8 control validity and composition validity together; a separate unchanged-G8 trial is unnecessary. All3 actual JIT gates run before any candidate math; broad frozen G8 numerics already passed attempt01. The storage-bit loop does invoke composition math before the subsequent full G8 numerical validator, but all full candidate validation is ordered DS/G8/composition before timing. This is safe under the retained G8 broad correctness evidence and actual-JIT prelaunch gates, without asserting any old timing is valid. A G8 validation failure invalidates the job's comparisons. Root remains sole GPU executor; cleanup still requires independent node checks after this new attempt.

## Strict GS runtime group8-stmatrix-01 — VALID REGRESSION

Reviewed local root archive `proj_rsi/diagnostics/group8-stmatrix-01/` (JSON, host log, DS/G8/new actual JIT SASS). Host log terminates exit0 at `2026-10-27T02:31:02Z`. Root reports terminal PID390213/container absent and GPU0/context query empty before the subsequent diagnostic; this review made no remote/profile/export calls while the root's next diagnostic runs. Graph-level cleanup is independently verified from frozen driver control flow and runtime JSON `RESET_ALL_FOUR`; node cleanup is root-observed for this job, unlike the direct Sol03 node check for G8attempt01.

**Decision: VALID_REGRESSION_NO_INCUMBENT_GAIN**, source retained; no integration or official promotion. DS/G8/GS source identities match the expected frozen `672dab44…`/`e92744c6…`/`8ef15e71…`. All three archived actual JIT SASS hashes match JSON. Re-executed the correct DS/G8/GS physical checkers and producer store-form gates on those exact actual JIT binaries; all pass. All runtime main attrs122regs/0local/103600B/512threads/pool65536/occupancy1, pack22regs/0local. Full4096 onehot and randomFP64, task normal/outlier and retained replay validators pass for all candidates, incumbent onehot/FP64/task normal/outlier pass. All3 bit-storage maps, all3 fresh workspace/output validations,20 actual graph mutations with15XP poison, complete output NaN overwrite, source immutability/counters, and exactly2 intended pack+main nodes pass.

The repaired strict nested records under `timing.strict_immediate_actual_timed_output_checks` pass at **both F1 and paired boundaries**, before later input/output/workspace writes or any replay/profile. These include8 oracle checks of actual final timed outputs,6 bitwise XP checks, currentX matching the actual last call, immutable W/packed/escape and zero counters. Independently verified each recorded policy seed against the chronological raw ledger: F1 last incumbent240146/DS240125/G8240104/GS240083; final paired all250031.

Recomputed all295 chronological calls:20 warmups +147F1 +128paired (=32 four-arm rows). All raw times and seeds match the seven F1 arrays and every paired row; all8 rotation/reverse orders appear exactly4 times and each pair shares one fresh fixture across all4 policies. No subset or alternate estimator was used.

| comparator | lower before/after F1 median us | GS gain F1 us | GS paired median gain us | GS wins/32 |
|---|---:|---:|---:|---:|
| actual selected incumbent s4s5hz/KS2 | 23.967999965 | -11.039998382 | -11.295999400 | 0 |
| DS | 33.663999289 | -1.343999058 | -1.359999180 | 5 |
| G8 | 34.432001412 | -0.575996935 | -0.703999773 | 8 |

GS F1 median35.007998347us. Secondary directly derived strict G8-versus-DS paired difference is DS−G8 =−0.399999321us, G8 wins9/32. This job provides new strict G8 control evidence but does not retroactively admit attempt01. No DP arm exists in this job, so a same-run DP-versus-G8 speed claim cannot be derived here. Neither method supports this GS composition. Its scalar-G8 parent is also behind DS in these same paired rows.

Limits: these are one H100 development job and its selected incumbent, not H200 official score estimates. F1 uses different normal seeds across serial blocks; only paired trials match each fresh fixture across arms. Strict checks validate each method's terminal outputs plus the explicit mutation suite, not every intermediate timed output. Static fewer fences and profile durations cannot identify unique latency causes. G8's128K versus DP's192K A lookahead and coarser paired B retirement remain causal confounds. Preserve all source versions, manifests, attempts and raws; use the new root-controlled diagnostic to choose a concrete next mechanism rather than rejecting the family.

Reproduction: `python3 proj_rsi/audits/sol03-kernel-validation-data/check-strict-runtime.py`; machine-readable result `sol03-kernel-validation-data/strict-runtime-review.json`. Script imports CPU SASS checkers only and performs no CUDA context, extension, prepare or launch.

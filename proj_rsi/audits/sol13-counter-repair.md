# Smallm counter-contract repair v3

Worker13 CPU-only driver repair. Read ITERATION.md. No CUDA context, extension
load, prepare, compile, benchmark, remote mutation, commit or root ledger edit.

Root reports attempt smallm-increment01 PID394798/container
okc-proj-smallm-increment01 terminated 2026-10-27 03:07:57 UTC exit1 at frozen
v2 tools/smallm_increment_lab.py:162, with GPU node subsequently at0 MiB.
That assertion loses the failing tensor name. Its exact failed counter cannot
be recovered from this assertion alone. Static source demonstrates that dormant
P1 ticket/flag state can validly remain nonzero after prepare; it does not prove
that P1 was the observed failing tensor. Attempt01 is
INVALID_NO_PERFORMANCE_ADMISSION; saved partial timings remain invalid.
Authoritative archive is results/proj-rsi/takeover/20261027/smallm-increment01/;
history/failed-attempt01-provenance.json binds all11 files by SHA256 and size.
Result has raw_samples0: no timings exist to admit. All three full prepares and
qkv32 basis FP64 validations completed; first normal graph check reached the
global invariant and failed. Exact failing tensor remains UNKNOWN.

Frozen v2 lab remains SHA256
5f1534620d3e62d017fbd8a167b7763aed0d8656ac938bcca8411572c778a308;
old manifest remains
7c5e90b6b6854f89a58affac6c6bdf8fe14b8578e7ca9ae2784764d6fc7e6235.
Exact copies and earlier measurement history are preserved under
candidates/smallm_increment_trial_v3/history. Candidate and all parents unchanged.
New driver tools/smallm_increment_lab_v3.py is a new measurement version.

## Narrow semantic change

V2 required every state tensor whose path contains cnt to be zero except rw_cnt.
V3 binds dispatch contracts to exact frozen candidate hashes and AST-derived
kernel_fn/_r6/_v12/_gvcall/_rscall/_rqcall source hashes. It resolves actual shape,
n equality, selected RQ object and physical counter storage. Selected RQ counters
must be strictly zero, including shared plan/offset aliases. A host wrapper around
original _rqcall asserts return0, so an unvalidated fallback is held explicitly;
it never changes a returned success or launches extra device work. All otherwise
dormant counter storage must exactly match the full prepare baseline, including
nonzero P1/gvcnt/rw_cnt. Counter reads/checks never zero or alter candidate state.

Selected GV P1 bits27/29 and router monotonic paths HOLD pending their own exact
progress contract. On ordinary GV, S1 and red.add modes use no global counter;
V&4 last-CTA reduction has reset0 obligation. Architecture-specific cluster path
HOLDs instead of assuming its pre-sm90 fallback. V12 run/run2/run3 counters have
explicit source reset stores; LT/RR have no counters. Conservative whole-storage
reset0 may HOLD a valid active GV buffer with nonzero unused words; no partial
range relaxation is admitted here. If observed, diagnose and version that contract.

Full storage key is device, allocation pointer and allocation byte size. Every
counter snapshot reads the full int32 allocation even when a selected view has
nonzero offset. Every alias receives the same active storage obligation. Immutable
weight/pack/escape views retain their own offset/shape/stride checks; workspace
and sink scratch are separately listed. Full prepared policy across all requested
shapes and labels is saved immediately after all three actual prepares. Prepare
baseline records include tensor name/storage/dtype/shape/offset/stride/digest,
nonzero count and first32 values. On failure, every counter has initial/final
metadata, digest, small values, contract, context and verdict; failing counters
add complete initial/final arrays. Immutable failures are reported separately.

## Preserved validity and CPU evidence

All old first-math JIT/SASS/resource gates, source pins, cursor/barrier proofs,
full-K basis FP64, original actual kernel_fn selection, current-input fixtures,
fresh/zero/outlier/probe/NaN workspace/output graph checks, immutable checks,
strict immediate final timed output, F1 bracket controls, balanced30 paired rounds,
all-five shape CLI, raw journal and terminal failure/cleanup handling remain.
Invariant checks now also run before math and after each basis oracle. No saved
attempt01 timings are imported into v3. Timing loops and graph launches unchanged.

CPU gate shares counter_verdict and counter_storage_contract with GPU checks.
Positive: valid nonzero dormant and selected zero pass. Negative: selected RQ
nonzero, dormant mutation and unrelated rw change fail. Dispatch n mismatch takes
fallback, selected active persistent P1 HOLDs, GV no-counter and regular reset
routes differ. Shared RQ plans at m16/m32 and an offset alias share physical
storage obligation; a nonzero outside the first view fails and full arrays/digest
preserve evidence. CPU check imports no torch and performs no GPU operation.

Independent critic worker14 reviews the final frozen manifest/hash. GPU remains
pending root independent critic GO. Bounds retained at inner1140/outer1200 seconds;
no speed estimate or promotion claim. Root-only next command after GO:

```sh
PROJ_GPU_CONTAINER=okc-proj-smallm-increment-v3-01 timeout --signal=TERM --kill-after=15s 1200s /root/okc_run.sh timeout --signal=TERM --kill-after=10s 1140s python proj_rsi/tools/smallm_increment_lab_v3.py --output proj_rsi/candidates/smallm_increment_trial_v3/attempt01 --root-review-token proj_rsi/audits/sol14-counter-contract.md --shapes qkv32 qkv16 pooler32 unpooler32 o32
```

Supports the hypothesis if a valid nonzero dormant baseline is preserved while
selected counters reset and all original correctness gates pass. Refutes a mere
lab-overconstraint diagnosis if selected RQ remains nonzero or dormant storage
changes; preserve the named failure and diagnose candidate reset/alias writes.
Either outcome leaves the family active for causal diagnosis under ITERATION.md.

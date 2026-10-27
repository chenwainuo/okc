# GO: serialized read-only native-profile NCU export, strict version token

Independent reviewer: sol17. Author: sol13. Date:2026-10-27.
**GO is scoped to the exact helper and literal descriptor below, normal
120/180-second serialized guard, and offline import only.** Root alone registers,
resolves the CPU-review dependency, pins this review and launches. No queue,
remote, NCU or GPU action was performed by this reviewer.

Frozen reviewed identities:

- Helper `proj_rsi/tools/native_profile_export.py`:
  `84edd702b2399aaf90af32c492e5f9e302eea3844e153c17096d00e95955057b`.
- Author CPU proof `proj_rsi/audits/native-profile-01/export-cpu-proof.json`:
  `9fc9e4910538f7d2546748409a804d003e15462a7190a61b12801251c962c057`.
- Pre-review descriptor `proj_rsi/scheduling/jobs/native-profile-export-01.json`:
  `0d5e0cd762ffc54540e6fd0d7d74e705c3defb859029cfcd4d7363095b6889b1`.
  This identity deliberately precedes root's required review SHA/GO replacement;
  such metadata changes do not authorize changing executable pins/argv/bounds.
- Exact report `proj_rsi/audits/native-profile-01/input/native-profile-01.ncu-rep`:
  `4bbde4f4d2a157dde8504dee2764c1a80706dd4ae1ff735725703681b39c6ddf`,
  1080736bytes. All eight descriptor file hashes independently match local bytes.

## Scope, launcher and resource ownership

The descriptor's actual target is literal `python3` followed immediately by the
source-pinned `/okc/proj_rsi/tools/native_profile_export.py`. There is no shell,
`-c`, `-m`, preload, CUDA/torch/report-SDK import, or substituted executable.
Only stdlib imports occur. `socket.gethostname()` records local metadata; the
script does not open an outbound socket, SSH or call a hardware API.

It requests exactly three external invocations: the fixed target NCU binary's
`--version` (5s), then two fixed `--import <exact report>` operations (45s each).
The import arguments are raw-page CSV and source-page SASS CSV. There is no
capture/replay/target program, process attachment, metrics collection, Python
target appended to NCU, or shell expansion. Every subprocess uses `shell=False`.
The script requests no kernel or CUDA context. This is a source/API scope claim;
vendor NCU internals have not been executed or observed in this CPU review.

The normal pinned guard runs inner120/outer180, `profile:false`, with only
`MAX_JOBS=1` in descriptor env. It does not ask for `SYS_ADMIN` or modify profiling
policy. The normal wrapper still serializes the device lock and gives root the
existing actual-probe/terminal/container/PID/memory cleanup gates. No import may
run alongside another timing job. A timeout/failure remains invalid/incomplete,
with partial stdout/stderr/host logs retained; no automatic retry or promotion.

The helper accepts only the immutable report's canonical nonsymlink path,
exact size and SHA. Both the report and resolved client binary hash are checked
again after export. The exact target binary path is
`/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu`.
Runtime export-client SHA/version and OS-release/hostname metadata are recorded.
These are not a cryptographic Docker image digest, and the pinned wrapper uses
an image tag. The historical capture-client binary SHA was never recorded:
it remains explicitly UNKNOWN. Never substitute export-client identity for
capture provenance.

## Outputs and parser claims

The exact literal output child is
`/okc/logs/scheduler/native-profile-export-01/run`; its parent is collected from
`/root/okc/logs/scheduler/native-profile-export-01`. The child must not already
exist. Runtime rejects relative/outside-scheduler paths, canonical-path escapes
and symlink parents before creating outputs. The fixed reviewed argv assigns
this job its dedicated artifact child; approval does not cover arbitrary other
output paths accepted by the generic helper. Do not precreate/erase/reuse the
child; a new attempt needs its own immutable descriptor/container/output identity.

Expected retained files: `raw.csv`, `source.csv`, both stderr files,
`raw-kernel-rows.json`, `PC-source-rows.json`, `ncu-version.txt`, and `result.json`.
Seven non-result artifacts receive SHA hashes. `result.json` includes command
argv/return codes, provenance, bounds and status. Failure preserves raw partial
files and an invalid/incomplete traceback result when Python can still write it;
hard guard termination may leave partial files, so exit/probe/inventory evidence
must still be checked independently.

The parser requires exactly the three expected nodes `rx_pack`, `rq_px_dp` and
`rq_tma_x`, retaining complete raw rows and per-PC source rows. It compares
available `# Samples` with `smsp__pcsamp_sample_count`; absent/non-integer metrics
stay null, and mismatches remain explicitly analyst-review-required. A successful
export status still says source/counter analysis PENDING. Synthetic parser tests
do not establish compatibility of actual vendor CSV until the real import runs.

No sample total, reason, phase distribution, causal bottleneck, latency or speed
gain is approved here. Author13 must bind actual source PCs to the frozen SASS/
CFG roles, retain common/unmapped and all denominators, reconcile samples and
sampling metadata, apply preregistered sample sufficiency and preserve missing
metrics/warnings. Counts are not microseconds; capture replay passes are not
independent performance repetitions. No H100-to-H200 score inference follows.

## Independent CPU evidence and repaired negative

Reexecuted `cpu_check()` with `subprocess.run` patched to raise on any real call:
PASS, including actual input SHA/size and eight unsupported-version negatives.
Separately checked all descriptor file pins and source stdlib-only imports.

A fully mocked runtime in a temporary artifact root exercises successful export:
exactly3 fixed calls, timeouts5/45/45, all `shell=False`, all8 expected output files,
7 artifact hashes, immutable-input/client final checks, and
`OFFLINE_EXPORT_COMPLETE_SOURCE_COUNTER_ANALYSIS_PENDING`. This executes no NCU
binary. Existing output child and symlink-escape output negatives reject before
any external invocation. Changed-report, absent raw sample metric and unsupported
version negatives preserve failure/unknown semantics.

The predecessor helper `05dff4bd6eae17ea92aa16116fc4bdd7033b1d8732ab3562f6bcd7f023ee5480`
incorrectly accepted `Version 2025.1.1beta` and
`Version 2025.1.1.0garbage`. The independent review reproduced both failures.
An intermediate word-boundary repair still accepted `2025.1.1-rc1` and
`2025.1.1+build`; review caught these before launch as well. The author finally
requires a whitespace/end token boundary `(?=\s|$)`, added all four suffix
negatives, regenerated proof and repinned the descriptor before this GO.
Independent reexecution accepts the tested plain2025.1.1/2025.1.1.0 forms and
rejects those suffixes plus the prior unsupported-version cases. No execution
result is transferred from either predecessor source.

The job has zero direct gain. Its information and follow-up priors are explicitly
subjective and conditional, with the complete120s estimated reservation,300s
contingent follow-up and15CPUmin cost. It does not double count capture/export as
achieved gains. Its separate CPU-review dependency avoids a circular dependency
on the capture job that remains VALIDATING pending export/interpretation.

GO permits root's bounded offline export after the exact review pin/dependency
and ordinary dispatcher admission gates. It does not finish capture validation,
approve counter interpretation, or change selected serving policy.

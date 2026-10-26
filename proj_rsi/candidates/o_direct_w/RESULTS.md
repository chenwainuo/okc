# o-direct-w01 — CLOSED_FIXED_POLICY_NO_SWEEP

Root-owned exclusive H100 run, 2026-10-26T21:14:50UTC, LAB_EXIT0. Generated source `2f571e925469ea90476aa1ec87d72d86997d460876a7250603113c1653d3feb1`; pw36 baseline `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Frozen lab/kernel/prototype unchanged.

|M|actual pw36 path|F1 baseline/prototype us|F1 gain us|paired median gain us|slower pairs|
|---:|---|---|---:|---:|---:|
|32|rq:t32s5hze KS2|9.696/11.936|-2.240|-2.208|21/21|
|128|r6 tunedLt200014|10.656/14.720|-4.064|-3.936|21/21|

Both exact-M groups pass all1024 onehot columns, strict FP64 normal/outlier checks, task probes and retained original/zero/fresh-outlier replay tests. Real Torch extension builds successfully; both232registers/local0. Every cold-L2 pair loses; neither>=0.5us dual timing gate passes.

Close this Q4/direct-W/X8/HALF/full-K/plain-store policy. No retry, sweep, port, incumbent edit or official slot. No H200 score gain or universal direct-W limitation claimed. Stronger actual timed-baseline/mutated-graph checks were promotion conditions; no survivor requires them.

Independent full audit: `proj_rsi/audits/o-direct-w01-preflight.md`. Raw: `results/proj-rsi/takeover/20261026/o-direct-w01.{json,host.log}`. JSON SHA256 `5d772ece0a12737336add57e33113d1ced27aae4f29d0f7d0e80938f59021256`; log `d15f9064b9c86c1dea36acdd1314c09829d05e6c1eeed74fe2bc57822694891f`.

README/RESULTS are post-run documentation. The manifest preserves preflight document hashes; frozen executable source/lab hashes remain unchanged.

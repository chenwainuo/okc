# lt-graph01 — CLOSE_POLICY_NO_SWEEP

Root-owned exclusive H100 run, 2026-10-26T21:16:33UTC, LAB_EXIT0. Frozen lab `8245f822dac5171aaf8a220899729c9ab5d5eacbb8a9bfae2e8404e4503578b3`; pw36 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. No lab/baseline code edit.

One rank0 legal o32 layout2 candidate: code200008, config `[66,13,1,0,0,1,35,0,3]`; forced splitK1/reduction0, no timed search. Actual baseline r6 gv45. All1024 onehot columns, FP64 normal/outliers and retained original/zero/fresh-outlier graph checks pass.

F1 before/after baseline9.920/9.888us, candidate10.752us, gain-0.864us.21 paired graph median gain-0.864us; candidate loses20/21 pairs. Both>=0.5us gates fail. Profiles each show one kernel (baseline gv, candidate Nvjet NNT); their single sampled durations do not override raw graph timing.

Close the fixed first-legal policy; no retry, layout/config/cluster sweep, o128 expansion, port or judge slot. Broader Lt impossibility and a recoverable benchmark/profile gap are not established. Source logging stringifies gv weights outside timing; this is preserved evidence, not a performance mechanism.

Independent audit: `proj_rsi/audits/lt-graph01-preflight.md`. Raw: `results/proj-rsi/takeover/20261026/lt-graph01.{json,host.log}`. JSON SHA256 `7747ff41b187f99ef0f207d86b51ab1d048a7c431f6d91a16556e7c57823e98f`; log `ffffe95944bc5f67b819dda13840afc191a0a8887441b32d1fe5943776cb4bfa`.

README/RESULTS are post-run documentation. Manifest preflight document hashes are retained; frozen lab hash remains unchanged.

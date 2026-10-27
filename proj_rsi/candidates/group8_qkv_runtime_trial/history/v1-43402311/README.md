# QKV128 split4 actual-policy runtime trial

State: CPU implementation only; GPU_UNRUN and independent critic required.
No candidate/parent CUDA or wrapper source was changed. This driver consumes the
frozen `group8_qkv_trial/runtime.py` Pipeline and exact manifests. It is an actual
root-executable measurement harness, not a performance projection.

`tools/group8_qkv_runtime_lab.py` imports only stdlib until root explicitly runs
runtime mode. `--cpu-check` verifies dependency hashes, 524288-word XP/output/
CSR/schedule/physical proofs, source-bound counter contracts, all six paired
orders five times each, two negative terminal-fixture cases and read-only
terminal-gate AST. No extension, prepare, CUDA context, NVCC or benchmark runs
under that mode. Frozen CPU wrapper compilation evidence is retained upstream.

Root-only runtime, after independent GO and exclusive hardware admission:

```
python3 proj_rsi/tools/group8_qkv_runtime_lab.py --output logs/qkv-split4-attempt01 --root-review-token proj_rsi/audits/EXACT-INDEPENDENT-REVIEW.md
```

Root must use its existing serialized H100 guard, a new named container and a
reviewed proposed inner1140/outer1200-second bound; the estimate is a resource
budget, not a measured run duration. No official submission follows this trial.
Full wall time includes exact extension build, two original full prepares,
packing, all numerical/graph checks and collection. Source/driver hashes must
remain fixed through launch/termination. The review placeholder above must be
replaced with the real independent artifact; file existence alone is not GO.

Before candidate math, the actual loaded `.so` is copied and hashed, main/pack/
reduce SASS exactly matches upstream offline host SASS, physical A/D/fence gates
pass, and runtime register/local/shared/occupancy gates pass. Main122, pack22,
reduce16 registers; no local spills;512-thread main rounded pool65536. The main
opt-in query precedes all capture. Compiler differences HOLD setup; they are
not permission to skip a mismatching gate. Parent production prepares naturally
execute their own baked kernels; their actual selected extension/source/SASS/
resources are recorded before external validation or timed graphs. No forced
RQ path or prepare-clock reset is used. Selected non-RQ baselines are allowed as
actual controls and have their real node profiles recorded.

The real task weight codec must roundtrip BF16 storage bits exactly and derive
CSR capacity from actual escapes. All8192 basis columns are exact at x8 for all
three policies; normal/outlier task and FP64 relativeL2<=4e-3/row<=2e-2 pass.
The full candidate call is fresh pack -> split4 main -> ordered FP32 reduction.
XP bytes exactly match a shape-only oracle; every partial is finite after NaN
poison and output equals (((partial0+partial1)+partial2)+partial3).to(BF16).
All65536 raw BF16 storage patterns are probed through the actual pack separately:
nonfinite arithmetic in that storage diagnostic is not claimed numerically
valid. Fresh valid fixtures then overwrite/revalidate every workspace/output.
A separately allocated Pipeline tests fresh workspace/output ownership.

Each control is its complete freshly prepared original `kernel_fn`; captured
outputs are retained separately, with no added output copy in timed graphs.
RQ launch-decline fallbacks are rejected when RQ is the prepared selection.
Immutable packed/raw/escape tensors are compared exactly. Counter contracts
reuse the pinned v3 source-bound proof: actual selected storage resets0,
dormant monotonic storage equals its prepare baseline, including offset aliases.
No harness counter zeroing occurs. Full failure values and successful final
counter digests are saved.

Timing preserves210 raw calls:15warm +105F1 +90paired. F1 is21 each in
PW36-before/RS32-before/new/RS32-after/PW36-after blocks. Paired is30 rows,
six permutations each five times, same fresh fixture across three arms,256MiB
read-only L2 flush before each replay, no sample deletion. At BOTH method
boundaries, before any replay/mutation/profile/input copy, the actual last
outputs are checked against each arm's last fixture; current X/XP/FP32 partial,
immutable bytes and counter contracts are verified. Raw calls immediately
append to `raw.jsonl` and summary retains every sample/ratio/difference.

Only after both strict terminal checks, CUPTI profiles must identify exactly
three candidate nodes (pack/main/reduce); outputs are revalidated immediately.
All three fixed graphs also pass original/zero/fresh normal/outlier/all-row
probe mutations with NaN output poison, plus XP/partial poison for candidate.
Graphs drain/reset on success/failure. Root must independently verify guard,
container/compute PID disappearance and memory release after exit. Exit0 alone
is not correctness or promotion admission. H100 timing is not an H200 forecast;
attainable gain remains UNKNOWN until a valid measured actual-control win.

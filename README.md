# Open Kernel Challenge serving experiments

Optimize SGLang serving for Muse Spark Open. Develop and profile individual expert
components on one H200, then validate the complete serving patch on eight H200s.
The patch stack is experimental: benchmark results determine which changes survive.

See the [architecture review and optimization candidates](docs/architecture-review.md)
for the serving path, bottleneck hypotheses, experiments and validation gates.
The [structured review](docs/architecture-review.json) is the current agent handoff record.

## Setup

Use a Linux GPU host with Docker, NVIDIA drivers, and NVIDIA Container Toolkit.
The launcher uses the exact challenge container digest and preserves compilation caches.

```bash
git clone git@github.com:chenwainuo/okc.git
cd okc
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
python3 bootstrap.py
```

Bootstrap downloads the official problem fixtures, verifies their hashes, fetches
SGLang at the required commit, and applies `serving/patch.diff`. Upstream sources
and challenge fixtures retain their original licenses and are fetched separately.
Bootstrap refuses to overwrite an unrelated SGLang worktree.

## Measure

```bash
# Short correctness + timing smoke test; two pairs are not statistical evidence.
bash benchmarks/run_h200.sh --pairs 2 --batches 8 32

# Actual stock serving method versus patched serving method, with profiler traces.
bash benchmarks/run_h200.sh --profile

# Each isolated change, intermediate/full stacks, and actual serving method.
bash benchmarks/run_h200.sh --matrix --profile

# Representative prefill regression diagnostics; increase pairs for conclusions.
bash benchmarks/run_h200.sh --matrix --prefill --pairs 20

# Marlin thread configuration / occupancy sweep with an unchanged JIT control.
bash benchmarks/run_h200.sh --execution --minutes 45

# Tile/padding candidates, with independent code artifacts and source hashes.
bash benchmarks/run_h200.sh --tiles
```

Results and profiles go into `results/`. Diagnostics include named eager stage ranges
(alignment, workspace, gate/up GEMM, SwiGLU, down GEMM, finalizer), graph kernel
traces, structured event tables, per-expert route counts and GPU memory snapshots. Clean timing uses CUDA graph replay,
cold L2, alternating execution order, fresh inputs and correctness checks against
the official FP32 reference. Profiler runs are separate from timing. Bootstrap
confidence intervals describe within-run variation only; repeat independent runs
before retaining a change. Local kernel speedups are **not** serving scores.

## Optimization loop

Run the resumable experiment queue with `bash benchmarks/run_h200.sh --loop --minutes 45`.
State, candidate sources, logs, and measurements persist under `results/optimization-loop/`.
The [ranked experiment queue](docs/optimization-queue.json) tracks potential ROI,
evidence, stage, and next action. Executable queued items run in rank order;
profiling and implementation tasks remain agent work.
See [the agent loop protocol](docs/optimization-loop.json) for gates, stop/resume,
budget boundaries, and promotion requirements. The finite queue stops when exhausted;
new hypotheses and active-patch promotion require agent review.

Baseline → profile → explicit bottleneck hypothesis → isolated implementation →
correctness → paired measurements → retain/revert → re-profile the stack →
full serving validation. Do not infer a speedup from fewer operations alone.

The active changes in `serving/patches/` enable small-batch fused finalization and
cap Hopper EP8 expert tiles at 16 through 256 tokens and 32 through 512 tokens.
Larger prefills retain stock tile selection. The combined artifact is `serving/patch.diff`.
Earlier routing candidates are parked in `serving/experiments/initial-stack/`;
their standalone measurements did not establish a useful gain.

## Agent-driven RSI

The [RSI protocol](rsi/protocol.json) assigns planner, implementer, critic, executor,
and validator roles. The [kernel rewrite plan](docs/kernel-rewrite-plan.json) ranks
counter-backed opportunities and labels speedup ranges as engineering priors.
The [independent audit](docs/rsi-measurement-audit.json) records measurement gaps.

```bash
# Read the current ledger; all GPU work remains supervisor-controlled.
python3 -m rsi status
# First-time initialization only, with an explicit run-accounting budget:
python3 -m rsi init --budget-usd 5
```

See [CLI usage](rsi/usage.json) and proposal/review schemas under `rsi/`.
Source bundles, exact shape coverage, fresh holdouts, raw samples and independent
review gate local promotion. Ledger promotion neither applies a patch nor proves
full serving speedup. Legacy reports are observational only. The local ledger
coordinates trusted agents; actor labels are not authentication and deadline
records do not themselves stop a process. Root must enforce execution deadlines.
Optimizer-method promotion requires the separate frozen portfolio evaluator
specified in the protocol; that evaluator is not implemented yet.

## Serving requirements

The [official rules](https://openkernelchallenge.dev/rules) and
[serving API](https://openkernelchallenge.dev/api/v1/serving) are authoritative.
`serving/benchmark-contract.json` records the observed configuration.

- Hidden multi-turn agentic coding conversations, concurrency 1, 4, 16, 32.
- Score compares throughput at stock interactivity across loads.
- Outputs must agree nearly as well as stock agrees with itself.
- Exactly 220 output tokens per request; TTFT at most twice stock at a load for
  that load to improve the score.
- Preserve precision, all eight routed experts, norms, attention window and DFlash.

The single-GPU adapter excludes loading, communication, scheduling, attention and
the pre-expert norm. Full-node correctness and performance remain mandatory.

## CPU checks

```bash
python3 -m unittest discover -s benchmarks -p 'test_*.py' -v
```

CPU tests cover dispatch and measurement logic, not GPU numerical correctness.
GPU checks include invalid routes, empty local routing, and NaNs in ignored slots.

# Existing full-expert and full-model tracing capability (CPU-only inspection)

The shortest existing full-expert trace path is `benchmarks/moe_local.py --profile`
with candidate `benchmarks/serving_moe.py`. That adapter invokes the actual patched
`MuseSpark1xFp4Experts._forward_marlin`, not the projection-only benchmark or the
isolated SS stage screen. No implementation change is needed for this trace.

Standalone launcher command (for a dedicated GPU host; root must adapt it to the
shared serialized queue/lock rather than invoke this Docker launcher beside work):

```bash
bash benchmarks/run_h200.sh --profile --pairs 2 \
  --batches 8 32 128 256 1 16 64 257 384 512 513 753 3012 16384 \
  --output results/full-expert-hotpath-14.json
```

Two pairs are a bounded diagnostic smoke setting, not comparative performance
validation. The default batch list is only six cases; explicit14-shape arguments
are necessary. The launcher uses exact image
`lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`,
mounts the checkout and compiler cache, sets
`PYTHONPATH=/workspace/serving/sglang/python`, then runs `benchmarks/preflight.py`,
`check_local_alignment.py`, `check_finalize.py` before `moe_local.py`.
The equivalent INNER profiling command is:

```bash
PYTHONPATH=/workspace/serving/sglang/python python3 benchmarks/moe_local.py \
  --baseline benchmarks/serving_stock_moe.py \
  --candidate benchmarks/serving_moe.py --profile --pairs 2 \
  --batches 8 32 128 256 1 16 64 257 384 512 513 753 3012 16384 \
  --output results/full-expert-hotpath-14.json
```

Do not equate its default `serving_stock_moe.py` control with the user's
stock861333f/source05ec9a8a full-goal anchor: this stock adapter loads upstream
SGLang git object a55ab7d5 directly. It is an available tracing/control mechanism,
not an admitted3x comparison. Promotion against861333f requires the entire frozen
baseline-v2 bundle and existing reviewed comparison protocol.

## What it logs

`benchmarks/diagnostics.py` already supplies `stage_ranges` and `save_profile`.
The runner emits separate baseline/candidate graph traces (five replays) and eager
stage traces (three calls, shapes/stacks), plus structured event tables and text:

- `<output-stem>-profiles/bs{B}-{baseline,candidate}.trace.json`
- corresponding `.events.json` and `.txt`
- corresponding `-eager-stages.trace.json/.events.json/.txt`

Eager `okc_stage/` ranges identify alignment, workspace, gate/up GEMM, SwiGLU,
down GEMM and finalizer. Wrapper dispatch distinguishes GEMMs using actual top_k
keyword (up>1,down=1); the current serving source uses those keywords. Kernel traces
show actual CUDA graph execution rather than Python attribution. Routing histograms,
local-route fractions, fresh-input checks, empty-route checks, source hashes/diff,
GPU memory and paired raw event samples are already retained. Post-timing profiling
is separate from clean timing. Eager and graph counts are explicitly3 and5 synthetic
calls, not production model call frequencies or model-time shares.

The local full-expert scope includes route alignment/local expert map, both GEMMs,
SwiGLU, per-expert post-norm and router-weighted sum. The adapter deliberately skips
the real constructor/distributed setup, starts after pre-expert norm, uses synthetic
reference weights/routing and invokes `_forward_marlin` directly. Model loading,
attention, token scheduler, actual router computation, pre-expert norm, distributed
collectives, TTFT, request throughput and8H200 serving are outside this experiment.
Router WEIGHTED SUM inside finalizer is not the model router itself.

## Runtime evidence and requirements

Read-only checkout HEAD is required a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
Preflight requires CUDA Torch, Triton, sgl_kernel, FlashInfer and SGLang loaded from
the patched checkout, plus GPU compiler/runtime support for Marlin/Triton. It warns
for non-H200 hardware. Current author CPU environment has Torch2.13.0+cpu and no
top-level triton/sgl_kernel/flashinfer/sglang/nvtx installation (find_spec only,
no module imports or CUDA initialization). This is concrete local missing-dependency
evidence; it does NOT establish the remote GPU image's package inventory.

The existing root PyTorch2.8/CUDA12.8 image is sufficient for the self-contained SS
stage screen but has not been verified here to contain the full SGLang dependencies.
Do not assume it can run this full-expert trace. The existing pinned SGLang launcher
is the executable dependency specification; root should inspect/reuse/pull that
image under its own operational policy. Image availability/transfer and full14
warmup/compile/oracle/profile cost are unmeasured. No720-second screen budget is
silently reused for this broader experiment.

## Existing full-model profiler/NVTX entrypoints for later8H200

After the actual model server is launched with its reviewed model/checkpoint and
8H200 TP/EP/DP configuration, no new profiling service is required:

1. `python3 -m sglang.launch_server` is the existing server entrypoint. A concrete
   model path/tokenizer/config and approved distributed launch arguments are still
   needed; no valid8H200 model launch can be inferred from the synthetic adapter.
2. HTTP `/start_profile` and `/stop_profile` already exist in
   `srt/entrypoints/http_server.py`; admin authentication applies when configured.
   `ProfileReq` supports output_dir,num_steps,activitiesCPU/GPU,profile_by_stage,
   with_stack,record_shapes,merge_profiles,profile_stages,detailed_annotations.
   A bounded num_steps request automatically stops; requests/load must execute
   while capture is active.
3. `python3 -m sglang.profiler --url http://127.0.0.1:30000 --num-steps 5
   --merge-profiles --output-dir /traces/full-model` is an
   existing controller. It writes server_args.json and calls the above API;
   it needs concurrent request load and can wait for the configured forward steps.
4. Existing `python3 -m sglang.benchmark.serving` drives requests and profiling:
   `--backend sglang --base-url http://127.0.0.1:30000 --dataset-name random
   --random-input-len 128 --random-output-len 32 --num-prompts 16 --profile
   --profile-num-steps 5 --profile-output-dir /traces/full-model`.
   This is a bounded smoke workload, not the official serving workload or a3x gate.
   Use global-step capture first; optional --profile-by-stage requires sufficient
   batches of BOTH prefill and decode and can wait if one stage is under-supplied.
5. Set `SGLANG_TORCH_PROFILER_DIR=/traces/full-model` on server launch. Optional
   `SGLANG_ENABLE_NVTX_SCHEDULER=1` and `SGLANG_ENABLE_NVTX_OPERATIONS=1` enable
   existing NVTX spans if Python nvtx is installed. `srt/utils/nvtx_utils.py` warns
   and disables NVTX if absent; Torch record_function spans still work without it.
   ModelRunner forward and scheduler spans already use this helper. Nsight Systems
   can consume NVTX later; model/server/collective traces remain separate from local
   expert/projection traces. No detailed expert-per-layer NVTX ranges are currently
   inserted into this class by the local diagnostics unless its wrappers are used.

All findings are CPU source inspection only. No GPU, remote command, API call,
model launch, queue action, trace capture or measured hotpath attribution occurred.

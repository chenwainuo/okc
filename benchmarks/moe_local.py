#!/usr/bin/env python3
"""Single-GPU MoE experiment; run inside the challenge's SGLang image.

From the workspace root:
  python3 benchmarks/moe_local.py --output results/local-buckets.json

Uses synthetic official reference inputs, not model checkpoint weights. Validates
both implementations before timing and checks captured outputs on fresh inputs.
Reports raw CUDA-event replay latency with cold L2; this is not the judge score
(no replay-cost subtraction or official confidence interval). Does not measure
collectives, scheduling, attention, or full serving throughput.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from measurement import summarize
from diagnostics import gpu_snapshot, routing_summary, save_profile, stage_ranges

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / "muse_moe_w4a16"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path,
                        default=PROBLEM / "candidate_local_buckets.py")
    parser.add_argument("--baseline", type=Path, default=PROBLEM / "baseline.py")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local-buckets.json")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 128, 256],
                        help="Token counts, e.g. 8 32 128 256 753 3012 16384")
    parser.add_argument("--profile", action="store_true",
                        help="Save separate CUDA graph profiler traces after timing each shape")
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error("--pairs must be positive, even, and at least 2")
    if any(bs < 1 for bs in args.batches):
        parser.error("--batches must be positive")

    import torch
    if not torch.cuda.is_available():
        parser.exit(2, "CUDA GPU required; use the challenge container on your H200.\n")
    sys.path.insert(0, str(PROBLEM))
    reference = load("moe_reference", PROBLEM / "reference.py")
    baseline = load("moe_baseline", args.baseline)
    candidate = load("moe_candidate", args.candidate)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)

    checkout = ROOT / "serving/sglang"
    source_diff = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD"])
    version = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
    report = dict(status="initializing", experiment_commit=version.stdout.strip() or None,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, seed=args.seed, pairs=args.pairs,
                  candidate=str(args.candidate), baseline=str(args.baseline),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  serving_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
                  serving_diff=source_diff.decode(),
                  measurement="raw CUDA graph replay, cold L2, synthetic inputs; not official score",
                  results=[])

    def checkpoint(status):
        report["status"] = status
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    checkpoint("generating_weights")

    print("Generating synthetic NVFP4 weights", flush=True)
    weights = reference.generate_weights(args.seed)
    # Isolate preparation from the reference and from the other implementation.
    modules = (baseline, candidate)
    states = []
    for module in modules:
        checkpoint("preparing_" + module.__name__)
        print(f"Preparing {module.__name__}", flush=True)
        started = time.perf_counter()
        states.append(module.prepare({k: v.clone() for k, v in weights.items()}))
        torch.cuda.synchronize()
        print(f"Prepared {module.__name__} in {time.perf_counter() - started:.1f}s", flush=True)

    def check(data, output):
        ok, detail = reference.check_implementation(data, output, weights)
        if not ok:
            raise AssertionError(detail)
        return detail

    # Empty local routing is valid in serving even though the challenge's
    # busiest-rank reference checker intentionally rejects it as uninformative.
    no_local = reference.generate_input(4, "ep8", args.seed + 99)
    no_local["topk_ids"][:] = torch.arange(32, 40, device="cuda", dtype=torch.int32)
    for module, state in zip(modules, states):
        checkpoint("checking_no_local_" + module.__name__)
        print(f"Checking no-local routes: {module.__name__} (first call may compile)", flush=True)
        out = module.kernel_fn(no_local, state)
        torch.testing.assert_close(out, torch.zeros_like(no_local["hidden_states"]),
                                   rtol=0, atol=0)
        print(f"no-local correctness {module.__name__}: passed", flush=True)

    for i, (bs, dist) in enumerate(((4, "normal"), (33, "normal"), (257, "normal"),
                                    (128, "skewed"), (1, "normal"))):
        data = reference.generate_input(bs, "ep8", args.seed + 100 + i, dist=dist)
        for module, state in zip(modules, states):
            checkpoint(f"checking_{module.__name__}_bs{bs}")
            print(f"correctness {module.__name__} bs={bs}: "
                  f"{check(data, module.kernel_fn(data, state))}", flush=True)

    # Larger than H200 L2 (50 MiB); evict before, not within, each timed replay.
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = report["results"]
    for bs in args.batches:
        checkpoint(f"measuring_bs{bs}")
        data = reference.generate_input(bs, "ep8", args.seed + 1000 + bs)
        graphs, outputs = [], []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for module, state in zip(modules, states):
                for _ in range(3):
                    module.kernel_fn(data, state)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for module, state in zip(modules, states):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = module.kernel_fn(data, state)
            graphs.append(graph)
            outputs.append(out)

        # Replay with no local routes after warming/capturing on valid routes.
        # This catches stale or uninitialized outputs without timing an artificial
        # routing distribution as if it represented the serving workload.
        original_ids = data["topk_ids"].clone()
        data["topk_ids"][:] = torch.arange(32, 40, device="cuda", dtype=torch.int32)
        for graph, out in zip(graphs, outputs):
            graph.replay()
            torch.testing.assert_close(out, torch.zeros_like(data["hidden_states"]),
                                       rtol=0, atol=0)
        data["topk_ids"].copy_(original_ids)

        timings = [[], []]
        ratios = []
        routing_samples = []
        # CUDA events are lazily initialized. Do that outside measurement and
        # reuse them, avoiding host-side event creation between start and end.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()
        for pair in range(args.pairs):
            fresh = reference.generate_input(bs, "ep8", args.seed + 2000 + bs * 100 + pair)
            routing_samples.append(routing_summary(fresh))
            for key in ("hidden_states", "topk_ids", "topk_weights"):
                data[key].copy_(fresh[key])
            order = (0, 1) if pair % 2 == 0 else (1, 0)
            elapsed = {}
            for index in order:
                flush.zero_()
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                elapsed[index] = start.elapsed_time(end) * 1000
                timings[index].append(elapsed[index])
                check(data, outputs[index])
            ratios.append(elapsed[0] / elapsed[1])
        row = dict(bs=bs, baseline_us=statistics.median(timings[0]),
                   candidate_us=statistics.median(timings[1]),
                   paired_speedup=statistics.median(ratios),
                   baseline_samples_us=timings[0], candidate_samples_us=timings[1])
        row.update(summarize(*timings))
        row["routing_samples"] = routing_samples
        row["gpu_memory"] = gpu_snapshot()
        results.append(row)
        checkpoint(f"measured_bs{bs}")
        print(f"bs={bs}: {row['paired_speedup']:.3f}x "
              f"({row['baseline_us']:.2f} -> {row['candidate_us']:.2f} us); "
              f"{row['assessment']}, exploratory CI={row['bootstrap_95']}", flush=True)
        if args.profile:
            checkpoint(f"profiling_bs{bs}")
            profile_dir = args.output.parent / (args.output.stem + "-profiles")
            profile_dir.mkdir(parents=True, exist_ok=True)
            for index, (label, graph) in enumerate(zip(("baseline", "candidate"), graphs)):
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        with torch.profiler.record_function("cache_flush"):
                            flush.zero_()
                        with torch.profiler.record_function(f"{label}_graph_replay"):
                            graph.replay()
                    torch.cuda.synchronize()
                prefix = profile_dir / f"bs{bs}-{label}"
                save_profile(prof, prefix)
                with stage_ranges(modules[index], states[index]):
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, with_stack=True,
                    ) as eager_prof:
                        for _ in range(3):
                            modules[index].kernel_fn(data, states[index])
                        torch.cuda.synchronize()
                save_profile(eager_prof, str(prefix) + "-eager-stages")
            row["profile_directory"] = str(profile_dir)
        del graphs, outputs, graph, out, original_ids

    report["geometric_mean_speedup"] = math.exp(statistics.mean(
        math.log(row["paired_speedup"]) for row in results))
    checkpoint("complete")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

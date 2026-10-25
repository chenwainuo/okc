"""Time one muse_bf16_proj submission against SGLang's baseline on a local H200.

CUDA-graph replay, cold L2, alternating order. The geomean matches the official
load definition (five layers times weight 62, six loads equally weighted).
This is a local proxy: promotion still wants the interval from this report, and
the official board can disagree by the judge's fixed replay overhead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
sys.path.insert(0, str(ROOT))

import reference  # noqa: E402
from proj_rsi.score import FLOORS_US, LAYERS, LOADS, floor_load_us, geomean, speedup_from_load_times  # noqa: E402


# The judge subtracts about 4 us of CUDA-graph replay cost from both sides.
# A local empty-graph measurement is noisier than that constant, so the score
# proxy uses 4 us and the report keeps the measured value beside it.
JUDGE_OVERHEAD_US = 4.0

OFFICIAL = (
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(fn, data, state, static_x=None):
    if static_x is None:
        static_x = data["x"].clone()
    static = {"x": static_x, "config": data["config"]}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(static, state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(static, state)
    return graph, static_x, output


def flush_l2(buffer: torch.Tensor) -> None:
    # Read a buffer larger than L2 so the timed replay starts cold.
    buffer.view(torch.float32).sum()
    torch.cuda.synchronize()


def replay_us(graph, start, end, scratch) -> float:
    flush_l2(scratch)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def measure_overhead_us(scratch, repeats: int = 30) -> float:
    ones = torch.zeros(1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ones.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ones.add_(1)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = [replay_us(graph, start, end, scratch) for _ in range(repeats)]
    samples.sort()
    return samples[len(samples) // 2]


def bootstrap_geomean(baseline_loads, candidate_loads, seed: int = 71237):
    """Resample pairs, take each load's median, then the official geomean."""
    pairs = len(next(iter(baseline_loads.values())))
    if pairs < 20 or pairs % 2:
        return None
    names = list(baseline_loads)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        indexes = [rng.randrange(pairs) for _ in range(pairs)]
        ratios = []
        for name in names:
            base = sorted(baseline_loads[name][i] for i in indexes)
            cand = sorted(candidate_loads[name][i] for i in indexes)
            mid = pairs // 2
            ratios.append(base[mid] / cand[mid])
        draws.append(geomean(ratios))
    draws.sort()
    return [draws[49], draws[1949]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "muse_bf16_proj" / "baseline.py")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5151)
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        raise SystemExit("pairs must be even and at least 2")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    candidate = load_module(args.submission.resolve(), "proj_candidate")
    baseline = load_module(args.baseline.resolve(), "proj_baseline")
    weights = reference.generate_weights(args.seed, device="cuda")
    weights_c = {name: tensor.clone() for name, tensor in weights.items()}
    weights_b = {name: tensor.clone() for name, tensor in weights.items()}
    candidate_state = candidate.prepare(weights_c)
    baseline_state = baseline.prepare(weights_b)

    cases = []
    for index, (m, layer, dist) in enumerate(OFFICIAL):
        data = reference.generate_input(m, layer, args.seed + 1000 + index, dist=dist, device="cuda")
        output = candidate.kernel_fn(data, candidate_state)
        ok, message = reference.check_implementation(data, output, weights)
        cases.append({"m": m, "layer": layer, "dist": dist, "ok": bool(ok), "message": message})
        if not ok:
            report = {"source": "local_h200", "correct": False, "cases": cases, "submission": str(args.submission)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            raise SystemExit(1)

    scratch = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    overhead_us = measure_overhead_us(scratch)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)

    per_shape = {}
    raw_candidate = {}
    raw_baseline = {}
    kernel_candidate = {}
    kernel_baseline = {}
    for load, m in LOADS:
        for layer in LAYERS:
            data = reference.generate_input(m, layer, args.seed + 17, device="cuda")
            graph_c, static_x, output_c = capture(candidate.kernel_fn, data, candidate_state)
            graph_b, _, _ = capture(
                baseline.kernel_fn, {"x": static_x, "config": data["config"]}, baseline_state, static_x
            )
            graph_c.replay()
            torch.cuda.synchronize()
            ok, message = reference.check_implementation(
                {"x": static_x, "config": data["config"]}, output_c, weights
            )
            for _ in range(10):
                graph_b.replay()
                graph_c.replay()
            torch.cuda.synchronize()
            base_samples = []
            cand_samples = []
            base_kernel = []
            cand_kernel = []
            for pair in range(args.pairs):
                fresh = reference.generate_input(m, layer, args.seed + 3000 + pair, device="cuda")
                static_x.copy_(fresh["x"])
                order = (
                    ((graph_b, base_samples, base_kernel), (graph_c, cand_samples, cand_kernel))
                    if pair % 2 == 0
                    else ((graph_c, cand_samples, cand_kernel), (graph_b, base_samples, base_kernel))
                )
                for graph, bucket, kernel_bucket in order:
                    raw = replay_us(graph, start, end, scratch)
                    bucket.append(max(raw - JUDGE_OVERHEAD_US, 0.05))
                    kernel_bucket.append(max(raw - overhead_us, 0.05))
            key = f"{load}|{layer}|m{m}"
            per_shape[key] = {
                "load": load,
                "layer": layer,
                "m": m,
                "graph_ok": bool(ok),
                "graph_message": message,
                "baseline_median_us": sorted(base_samples)[len(base_samples) // 2],
                "candidate_median_us": sorted(cand_samples)[len(cand_samples) // 2],
                "speedup": (sorted(base_samples)[len(base_samples) // 2]
                            / sorted(cand_samples)[len(cand_samples) // 2]),
            }
            raw_candidate.setdefault(load, {name: [] for name in LAYERS})
            raw_baseline.setdefault(load, {name: [] for name in LAYERS})
            raw_candidate[load][layer] = cand_samples
            raw_baseline[load][layer] = base_samples
            kernel_candidate.setdefault(load, {name: [] for name in LAYERS})
            kernel_baseline.setdefault(load, {name: [] for name in LAYERS})
            kernel_candidate[load][layer] = cand_kernel
            kernel_baseline[load][layer] = base_kernel
            if not ok:
                cases.append({"m": m, "layer": layer, "dist": "graph", "ok": False, "message": message})

    candidate_load_pairs = {}
    baseline_load_pairs = {}
    for load, _m in LOADS:
        candidate_load_pairs[load] = []
        baseline_load_pairs[load] = []
        for pair in range(args.pairs):
            candidate_load_pairs[load].append(sum(raw_candidate[load][layer][pair] for layer in LAYERS) * 62)
            baseline_load_pairs[load].append(sum(raw_baseline[load][layer][pair] for layer in LAYERS) * 62)

    candidate_load_us = {load: sorted(values)[len(values) // 2] for load, values in candidate_load_pairs.items()}
    baseline_load_us = {load: sorted(values)[len(values) // 2] for load, values in baseline_load_pairs.items()}

    def _kernel_loads(side):
        loads = {}
        for load, _m in LOADS:
            series = []
            for pair in range(args.pairs):
                series.append(sum(side[load][layer][pair] for layer in LAYERS) * 62)
            loads[load] = sorted(series)[len(series) // 2]
        return loads

    candidate_kernel_load_us = _kernel_loads(kernel_candidate)
    baseline_kernel_load_us = _kernel_loads(kernel_baseline)
    point = speedup_from_load_times(candidate_load_us, baseline_load_us)
    interval = bootstrap_geomean(baseline_load_pairs, candidate_load_pairs)
    floors = {load: floor_load_us(load) for load in FLOORS_US}
    fraction = geomean([floors[load] / candidate_load_us[load] for load in FLOORS_US])
    correct = all(item["ok"] for item in cases) and all(item["graph_ok"] for item in per_shape.values())
    report = {
        "source": "local_h200",
        "gpu": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "")),
        "submission": str(args.submission),
        "pairs": args.pairs,
        "overhead_us": JUDGE_OVERHEAD_US,
        "measured_empty_graph_us": overhead_us,
        "correct": correct,
        "cases": cases,
        "per_shape": per_shape,
        "candidate_load_us": candidate_load_us,
        "baseline_load_us": baseline_load_us,
        "floor_load_us": floors,
        "speedup": point,
        "ci_low": None if interval is None else interval[0],
        "ci_high": None if interval is None else interval[1],
        "geomean_us": geomean(list(candidate_load_us.values())),
        "baseline_geomean_us": geomean(list(baseline_load_us.values())),
        "geomean_kernel_us": geomean(list(candidate_kernel_load_us.values())),
        "baseline_geomean_kernel_us": geomean(list(baseline_kernel_load_us.values())),
        "candidate_kernel_load_us": candidate_kernel_load_us,
        "baseline_kernel_load_us": baseline_kernel_load_us,
        "fraction_of_floor": fraction,
        "published_sglang_us": 4830.8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "correct": correct,
        "speedup": point,
        "ci": interval,
        "geomean_us": report["geomean_us"],
        "baseline_geomean_us": report["baseline_geomean_us"],
        "geomean_kernel_us": report["geomean_kernel_us"],
        "baseline_geomean_kernel_us": report["baseline_geomean_kernel_us"],
        "fraction_of_floor": fraction,
        "overhead_us": JUDGE_OVERHEAD_US,
        "measured_empty_graph_us": overhead_us,
        "output": str(args.output),
    }, indent=2))
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

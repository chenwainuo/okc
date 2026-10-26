"""Validate retained CUDA-graph outputs while mutating one fixed input allocation.

This is a correctness check, with no timing or performance comparison. Run only
under the root scheduler's exclusive GPU handoff. --sub is loaded as a black box.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
MS = (1, 8, 32)


def capture(mod, data, state):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.kernel_fn(data, state)
        mod.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mod.kernel_fn(data, state)
    return graph, output


def unchanged(x, expected, storage_ptr, label):
    assert x.data_ptr() == storage_ptr, f"{label}: input allocation changed"
    assert torch.equal(x, expected), f"{label}: kernel modified the input"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=Path, default=Path(__file__).with_name("submission.py"))
    ap.add_argument("--seed", type=int, default=41237)
    args = ap.parse_args()
    # Set before import so the submitted default is tested even if the caller's
    # shell retains a previous LATENCY_DBG diagnostic setting.
    os.environ["LATENCY_DBG"] = "512"
    spec = importlib.util.spec_from_file_location("replay_candidate", args.sub)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"DEVICE {torch.cuda.get_device_name()} torch={torch.__version__}", flush=True)
    print(f"CANDIDATE {args.sub} LATENCY_DBG=512", flush=True)
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    start = time.time()
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    print(f"PREPARE {time.time() - start:.2f}s", flush=True)
    checks = 0
    for li, layer in enumerate(LAYERS):
        for m in MS:
            seed = args.seed + 1000 * li + m
            data = reference.generate_input(m, layer, seed)
            x = data["x"]
            storage_ptr = x.data_ptr()
            original_x = x.clone()
            graph, output = capture(mod, data, state)
            torch.cuda.synchronize()
            unchanged(x, original_x, storage_ptr, f"{layer} m={m} capture")

            graph.replay()
            torch.cuda.synchronize()
            unchanged(x, original_x, storage_ptr, f"{layer} m={m} original")
            ok, msg = reference.check_implementation(data, output, weights)
            assert ok, f"{layer} m={m} original: {msg}"
            checks += 1
            print(f"REPLAY {layer} m={m} original PASS {msg}", flush=True)

            x.zero_()
            zero_x = torch.zeros_like(x)
            graph.replay()
            torch.cuda.synchronize()
            unchanged(x, zero_x, storage_ptr, f"{layer} m={m} zero")
            assert torch.equal(output, torch.zeros_like(output)), f"{layer} m={m} zero: output is not exactly zero"
            checks += 1
            print(f"REPLAY {layer} m={m} zero PASS exact zero", flush=True)

            fresh = reference.generate_input(m, layer, seed + 500003, dist="outliers")
            fresh_x = fresh["x"].clone()
            x.copy_(fresh_x)
            # Match the new input's config/probe state while retaining the exact
            # X allocation captured above. The reference finds probes from X.
            fresh_data = {"x": x, "config": fresh["config"]}
            graph.replay()
            torch.cuda.synchronize()
            unchanged(x, fresh_x, storage_ptr, f"{layer} m={m} fresh")
            ok, msg = reference.check_implementation(fresh_data, output, weights)
            assert ok, f"{layer} m={m} fresh: {msg}"
            checks += 1
            print(f"REPLAY {layer} m={m} fresh_outliers PASS {msg}", flush=True)
            del graph, output
    assert checks == 45, checks
    print("REPLAY_VALIDATION 45/45 PASS; no performance claim", flush=True)


if __name__ == "__main__":
    main()

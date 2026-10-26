"""Two frozen versions, counter diagnosis only; no performance admission or tuning."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference

VERSIONS = {
    "v1": ("operand_reverse", "ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8", 168),
    "v2": ("operand_reverse_v2", "97664bdee7593d46e0493b9599062c37af150262d597002a722016f060548faf", 162),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "STARTED", "purpose": "COUNTERS_ONLY_NO_BENCHMARK_CLAIM", "versions": {}}
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        modules = {}
        for label, (directory, expected, regs) in VERSIONS.items():
            path = ROOT / "proj_rsi/candidates" / directory / "prototype.py"
            spec = importlib.util.spec_from_file_location("ncu_compare_" + label, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            actual = hashlib.sha256(mod.source().encode()).hexdigest()
            assert actual == expected, label + " source drift"
            ext = mod.extension()
            empty = torch.empty(0, device="cuda", dtype=torch.int64)
            sink = torch.zeros(4, device="cuda", dtype=torch.int32)
            info = lambda c: int(ext.rqrun(0, empty, empty, empty, empty, empty, 0, 0, 0, empty, sink, c))
            assert info(1) == regs and info(2) == 0 and info(4) >= 1
            modules[label] = (mod, ext, empty, sink)
            result["versions"][label] = {"source_sha256": actual, "regs": regs, "smem": info(3)}
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        w = weights["unpooler"]
        helper = modules["v1"][0].S
        packed, table, esc6 = helper._rqpack(w)
        assert torch.equal(helper._rqunpack(packed, table, esc6, *w.shape), w.view(torch.int16).int() & 65535)
        escapes, _ = helper._rqesc(esc6, packed.shape[0], packed.shape[1], 1, "cuda")
        data = reference.generate_input(128, "unpooler", 5237)
        x = data["x"].clone()
        graphs = {}
        counters = {}
        for label, (_, ext, empty, sink) in modules.items():
            out = torch.empty((128, 8192), device="cuda", dtype=torch.bfloat16)
            cnt = torch.zeros(4 * packed.shape[0], device="cuda", dtype=torch.int32)
            counters[label] = cnt

            def call(ext=ext, out=out, cnt=cnt, empty=empty, sink=sink):
                rc = int(ext.rqrun(0, packed, x, out, escapes, cnt, 1, table, 0, empty, sink, 0))
                assert rc == 0, rc

            call(); call(); torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graphs[label] = (graph, out)
        assert len({out.data_ptr() for _, out in graphs.values()}) == 2
        result["profile_order"] = list(graphs)
        result["checks"] = []
        save()
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        flush_sink = torch.empty((), device="cuda", dtype=torch.float32)
        for label, (graph, out) in graphs.items():
            out.fill_(float("nan"))
            torch.sum(flush, 0, out=flush_sink)
            torch.cuda.synchronize()
            print("PROFILE_BEGIN " + label, flush=True)
            torch.cuda.nvtx.range_push("profile")
            try:
                graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            assert torch.equal(x, data["x"]), "X changed"
            ok, message = reference.check_implementation({"x": x, "config": data["config"]}, out, {"unpooler": w})
            assert ok, (label, message)
            assert not int(counters[label].abs().sum())
            result["checks"].append({"version": label, "immediate_profiled_output_oracle": message,
                                     "NaN_overwritten": True, "input_immutable": True})
            print("PROFILE_END " + label, flush=True)
            save()
        for graph, _ in graphs.values():
            graph.reset()
        result["status"] = "COUNTER_DRIVER_COMPLETE_EXPORT_REVIEW_REQUIRED"
        save()
        print("VERSION_COUNTER_DRIVER_DONE", flush=True)
    except Exception as error:
        result.update(status="INVALID", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()

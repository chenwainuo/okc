"""One counter diagnostic of two frozen actual graph paths; no timing admission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "proj_rsi/candidates/operand_reverse"
SOURCE_SHA = "ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8"
PARENT_SHA = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
sys.path.insert(0, str(CANDIDATE))
import prototype as P
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "STARTED", "purpose": "COUNTERS_ONLY_NO_SPEED_CLAIM",
              "source_sha256": hashlib.sha256(P.source().encode()).hexdigest()}
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        assert result["source_sha256"] == SOURCE_SHA
        parent = CANDIDATE.parent / "pw36/submission.py"
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA
        spec = importlib.util.spec_from_file_location("reverse_ncu_parent", parent)
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        ext = P.extension()
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        sink = torch.zeros(4, dtype=torch.int32, device="cuda")
        info = lambda k: int(ext.rqrun(0, empty, empty, empty, empty, empty, 0, 0, 0, empty, sink, k))
        assert info(1) == 168 and info(2) == 0 and info(4) >= 1
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        state = base.prepare(weights)
        w = weights["unpooler"]
        packed, table, escapes = P.S._rqpack(w)
        assert torch.equal(P.S._rqunpack(packed, table, escapes, *w.shape), w.view(torch.int16).int() & 65535)
        escapes, _ = P.S._rqesc(escapes, packed.shape[0], packed.shape[1], 1, "cuda")
        count = torch.zeros(4 * packed.shape[0], dtype=torch.int32, device="cuda")
        data = reference.generate_input(128, "unpooler", 5237)
        x = data["x"].clone()
        output = torch.empty((128, 8192), dtype=torch.bfloat16, device="cuda")

        def old():
            return base.kernel_fn({"x": x, "config": data["config"]}, state)

        def new():
            rc = int(ext.rqrun(0, packed, x, output, escapes, count, 1, table, 0, empty, sink, 0))
            assert rc == 0, rc
            return output

        graphs = {}
        for name, fn in (("baseline", old), ("candidate", new)):
            fn(); fn(); torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output = fn()
            graphs[name] = (graph, captured_output)
        rp = state.get("rq", {}).get(("unpooler", 128))
        result["baseline_pick"] = ({"kind": "rq", "variant": base.RQPROD[rp["i"]], "KS": rp["KS"]}
                                   if rp else {"kind": "kernel_fn", "Lt_code": state.get("lt", {}).get(("unpooler", 128))})
        result["parent_sha256"] = PARENT_SHA
        result["profile_order"] = list(graphs)
        result["checks"] = []
        save()
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        flush_sink = torch.empty((), device="cuda", dtype=torch.float32)

        def oracle(name, out, tag):
            torch.cuda.synchronize()
            assert torch.equal(x, data["x"]), "Input mutation"
            ok, message = reference.check_implementation({"x": x, "config": data["config"]}, out, {"unpooler": w})
            assert ok, message
            result["checks"].append({"path": name, "stage": tag, "oracle": message, "input_immutable": True})

        for name, (graph, out) in graphs.items():
            out.fill_(float("nan"))
            torch.sum(flush, 0, out=flush_sink)
            torch.cuda.synchronize()
            print("PROFILE_BEGIN " + name, flush=True)
            torch.cuda.nvtx.range_push("profile")
            try:
                graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            # Inspect precisely this profiled replay, before any further replay.
            oracle(name, out, "immediately_after_profiled_graph")
            print("PROFILE_END " + name, flush=True)
            save()
        result["status"] = "DIAGNOSTIC_COMPLETE"
        save()
        print("NCU_DRIVER_DONE counters require separate exported-report review", flush=True)
    except Exception as error:
        result.update(status="FAILED", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()

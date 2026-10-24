"""Rich local diagnostics; never used in clean timing or submitted serving code."""
from contextlib import contextmanager, ExitStack
import functools
import importlib
import json
from pathlib import Path
from unittest.mock import patch


def routing_summary(data):
    import torch
    cfg = data["config"]
    ids = data["topk_ids"].detach().cpu().long()
    local = ids - cfg["local_start"]
    valid = (local >= 0) & (local < cfg["local_count"])
    hist = torch.bincount(local[valid], minlength=cfg["local_count"])
    return {"tokens": ids.shape[0], "route_slots": ids.numel(),
            "local_routes": int(valid.sum()), "local_fraction": float(valid.float().mean()),
            "tokens_without_local_routes": int((~valid.any(dim=1)).sum()),
            "distinct_local_experts": int((hist > 0).sum()),
            "routes_per_local_expert": hist.tolist()}


def gpu_snapshot():
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


@contextmanager
def stage_ranges(module, state):
    """Temporarily annotate eager calls, restoring all bindings on exit.

    Python stage attribution is diagnostic only. CUDA graph traces separately
    show the kernels actually executed in timed graph replays.
    """
    import torch
    targets = {
        "moe_align_block_size": ("sglang.srt.layers.moe.fused_moe_triton", "alignment"),
        "moe_wna16_marlin_gemm": ("sglang.kernels.ops.moe.moe_wna16_marlin", "gemm"),
        "silu_and_mul": ("sglang.kernels.ops.activation.activation", "swiglu"),
        "fused_post_expert_norm_weighted_sum": (
            "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton", "finalize"),
        "marlin_make_workspace": ("sglang.srt.layers.quantization.marlin_utils", "workspace"),
    }

    def wrap(fn, label):
        @functools.wraps(fn)
        def instrumented(*args, **kwargs):
            stage = label
            if label == "gemm":
                stage = "gemm_gate_up" if kwargs.get("top_k", 1) > 1 else "gemm_down"
            with torch.profiler.record_function("okc_stage/" + stage):
                return fn(*args, **kwargs)
        return instrumented

    bindings = [module.kernel_fn.__globals__]
    if hasattr(state, "_forward_marlin"):
        bindings.append(state._forward_marlin.__func__.__globals__)
    with ExitStack() as stack:
        for name, (path, label) in targets.items():
            for namespace in bindings:
                if name in namespace:
                    stack.enter_context(patch.dict(namespace, {name: wrap(namespace[name], label)}))
            owner = importlib.import_module(path)
            stack.enter_context(patch.object(owner, name, wrap(getattr(owner, name), label)))
        yield


def save_profile(prof, prefix):
    prefix = Path(prefix)
    prof.export_chrome_trace(str(prefix) + ".trace.json")
    averages = prof.key_averages()
    Path(str(prefix) + ".txt").write_text(averages.table(
        sort_by="self_device_time_total", row_limit=80) + "\n")
    rows = [{"name": row.key, "count": row.count,
             "device_type": str(row.device_type),
             "is_user_annotation": row.is_user_annotation,
             "self_cpu_us": row.self_cpu_time_total,
             "cpu_total_us": row.cpu_time_total,
             "device_total_us": row.device_time_total,
             "self_device_us": row.self_device_time_total} for row in averages]
    Path(str(prefix) + ".events.json").write_text(json.dumps(rows, indent=2) + "\n")

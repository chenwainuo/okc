"""Cost ladder: time the kernel in modes 1 (stream only), 2 (+LDS+decode), 3 (+wgmma), 0 (full) for given shapes."""
import sys, os, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, reference
from bench_util import time_graph, flush_buffer
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["qkv_gate"]
ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 16, 128, 256]
modes = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2, 3, 0]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 20
weights = reference.generate_weights(1234)
state = mod.prepare(weights); torch.cuda.synchronize()
flush = flush_buffer()
tiny = torch.zeros(32, device="cuda")
print(f"REPLAY_OVERHEAD {time_graph(lambda: tiny.add_(1.0), flush, reps):.2f} us")
for layer in layers:
    for m in ms:
        data = reference.generate_input(m, layer, 1234 + m, dist="normal")
        n = data["config"]["n"]
        x = data["x"]
        if layer == "router":
            w = weights[layer]; hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16); hi_t, lo_t = hi.t().contiguous(), lo.t().contiguous()
            def tfn():
                acc = torch.mm(x, hi_t, out_dtype=torch.float32); acc += torch.mm(x, lo_t, out_dtype=torch.float32); return acc
        else:
            wt = weights[layer][:n].t().contiguous()
            def tfn(): return torch.mm(x, wt)
        tb = time_graph(tfn, flush, reps)
        row = f"{layer:9s} m={m:<4d} torch {tb:7.2f}"
        for mode in modes:
            mod.MODE = mode
            t = time_graph(lambda: mod.kernel_fn(data, state), flush, reps)
            row += f" | mode{mode} {t:7.2f}"
        mod.MODE = 0
        out = mod.kernel_fn(data, state); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, weights)
        print(row + f" | {'ok' if ok else 'FAIL'}", flush=True)

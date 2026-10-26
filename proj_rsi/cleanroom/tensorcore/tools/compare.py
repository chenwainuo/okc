"""Time several submission variants on the same shapes in one GPU-lock session (graph replay, L2 flush)."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, reference
from bench_util import time_graph, flush_buffer
subs = sys.argv[1].split(",")
shapes = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2].split(",")]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
weights = reference.generate_weights(1234)
mods, states = [], []
for p in subs:
    spec = importlib.util.spec_from_file_location("sub_" + Path(p).stem, p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mods.append(mod); states.append(mod.prepare(weights))
torch.cuda.synchronize()
flush = flush_buffer()
tiny = torch.zeros(32, device="cuda")
print(f"REPLAY_OVERHEAD {time_graph(lambda: tiny.add_(1.0), flush, reps):.2f} us")
print("shape            torch   " + "  ".join(f"{Path(p).stem[:12]:>12s}" for p in subs))
for layer, m in shapes:
    data = reference.generate_input(m, layer, 1234 + m, dist="normal")
    n = data["config"]["n"]; x = data["x"]
    if layer == "router":
        w = weights[layer]; hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16); hi_t, lo_t = hi.t().contiguous(), lo.t().contiguous()
        def tfn():
            acc = torch.mm(x, hi_t, out_dtype=torch.float32); acc += torch.mm(x, lo_t, out_dtype=torch.float32); return acc
    else:
        wt = weights[layer][:n].t().contiguous()
        def tfn(): return torch.mm(x, wt)
    row = f"{layer:9s} m={m:<4d} {time_graph(tfn, flush, reps):7.2f} "
    for mod, st in zip(mods, states):
        t = time_graph(lambda: mod.kernel_fn(data, st), flush, reps)
        out = mod.kernel_fn(data, st); torch.cuda.synchronize()
        ok = reference.check_implementation(data, out, weights)[0]
        row += f"  {t:10.2f}{'' if ok else '!'} "
    print(row, flush=True)

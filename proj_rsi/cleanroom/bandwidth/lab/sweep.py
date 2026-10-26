"""Quick config sweep (run under the GPU lock): python sweep.py <layer> <m,...> <variant;variant...> [reps]
variant = key=val,key=val  with keys S,G,D,MT,NB2,kind(small|big|rfma|torch).  Prints graph-replay median us."""
import sys, importlib.util, statistics
sys.path.insert(0, "/okc/muse_bf16_proj"); sys.path.insert(0, "/okc/proj_rsi/tools")
import torch, reference
from h100_bench import time_graph, flush_buffer, torch_baseline
spec = importlib.util.spec_from_file_location("submission", "/okc/proj_rsi/cleanroom/bandwidth/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layer, ms, variants = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3].split(";")
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 15
w = reference.generate_weights(1234); st = mod.prepare(w); ext = mod._ext()
flush = flush_buffer()
tiny = torch.zeros(32, device="cuda")
print(f"REPLAY_OVERHEAD {time_graph(lambda: tiny.add_(1.0), flush, reps):.2f} us")
for m in ms:
    data = reference.generate_input(m, layer, 1234 + m); x = data["x"]; N, K = data["config"]["n"], data["config"]["k"]
    L = st[layer]; router = layer == "router"
    tb = time_graph(torch_baseline(layer, data, w), flush, reps)
    print(f"{layer} m={m} torch {tb:.2f} us  sol(H100) {reference.speed_of_light_ns(m, layer, {'hbm_bytes_per_s': 3.35e12, 'bf16_dense_flops_per_s': 989e12})/1e3:.2f}")
    for v in variants:
        kv = dict(p.split("=") for p in v.split(",") if p)
        kind = kv.pop("kind", "small" if m <= 32 else "big")
        c = {k: int(val) for k, val in kv.items()}
        if kind == "small":
            MT = c.get("MT", 1 if m <= 16 else 2); S = c.get("S", 4); D = c.get("D", 4)
            out = torch.empty((m, N), dtype=torch.float32 if router else torch.bfloat16, device="cuda")
            fn = lambda: ext.small_launch(L["packed"], x, out, L["esc_off"], L["esc"], m, K, N, L["base2"], S, MT, router, D)
        elif kind == "rfma":
            S = c.get("S", 8); D = c.get("D", 4)
            out = torch.empty((m, N), dtype=torch.float32, device="cuda")
            fn = lambda: ext.rfma_launch(L["packed"], x, out, L["esc_off"], L["esc"], m, K, N, L["base2"], S, D)
        elif kind == "big":
            MT = c.get("MT", 4 if m <= 64 else (8 if m <= 128 else 16)); S = c.get("S", 4); G = c.get("G", 1); NB2 = c.get("NB2", 1 if router else 2); D = c.get("D", 12)
            out = torch.empty((m, N), dtype=torch.float32 if router else torch.bfloat16, device="cuda")
            fn = lambda: ext.big_launch(L["packed"], x, out, L["esc_off"], L["esc"], L["scratch"], L["counters"], m, K, N, L["base2"], S, G, MT, NB2, router, D)
        else:
            fn = lambda: mod.kernel_fn(data, st)
        try:
            fn(); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, fn() if kind == "torch" else out, w)
            t = time_graph(fn, flush, reps)
            print(f"   {v:40s} {t:8.2f} us  x_torch {tb/t:.3f}  {'ok' if ok else 'FAIL ' + msg}")
        except Exception as ex:
            print(f"   {v:40s} EXC {str(ex)[:200]}")

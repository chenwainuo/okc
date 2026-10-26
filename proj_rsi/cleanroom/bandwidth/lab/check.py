"""Correctness only (no GPU lock needed): prepare, then check_implementation on the given (layer, m) cases."""
import sys, time, importlib.util
sys.path.insert(0, "/okc/muse_bf16_proj")
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
w = reference.generate_weights(1234)
t0 = time.time(); st = mod.prepare(w); torch.cuda.synchronize(); print(f"PREPARE {time.time()-t0:.1f}s", flush=True)
for name in ("qkv_gate", "o", "pooler", "unpooler", "router"):
    print(name, "base", st[name]["base"], "escapes", st[name]["n_esc"])
cases = sys.argv[2] if len(sys.argv) > 2 else "all"
if cases == "all":
    cases = [(l, m, d) for l in ("qkv_gate", "o", "pooler", "unpooler", "router") for m in (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256) for d in ("normal",)]
    cases += [("router", 9, "outliers"), ("qkv_gate", 512, "outliers"), ("unpooler", 1000, "outliers"), ("unpooler", 5, "normal")]
else:
    cases = [(c.split(":")[0], int(c.split(":")[1]), "normal") for c in cases.split(",")]
bad = 0
for layer, m, dist in cases:
    data = reference.generate_input(m, layer, 1234 + m, dist=dist)
    try:
        out = mod.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
    except Exception as ex:
        ok, msg = False, f"EXC {type(ex).__name__}: {str(ex)[:300]}"
    bad += (not ok)
    print(f"CHECK {layer:9s} m={m:<5d} {dist:8s} {'ok ' if ok else 'FAIL'} {msg}", flush=True)
print("ALL OK" if not bad else f"{bad} FAILURES")

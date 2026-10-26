import sys, importlib.util
sys.path.insert(0, "/okc/muse_bf16_proj")
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
w = reference.generate_weights(1234); st = mod.prepare(w)
for layer, m in (("pooler", 16), ("unpooler", 32), ("qkv_gate", 8)):
    data = reference.generate_input(m, layer, 1234 + m)
    out = mod.kernel_fn(data, st); torch.cuda.synchronize()
    ref = reference.ref_kernel(data, w)
    rows = reference.probe_rows(data["x"])
    x = data["x"]
    for r in rows.tolist():
        k0 = int((x[r] != 0).nonzero().flatten()[0]); xv = float(x[r, k0])
        o, rf = out[r].float(), ref[r].float()
        bad = (o != rf).nonzero().flatten()
        exact_col = (w[layer][:out.shape[1], k0].float() * xv)
        print(layer, m, "row", r, "k0", k0, "x", xv, "mismatches", bad.numel(), [(int(b), float(o[b]), float(rf[b]), float(exact_col[b])) for b in bad[:4].tolist()])

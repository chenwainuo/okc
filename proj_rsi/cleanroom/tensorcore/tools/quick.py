"""Quick correctness run: prepare + check a few shapes (no timing)."""
import sys, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["qkv_gate", "o", "pooler", "unpooler"]
ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 8, 16, 32, 128, 256]
weights = reference.generate_weights(1234)
t0 = time.time(); state = mod.prepare(weights); torch.cuda.synchronize(); print(f"PREPARE {time.time()-t0:.1f}s", flush=True)
for layer in layers:
    for m in ms:
        data = reference.generate_input(m, layer, 1234 + m, dist="normal")
        out = mod.kernel_fn(data, state); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, weights)
        print(f"CHECK {layer:9s} m={m:<4d} {'ok ' if ok else 'FAIL'} {msg}", flush=True)

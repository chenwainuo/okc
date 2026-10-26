"""contrarian: correctness on every judge m for every layer (task checker), no timing."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj")); sys.path.insert(0, str(ROOT / "proj_rsi/tools"))
import reference
from h100_bench import load_submission

weights = reference.generate_weights(1234)
mod = load_submission(Path(__file__).with_name("submission.py"))
state = mod.prepare(weights)
fails = 0
for layer in ["qkv_gate", "o", "pooler", "unpooler", "router"]:
    for m, dist in [(1, "normal"), (5, "normal"), (7, "normal"), (8, "normal"), (9, "outliers"), (16, "normal"), (32, "normal"),
                    (33, "normal"), (128, "normal"), (130, "normal"), (256, "normal"), (512, "outliers"), (1000, "outliers")]:
        data = reference.generate_input(m, layer, 1234 + m, dist=dist)
        out = mod.kernel_fn(data, state)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, weights)
        fails += (not ok)
        print(f"CHECK {layer:9s} m={m:<5d} {dist:8s} {'ok ' if ok else 'FAIL'} {msg}", flush=True)
print("FAILS", fails)

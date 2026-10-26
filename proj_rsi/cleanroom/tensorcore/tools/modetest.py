"""Eager mode test: run each ladder mode a few times with progress prints (hang diagnosis)."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layer, m = sys.argv[2], int(sys.argv[3])
modes = [int(v) for v in sys.argv[4].split(",")]
weights = reference.generate_weights(1234)
state = mod.prepare(weights); torch.cuda.synchronize()
data = reference.generate_input(m, layer, 1234 + m, dist="normal")
for mode in modes:
    mod.MODE = mode
    for rep in range(3):
        out = mod.kernel_fn(data, state); torch.cuda.synchronize()
        print(f"mode {mode} rep {rep} done", flush=True)
    if mode == 0:
        print(reference.check_implementation(data, out, weights), flush=True)

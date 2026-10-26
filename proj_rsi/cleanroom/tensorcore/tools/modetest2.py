import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
weights = reference.generate_weights(1234)
state = mod.prepare(weights); torch.cuda.synchronize()
data = reference.generate_input(16, "qkv_gate", 1250, dist="normal")
ws = state["layers"]["qkv_gate"]["ws"]
for mode in [4, 0, 3, 0]:
    mod.MODE = mode
    out = mod.kernel_fn(data, state); torch.cuda.synchronize()
    print(f"mode {mode}: ws max {float(ws.abs().max()):.4f} counters {state['layers']['qkv_gate']['counters'].tolist()[:4]}", reference.check_implementation(data, out, weights)[1][:40], flush=True)

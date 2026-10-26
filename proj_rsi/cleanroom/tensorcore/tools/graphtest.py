"""Graph-replay hang diagnosis: capture kernel_fn as a CUDA graph and replay it a few times with progress prints."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import torch, reference
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layer, m = sys.argv[2], int(sys.argv[3]); modes = [int(v) for v in sys.argv[4].split(",")]
weights = reference.generate_weights(1234)
state = mod.prepare(weights); torch.cuda.synchronize()
data = reference.generate_input(m, layer, 1234 + m, dist="normal")
for mode in modes:
    mod.MODE = mode
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): mod.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); print(f"mode {mode} warmup ok", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod.kernel_fn(data, state)
    torch.cuda.synchronize(); print(f"mode {mode} captured", flush=True)
    for i in range(5):
        g.replay(); torch.cuda.synchronize(); print(f"mode {mode} replay {i} ok", flush=True)
    if mode == 0: print(reference.check_implementation(data, out, weights), flush=True)

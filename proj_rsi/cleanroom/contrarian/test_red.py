"""contrarian: focused correctness test of split-K configs. python test_red.py layer m cfg1 cfg2 ..."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj")); sys.path.insert(0, str(ROOT / "proj_rsi/tools"))
import reference
from h100_bench import load_submission  # noqa

def main():
    layer, m = sys.argv[1], int(sys.argv[2])
    cfgs = [tuple(int(v) for v in c.split(",")) for c in sys.argv[3:]]
    weights = reference.generate_weights(1234)
    mod = load_submission(Path(__file__).with_name("submission.py"))
    mod._BAKE = 0
    state = mod.prepare(weights)
    data = reference.generate_input(m, layer, 1234 + m)
    n = data["config"]["n"]
    x = data["x"]
    for cfg in cfgs:
        for rep in range(3):
            out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device="cuda")
            try:
                mod._launch(x, layer, n, state, out, cfg)
                torch.cuda.synchronize()
            except Exception as e:  # noqa
                print("cfg", cfg, "rep", rep, "EXC", e); break
            ok, msg = reference.check_implementation(data, out, weights)
            cnt = state["counters"][:512]
            print("cfg", cfg, "rep", rep, "ok" if ok else "FAIL", msg[:70], "| nonzero counters:", int((cnt != 0).sum()), flush=True)

if __name__ == "__main__":
    main()

"""Official nine cases on a CPU model of the packed weights and Python dispatch.

This does not execute or validate CUDA indexing, barriers, graph capture, or timing.
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402

CASES = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"),
         (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"),
         (9, "router", "outliers"), (512, "qkv_gate", "outliers"),
         (1000, "unpooler", "outliers")]


def decode_plane(p, pl):
    nb, packed = p["NB"], p["packed"]
    nc, ks, _ = packed.shape
    b = packed[:, :, pl * 8 * nb * 96:(pl + 1) * 8 * nb * 96]
    low = b[:, :, :8 * nb * 64].to(torch.int32)
    hbyte = b[:, :, 8 * nb * 64:].to(torch.int32)
    nib = torch.stack((hbyte & 15, hbyte >> 4), -1).flatten(2)
    hb = p["lut"][2 * pl] & 255
    bits = low | (((nib & 7) + hb) << 8) | ((nib >> 3) << 15)
    w = bits.to(torch.int16).view(torch.bfloat16).reshape(nc, ks, 8, nb, 4, 2, 8)
    return w.permute(0, 3, 1, 2, 5, 4, 6).reshape(p["N"], p["K"]).float()


class CpuPackedModel:
    def __init__(self):
        self.weights = {}

    def add(self, p, original):
        planes = []
        for pl, (ni, ki, delta) in enumerate(p["exc_parts"]):
            w = decode_plane(p, pl)
            w[ni, ki] += delta
            if p["mode"] == 0:
                target = original
            else:
                hi = original.to(torch.bfloat16)
                target = hi if pl == 0 else (original - hi.float()).to(torch.bfloat16)
            assert torch.equal(w.to(torch.bfloat16), target), "BF16 reconstruction differs"
            planes.append(w)
        self.weights[p["packed"].data_ptr()] = sum(planes)

    def latency_gemv_w_run(self, packed, x, excw, ecap, ksplit, nb, n, nstages, s,
                           lut0, lut1, lut2, lut3, smem, mode, dbg):
        w = self.weights[packed.data_ptr()]
        y = x.float() @ w.T
        return y if mode else y.to(torch.bfloat16)


def main():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("latency_submission", Path(__file__).with_name("submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = CpuPackedModel()
    mod._ext = lambda: model
    weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cpu")
    state = mod.prepare(weights)
    for key in ("qkv_gate", "pooler", "unpooler_slice", "router"):
        nb = mod.PLAN[key][0][1]
        model.add(state["packs"][(key, nb)], mod._src_weight(weights, key))
        print(f"RECONSTRUCT {key}: bitwise BF16 planes pass", flush=True)
    for i, (m, layer, dist) in enumerate(CASES):
        data = reference.generate_input(m, layer, 100 + i, dist=dist, device="cpu")
        out = mod.kernel_fn(data, state)
        ok, msg = reference.check_implementation(data, out, weights)
        print(f"CPU_MODEL {layer} m={m} dist={dist}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
        if not ok:
            raise AssertionError(msg)
    print("CPU_MODEL official 9/9 PASS; CUDA execution remains untested", flush=True)


if __name__ == "__main__":
    main()

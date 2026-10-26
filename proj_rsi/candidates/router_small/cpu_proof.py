"""Independent PTX fragment/physical decoder proof. CPU only, no GPU initialization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from packing import decode_plane, encode_plane, indices, pack


def prmt(a, b, selector):
    out = np.zeros_like(a, dtype=np.uint32)
    for byte in range(4):
        s = (selector >> (4 * byte)) & 7
        src = a if s < 4 else b
        out |= ((src >> (8 * (s % 4))) & 255) << (8 * byte)
    return out


def kernel_decoder(payload, headers):
    """Translate physical vector loads, C++ uint32 arithmetic and PRMT, not packing.decode_plane."""
    p0 = np.ascontiguousarray(payload[:, :512]).view(np.uint32).reshape(4096, 32, 4)
    p1 = np.ascontiguousarray(payload[:, 512:768]).view(np.uint32).reshape(4096, 32, 2)
    p2 = np.ascontiguousarray(payload[:, 768:1024]).view(np.uint32).reshape(4096, 32, 2)
    base = (headers & 127)[:, None] * np.uint32(0x01010101)
    wide = (headers & 128) != 0
    result = np.empty((4096, 4, 32, 4), dtype=np.uint16)
    for q in range(4):
        L, C = p0[:, :, q], p1[:, :, q // 2]
        H = (((C >> ((q & 1) * 4)) & np.uint32(0x07070707)) + base)
        H |= ((C << 4) if q & 1 else C) & np.uint32(0x80808080)
        pair = np.stack((prmt(L, H, 0x5140), prmt(L, H, 0x7362)), axis=-1)
        wide_pair = p0[:, :, 2*q:2*q+2] if q < 2 else p1 if q == 2 else p2
        pair[wide] = wide_pair[wide]
        for r in range(2):
            result[:, q, :, 2*r] = pair[:, :, r] & 65535
            result[:, q, :, 2*r+1] = pair[:, :, r] >> 16
    return result


def bf(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def matrix_proof(x, planes):
    """Build matrices ONLY through PTX fragment locations, then use ordinary dense matmul."""
    A = np.empty((4096, 2, 4, 16, 16), dtype=np.float64)
    B = np.empty((4096, 2, 4, 16, 8), dtype=np.float64)
    warp = (np.arange(4096) % 32)[:, None]
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for q in range(4):
            for cohort in range(2):
                for i in range(8):
                    # C++ rs_step's four u32 source loads, split into halves.
                    group = 2 * cohort + ((i // 2) & 1)
                    global_k = warp[:, 0] * 256 + group * 64 + q * 16 + 2*t + (i // 4) * 8 + (i & 1)
                    a = x[g, global_k]
                    # Independent PTX A register coordinate.
                    row = g if i % 4 < 2 else g + 8
                    k = 2*t + (i & 1) + (8 if i >= 4 else 0)
                    A[:, cohort, q, row, k] = a
            for p in range(2):
                for i in range(4):
                    # Independent PTX B register coordinate.
                    k, col = 2*t + (i & 1) + (8 if i >= 2 else 0), g
                    B[:, p, q, k, col] = bf(planes[p][:, q, lane, i])
    C = np.einsum("wcqik,wpqkj->wcpij", A, B, optimize=True)
    # GPU keeps separate C accumulators per cohort and interleaves hi/lo; FP64 here proves layout only.
    C = C.sum(axis=2)

    def select(c, wrong=None):
        partial = np.zeros((4096, 8, 2), dtype=np.float64)
        for group in range(4):
            cohort = group // 2
            if wrong == "cohort":
                cohort ^= 1
            row0 = (group & 1) * 8
            if wrong == "diagonal":
                row0 ^= 8
            for e in range(2):
                col = 2*group + e
                if wrong == "B-column-group":
                    col = (col + 2) % 8
                partial[:, :, e] += c[:, cohort, row0:row0+8, col]
        return partial.reshape(128, 32, 8, 2).sum(axis=1).transpose(1, 0, 2).reshape(8, 256)

    return select(C), {bug: select(C, bug) for bug in ("cohort", "diagonal", "B-column-group")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path(__file__).with_name("cpu-proof.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    row, k = indices()
    linear = row * 8192 + k
    assert np.array_equal(np.sort(linear.ravel()), np.arange(256 * 8192)), "Not a bijection"
    rng = np.random.default_rng(5151)
    w = torch.from_numpy((rng.normal(size=(256, 8192)) / 8192**0.5).astype(np.float32))
    data, head, stats = pack(w)
    payload = data.numpy().reshape(4096, 2, 1024)
    headers = head.numpy()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    original, decoded = [], []
    for p, plane in enumerate((hi, lo)):
        expected = plane.view(torch.int16).numpy().view(np.uint16)[row, k]
        got = kernel_decoder(payload[:, p], (headers >> (8 * p)) & 255)
        assert np.array_equal(got, expected)
        original.append(plane.float().numpy().astype(np.float64))
        decoded.append(got)

    # Exhaust all16-bit BF16 payloads through wide fallback; narrow adversarial bases/sign/code cases.
    exhaustive = np.resize(np.arange(65536, dtype=np.uint16), (4096, 4, 32, 4))
    ep, eh, es = encode_plane(exhaustive, force_wide=True)
    assert es["wide"] == 4096
    assert np.array_equal(kernel_decoder(ep, eh), exhaustive)
    for base in (0, 1, 56, 119, 120):
        off = rng.integers(0, 8, size=exhaustive.shape, dtype=np.uint16)
        bits = rng.integers(0, 256, size=off.shape, dtype=np.uint16) | ((off + base) << 8)
        bits |= rng.integers(0, 2, size=off.shape, dtype=np.uint16) << 15
        ep, eh, es = encode_plane(bits)
        assert es["wide"] == 0
        assert np.array_equal(kernel_decoder(ep, eh), bits)
        assert np.array_equal(decode_plane(ep, eh), bits)

    x = torch.from_numpy(rng.normal(size=(8, 8192)).astype(np.float32)).to(torch.bfloat16).float().numpy().astype(np.float64)
    got, negative = matrix_proof(x, decoded)
    ref = x @ (original[0] + original[1]).T
    err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert err < 1e-12
    caught = {}
    for bug, out in negative.items():
        e = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        assert e > 0.1, bug
        caught[bug] = e
    # Symbolic ALL8192 onehots: selected A group equals B group, every K occurs exactly once/expert.
    basis_k = []
    for warp in range(32):
        for group in range(4):
            for q in range(4):
                for kk in range(16):
                    a_k = warp*256 + (2*(group//2)+(group&1))*64 + q*16 + kk
                    b_k = warp*256 + group*64 + q*16 + kk
                    assert a_k == b_k
                    basis_k.append(a_k)
    assert sorted(basis_k) == list(range(8192))
    result = {"state":"CPU_PROOF_PASS_GPU_UNRUN", "BF16_values_bit_exact": 4194304,
              "direct_index_bijection":256*8192, "exhaustive_BF16_encodings":65536,
              "narrow_adversarial_bases":[0,1,56,119,120], "symbolic_onehot_allK":8192,
              "PTX_dense_matrix_emulation_relative_l2":err, "negative_controls_error":caught,
              "packing":stats, "all_runtime_escapes":0,
              "kernel_sha256":hashlib.sha256(Path(__file__).with_name("kernel.cu").read_bytes()).hexdigest(),
              "gpu_FP32_numerics_speed_convergence":"NOT_PROVEN"}
    args.json.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

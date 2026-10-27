"""CPU-only reversible representation proof/local-pattern observer; no ILC predictor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

BASE = 0x3C00
MASK15 = 0x7FFF


def encode(u):
    u = np.asarray(u, dtype=np.uint32)
    d = ((u & MASK15) - BASE) & MASK15
    zz = ((d << 1) ^ (np.uint32(0) - (d >> 14))) & MASK15
    return ((zz << 1) | (u >> 15)).astype(np.uint16)


def decode(e):
    e = np.asarray(e, dtype=np.uint32)
    zz = e >> 1
    d = (zz >> 1) ^ (np.uint32(0) - (zz & 1))
    return (((d + BASE) & MASK15) | ((e & 1) << 15)).astype(np.uint16)


def decode_pair(e):
    """Two independent u16s in one u32; unsigned15 sum cannot carry across halfwords."""
    z = (e >> 1) & np.uint32(0x7FFF7FFF)
    d = ((z >> 1) & np.uint32(0x3FFF3FFF)) ^ ((z & np.uint32(0x00010001)) * np.uint32(0x7FFF))
    mag = (d + np.uint32(BASE * 0x00010001)) & np.uint32(0x7FFF7FFF)
    return mag | ((e & np.uint32(0x00010001)) << 15)


def encode_xor(u):
    u = np.asarray(u, dtype=np.uint32)
    return ((((u & MASK15) ^ BASE) << 1) | (u >> 15)).astype(np.uint16)


def decode_xor(e):
    e = np.asarray(e, dtype=np.uint32)
    return (((e >> 1) ^ BASE) | ((e & 1) << 15)).astype(np.uint16)


def decode_xor_pair(e):
    mag = ((e >> 1) & np.uint32(0x7FFF7FFF)) ^ np.uint32(BASE * 0x00010001)
    return mag | ((e & np.uint32(0x00010001)) << 15)


def planes(u):
    """64 scalar weights/128B fixed tile: 64 low bytes followed by 64 high bytes."""
    b = np.asarray(u, dtype='<u2').view(np.uint8).reshape(-1, 64, 2)
    return b.transpose(0, 2, 1).copy().reshape(-1)


def unplanes(b):
    return b.reshape(-1, 2, 64).transpose(0, 2, 1).copy().reshape(-1).view('<u2')


def entropy(u, bins):
    c = np.bincount(u.reshape(-1), minlength=bins)
    p = c[c > 0].astype(np.float64) / u.size
    return float(-(p * np.log2(p)).sum())


def physical_patterns(b):
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    sectors = b.reshape(-1, 32)
    sorted_b = np.sort(sectors, axis=1)
    unique = 1 + np.count_nonzero(sorted_b[:, 1:] != sorted_b[:, :-1], axis=1)
    words = b.view('<u2').reshape(-1, 16)
    bit_or = np.bitwise_or.reduce(words, axis=1).astype(np.uint32)
    width = np.where(bit_or == 0, 0, np.floor(np.log2(np.maximum(bit_or, 1))) + 1)
    return {
        'bytes': int(b.size),
        'sha256': hashlib.sha256(b.tobytes()).hexdigest(),
        'byte_symbol_entropy_bits': entropy(b, 256),
        'zero_byte_fraction': float(np.mean(b == 0)),
        'all_zero_32B_sector_fraction': float(np.mean(np.all(sectors == 0, axis=1))),
        'constant_byte_32B_sector_fraction': float(np.mean(unique == 1)),
        'unique_bytes_per_32B_sector_mean': float(unique.mean()),
        'u16_common_zero_high_bits_per_32B_sector_mean': float((16 - width).mean()),
        'all_u16_fit_unsigned8_32B_sector_fraction': float(np.mean(bit_or < 256)),
        'interpretation': 'Physical-byte local-pattern summaries only; not NVIDIA codec formats or ratio predictions',
    }


def main():
    assert sys.byteorder == 'little'
    all_u = np.arange(65536, dtype=np.uint32)
    all_e = encode(all_u)
    assert np.unique(all_e).size == 65536
    assert np.array_equal(decode(all_e), all_u.astype(np.uint16))
    assert np.array_equal(encode(decode(all_u)), all_u.astype(np.uint16))
    # Full coverage of each half's values, including every NaN/inf/signed-zero pattern.
    hi = (all_u * 40503 + 7919) & 65535
    pairs = all_u | (hi << 16)
    ep = encode(all_u).astype(np.uint32) | (encode(hi).astype(np.uint32) << 16)
    assert np.array_equal(decode_pair(ep), pairs)
    xor_all = encode_xor(all_u)
    assert np.unique(xor_all).size == 65536 and np.array_equal(decode_xor(xor_all), all_u.astype(np.uint16))
    assert np.array_equal(encode_xor(decode_xor(all_u)), all_u.astype(np.uint16))
    xp = encode_xor(all_u).astype(np.uint32) | (encode_xor(hi).astype(np.uint32) << 16)
    assert np.array_equal(decode_xor_pair(xp), pairs)
    assert 0x7FFF + BASE < 0x10000  # no inter-half carry in the packed add
    n = 1 << 22
    torch.set_num_threads(2)
    gen = torch.Generator(device='cpu').manual_seed(5151)
    # Same unpooler marginal distribution, NOT the GPU RNG's exact weight payload.
    w = (torch.randn(n, generator=gen, device='cpu') / 64).to(torch.bfloat16)
    u = (w.view(torch.int16).numpy().astype(np.uint32) & 65535).astype(np.uint16)
    e = encode(u)
    ex = encode_xor(u)
    assert np.array_equal(decode(e), u)
    assert np.array_equal(unplanes(planes(u)), u)
    assert np.array_equal(decode(unplanes(planes(e))), u)
    assert np.array_equal(decode_xor(ex), u)
    assert np.array_equal(decode_xor(unplanes(planes(ex))), u)
    z = e.astype(np.uint32)
    reps = {'native16': u.view(np.uint8), 'centered_zigzag16': e.view(np.uint8), 'signlow_xor16': ex.view(np.uint8),
            'native16_byteplanes64': planes(u), 'centered_zigzag16_byteplanes64': planes(e)}
    result = {
        'state': 'CPU_PROOF_ONLY_NO_CUDA_NO_COMPRESSION_RATIO',
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'BASE': BASE,
        'proof': {'all_65536_patterns_bijection': True, 'both_roundtrips': True,
                  'paired_all_low_and_high_values': True, 'paired_not_exhaustive_2pow32': True,
                  'paired_add_no_interhalf_carry': True, 'byteplane_roundtrips': True},
        'xor_proof': {'all_65536_patterns_bijection': True, 'both_roundtrips': True,
                      'paired_all_low_and_high_values': True, 'byteplane_roundtrips': True},
        'sample': {'weights': n, 'device': 'cpu', 'seed': 5151, 'distribution': 'round_BF16_normal(0,1/64)',
                   'actual_H100_weight_payload': False, 'numeric_recode': False},
        'symbol_entropy_bits_per_weight': entropy(u, 65536),
        'encoded_symbol_entropy_bits_per_weight': entropy(e, 65536),
        'encoded_value_fractions': {str(bound): float(np.mean(z < bound)) for bound in [256, 1024, 2048, 4096, 8192]},
        'representations': {name: physical_patterns(b) for name, b in reps.items()},
        'limits': ['No GPU/API invocation', 'No hardware compression ratio or throughput forecast',
                   'CPU RNG differs from actual GPU weights', 'Local pattern formats are not claimed NVIDIA formats'],
    }
    target = Path(__file__).with_name('cpu_patterns.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

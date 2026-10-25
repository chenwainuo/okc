"""Official muse_bf16_proj score arithmetic.

A load's time is the sum of its five layer times, each times 62. The speedup is the
geometric mean of SGLang's load time over the candidate's, and every load counts
equally. Floors are the published bytes/FLOP ceilings, not measured SGLang times.
"""

from __future__ import annotations

import math
from typing import Mapping

WEIGHT = 62

# Published floors in microseconds, TASK.md. Order is qkv_gate, o, pooler, unpooler, router.
FLOORS_US = {
    "decode 1": (7.9, 3.5, 14.0, 1.7, 1.8),
    "decode 16": (7.9, 3.6, 14.1, 14.1, 1.8),
    "decode 32": (8.0, 3.6, 14.1, 14.1, 1.9),
    "verify 1": (7.9, 3.5, 14.0, 1.8, 1.8),
    "verify 16": (8.4, 4.0, 14.6, 14.6, 2.2),
    "verify 32": (9.8, 4.5, 17.4, 17.4, 2.7),
}

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")

LOADS = (
    ("decode 1", 1),
    ("decode 16", 16),
    ("decode 32", 32),
    ("verify 1", 8),
    ("verify 16", 128),
    ("verify 32", 256),
)

# Hardware constants the floor uses.
HBM_BYTES_PER_S = 4.8e12
BF16_FLOPS_PER_S = 989e12

LAYER_NK = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
    "router": (256, 8192),
}


def out_features(layer: str, m: int) -> int:
    if layer == "unpooler" and m <= 8:
        return 1024
    return LAYER_NK[layer][0]


def geomean(values: list[float]) -> float:
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError(f"geomean needs finite positive values, got {values}")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_time_us(per_layer_us: Mapping[str, float]) -> float:
    missing = [name for name in LAYERS if name not in per_layer_us]
    if missing:
        raise ValueError(f"load is missing layers {missing}")
    return float(sum(per_layer_us[name] for name in LAYERS)) * WEIGHT


def floor_layer_us(m: int, layer: str) -> float:
    n = out_features(layer, m)
    k = LAYER_NK[layer][1]
    w_bytes = 4 if layer == "router" else 2
    bytes_moved = n * k * w_bytes + m * k * 2 + m * n * w_bytes
    flops = 2.0 * m * n * k
    return 1e6 * max(bytes_moved / HBM_BYTES_PER_S, flops / BF16_FLOPS_PER_S)


def floor_load_us(load: str) -> float:
    m = dict(LOADS)[load]
    return load_time_us({layer: floor_layer_us(m, layer) for layer in LAYERS})


def score_from_layer_times(by_load: Mapping[str, Mapping[str, float]]) -> dict:
    """by_load[load][layer] = microseconds for one call of that shape."""
    unknown = set(by_load) - set(FLOORS_US)
    if unknown or set(by_load) != set(FLOORS_US):
        raise ValueError(f"expected exactly the six scored loads, got {sorted(by_load)}")
    load_us = {load: load_time_us(layers) for load, layers in by_load.items()}
    floors = {load: floor_load_us(load) for load in FLOORS_US}
    ratios = [floors[load] / load_us[load] for load in FLOORS_US]
    return {
        "load_us": load_us,
        "floor_load_us": floors,
        "geomean_us": geomean(list(load_us.values())),
        "floor_geomean_us": geomean(list(floors.values())),
        "fraction_of_floor": geomean(ratios),
    }


def speedup_from_load_times(
    candidate_load_us: Mapping[str, float], baseline_load_us: Mapping[str, float]
) -> float:
    if set(candidate_load_us) != set(FLOORS_US) or set(baseline_load_us) != set(FLOORS_US):
        raise ValueError("both sides need the six scored loads")
    ratios = [baseline_load_us[load] / candidate_load_us[load] for load in FLOORS_US]
    return geomean(ratios)

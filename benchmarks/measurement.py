"""Exploratory uncertainty estimates for alternating-order local measurements.

Adjacent runs form a block (AB then BA). Resample blocks, not individual timings.
Intervals describe within-run sampling variation, not thermal drift, GPU-to-GPU
variation, or the hidden serving workload. Repeat independent runs before deciding.
"""
import math
import random
import statistics


def summarize(baseline, candidate):
    if len(baseline) != len(candidate) or len(baseline) < 2 or len(baseline) % 2:
        raise ValueError("Equal, even sample counts of at least two are required")
    if any(not math.isfinite(x) or x <= 0 for x in [*baseline, *candidate]):
        raise ValueError("Timings must be finite and positive")
    logs = [math.log(a / b) for a, b in zip(baseline, candidate)]
    blocks = [statistics.mean(logs[i:i + 2]) for i in range(0, len(logs), 2)]
    estimate = math.exp(statistics.median(blocks))
    result = {"balanced_speedup": estimate, "order_balanced_blocks": len(blocks),
              "bootstrap_95": None, "assessment": "insufficient_samples"}
    if len(blocks) < 10:
        return result
    rng = random.Random(71237)
    samples = sorted(math.exp(statistics.median(rng.choices(blocks, k=len(blocks))))
                     for _ in range(2000))
    lower, upper = samples[49], samples[1949]
    result["bootstrap_95"] = [lower, upper]
    # A practical 2% screening margin, not an official significance threshold.
    result["assessment"] = ("promising" if lower > 1.02 else
                            "regression" if upper < 1 / 1.02 else "inconclusive")
    return result

#!/usr/bin/env python3
"""Compare the opted-in fused small-batch finalize to the original PyTorch path.

Includes NaNs in invalid expert slots: those slots must never affect the output.
The FP64 oracle independently checks norm order, masking and weighted summation.
"""
import itertools
import torch


def oracle(x, scores, weight, ids, normalize, weight_first, eps):
    live = (ids >= 0).unsqueeze(-1)
    y = torch.where(live, x.double(), 0.0)
    w = None if weight is None else weight.double()
    if normalize:
        if w is not None and weight_first:
            y = y * w
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        if w is not None and not weight_first:
            y = y * w
    return (y * scores.double().unsqueeze(-1)).sum(1).to(x.dtype)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; run inside the challenge container.")
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum as finalize,
    )
    gen = torch.Generator(device="cuda").manual_seed(23851)
    count = 0
    for tokens, hidden in itertools.product((0, 1, 7, 8, 16, 31, 32, 33), (64, 4096)):
        x = torch.randn((tokens, 8, hidden), generator=gen, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(-1, 3, (tokens, 8), generator=gen, device="cuda", dtype=torch.int32)
        if tokens:
            ids[0] = -1  # An entire token with no local routes must produce zeros.
        x = x.masked_fill((ids < 0).unsqueeze(-1), float("nan"))
        scores = torch.rand((tokens, 8), generator=gen, device="cuda")
        weight = torch.randn((hidden,), generator=gen, device="cuda", dtype=torch.bfloat16)
        for normalize, weighted, weight_first in itertools.product((False, True), repeat=3):
            w = weight if weighted else None
            expected = oracle(x, scores, w, ids, normalize, weight_first, 1e-8)
            for fused in (False, True):
                actual = finalize(x, scores, norm_weight=w, apply_norm=normalize,
                                  weight_first=weight_first, slot_expert_ids=ids,
                                  fuse_small_tokens=fused)
                # Allow BF16 rounding at the final cast; compare with the independent
                # oracle instead of requiring identical floating-point reductions.
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                count += 1
    print(f"Passed {count} finalize correctness checks")


if __name__ == "__main__":
    main()

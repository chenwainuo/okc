#!/usr/bin/env python3
"""GPU regression check for replacing 256 alignment buckets with 32 local ones.

Checks the output contract, not identical sorting order (CUDA atomics may reorder
tokens within an expert). Includes no-local routes, invalid slots, and boundary IDs.
Run in the challenge container: python3 benchmarks/check_local_alignment.py
"""
import torch


def validate(ids, block_size, result, skip_nonlocal=False):
    sorted_ids, experts, padded = [x.cpu() for x in result]
    count = int(padded.item())
    assert count % block_size == 0
    assert count <= sorted_ids.numel()
    flat = ids.cpu().flatten().tolist()
    seen = []
    for offset in range(0, count, block_size):
        expert = int(experts[offset // block_size])
        assert -1 <= expert < 32
        for token in sorted_ids[offset:offset + block_size].tolist():
            if token == len(flat):
                continue
            assert 0 <= token < len(flat)
            assert flat[token] == expert
            seen.append(token)
    expected = [i for i, expert in enumerate(flat) if not skip_nonlocal or expert >= 0]
    assert sorted(seen) == expected, "lost or duplicated route slots"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; run inside the challenge container.")
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    gen = torch.Generator(device="cuda").manual_seed(98471)
    cases = [torch.full((4, 8), -1, device="cuda", dtype=torch.int32),
             torch.zeros((4, 8), device="cuda", dtype=torch.int32),
             torch.tensor([[-1, 0, 31, 0, -1, 31, 2, 3]], device="cuda", dtype=torch.int32)]
    for bs in (1, 4, 8, 16, 32, 33, 128, 256, 257, 753, 1024):
        cases.append(torch.randint(-1, 32, (bs, 8), generator=gen,
                                   device="cuda", dtype=torch.int32))
    for ids in cases:
        for block in (8, 16, 32, 48, 64):
            for buckets in (256, 32):
                for skip in (False, True):
                    result = moe_align_block_size(ids, block, buckets, ignore_invalid_expert=skip)
                    validate(ids, block, result, skip)
    print(f"Passed {len(cases) * 5 * 2 * 2} alignment contract checks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One warmed expert call under CUDA profiler control; hardware counters only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'muse_moe_w4a16'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=16001)
    args = parser.parse_args()
    import torch
    import torch.cuda.profiler as cuda_profiler
    import reference
    from serving_moe import prepare, kernel_fn
    from diagnostics import routing_summary
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = reference.generate_weights(args.seed)
    data = reference.generate_input(args.batch, 'ep8', args.seed + 1)
    state = prepare(weights)
    for _ in range(4):
        out = kernel_fn(data, state)
    torch.cuda.synchronize()
    ok, detail = reference.check_implementation(data, out, weights)
    if not ok:
        raise RuntimeError(detail)
    print(json.dumps(dict(batch=args.batch,seed=args.seed,correctness=detail,
                          routing=routing_summary(data))), flush=True)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    flush.zero_()
    torch.cuda.synchronize()
    cuda_profiler.start()
    out = kernel_fn(data, state)
    torch.cuda.synchronize()
    cuda_profiler.stop()
    print('Hardware counter capture complete; profiler timings are not benchmark results.', flush=True)


if __name__ == '__main__':
    main()

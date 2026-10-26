"""Weight statistics: exponent histogram and escape rate for a 16-code exponent window, per layer."""
import sys, math, torch
sys.path.insert(0, "/okc/muse_bf16_proj")
import reference
w = reference.generate_weights(1234)
for name, t in w.items():
    if t.dtype == torch.float32:
        b = t.view(torch.int32)
        e = (b >> 23) & 0xff
    else:
        b = t.view(torch.int16).to(torch.int32) & 0xffff
        e = (b >> 7) & 0xff
    h = torch.bincount(e.flatten(), minlength=256).double()
    n = h.sum()
    cs = torch.cumsum(h, 0)
    # best 16-wide window
    best = None
    for base in range(0, 240):
        cov = cs[base + 15] - (cs[base - 1] if base > 0 else 0)
        if best is None or cov > best[1]:
            best = (base, cov)
    base, cov = best
    nz = (h > 0).nonzero().flatten().tolist()
    print(f"{name:9s} N*K={int(n)} emax={max(nz)} emin={min(nz)} base16={base} escapes={int(n-cov)} rate={(n-cov)/n:.2e} ; top codes: " +
          " ".join(f"{i}:{h[i]/n:.4f}" for i in range(base-2, base+16) if h[i] > 0))
    # 15-wide window + escape code
    best15 = max(((bb, cs[bb+14] - (cs[bb-1] if bb > 0 else 0)) for bb in range(1, 240)), key=lambda z: z[1])
    print(f"          window15 base={best15[0]} escapes={int(n-best15[1])} rate={(n-best15[1])/n:.2e}")

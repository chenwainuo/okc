"""pw24 plain bf16 GEMM (bg): CPU emulation of the data path and a randomized simulation of its two-producer,
two-ring protocol (cluster pair with W multicast and remote empty arrives, or no cluster).

Data path per CTA (token tile tt, n-block nb): W slot image = two 32-row TMA boxes (SWIZZLE_128B) at +0 and +4096 B
(cluster: rank r issues rows 32 r .. into both CTAs; same bytes either way), x slot = a 64-token box (rows >= m zero);
wgmma m64n64k16 SS through the descriptors (start + 32 q, SBO 1024, swizzle on absolute address bits); D layout ->
bf16 smem tile [tok][n] -> 16-B row stores for tokens < m. Checked bit-exact against bf16(x @ W^T) with fp64 sums
(fp32 order is the only difference on the GPU; one-hot rows are exact either way), odd m, m = 256 (4 token tiles).

Protocol: per CTA a W producer lane and an x producer lane (each waits only on its own ring's empties), 4 MMA warps
(wait W full + x full, issue = asynchronous read of both slots, wait_group(1) retires s-1 and releases it: local W
empty, the peer's W empty (cluster), local x empty). W full expects both halves (the peer's multicast half may land
before the local expect). Mutations that must be caught: "early_release", "no_remote" (peer never told), and
"local_count" (W empty counts only the local warps, so a slot is refilled while the peer still reads it).

python3 -m proj_rsi.tools.bg_emu"""
import random
import torch
from proj_rsi.tools.tw_emu import bf, desc_read, Bar


def tma_box(rows_bits, nrows):
    img = torch.zeros(nrows * 64, dtype=torch.long)
    for r in range(rows_bits.shape[0]):
        for c in range(8):
            a = r * 128 + ((c ^ (r & 7)) << 4)
            img[a // 2: a // 2 + 8] = rows_bits[r, 8 * c: 8 * c + 8]
    return img


def emulate(x, W, cluster):
    m, K = x.shape
    N = W.shape[0]
    xb = x.view(torch.int16).long() & 0xffff
    wb = W.view(torch.int16).long() & 0xffff
    TT = ((m + 127) // 128) * 2
    lanes = torch.arange(32); g, t = lanes // 4, lanes % 4
    out = torch.full((m, N), float("nan"), dtype=torch.float64)
    for tt in range(TT):
        for nb in range(N // 64):
            D = torch.zeros(64, 64, dtype=torch.float64)  # [W row][token]
            for s in range(K // 64):
                wimg = torch.zeros(64 * 64, dtype=torch.long)
                for half in range(2):  # rank `half` (cluster) or the CTA itself issues rows 32 half ..
                    box = tma_box(wb[nb * 64 + 32 * half: nb * 64 + 32 * half + 32, s * 64: s * 64 + 64], 32)
                    wimg[half * 2048: half * 2048 + 2048] = box
                rows = torch.zeros(64, 64, dtype=torch.long)
                for r in range(64):
                    if tt * 64 + r < m:
                        rows[r] = xb[tt * 64 + r, s * 64: s * 64 + 64]
                ximg = tma_box(rows, 64)
                for q in range(4):
                    D += bf(desc_read(wimg, 32 * q, 64)) @ bf(desc_read(ximg, 32 * q, 64)).t()
            tile = torch.zeros(64, 64, dtype=torch.float64)  # [tok][n] bf16
            for w in range(4):
                r = 16 * w + g
                for i in range(8):
                    tk = 8 * i + 2 * t
                    for e, (dt, dr) in enumerate(((0, 0), (1, 0), (0, 8), (1, 8))):
                        tile[tk + dt, r + dr] = D[r + dr, tk + dt].float().to(torch.bfloat16).double()
            for tk in range(64):
                if tt * 64 + tk < m:
                    out[tt * 64 + tk, nb * 64: nb * 64 + 64] = tile[tk]
    return out


def sim(ST, KC, cluster, seed, bug=None):
    rnd = random.Random(seed)
    NC = 2 if cluster else 1
    C = []
    for c in range(NC):
        C.append({"wf": [Bar(1) for _ in range(ST)], "we": [Bar(8 if (cluster and bug != "local_count") else 4) for _ in range(ST)],
                  "xf": [Bar(1) for _ in range(ST)], "xe": [Bar(4) for _ in range(ST)],
                  "wland": [set() for _ in range(ST)], "xland": [None] * ST, "wread": [set() for _ in range(ST)], "xread": [set() for _ in range(ST)],
                  "ws": 0, "xs": 0, "mma": [{"s": 0, "st": "in", "pend": []} for _ in range(4)]})
    inflight = []

    def issue_w(c, s):
        sl = s % ST
        C[c]["wf"][sl].arrive(tx=2)  # local expect: both halves
        for d in range(NC):
            assert not C[d]["wread"][sl], "W slot overwritten while a wgmma reads it"
            if NC == 1:  # no cluster: the CTA issues both halves
                inflight.append((d, "w", sl, s, 0)); inflight.append((d, "w", sl, s, 1))
            else:
                inflight.append((d, "w", sl, s, c))

    def issue_x(c, s):
        sl = s % ST
        assert not C[c]["xread"][sl], "x slot overwritten while a wgmma reads it"
        C[c]["xf"][sl].arrive(tx=1); inflight.append((c, "x", sl, s, 0))

    def ready(bar, par):
        return bar.ready(par)

    while True:
        acts = []
        for c, S in enumerate(C):
            ws, xs = S["ws"], S["xs"]
            if ws < KC and (ws < ST or ready(S["we"][ws % ST], ((ws // ST) - 1) & 1)):
                acts.append((c, "pw"))
            if xs < KC and (xs < ST or ready(S["xe"][xs % ST], ((xs // ST) - 1) & 1)):
                acts.append((c, "px"))
            for i, k in enumerate(S["mma"]):
                s = k["s"]
                if k["st"] == "in" and s < KC and ready(S["wf"][s % ST], (s // ST) & 1) and ready(S["xf"][s % ST], (s // ST) & 1):
                    acts.append((c, ("mi", i)))
                elif k["st"] == "wait":
                    acts.append((c, ("mw", i)))
                elif k["st"] == "in" and s == KC and k["pend"]:
                    acts.append((c, ("mf", i)))
        if inflight:
            acts.append((None, "land"))
        if not acts:
            break
        c, a = rnd.choice(acts)
        if a == "land":
            d, ring, sl, s, half = inflight.pop(rnd.randrange(len(inflight)))
            if ring == "w":
                C[d]["wland"][sl] = {e for e in C[d]["wland"][sl] if e[0] == s} | {(s, half)}
                C[d]["wf"][sl].ctx(1)
            else:
                C[d]["xland"][sl] = s; C[d]["xf"][sl].ctx(1)
            continue
        S = C[c]
        if a == "pw":
            issue_w(c, S["ws"]); S["ws"] += 1
        elif a == "px":
            issue_x(c, S["xs"]); S["xs"] += 1
        else:
            kind, i = a; k = S["mma"][i]; s = k["s"]
            def release(v):
                if bug != "early_release":
                    S["wread"][v % ST].discard((i, v)); S["xread"][v % ST].discard((i, v))
                    S["we"][v % ST].arrive()
                    if NC == 2 and bug != "no_remote":
                        C[c ^ 1]["we"][v % ST].arrive()
                    S["xe"][v % ST].arrive()
            if kind == "mi":
                sl = s % ST
                assert S["wland"][sl] == {(s, 0), (s, 1)}, "wgmma reads an incomplete W tile"
                assert S["xland"][sl] == s, "wgmma reads an x slot not holding this stage"
                S["wread"][sl].add((i, s)); S["xread"][sl].add((i, s))
                if bug == "early_release":
                    S["we"][sl].arrive()
                    if NC == 2:
                        C[c ^ 1]["we"][sl].arrive()
                    S["xe"][sl].arrive()
                    S["wread"][sl].discard((i, s)); S["xread"][sl].discard((i, s))
                    S.setdefault("ghost", []).append((sl, s))  # the hardware read is still in flight
                k["pend"].append(s); k["s"] += 1; k["st"] = "wait"
            elif kind == "mw":
                while len(k["pend"]) > 1:
                    release(k["pend"].pop(0))
                k["st"] = "in"
            else:
                for v in k["pend"]:
                    release(v)
                k["pend"] = []
        if bug == "early_release":  # a slot refilled while an un-retired wgmma still reads it
            for d in range(NC):
                for sl, s in C[d].get("ghost", []):
                    ok = all(m_["s"] > s + 1 or (m_["s"] == s + 1 and m_["st"] == "in") for m_ in C[d]["mma"])
                    if not ok and C[d]["wland"][sl] and all(e[0] != s for e in C[d]["wland"][sl]):
                        raise AssertionError("W slot overwritten while a wgmma reads it")
    for S in C:
        assert S["ws"] == KC and S["xs"] == KC and all(k["s"] == KC and not k["pend"] for k in S["mma"]), "deadlock"


def main():
    ok = True
    gen = torch.Generator().manual_seed(5)
    for m, N, K, cl in ((128, 128, 256, True), (121, 64, 192, True), (256, 64, 128, False), (250, 64, 128, True), (128, 64, 128, False)):
        W = (torch.randn(N, K, generator=gen) / K ** 0.5).to(torch.bfloat16)
        x = torch.randn(m, K, generator=gen).to(torch.bfloat16)
        o = emulate(x, W, cl)
        ref = (x.double() @ W.double().t()).float().to(torch.bfloat16).double()
        exact = torch.equal(o, ref)
        xo = torch.zeros(m, K); xo[torch.arange(m), torch.randint(K, (m,), generator=gen)] = 8.0; xo = xo.to(torch.bfloat16)
        ex2 = torch.equal(emulate(xo, W, cl), (xo.double() @ W.double().t()).to(torch.bfloat16).double())
        print("   data m=%d N=%d K=%d cluster=%d: bit-exact %s, one-hot exact %s" % (m, N, K, cl, exact, ex2))
        ok &= exact and ex2
    for cl in (True, False):
        bad = 0
        for seed in range(200):
            for KC in (1, 3, 13, 40):
                try:
                    sim(12 if seed % 2 else 6, KC, cl, seed)
                except AssertionError as e:
                    bad += 1; print("   sim cluster=%d seed %d KC %d: %s" % (cl, seed, KC, e))
        print("   protocol cluster=%d: %d/800 schedules clean" % (cl, 800 - bad))
        ok &= bad == 0
    for bug in ("early_release", "no_remote", "local_count"):
        caught = 0
        for seed in range(200):
            try:
                sim(6, 40, True, seed, bug)
            except AssertionError:
                caught += 1
        print("   mutation %-14s caught in %d/200" % (bug, caught))
        ok &= caught > 0
    print("BG EMU", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

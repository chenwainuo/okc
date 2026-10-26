"""Source/index/event proof; no torch, CUDA, compiler or remote calls."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from variant import CONFIGS, INSERT, SHIPPED_SHA256, source_snapshot


def desc(a):
    return ((a & 0x3FFFF) >> 4) | (1 << 16) | (64 << 32) | (1 << 62)


def desc_address(d, row, k16):
    start = (d & 0x3FFF) << 4
    sbo = ((d >> 32) & 0x3FFF) << 4
    logical = start + (row // 8) * sbo + (row % 8) * 128 + 2 * k16
    return logical ^ (((logical >> 7) & 7) << 4)


def index_proof(cfg, raw):
    kr, slots, toks, wn, _, _ = cfg
    stages = kr // 64
    slots = min(slots, stages)
    sb = (toks + 128) * 128
    base = (raw + 1023) & ~1023
    sm = slots * sb + 1024 + 16 * slots
    assert base - raw + slots * sb + 16 * slots <= sm
    checked = 0
    for sl in range(slots):
        for q in range(4):
            for rows, offset in ((toks, 0), (64, toks * 128), (64, (toks + 64) * 128)):
                a = base + sl * sb + offset + q * 32
                d = desc(a)
                for row in range(rows):
                    for k16 in range(16):
                        # TMA SW128 byte address vs independent wgmma descriptor interpretation.
                        byte = 2 * (q * 16 + k16)
                        expected = base + sl * sb + offset + row * 128 + (byte ^ ((row & 7) * 16))
                        assert desc_address(d, row, k16) == expected
                        checked += 1
    # Consumer fragment -> smem epilogue: exactly one writer for every (token,n).
    stores = set()
    for tid in range(128):
        lane, warp = tid & 31, tid >> 5
        g, t = lane >> 2, lane & 3
        for h in range(toks // wn):
            for i in range(wn // 8):
                tok, n = h * wn + 8 * i + 2 * t, 16 * warp + g
                for r, c in ((tok, n), (tok + 1, n), (tok, n + 8), (tok + 1, n + 8)):
                    assert (r, c) not in stores
                    stores.add((r, c))
    assert stores == {(r, c) for r in range(toks) for c in range(64)}
    return checked


def ring_proof(cfg, seed, bug=None):
    """Random fair event schedules. Full count1/bytesSB; empty count4; retire after next commit."""
    kr, slots, toks, _, _, mode = cfg
    stages, slots = kr // 64, min(slots, kr // 64)
    sb = (toks + 128) * 128
    gated = bool(mode & 128)
    rng = random.Random(seed)
    issued, consumed, gate_seen = 0, 0, not gated
    inflight, pending, landed, retired = {}, {}, set(), set()
    full_gen = [0] * slots
    empty_gen = [0] * slots
    events = 0
    while consumed < stages or issued < stages or not gate_seen:
        choices = []
        if gated and issued == 0 and bug == "wait-before-issue":
            pass
        elif issued < stages and (issued == 0 or gate_seen):
            s, sl, gen = issued, issued % slots, issued // slots
            if s < slots or empty_gen[sl] >= gen or bug == "skip-empty":
                choices.append(("issue", s))
        for s, pieces in pending.items():
            for p in pieces:
                choices.append(("land", s, p))
        if gated and issued > 0 and not gate_seen:
            if bug != "wrong-phase" and 0 in landed:
                choices.append(("gate",))
        if consumed in landed:
            choices.append(("consume", consumed))
        assert choices, f"No progress, seed={seed}, bug={bug}, issued={issued}, consumed={consumed}"
        e = rng.choice(choices)
        if e[0] == "issue":
            s, sl, gen = e[1], e[1] % slots, e[1] // slots
            assert sl not in inflight, "Ring overwrote a stage with live TMA/MMA"
            assert full_gen[sl] == gen
            if s >= slots:
                assert empty_gen[sl] >= gen, "Producer skipped empty-barrier phase"
            inflight[sl] = s
            # One expect_tx and exactly three nonoverlapping TMA pieces.
            pending[s] = {0: toks * 128, 1: 64 * 128, 2: 64 * 128}
            assert sum(pending[s].values()) == sb
            issued += 1
        elif e[0] == "land":
            s, p = e[1:]
            del pending[s][p]
            if not pending[s]:
                del pending[s]
                landed.add(s)
                full_gen[s % slots] += 1
        elif e[0] == "gate":
            # New waiter sees phase0 as immediately preceding; producer has not reused slot0.
            assert full_gen[0] == 1 and issued == 1
            gate_seen = True
        else:
            s = e[1]
            assert full_gen[s % slots] == s // slots + 1
            assert inflight[s % slots] == s
            # Identical hi,lo order; wait_group1 retires previous stage, four warps arrive.
            if s >= 1:
                old = s - 1
                assert inflight[old % slots] == old
                del inflight[old % slots]
                retired.add(old)
                if old + slots < stages:
                    empty_gen[old % slots] += 1
            consumed += 1
        events += 1
        assert events <= stages * 12
    assert consumed == issued == stages
    assert gate_seen
    assert retired == set(range(stages - 1))
    return events


def zero_proof(cfg, m):
    kr, _, toks, _, _, _ = cfg
    G = 8192 // kr
    for t0 in range(0, m, toks):
        E = min(toks, m - t0) * 16
        seen = set()
        for split in range(G):
            e0, e1 = split * E // G, (split + 1) * E // G
            for lane in range(32):
                for i in range(e0 + lane, e1, 32):
                    assert i not in seen
                    seen.add(i)
        assert seen == set(range(E))
        # Stream-ordered calls: one group of G release arrivals; poll target equals group end.
        for call in range(5):
            targets = {(arrived // G + 1) * G for arrived in range(call * G, (call + 1) * G)}
            assert targets == {(call + 1) * G}
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    before, after = source_snapshot()
    assert after.replace(INSERT, "") == before
    for needle in ("if (tid >= 160)", "bar.sync 2, 160", "atom.add.release.gpu.global.s32",
                   "return -30", "return -31", "rw_mma<WN>(d + h * (WN / 2), dh, db)",
                   "rw_mma<WN>(d + h * (WN / 2), dl, db)"):
        assert before.count(needle) == after.count(needle) and before.count(needle)
    for v, parent in ((300, 200), (301, 201)):
        assert CONFIGS[v][:-1] == CONFIGS[parent][:-1]
        assert CONFIGS[v][-1] ^ CONFIGS[parent][-1] == 128
    idx, traces, events = 0, 0, 0
    for cfg in CONFIGS.values():
        for raw in (0, 400, 1023):
            idx += index_proof(cfg, raw)
        for m in (121, 128, 250, 256):
            zero_proof(cfg, m)
        for seed in range(512):
            events += ring_proof(cfg, seed)
            traces += 1
    negative = {}
    for bug in ("wait-before-issue", "wrong-phase", "skip-empty"):
        caught = 0
        for seed in range(64):
            try:
                ring_proof(CONFIGS[301], seed, bug)
            except AssertionError:
                caught += 1
        assert caught, f"Negative control survived: {bug}"
        negative[bug] = caught
    result = {"state": "CPU_PROOF_PASS_GPU_UNRUN", "parent_sha256": SHIPPED_SHA256,
              "cuda_before_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "cuda_after_sha256": hashlib.sha256(after.encode()).hexdigest(),
              "changed_cuda": "one startup phase0 wait, MODE bit128",
              "descriptor_index_checks": idx, "random_ring_traces": traces,
              "events": events, "negative_controls_caught_of64": negative,
              "numerics_gpu_convergence_speed": "NOT_PROVEN"}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

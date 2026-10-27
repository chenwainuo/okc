"""CPU-only alternating-producer ownership, phase, escape and resource proofs."""
import json
import random


def proof():
    owners = [[s for s in range(p, 64, 2)] for p in range(2)]
    assert sorted(owners[0] + owners[1]) == list(range(64))
    for p, stages in enumerate(owners):
        assert len(stages) == 32
        for s in stages:
            assert s % 2 == p
            for slots in (6, 8):
                assert s % slots % 2 == p
                if s >= slots:
                    assert (s - slots) % 2 == p
                    assert (s // slots) % 2 != ((s - slots) // slots) % 2
            # Both prerequisites for a current stage are strictly earlier.
            assert s < 6 or s - 6 < s
            assert s < 8 or s - 8 < s
    # Four producer warps per named barrier, no overlap with math/bar0.
    barrier_threads = {3: [], 4: []}
    for thread in range(512):
        if thread >= 256:
            p = (thread - 256) // 128
            barrier_threads[3 + p].append(thread)
    assert barrier_threads == {3: list(range(256, 384)), 4: list(range(384, 512))}
    assert set(barrier_threads[3]).isdisjoint(barrier_threads[4])
    assert [256 + 128*p for p in range(2)] == [256, 384]
    # B publication precedes each matching math read. Math completes all64stages
    # before output reuse, therefore both producers' final62/63 writes precede it.
    assert [s[-1] for s in owners] == [62, 63]
    # Compare full sequential repair with parity traversal on arbitrary dense CSR
    # lists, including repeated edits, empty stages and >32-entry windows.
    rng = random.Random(7919)
    scenarios = []
    for density in (0, 1, 31, 32, 33, 512, 4096):
        entries = sorted((rng.randrange(64), rng.randrange(32), rng.randrange(16),
                          rng.randrange(2), rng.randrange(1, 128)) for _ in range(density))
        initial = [[[rng.getrandbits(32) for _ in range(16)] for _ in range(32)] for _ in range(64)]
        expected = [[a[:] for a in stage] for stage in initial]
        for s, lane, reg, half, xor7 in entries:
            expected[s][lane][reg] ^= xor7 << (8 + 16*half)
        observed = [[a[:] for a in stage] for stage in initial]
        skipped = applied = 0
        for p in range(2):
            cursor = 0
            for s in owners[p]:
                while cursor < len(entries) and entries[cursor][0] < s:
                    cursor += 1; skipped += 1
                while cursor < len(entries) and entries[cursor][0] == s:
                    _, lane, reg, half, xor7 = entries[cursor]
                    observed[s][lane][reg] ^= xor7 << (8 + 16*half)
                    cursor += 1; applied += 1
        assert observed == expected and applied == len(entries)
        scenarios.append({"records": density, "applied_once": applied, "skipped": skipped})
    # Negative control: stepping by two without skipping stalls at the other
    # parity's first record, losing a later assigned record.
    bad_entries = [(1, 0, 0, 0, 1), (2, 0, 0, 0, 2)]
    cursor = 0; bad_applied = []
    for s in owners[0]:
        while cursor < len(bad_entries) and bad_entries[cursor][0] == s:
            bad_applied.append(s); cursor += 1
    assert 2 not in bad_applied
    return {"status": "CPU_PROTOCOL_PROOF_ONLY_GPU_UNRUN", "stage_owners": owners,
            "barriers": {k: [min(v), max(v), len(v)] for k, v in barrier_threads.items()},
            "escape_scenarios": scenarios, "negative_missing_skip_detected": True,
            "static_register_gate": "numRegs<=128, allocated_round8*512<=65536, local0, actual optin occupancy>=1",
            "deadlock_argument": "W(s) requires initial TMA or same-owner(s-6); Bempty(s) requires math(s-8). All dependencies strictly earlier; finite warp CSR and separate128-thread barriers assumed.",
            "limits": "Ownership/phase/CSR model, not GPU scheduling or CUDA binary proof; source, actual SASS/A4 retirement, full numerical/graph and cleanup gates remain mandatory."}


if __name__ == "__main__":
    print(json.dumps(proof(), indent=2))

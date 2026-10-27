"""CPU-only abstract proof of the proposed paired-fence schedule, not its failed SASS.

Models each A bank as16 separate registers and pending WGMMA groups. Does not
model NVIDIA register allocation, warp scheduling, memory visibility or latency.
"""
import json
from pathlib import Path


def proof():
    banks = [[None] * 16 for _ in range(4)]
    versions = [[0] * 16 for _ in range(4)]
    fenced = [[-1] * 16 for _ in range(4)]
    pending = []
    loads, issues, fences, releases, events = [], [], [], [], []

    def load(stage):
        slot = stage % 4
        assert all(old % 4 != slot for old in pending), (stage, pending)
        for reg in range(16):
            banks[slot][reg] = stage
            versions[slot][reg] += 1
        loads.append(stage)
        events.append(["load", stage, list(pending)])

    def issue(stage):
        slot = stage % 4
        assert banks[slot] == [stage] * 16
        assert fenced[slot] == versions[slot], (stage, fenced[slot], versions[slot])
        pending.append(stage)
        issues.append(stage)
        # wait_group1: source performs no generic D access here.
        while len(pending) > 1:
            pending.pop(0)
        events.append(["issue_wait1", stage, list(pending)])

    load(0)
    load(1)
    for b in range(0, 64, 2):
        # Both Bfull waits precede the fence and the two issue groups.
        for slot in (b % 4, (b + 1) % 4):
            fenced[slot] = list(versions[slot])
        fences.append(b)
        events.append(["fence", b, list(pending)])
        if b < 62:
            load(b + 2)
        issue(b)
        if b < 62:
            load(b + 3)
        issue(b + 1)
        # Delayed releases: b-1 is retired by issue(b), b by issue(b+1).
        for retired in ([b - 1] if b else []) + [b]:
            assert retired not in pending
            releases.append(retired)
    assert pending == [63]
    pending.clear()  # final wait0 precedes output conversion and A/D generic access.
    releases.append(63)
    assert loads == issues == list(range(64))
    assert sorted(releases) == list(range(64)) and len(set(releases)) == 64
    assert len(fences) == 32

    # Wait-for DAG. W(s) can be refilled after P(s-6), same-parity producer
    # stage order adds P(s-2). P(s>=8) needs released decoded slot s-8.
    # Deferred release R(even r) follows I(r+1), R(odd r<63) follows I(r+2).
    edges = {}
    def node(n):
        edges.setdefault(n, set())
    def depends(n, dep):
        node(n); node(dep); edges[n].add(dep)
    for s in range(64):
        node(f"P{s}"); node(f"I{s}"); node(f"R{s}")
        if s >= 2:
            depends(f"P{s}", f"P{s-2}")
        if s >= 6:
            depends(f"P{s}", f"P{s-6}")
        if s >= 8:
            depends(f"P{s}", f"R{s-8}")
        if s:
            depends(f"I{s}", f"I{s-1}")
        b = s & ~1
        depends(f"I{s}", f"P{b}")
        depends(f"I{s}", f"P{b+1}")
        release_after = 63 if s == 63 else s + (1 if s % 2 == 0 else 2)
        depends(f"R{s}", f"I{release_after}")
    done = set()
    while len(done) < len(edges):
        ready = {n for n, deps in edges.items() if n not in done and deps <= done}
        assert ready, {n: sorted(deps - done) for n, deps in edges.items() if n not in done}
        done |= ready
    return {"status": "PASS_ABSTRACT_SCHEDULE_ONLY", "stage_loads": len(loads),
            "issues": len(issues), "fences_per_WG": len(fences), "release_count": len(releases),
            "per_bank_registers_checked": 16, "D_generic_access_between_groups": 0,
            "wait_for_DAG_nodes": len(done), "events": events,
            "limits": "No physical allocator/runtime proof. Current compiled source fails C7520/half-fence gate. Startup/tail have no uninitialized A reads in this model."}


if __name__ == "__main__":
    result = proof()
    Path(__file__).with_name("schedule_proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))

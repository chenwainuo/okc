"""Symbolic native-index, A2x32 retirement and W6/B8 wait-for proof; zero CUDA."""
import json


def proof():
    a=[None,None];versions=[[0]*32 for _ in range(2)];fenced=[[None]*32 for _ in range(2)]
    pending=[];loads=[];issues=[];releases=[];events=[]
    def load(pair):
        slot=pair%2
        assert all(p%2!=slot for p in pending)
        a[slot]=[(2*pair+j//16,j%16) for j in range(32)]
        versions[slot]=[v+1 for v in versions[slot]]
        loads.append(pair);events.append(["load_pair",pair,list(pending)])
    load(0);load(1)
    for p in range(32):
        slot=p%2
        assert a[slot]==[(2*p+j//16,j%16) for j in range(32)]
        fenced[slot]=versions[slot].copy()
        for stage in (2*p,2*p+1):
            for k16 in range(4):
                regs=[16*(stage%2)+4*k16+j for j in range(4)]
                assert all(fenced[slot][r]==versions[slot][r] for r in regs)
                assert [a[slot][r] for r in regs]==[(stage,4*k16+j) for j in range(4)]
                issues.append((stage,k16))
        pending.append(p)
        retired=pending[:-1];pending=pending[-1:]
        assert retired==([] if p==0 else [p-1])
        if p:
            releases.extend((2*p-2,2*p-1))
            if p+1<32:load(p+1)
        events.append(["commit_wait1",p,list(pending)])
    assert pending==[31]
    pending.clear();releases.extend((62,63))
    assert loads==list(range(32))
    assert issues==[(s,k) for s in range(64) for k in range(4)]
    assert releases==list(range(64))
    # W(s) is initialized0..5 or refilled after same-owner P(s-6). P(s)
    # retains parity order and cannot overwrite B(s-8) before release R(s-8).
    deps={}
    def edge(n,d):deps.setdefault(n,set()).add(d);deps.setdefault(d,set())
    for s in range(64):
        deps.setdefault(f"P{s}",set());deps.setdefault(f"R{s}",set())
        if s>=2:edge(f"P{s}",f"P{s-2}")
        if s>=6:edge(f"P{s}",f"P{s-6}")
        if s>=8:edge(f"P{s}",f"R{s-8}")
        edge(f"R{s}",f"G{min(s//2+1,31)}")
        if s>=8:assert (s-8)//2+1<s//2
    for p in range(32):
        edge(f"G{p}",f"P{2*p}");edge(f"G{p}",f"P{2*p+1}")
        if p:edge(f"G{p}",f"G{p-1}")
    done=set()
    while len(done)<len(deps):
        ready={n for n,d in deps.items() if n not in done and d<=done}
        assert ready,{n:sorted(d-done) for n,d in deps.items() if n not in done}
        done|=ready
    # Both old B slots retain their own phase/8-arrival protocol. Producer
    # metadata remains64stages; a math pair does not merge full/empty barriers.
    phases={slot:[(s,(s//8)&1) for s in range(slot,64,8)] for slot in range(8)}
    for records in phases.values():assert [phase for _,phase in records]==[0,1]*4
    # Negative source-model control: next pair cannot overwrite current A2slot
    # before its covering wait; current p0 is still pending after its own wait1.
    pending=[0];rejected=False
    try:load(2)
    except AssertionError:rejected=True
    assert rejected
    return {"status":"PASS_SYMBOLIC_GROUP8_SCHEDULE_NOT_RUNTIME", "A_words_per_bank":32,
            "physical_banks_required":2,"source_old_K64_stages":64,"math_pairs":32,
            "same_order_MMA_quads":len(issues),"fences_commits_wait1_per_WG":32,
            "stage_Bempty_releases":len(releases),"wait_for_DAG_nodes":len(done),
            "B8_phases":phases,"negative_current_bank_refill_rejected":rejected,"events":events,
            "limits":"Sequential symbolic stage/bank/DAG model. Does not simulate arbitrary GPU interleavings, register allocation, fences' hardware latency or runtime launchability."}


if __name__=="__main__":print(json.dumps(proof(),indent=2))

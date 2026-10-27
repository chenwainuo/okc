"""CPU gate for actual SM90a main binaries: async operands and fence coverage.

Fixed loop/descriptor/output policy, discovered physical A/D registers. Not a
general SASS verifier or a hardware wait-mask decoder. Conditional math-loop
writes execute conservatively in the model; all external paths are inspected.
"""
import re


def parse(text):
    return [(int(m[1],16),m[2].strip()) for m in re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);",text)]


def plain(ins):
    return re.sub(r"^@!?P\d+\s+", "", ins).replace(".reuse", "")


def registers(ins):
    regs=set()
    for m in re.finditer(r"(?<![\w])R(\d+)(?:\.(64|128))?",ins):
        width=int(m[2])//32 if m[2] else 1
        regs.update(range(int(m[1]),int(m[1])+width))
    return regs


def writes(ins):
    ins=plain(ins)
    if ins.startswith("HGMMA."):return set()
    m=re.match(r"([A-Z0-9_.]+)\s+(?:P\d+,\s*)?R(\d+)(?:,|\s)",ins)
    if not m:return set()
    op,start=m[1],int(m[2])
    width=4 if ".128" in op else 2 if (".64" in op or ".WIDE" in op or op=="CS2R") else 1
    return set(range(start,start+width))


def accesses(ins):
    ins=plain(ins)
    regs=registers(ins)|writes(ins)
    # SASS vector-memory operands name the first data register only.
    if ins.startswith(("STS.","STG.","STL.")):
        data=re.search(r", R(\d+)(?:\b|\.)",ins)
        if data:
            width=4 if ".128" in ins.split()[0] else 2 if ".64" in ins.split()[0] else 1
            regs.update(range(int(data[1]),int(data[1])+width))
    return regs


def branch(ins):
    m=re.fullmatch(r"BRA(?:\.[A-Z]+)* 0x([0-9a-f]+)",plain(ins))
    return int(m[1],16) if m else None


def check_one(table, strict=True):
    rows=parse(table); assert rows
    aligned=[(pc,ins) for pc,ins in rows if plain(ins).startswith(("HGMMA.","WARPGROUP.ARRIVE","WARPGROUP.DEPBAR.LE"))]
    assert all(not ins.startswith("@") for _,ins in aligned),"Predicated aligned WGMMA/fence/wait forbidden"
    if strict:
        assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b",table),"Local memory/dynamic register redistribution forbidden"
        assert sum("WARPGROUP.ARRIVE" in ins for _,ins in rows)==2
    bypc=dict(rows);pcs=[pc for pc,_ in rows];nextpc={a:b for a,b in zip(pcs,pcs[1:])}
    h=[(pc,plain(ins)) for pc,ins in rows if plain(ins).startswith("HGMMA.")]
    assert h
    groups=[];current=[];dregs=None
    for pc,ins in h:
        m=re.search(r"HGMMA\.[^ ]+ R(\d+), R(\d+),",ins);assert m,ins
        d,a=int(m[1]),int(m[2]);ds=set(range(d,d+32));ar=set(range(a,a+4))
        assert not ds&ar
        if dregs is None:dregs=ds
        assert dregs==ds,"D must be one same-shape asynchronous chain"
        current.append((pc,ar))
        if "gsb0" in ins:groups.append(current);current=[]
    assert not current and groups
    sets=[set.union(*(ar for _,ar in g)) for g in groups]
    all_a=set.union(*sets)
    assert not all_a&dregs
    if strict:
        assert len(h)==16 and len(groups)==2
        assert [len(g) for g in groups]==[8,8]
        assert [len(s) for s in sets]==[32,32] and len(all_a)==64
        assert not sets[0]&sets[1]
    wait0=min(pc for pc,ins in rows if "WARPGROUP.DEPBAR.LE" in ins and ", 0x0" in ins)
    back=[(pc,branch(ins)) for pc,ins in rows if h[-1][0]<pc<wait0 and branch(ins) is not None and branch(ins)<h[0][0]]
    assert back,"No recognized main-loop backedge"
    end,lo=max(back);body=[(pc,ins) for pc,ins in rows if lo<=pc<=end]
    assert sum("HGMMA." in ins for _,ins in body)==len(h)
    assert sum("WARPGROUP.DEPBAR.LE" in ins for _,ins in body)==len(groups)
    assert all(", 0x1" in ins for _,ins in body if "WARPGROUP.DEPBAR.LE" in ins)
    if strict:assert sum("WARPGROUP.ARRIVE" in ins for _,ins in body)==2
    # Check every reachable instruction in external slow paths, not merely the
    # first poll block. Stop only on a known return to the math loop.
    external={branch(ins) for _,ins in body if branch(ins) is not None and not lo<=branch(ins)<=end}
    outlined=set()
    for target in external:
        assert target>wait0,(hex(target),"unexpected math branch")
        todo=[target];seen=set();returns=set()
        while todo:
            pc=todo.pop()
            if lo<=pc<=end:returns.add(pc);continue
            if pc in seen:continue
            seen.add(pc);assert pc in bypc,hex(pc)
            ins=bypc[pc];p=plain(ins)
            assert not accesses(ins)&(all_a|dregs),("OUTLINED_GENERIC_A_D_ACCESS",hex(pc),ins)
            assert not p.startswith(("HGMMA.","WARPGROUP.","EXIT")),(hex(pc),ins)
            dest=branch(ins)
            if dest is not None:
                todo.append(dest)
                if ins.startswith("@"):todo.append(nextpc[pc])
            else:todo.append(nextpc[pc])
        assert returns,(target,seen)
        outlined.update(seen)
    # The initial16 payload vector loads write all64 A words. Producer-only
    # initialization does not clobber A; math D is explicitly zeroed by CS2R.
    initial=[(pc,ins) for pc,ins in rows if pc<lo and writes(ins)&all_a]
    assert initial and all(plain(ins).startswith("LDG.") and ".128" in plain(ins).split()[0] for _,ins in initial)
    initialized_a=set.union(*(writes(ins)&all_a for _,ins in initial))
    assert initialized_a==all_a
    initialized_d=set.union(*(writes(ins)&dregs for pc,ins in rows if pc<lo and "CS2R" in ins and "SRZ" in ins))
    assert initialized_d==dregs
    versions={r:0 for r in all_a|dregs};fenced={r:-1 for r in versions}
    for _,ins in initial:
        for r in writes(ins)&all_a:versions[r]+=1
    pending=[];chunk=set();groups_done=0;events=[];payload_writes=0;fence_count=0
    # Two iterations cover cyclic slot reuse. Startup/tail branch-skipped
    # writes also execute: this can reject safe code, never hide an overwrite.
    for iteration in range(2):
        gindex=0;since_fence=False
        for pc,raw in body:
            ins=plain(raw)
            if ins.startswith("WARPGROUP.ARRIVE"):
                assert not chunk
                fenced=dict(versions);fence_count+=1;since_fence=True
            elif ins.startswith("HGMMA."):
                m=re.search(r"HGMMA\.[^ ]+ R\d+, R(\d+),",ins);a=int(m[1]);ar=set(range(a,a+4))
                assert all(fenced[r]==versions[r] for r in ar|dregs),{
                    "error":"UNFENCED_GENERIC_A_D_ACCESS","iteration":iteration,"PC":hex(pc),"instruction":raw}
                assert since_fence
                chunk|=ar
                if "gsb0" in ins:
                    assert chunk==sets[gindex]
                    pending.append(set(chunk));chunk.clear();groups_done+=1;gindex+=1
            elif ins.startswith("WARPGROUP.DEPBAR.LE"):
                assert not chunk
                retired=max(0,len(pending)-1);pending=pending[-1:]
                events.append({"iteration":iteration,"PC":pc,"retired_groups":retired})
                since_fence=False
            else:
                touched=accesses(ins)&(all_a|dregs)
                live=set(chunk)
                for regs in pending:live|=regs
                if pending or chunk:live|=dregs
                assert not touched&live,{"error":"GENERIC_ASYNC_OPERAND_ACCESS_BEFORE_RETIRE", "iteration":iteration,
                    "PC":hex(pc),"instruction":raw,"clobbered_or_read":sorted(touched&live)}
                dst=writes(ins)&(all_a|dregs)
                if strict and since_fence:assert not dst,("Future A load moved into fenced MMA region",hex(pc),ins)
                for r in dst:versions[r]+=1
                if dst:payload_writes+=1
        assert gindex==len(groups)
    assert not chunk and len(pending)==1
    for pc,ins in rows:
        if end<pc<wait0:assert not accesses(ins)&(all_a|dregs),(hex(pc),ins)
    pending.clear() # final wait0; all generic output conversions follow it.
    return {"status":"PASS_PHYSICAL_ASYNC_A_AND_FENCE_COVERAGE","groups_per_static_loop":len(groups),
            "MMA_per_group":[len(g) for g in groups],"A_register_sets":[sorted(s) for s in sets],
            "D_registers":sorted(dregs),"loop_PC_range":[lo,end],"iterations":2,
            "groups_checked":groups_done,"fences_checked":fence_count,"initial_vector_loads":len(initial),
            "all_MMA_fence_wait_operations_unpredicated":True,
            "payload_writes_checked":payload_writes,"outlined_instruction_count":len(outlined),
            "outlined_entry_PCs":sorted(external),"retirement_events":events,
            "tail":"Final wait0 precedes first generic A/D epilogue access",
            "limits":"Fixed main-loop CFG, all predicate writes conservatively executed. Source proofs retain exact XP indices/B descriptors. No wait-mask/latency or general ISA verifier claim."}


def check(sass):
    tables=re.split(r"\n\s*Function : ",sass)[1:]
    mains=[t for t in tables if "rq_px_g8_stmx" in t.splitlines()[0]]
    assert mains,"No actual group8 main in JIT artifact"
    return {"status":"PASS_ALL_ACTUAL_GROUP8_BINARIES","binaries":[check_one(t) for t in mains]}

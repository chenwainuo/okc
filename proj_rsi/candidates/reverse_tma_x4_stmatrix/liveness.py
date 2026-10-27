"""CPU-only conservative physical register retirement gate for this fixed main kernel."""
import re


def parse(text):
    return [(int(m[1],16),m[2].strip()) for m in
            re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);",text)]


def plain(ins):return re.sub(r"^@!?P\d+\s+","",ins)


def writes(ins):
    ins=plain(ins)
    if ins.startswith("HGMMA."):return set()
    m=re.match(r"([A-Z0-9_.]+)\s+(?:P\d+,\s*)?R(\d+)(?:,|\s)",ins)
    if not m:return set()
    op,start=m[1],int(m[2])
    # Every payload LDG.128 writes4. Other possible first-GPR writes are
    # conservatively widened to4, never narrowed to hide a write hazard.
    width=4
    return set(range(start,start+width))


def check_one(table):
    rows=parse(table);assert rows
    h=[(pc,ins) for pc,ins in rows if plain(ins).startswith("HGMMA.")]
    assert len(h)==16
    stage_sets=[]
    for start in range(0,16,4):
        regs=set()
        for pc,ins in h[start:start+4]:
            m=re.search(r"HGMMA\.[^ ]+ R(\d+), R(\d+),",plain(ins));assert m,ins
            d,a=int(m[1]),int(m[2]);assert d==24
            assert not set(range(d,d+32))&set(range(a,a+4))
            regs.update(range(a,a+4))
        assert len(regs)==16;stage_sets.append(regs)
    assert len(set.union(*stage_sets))==64,"native4slots must be physically distinct"
    all_a=set.union(*stage_sets)
    wait0=min(pc for pc,ins in rows if "WARPGROUP.DEPBAR.LE" in ins and "0x0" in ins)
    last_h=h[-1][0]
    back=[]
    for pc,ins in rows:
        m=re.fullmatch(r"BRA(?:\.[A-Z]+)* 0x([0-9a-f]+)",plain(ins))
        if m and last_h<pc<wait0 and int(m[1],16)<h[0][0]:back.append((pc,int(m[1],16)))
    assert back,"no recognized fixed mainloop backedge"
    end,lo=max(back);body=[(pc,ins) for pc,ins in rows if lo<=pc<=end]
    assert sum("HGMMA." in ins for pc,ins in body)==16
    assert sum("WARPGROUP.DEPBAR.LE" in ins for pc,ins in body)==4
    # Outlined wait branches may block; they cannot write any A register.
    slow=[]
    for pc,ins in body:
        m=re.fullmatch(r"BRA(?:\.[A-Z]+)* 0x([0-9a-f]+)",plain(ins))
        if not m:continue
        target=int(m[1],16)
        if lo<=target<=end:continue
        assert target>wait0,(pc,ins)
        block=[]
        for q,x in rows:
            if q<target:continue
            block.append((q,x))
            ret=re.fullmatch(r"BRA 0x([0-9a-f]+)",plain(x))
            if ret and lo<=int(ret[1],16)<=end:break
        assert block and re.fullmatch(r"BRA 0x([0-9a-f]+)",plain(block[-1][1])),block[-1:]
        assert not any(writes(x)&all_a for q,x in block),(target,block)
        slow.append(target)
    pending=[];current=set();groups=0;refill_writes=0;events=[]
    # Two loop iterations cover cyclic inter-slot retirement as well as startup.
    # Execute all predicate-qualified writes: conservative for skipped startup/tail.
    for iteration in range(2):
        for pc,raw in body:
            ins=plain(raw)
            if ins.startswith("HGMMA."):
                a=int(re.search(r"HGMMA\.[^ ]+ R\d+, R(\d+),",ins)[1])
                current.update(range(a,a+4))
                if "gsb0" in ins:
                    assert len(current)==16
                    pending.append(set(current));current.clear();groups+=1
            elif ins.startswith("WARPGROUP.DEPBAR.LE"):
                keep=int(re.search(r", 0x([01])",ins)[1],16)
                assert not current and keep==1
                retired=max(0,len(pending)-keep)
                if retired:pending=pending[-keep:]
                events.append({"iteration":iteration,"PC":pc,"retired_groups":retired})
            else:
                dst=writes(ins)&all_a
                if not dst:continue
                live=set(current)
                for regs in pending:live.update(regs)
                assert not dst&live,{"error":"ASYNC_A_WRITE_BEFORE_RETIRE","iteration":iteration,
                                     "PC":hex(pc),"instruction":raw,"clobbered":sorted(dst&live)}
                refill_writes+=1
    assert groups==8 and len(pending)==1 and not current
    return {"status":"PASS_PHYSICAL_A4_RETIREMENT","iterations":2,"groups":groups,
            "A_register_sets":[sorted(regs) for regs in stage_sets],"payload_or_conservative_A_writes_checked":refill_writes,
            "outlined_wait_paths_no_A_writes":sorted(set(slow)),"retirement_events":events,
            "final":"wait0 retires lastgroup before epilogue","loop_PC_range":[lo,end],
            "limits":"Fixed unrolledmain CFG; predicatewrites conservatively all executed; slowwaitpaths checked for noAwrite. Not a general SASS control/dataflow verifier."}


def check(sass):
    tables=re.split(r"\n\s*Function : ",sass)[1:]
    mains=[table for table in tables if "rq_tma_x4_stmx" in table.splitlines()[0]]
    assert mains,"no rq_tma_x4_stmx kernel in actual artifact"
    return {"status":"PASS_ALL_ACTUAL_MAIN_BINARIES","binaries":[check_one(table) for table in mains]}

def store_form(sass,kernel_token,stmatrix,dual=False):
    """Check the intended store form and retained proxy/barrier in exact JIT SASS."""
    tables=[t for t in re.split(r"\n\s*Function : ",sass)[1:] if kernel_token in t.splitlines()[0]]
    assert tables,"missing expected store-version kernel"
    result=[]
    for table in tables:
        rows=parse(table)
        matrix=[(pc,ins) for pc,ins in rows if plain(ins).startswith("STSM.")]
        scalar=[(pc,ins) for pc,ins in rows if re.match(r"STS(?:\.|\s)",plain(ins))]
        assert len(matrix)==(4 if stmatrix else 0) and len(scalar)==(16 if stmatrix else 32)
        if stmatrix:
            assert all(ins.startswith("STSM.16.M88.4 ") for pc,ins in matrix),matrix
            last=max(pc for pc,ins in matrix)
        else:last=max(pc for pc,ins in scalar)
        proxy=min(pc for pc,ins in rows if pc>last and ins.startswith("FENCE.VIEW.ASYNC.S"))
        barrier_ids=(3,4) if dual else (3,)
        barriers={str(b):min(pc for pc,ins in rows if pc>proxy and plain(ins)==f"BAR.SYNC.DEFER_BLOCKING 0x{b:x}, 0x80") for b in barrier_ids}
        predicate_proof=None
        if dual:
            selected=[next(ins for pc,ins in rows if pc==barriers[str(b)]) for b in (3,4)]
            preds=[re.fullmatch(r"@(!?)P(\d+) BAR.SYNC.DEFER_BLOCKING 0x[34], 0x80",ins) for ins in selected]
            assert all(preds) and preds[0][2]==preds[1][2] and preds[0][1]!=preds[1][1],selected
            lo,hi=sorted(barriers.values())
            between=[ins for pc,ins in rows if lo<pc<hi]
            # Exact fixed artifacts use either adjacent BARs or only these
            # control instructions, none of which writes any predicate.
            assert all(re.fullmatch(r"(?:WARPSYNC.ALL|NOP|BSSY B\d+, 0x[0-9a-f]+)",plain(ins)) for ins in between),between
            predicate_proof={"same_predicate":"P"+preds[0][2],"complementary_bar3_bar4":True,
                             "instructions":selected,"intervening_no_predicate_writer_instructions":between}
        assert any(last<pc<proxy and ins=="MEMBAR.ALL.CTA" for pc,ins in rows)
        assert sum("HGMMA." in ins for pc,ins in rows)==16
        assert sum("LDG.E.NA.128.CONSTANT" in ins for pc,ins in rows)==0
        assert not any("SHFL.BFLY" in ins for pc,ins in rows)
        result.append({"STSM":len(matrix),"scalar_STS":len(scalar),"fullwarp_unpredicated_STSM":stmatrix,
                       "last_producer_store_PC":last,"proxy_fence_PC":proxy,"producer_barriers_PC":barriers,"owner_barrier_predicates":predicate_proof,
                       "shared_payload_bytes_per_stage":8192,"HGMMA":16,"X_LDG128":0})
    return {"status":"PASS_INTENDED_STORE_FORM_AND_RETAINED_PROXY_OWNER_BARRIERS","binaries":result}

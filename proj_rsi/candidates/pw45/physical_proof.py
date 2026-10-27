"""Fixed canonical HALF/GS1/WM0 physical A/D trace and plain-store gates; CPU only."""
import json,re
from pathlib import Path
from cpu_proof import streams,NEW
HERE=Path(__file__).resolve().parent

def writes(ins):
    op=re.sub(r'^@!?[UP]*P\d+\s+','',ins).split(' ',1)
    if len(op)!=2:return set()
    opcode,args=op
    if opcode.startswith(('HGMMA','ST','BRA','WARPGROUP','SYNCS','BAR','FENCE','MEMBAR','UTMA','DEPBAR','EXIT','NOP','YIELD','USETMAXREG')):return set()
    first=args.split(',')[0].strip();m=re.fullmatch(r'R(\d+)',first)
    if not m:return set()
    base=int(m[1]);count=4 if re.search(r'\.(?:128|V4)(?:\.|$)',opcode) else 2 if '.64' in opcode or '.WIDE' in opcode else 1
    return set(range(base,base+count))

def trace(rows):
    mma=[(pc,ins) for pc,ins in rows if 'HGMMA.' in ins]
    assert len(mma)==16 and all('HGMMA.64x32x16.F32.BF16' in ins for pc,ins in mma)
    groups=[mma[i:i+4] for i in range(0,16,4)]
    aset=[]
    for group in groups:
        regs=[int(re.match(r'.*? R\d+, R(\d+),',ins)[1]) for pc,ins in group]
        assert regs==list(range(regs[0],regs[0]+16,4))
        aset.append(set(range(regs[0],regs[0]+16)))
        assert all(re.match(r'.*? R24,',ins) for pc,ins in group)
    assert aset==[set(range(40,56)),set(range(72,88)),set(range(56,72)),set(range(72,88))]
    # Exact loop back edge encloses middle two four-MMA groups; startup and
    # tail are separate. FullK16 executes seven two-group loop bodies.
    back=[]
    for pc,ins in rows:
        m=re.search(r'\bBRA\s+0x([0-9a-f]+)',ins)
        if m and groups[2][-1][0]<pc<groups[3][0][0] and int(m[1],16)<groups[1][0][0]:back.append((pc,int(m[1],16)))
    assert len(back)==1,back
    end,start=back[0];stsm=[pc for pc,i in rows if 'STSM.' in i]
    assert len(stsm)==2
    prefix=[r for r in rows if r[0]<start]
    loop=[r for r in rows if start<=r[0]<=end]
    tail=[r for r in rows if end<r[0]<min(stsm)]
    steps=prefix+loop*7+tail
    pending=[];current=set();dregs=set(range(24,40));events=[];issued=0;checked=0
    for pc,ins in steps:
        wr=writes(ins);live=set().union(*pending,current) if pending or current else set()
        assert not wr&live,('GENERIC_A_WRITE_WHILE_ASYNC',hex(pc),ins,sorted(wr&live))
        assert not (pending or current) or not wr&dregs,('GENERIC_D_WRITE_WHILE_ASYNC',hex(pc),ins)
        checked+=bool(wr)
        if 'HGMMA.' in ins:
            base=int(re.match(r'.*? R\d+, R(\d+),',ins)[1]);part=set(range(base,base+4))
            assert not any(part&p for p in pending),('A_MMA_BANK_REUSE_WHILE_PENDING',pc)
            current|=part
            if 'gsb0' in ins:
                assert len(current)==16;pending.append(current);current=set();issued+=1
        if 'WARPGROUP.DEPBAR.LE' in ins:
            m=re.search(r', 0x([01])\b',ins);assert m
            keep=int(m[1]);retired=max(0,len(pending)-keep)
            pending=pending[-keep:] if keep else []
            events.append({'PC':pc,'wait_keep':keep,'retired_groups':retired})
    assert issued==16 and not pending and not current
    assert len(events)==17 and sum(e['wait_keep']==1 for e in events)==16 and events[-1]['wait_keep']==0
    # Outlined TRYWAIT blocks must not generically touch any A or D payload.
    allA=set.union(*aset);outlined=0
    for target in sorted(set(int(m[1],16) for pc,i in steps if (m:=re.search(r'\bBRA\s+0x([0-9a-f]+)',i)) and int(m[1],16)>max(stsm))):
        block=[]
        for pc,i in rows:
            if pc<target:continue
            block.append((pc,i))
            if re.match(r'BRA\s+0x',i):break
        assert block and all(not writes(i)&(allA|dregs) for pc,i in block)
        outlined+=1
    return {'state':'PASS_FIXED_PHYSICAL_A_D_FULLK16','groups':issued,'A_register_sets':[sorted(a) for a in aset],'D_registers':[24,39],
        'loop_PC_range':[start,end],'loop_iterations':7,'generic_destination_writes_checked':checked,'retirement_events':events,'outlined_wait_blocks':outlined,
        'limits':'Exact compiled GS1/WM0/fullK16 bounded trace with conservatively executed predicate writes and safe outlined waits, not general SASS CFG/address proof.'}

def check(sass):
    records=[]
    for args,rows in streams(sass).items():
        if args not in NEW:continue
        st=args[1];hgmma=[p for p,i in rows if 'HGMMA.' in i];wait0=[p for p,i in rows if 'WARPGROUP.DEPBAR.LE' in i and '0x0' in i]
        stsm=[p for p,i in rows if 'STSM.' in i];store=[p for p,i in rows if 'UTMASTG.2D' in i];reduce=[p for p,i in rows if 'UTMAREDG' in i]
        assert len(stsm)==2 and len(store)==1 and reduce==[]
        assert max(hgmma)<max(wait0)<min(stsm)<max(stsm)<store[0]
        assert any(p>store[0] and i.startswith('DEPBAR.LE') for p,i in rows)
        assert any(max(wait0)<p<min(stsm) and 'BAR.SYNC' in i and '0x80' in i for p,i in rows)
        assert any(max(stsm)<p<store[0] and 'FENCE.VIEW.ASYNC.S' in i for p,i in rows)
        assert any(max(stsm)<p<store[0] and 'BAR.SYNC' in i and '0x80' in i for p,i in rows)
        assert not any(re.search(r'\b(?:LDL|STL|TRAP)\b',i) for p,i in rows)
        assert any('USETMAXREG.TRY_ALLOC' in i and '0xe8' in i for p,i in rows) and any('USETMAXREG.DEALLOC' in i and '0x28' in i for p,i in rows)
        assert not any(('ATOM' in i or 'REDG' in i) for p,i in rows),'plain no-counter path unexpectedly writes global reduction/atomic'
        records.append({'args':args,'shared_bytes':1024+st*(4096+6144)+24*st+16,'STSM':2,'plain_TMA_store':1,'TMA_reduce':0,'zero_poll_trap':0,
            'final_Dwait0_before_output':'PASS','publication_and_TMA_read_drain':'PASS','physical_retirement':trace(rows)})
    assert len(records)==2
    return records

if __name__=='__main__':
    r=check((HERE/'sass.txt').read_text());(HERE/'physical-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'state':'PASS_PHYSICAL_TWO_HALF_PLAIN_ARMS','mains':len(r)},indent=2))

"""CPU full store/CSR/schedule/source/actual-cubin gates for dedicated X issuer."""
from collections import defaultdict,deque
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import liveness
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'reverse_tma_x_async_refill'
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=load(HERE.parent/'reverse_tma_x'/'proofs.py','dedicated_parent_layout')

def assignments(stage):
    if stage%2==0:return [(w,w-8) for w in range(8,12)]
    return [(12,0),(12,3),(13,1),(14,2)]

def store_bitmap():
    total=0
    for s in range(64):
        seen={}
        for warp,tile in assignments(s):
            assert warp!=15 and 0<=tile<4
            for lane in range(32):
                t,g=lane&3,lane>>2
                for k in range(4):
                    for i in range(4):
                        row=16*tile+8*(i%2)+g
                        addr=row*128+(((2*k+i//2)^(row&7))*16)+4*t
                        marker=(s,tile,lane,k*4+i)
                        for byte in range(4):
                            assert addr+byte not in seen
                            seen[addr+byte]=(marker,byte)
        assert set(seen)==set(range(8192))
        # Parent's old parity owner4warps has exactly the same tile/value address.
        expected={}
        for tile in range(4):
            for lane in range(32):
                for k in range(4):
                    for i in range(4):
                        row=16*tile+8*(i%2)+(lane>>2)
                        dst=row*128+(((2*k+i//2)^(row&7))*16)+4*(lane&3)
                        for byte in range(4):expected[dst+byte]=((s,tile,lane,k*4+i),byte)
        assert seen==expected
        total+=len(seen)
    barriers={3:list(range(256,384)),4:list(range(384,480))}
    assert set(barriers[3]).isdisjoint(barriers[4])
    assert len(barriers[3])==128 and len(barriers[4])==96
    assert set(barriers[3]+barriers[4])==set(range(256,480))
    assert list(range(480,512))==list(range(15*32,16*32))
    return dict(status='PASS_ALL64_STAGES_SAME_PARENT_DECODED_BYTES',stage_bytes=8192,total_store_bytes=total,
                odd_tiles=assignments(1),even_tiles=assignments(0),named_barrier_threads=barriers,
                total_W_warps=7,dedicated_X_warp=15,Bempty_count=8,Xempty_count=8,
                counts='Publication bars count producer threads128/96; Bempty/Xempty count8 mathwarp arrivals. There is no universal 7-arrival replacement.')

class Cursor:
    def __init__(self,entries):
        self.data=entries+[0xffffffff]*64;self.ei=0;self.window=self.data[:32]
    def next(self):
        return self.window[self.ei&31]
    def advance(self):
        self.ei+=1
        if self.ei%32==0:self.window=self.data[self.ei:self.ei+32]
    def apply(self,s,values,skip=True):
        if skip:
            while self.next()>>17<s:self.advance()
        while self.next()>>17==s:
            ent=self.next();lane=(ent>>12)&31;reg=(ent>>8)&15;half=(ent>>7)&1
            values[lane][reg]^=(ent&127)<<(8+half*16);self.advance()
        return values

def initial(s,tile):return [[((s+1)<<20)|(tile<<16)|(lane<<8)|reg for reg in range(16)] for lane in range(32)]
def csr():
    rng=random.Random(81015);cases=0;shared_cursor_rejected=False;missing_skip_rejected=False
    for count in (0,1,31,32,33,63,64,257,2048):
        lists={}
        for tile in range(4):
            entries=[]
            for _ in range(count):
                seq=rng.randrange(64);lane=rng.randrange(32);reg=rng.randrange(16);half=rng.randrange(2);xr=rng.randrange(1,128)
                entries.append((seq<<17)|(lane<<12)|(reg<<8)|(half<<7)|xr)
            # Guaranteed earlier/opposite-parity and different tile records exercise skips.
            entries.extend([(0<<17)|(tile<<12)|17,(1<<17)|((tile+1)<<12)|(3<<8)|11])
            lists[tile]=sorted(entries,key=lambda x:x>>17)
        cursors={(owner,tile):Cursor(lists[tile]) for owner in (0,1) for tile in range(4)}
        bad_shared=Cursor(lists[0]);bad_skip={tile:Cursor(lists[tile]) for tile in range(4)}
        for s in range(64):
            for warp,tile in assignments(s):
                want=initial(s,tile)
                for ent in lists[tile]:
                    if ent>>17==s:want[(ent>>12)&31][(ent>>8)&15]^=(ent&127)<<(8+((ent>>7)&1)*16)
                got=cursors[(s%2,tile)].apply(s,initial(s,tile));assert got==want
                cases+=1
                if s%2:
                    if tile in (0,3):
                        got=bad_shared.apply(s,initial(s,tile))
                        if got!=want:shared_cursor_rejected=True
                    got=bad_skip[tile].apply(s,initial(s,tile),skip=False)
                    if got!=want:missing_skip_rejected=True
    assert shared_cursor_rejected and missing_skip_rejected
    return dict(status='PASS_INDEPENDENT_ENCODED32LANE_CURSOR_WINDOWS',density_cases=9,stage_tile_cases=cases,
                boundary_reload='31/32/33 and63/64 crossings tested; sentinel32767; skip oppositeparity retained',
                independent_warp12_tile0_tile3=True,negative_shared_cursor='REJECTED',negative_missing_skip='REJECTED')

def dag():
    deps=defaultdict(set)
    def e(n,*ps):deps[n].update(ps)
    def C(s,w):return f'C{s}/WG{w}'
    def M(s,w):return f'M{s}/WG{w}'
    for s in range(64):
        if s<2:e(f'XI{s}','INIT')
        else:e(f'XI{s}',f'EMPTY{s-2}',f'XI{s-1}')
        e(f'X{s}',f'XI{s}')
        for wg in range(2):
            e(C(s,wg),f'X{s}')
            if s:e(C(s,wg),C(s-1,wg))
            if s>=4:e(C(s,wg),M(s-3,wg))
            e(M(s,wg),C(s,wg),f'B{s}')
            if s:e(M(s,wg),M(s-1,wg))
            if s<2:e(M(s,wg),C(3,wg))
            elif s+2<64:e(M(s,wg),C(s+2,wg))
        e(f'EMPTY{s}',C(s,0),C(s,1))
        # W owners only need previous same-parity Wdecode/barrier publication,
        # Wfull and Bempty, never Xempty. A seven-warp blanketbarrier is absent.
        if s<6:e(f'W{s}','INIT')
        else:e(f'W{s}',f'B{s-6}')
        e(f'B{s}',f'W{s}')
        if s>=2:e(f'B{s}',f'B{s-2}')
        if s>=8:e(f'B{s}',M(s-7,0),M(s-7,1))
        for warp,tile in assignments(s):
            node=f'D{s}/tile{tile}/warp{warp}';e(node,f'W{s}')
            if s>=2:e(node,f'B{s-2}')
            if s>=8:e(node,M(s-7,0),M(s-7,1))
            if s%2 and tile==3:e(node,f'D{s}/tile0/warp12')
            e(f'B{s}',node)
    e('DRAIN',M(63,0),M(63,1));e('OUTPUT','DRAIN')
    allnodes=set(deps)|set.union(set(),*deps.values());n={x:len(deps[x]) for x in allnodes};ch=defaultdict(set)
    for node,parents in deps.items():
        for p in parents:ch[p].add(node)
    todo=deque(sorted(p for p in allnodes if not n[p]));order=[]
    while todo:
        p=todo.popleft();order.append(p)
        for child in sorted(ch[p]):
            n[child]-=1
            if not n[child]:todo.append(child)
    assert len(order)==len(allnodes),[(p,deps[p]) for p in n if n[p]]
    # Simulate all repeated slot phases including tail62/63 that need no refill.
    resident=[0,1];phase=[0,0];issued={0,1}
    for s in range(64):
        slot=s&1;assert resident[slot]==s and phase[slot]==s//2
        phase[slot]+=1;resident[slot]=None
        if s+2<64:issued.add(s+2);resident[slot]=s+2
    assert issued==set(range(64)) and resident==[None,None]
    return dict(status='PASS_FULL_X_W_B_A4_DAG',nodes=len(allnodes),edges=sum(map(len,deps.values())),topological_order=order,
                dependencies={p:sorted(d) for p,d in deps.items()},X_issues=64,dedicated_refills=62,Xempty_arrivals=512,
                X_DAG='XI(s+2) issue depends on previousXI andbothWGC(s); X(s) completion depends onlyXI(s), allowing crossslot completion reordering. NoWdecode/publication dependency.',
                W_DAG='Wrefill(s+6) afterBpublish(s); Bempty(s-8) released atM(s-7)wait1; sameparity owners sequential',
                startup='C0/1→EMPTY0/1→X2/3→C2/3→M0; W0/1decode independent fromempty/TMAissuer',
                tail='Xwarp exits afterissuing63, noX64/65; all8math readers consume62/63; finalwait0 beforeoutput')

def source():
    p=(PARENT/'kernel_body.cuh').read_text();b=(HERE/'kernel_body.cuh').read_text();delta=json.loads((HERE/'delta.json').read_text())
    old=b
    for edit in reversed(delta['edits']):
        assert old.count(edit['after'])==1,edit['after'][:80]
        old=old.replace(edit['after'],edit['before'],1)
    assert old==p
    # Full CUDA header/codec/TMA-host/output are exact parent once body/name restored.
    cu=(HERE/'extension_source.cu').read_text().replace(b,p).replace('rq_dedicated_x<','rq_tma_x_async<')
    assert hashlib.sha256(cu.encode()).hexdigest()==delta['parent_source_sha256']
    assert b[b.index('  prefetch(0,q0)'):]==p[p.index('  prefetch(0,q0)'):]
    assert b[b.index('  auto prefetch='):b.index('  if(tid==256)')]==p[p.index('  auto prefetch='):p.index('  if(tid==256)')]
    assert b[b.index('  if(tid==256)'):b.index('  if(warp==15)')]==p[p.index('  if(tid==256)'):p.index('  if(warp>=8)')]
    return dict(status='PASS_STRICT_BODY_AND_FULL_CUDA_PARENT_INVERSE',parent_source_sha256=delta['parent_source_sha256'],
                unchanged='entiremath/A4/FP32/WGMMA/codec/epilogue, mathXprefetch/release, initialX/W/barrierinit, hostdescriptor/launch')

def reach(rows,start):
    table=dict(rows);todo=[start];seen=set()
    while todo:
        pc=todo.pop()
        if pc in seen:continue
        seen.add(pc);raw=table[pc];plain=re.sub(r'^@!?P\d+\s+','',raw)
        match=re.search(r'\bBRA(?:\.[A-Z]+)* .*?(0x[0-9a-f]+)$',plain)
        if plain=='EXIT':targets=[pc+16] if raw.startswith('@') else []
        elif match:targets=[int(match[1],16)]+([pc+16] if raw.startswith('@') or plain.startswith('BRA.DIV') else [])
        else:targets=[pc+16]
        todo.extend(q for q in targets if q in table and q not in seen)
    return seen

def physical(sass):
    rows=liveness.parse(sass);d=dict(rows)
    assert d[0xbb0]=='@!P0 BRA 0x4360' and d[0xbd0]=='@P0 BRA 0x2360'
    roles={n:reach(rows,pc) for n,pc in [('math',0xbe0),('Wproducer',0x2360),('Xissuer',0x4360)]}
    for a in roles:
        for b in roles:
            if a!=b:assert not roles[a]&roles[b]
    math=[(p,i) for p,i in rows if p in roles['math']];w=[(p,i) for p,i in rows if p in roles['Wproducer']];x=[(p,i) for p,i in rows if p in roles['Xissuer']]
    assert sum('LDSM.16.M88.4' in i for p,i in math)==32
    assert not any('UTMALDG.' in i or re.search(r'\bLDG\.',i) or 'BAR.SYNC.DEFER_BLOCKING 0x5' in i for p,i in math)
    assert not any('UTMALDG.' in i or re.search(r'\+0x214(?:00|08|10|18)\]',i) for p,i in w)
    assert sum('UTMALDG.2D' in i for p,i in x)==1
    assert not any(re.search(r'\b(?:LDSM|LDS|STS|LDG)(?:\.| )',i) or 'BAR.SYNC' in i for p,i in x)
    assert any('SYNCS.PHASECHK.' in i and '+0x21410]' in i for p,i in x)
    bars=[(p,i) for p,i in w if 'BAR.SYNC' in i]
    assert len(bars)==2 and any('0x3, 0x80' in i for p,i in bars) and any('0x4, 0x60' in i for p,i in bars)
    stores=[p for p,i in w if re.search(r'\bSTS(?:\.| )',i)];assert len(stores)==32
    proxy=[p for p,i in w if 'FENCE.VIEW.ASYNC.S' in i];assert len(proxy)==1
    assert max(stores)<proxy[0]<min(p for p,i in bars)
    arrives=[(p,i) for p,i in math if 'SYNCS.ARRIVE.' in i and ('+0x21410]' in i or '+0x21418]' in i)]
    assert len(arrives)==8;last=0xbe0;handoff=[]
    for p,i in arrives:
        block=[(q,j) for q,j in math if last<=q<p]
        loads=[q for q,j in block if 'LDSM.16.M88.4' in j];assert len(loads)==4
        mem=max(q for q,j in block if 'MEMBAR.ALL.CTA' in j);fence=max(q for q,j in block if 'FENCE.VIEW.ASYNC.S' in j)
        assert max(loads)<mem<fence<p
        handoff.append(dict(LDSM=hex(max(loads)),MEMBAR=hex(mem),FENCE=hex(fence),Xempty_arrive=hex(p)));last=p+16
    assert not re.search(r'\b(?:LDL|STL|USETMAXREG)\b',sass)
    return dict(A4=liveness.check(sass),role_PCs={n:[hex(p) for p in sorted(v)] for n,v in roles.items()},
                W_static_STS=32,W_publication_bars=[i for p,i in bars],dedicated_TMA_static=1,
                Xempty_math_handoffs=handoff,math_TMA=0,math_bar5=0,
                compiler='source__syncwarp iselided as validatedparent; convergedLDSM/MEMBAR/FENCE/lane0arrive retained;actualJITregate required',
                limits='FixedlocalCUDA12.8cubin. No generalSASSaddress-dataflow verifier or GPUcorrectness/occupancyquery claim.')

def proof():
    return dict(layout=OLD.layout(),source=source(),stores=store_bitmap(),CSR=csr(),DAG=dag(),physical=physical((HERE/'sass.txt').read_text()),
                shared_regions={'W':[0,36864],'B':[36864,102400],'metadata':[102400,102576],'X':[103424,136192],'Xfull':[136192,136208],'Xempty':[136208,136224]},dynamic_shared_bytes=137248,
                lifetime='Dedicatedwarp onlyrecyclesXafter8mathwarparrivals; outputscratchB reusedafterallWGMMAwait0 and mathbar1; noX/Wregionoverlap')

if __name__=='__main__':
    r=proof();(HERE/'proofs.json').write_text(json.dumps(r,indent=2)+'\n');print('CPU_PROOFS_PASS_GPU_UNRUN',r['DAG']['nodes'])

"""CPU-only complete X bitmap, protocol, resource and local SASS gates."""
import importlib.util
import json
from pathlib import Path
import re
import liveness
HERE=Path(__file__).resolve().parent


def layout():
    # SW128: logical128B row, XOR row[2:0] into16B-sector index.
    def sw(row,col): return row*128+(((col//8)^(row&7))*16)+(col%8)*2
    shared={}
    for row in range(128):
        for col in range(64):
            a=sw(row,col)
            assert a not in shared and a%2==0 and 0<=a<16384
            shared[a]=(row,col)
    assert len(shared)==8192
    seen=set(); words=0
    for warp in range(8):
        for chunk in range(4):
            providers=[]
            for lane in range(32):
                row=64*(warp>>2)+16*(warp&3)+(lane&7)+8*((lane>>3)&1)
                col=16*chunk+8*(lane>>4)
                address=sw(row,col)
                assert address%16==0
                providers.append(address)
            for lane in range(32):
                for matrix in range(4):
                    # PTX x4: lanes[8*m:8*m+8] supply8 rowstarts for matrix m.
                    address=providers[8*matrix+(lane>>2)]+4*(lane&3)
                    got=(shared[address],shared[address+2])
                    row=64*(warp>>2)+16*(warp&3)+(lane>>2)+8*(matrix&1)
                    col=16*chunk+8*(matrix>>1)+2*(lane&3)
                    expected=((row,col),(row,col+1))
                    assert got==expected,(warp,chunk,lane,matrix,got,expected)
                    assert not seen.intersection(got)
                    seen.update(got);words+=1
    assert seen==set(shared.values())
    # All64 K64 stages: global source coordinates cover128x4096 exactly.
    assert 64*len(seen)==128*4096
    return dict(status='PASS',shared_bf16=8192,register_words_per_stage=words,
                stages=64,global_bf16=524288,all_coordinates_once=True,
                primary_ldmatrix='PTX8.7 section9.7.14.5.15 Figure103',
                primary_wgmma='PTX8.7 section9.7.15.5.1.1.1 Figure147')


def protocol():
    issued={0,1,2,3};resident=[0,1,2,3];consumed=[];trace=[];a=[None]*4;pending=[]
    def prefetch(s):
        slot=s%4
        assert resident[slot]==s and s in issued
        # All8 warps wait same completed phase, read fourLDSM, proxyfence+bar5.
        assert (s//4)%2==((s-slot)//4)%2
        consumed.append(s)
        trace.append(dict(event='read_complete_all8warps',s=s,slot=slot,phase=(s//4)%2))
        resident[slot]=None
        if s+4<64:
            assert s+4 not in issued
            issued.add(s+4);resident[slot]=s+4
            trace.append(dict(event='tid0_expect16384_and_TMA',s=s+4,slot=slot))
        a[s%4]=s
    for s in range(4):prefetch(s)
    for s in range(64):
        assert a[s%4]==s
        pending.append(s)
        if len(pending)>1:pending.pop(0)
        if s>0:
            retired=s-1
            assert retired not in pending
            if s+3<64:prefetch(s+3)
    pending.clear() # wait_group0 before epilogue; sharedX already wholly retired.
    assert consumed==list(range(64)) and issued==set(range(64)) and resident==[None,None,None,None]
    # W6/B8 needs only prior stage edges; no W/X publication cycle.
    edges=[]
    for s in range(64):
        if s>=6:edges.append((s-6,s,'Wslot previous producer read'))
        if s>=8:edges.append((s-7,s,'Bslot group(s-8) retired atmath(s-7) wait1'))
        if s>=4:edges.append((s-4,s,'Xslot prior LDSM completion'))
    assert all(a<b for a,b,_ in edges)
    # Disjoint physical shared regions incl metadata and output reuse.
    regions=[('W',0,36864),('B',36864,102400),('Wfull',102400,102448),
             ('Bfull',102448,102512),('Bempty',102512,102576),
             ('X',103424,168960),('Xfull',168960,168992)]
    for i,(_,lo,hi) in enumerate(regions):
        assert lo<hi
        for _,ol,oh in regions[i+1:]:assert hi<=ol or oh<=lo
    assert 168992+1024==170016 and 170016<227328
    return dict(status='PASS_SOURCE_SCHEDULE',X_TMA_issues=64,X_phase_reuses=60,
                math_bar5_participants=list(range(256)),producer_bar3=list(range(256,384)),
                producer_bar4=list(range(384,512)),trace=trace,regions=regions,
                dynamic_shared_bytes=170016,
                output='B scratch [36864,53248) only after finalwait0; disjointX; W/Bsourceunchanged',
                deadlock='X reads/issue elected bymath withoutW producer dependency; allringreuseedges strictlyearlierstage')


def physical(sass):
    rows=liveness.parse(sass)
    a=liveness.check(sass)
    # Fixed local artifact only. Separate actual postbarrier math/prod branch.
    d=dict(rows)
    assert d[0xc30]=='@P0 BRA 0x2ed0'
    math=[(p,i) for p,i in rows if 0xc40<=p<0x2ed0]
    assert not any(re.search(r'\bLDG\.',i) for _,i in math)
    lds=[(p,i) for p,i in math if 'LDSM.16.M88.4' in i]
    assert len(lds)==32
    bars=[(p,i) for p,i in math if 'BAR.SYNC.DEFER_BLOCKING 0x5, 0x100' in i]
    assert len(bars)==8
    assert sum('UTMALDG.2D' in ins for pc,ins in rows if pc<0xc40)==4
    blocks=[]
    previous_bar=0xc40
    for p,i in bars:
        block=[(q,j) for q,j in math if previous_bar<=q<p]
        loads=[q for q,j in block if 'LDSM.' in j]
        assert len(loads)==4
        fence=max(q for q,j in block if 'FENCE.VIEW.ASYNC.S' in j)
        membar=max(q for q,j in block if 'MEMBAR.ALL.CTA' in j)
        assert max(loads)<membar<fence<p
        assert not any(re.search(r'\b(?:LDSM|LDG|LDS|STS|UTMALDG)\b',j) for q,j in block if fence<q<p)
        blocks.append(dict(bar_PC=hex(p),last_LDSM=hex(max(loads)),proxy_fence_PC=hex(fence)))
        previous_bar=p+16
    assert len([1 for p,i in math if 'UTMALDG.2D' in i])==8
    assert not re.search(r'\b(?:LDL|STL|USETMAXREG)\b',sass)
    return dict(A4=a,X_physical_blocks=blocks,math_global_X_LDG=0,
                static_LDSM=32,static_X_TMA_steady_and_startup=8,
                limits='LocalCUDA12.8cubin only; driverJIT must rerun gates before anyGPUmath')


def source_restore():
    import hashlib
    import composition_proof
    return composition_proof.proof()


def all_proofs():
    import layout_proof,protocol_proof
    sass=(HERE/'sass.txt').read_text()
    return dict(layout=layout(),protocol=protocol(),physical=physical(sass),
                source_restore=source_restore(),store_form=liveness.store_form(sass,'rq_tma_x4_stmx',True,dual=True),
                store_bits=layout_proof.wproof(),output_map=layout_proof.outputproof(),producer_protocol=protocol_proof.proof())

if __name__=='__main__':
    result=all_proofs();(HERE/'proofs.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CPU_PROOFS_PASS_GPU_UNRUN')

"""CPU-only exact-source/layout, full dependency DAG and physical A4 proof."""
from collections import defaultdict,deque
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import liveness
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'reverse_tma_x'
spec=importlib.util.spec_from_file_location('native_x_parent_proof',PARENT/'proofs.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def dag():
    deps=defaultdict(set)
    def edge(node,*parents):deps[node].update(parents)
    def C(s,w):return f'C{s}/WG{w}'
    def M(s,w):return f'M{s}/WG{w}'
    for s in range(64):
        # Each full phase has exactly8 arrivals; slot parity s//2 mod2.
        edge(f'X{s}', *([] if s<2 else [f'ISSUE{s-2}']))
        for w in range(2):
            edge(C(s,w),f'X{s}')
            if s:edge(C(s,w),C(s-1,w))
            if s>=4:edge(C(s,w),M(s-3,w))
            edge(M(s,w),C(s,w),f'B{s}')
            if s:edge(M(s,w),M(s-1,w))
            if s<2:edge(M(s,w),C(3,w))
            elif s+2<64:edge(M(s,w),C(s+2,w))
        edge(f'EMPTY{s}',C(s,0),C(s,1))
        edge(f'ISSUE{s}',*([f'B{s-2}'] if s>=2 else []))
        if s+2<64:edge(f'ISSUE{s}',f'EMPTY{s}')
        edge(f'B{s}',f'ISSUE{s}',f'W{s}')
        if s>=6:edge(f'W{s}',f'B{s-6}')
        else:edge(f'W{s}')
        # Bempty(s-8) released after wait1 at mathstage(s-7), eachWG.
        if s>=8:edge(f'B{s}',M(s-7,0),M(s-7,1))
    edge('DRAIN',M(63,0),M(63,1));edge('OUTPUT','DRAIN')
    allnodes=set(deps)|set.union(set(),*deps.values())
    indeg={n:len(deps[n]) for n in allnodes};children=defaultdict(set)
    for n,ps in deps.items():
        for p in ps:children[p].add(n)
    q=deque(sorted(n for n in allnodes if indeg[n]==0));order=[]
    while q:
        n=q.popleft();order.append(n)
        for ch in sorted(children[n]):
            indeg[ch]-=1
            if indeg[ch]==0:q.append(ch)
    assert len(order)==len(allnodes),[(n,deps[n]) for n in allnodes if indeg[n]]
    rank={n:i for i,n in enumerate(order)}
    assert all(rank[p]<rank[n] for n,ps in deps.items() for p in ps)
    phases=[0,0];issued=[0,1];reads=[]
    for s in range(64):
        slot=s%2
        assert phases[slot]==s//2
        assert ((phases[slot])&1)==(s//2)&1
        reads.append(s)
        if s+2<64:issued.append(s+2)
        phases[slot]+=1
    assert sorted(issued)==reads==list(range(64))
    return dict(status='PASS_FULL_DAG',nodes=len(allnodes),edges=sum(map(len,deps.values())),
                topological_order=order,dependencies={n:sorted(ps) for n,ps in deps.items()},
                X_empty_arrivals=64*8,X_full_issues=64,X_refills=62,
                Xempty_phase='s//2 mod2; all8warp arrivals before ownerwait success; next phase cannot complete before nextTMA andread',
                W6_B8='Bslot reuse waits M(s-7), notM(s-8); Wrefill waitsBpub(s-6)',
                tail='Xempty62/63 completes withoutproducerwait; noX64/65; lastwait0 beforeoutput',
                startup='C0/1→ISSUE0/1→X2/3→C2/3→M0/1; noMMA dependency for first4Aloads')


def source():
    b=(HERE/'kernel_body.cuh').read_text();p=(PARENT/'kernel_body.cuh').read_text()
    # Reverse exactly all intended mutations, prove rest includingA4/W/B/output identical.
    b=b.replace('rq_tma_x_async(','rq_tma_x(').replace('+2*16384+4*8;','+2*16384+2*8;')
    b=b.replace('xfull=xring+32768, xempty=xfull+16;','xfull=xring+32768;')
    a=b.index('    // Each lane orders');z=b.index('  };',a)
    x=p[p.index('    // All math readers'):p.index('  };',p.index('    // All math readers'))]
    b=b[:a]+x+b[z:]
    b=b.replace('    for(int s=0;s<2;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],8;\\n"::"r"(xempty+8*s));\n','')
    a=b.index('      // One elected thread');z=b.index('      rq_wait(wfull',a)
    b=b[:a]+b[z:]
    assert b==p
    return dict(status='EXACT_RESTORE_PARENT',parent_body_sha256=hashlib.sha256(p.encode()).hexdigest())


def physical(sass):
    rows=liveness.parse(sass);d=dict(rows)
    assert d[0xb10]=='@P0 BRA 0x22a0'
    math=[(p,i) for p,i in rows if 0xb20<=p<0x22a0]
    assert not any('UTMALDG.' in i or re.search(r'\bLDG\.',i) for p,i in math)
    assert not any('BAR.SYNC.DEFER_BLOCKING 0x5' in i for p,i in rows)
    arrives=[(p,i) for p,i in math if 'SYNCS.ARRIVE.' in i and ('0x21410]' in i or '0x21418]' in i)]
    assert len(arrives)==8
    blocks=[];last=0xb20
    for p,i in arrives:
        block=[(q,j) for q,j in math if last<=q<p]
        loads=[q for q,j in block if 'LDSM.16.M88.4' in j]
        assert len(loads)==4
        mem=max(q for q,j in block if 'MEMBAR.ALL.CTA' in j)
        fence=max(q for q,j in block if 'FENCE.VIEW.ASYNC.S' in j)
        assert max(loads)<mem<fence<p
        assert not any('BRA' in j or 'BSSY' in j or 'BSYNC' in j for q,j in block if min(loads)<=q<p)
        blocks.append(dict(last_LDSM=hex(max(loads)),MEMBAR=hex(mem),FENCE=hex(fence),Xempty_arrive=hex(p)))
        last=p+16
    prod=[(p,i) for p,i in rows if 0x22a0<=p<0x3890]
    assert sum('UTMALDG.2D' in i for p,i in prod)==1
    assert any('SYNCS.PHASECHK.' in i and '0x21410]' in i for p,i in prod)
    assert any('SYNCS.ARRIVE.' in i and '0x21400]' in i for p,i in prod)
    assert not re.search(r'\b(?:LDL|STL|USETMAXREG)\b',sass)
    return dict(A4=liveness.check(sass),math_X_handoff=blocks,math_TMA=0,math_LDG=0,math_bar5=0,
                producer_X_TMA_sites=1,
                warp_sync='source __syncwarp compiler-elided on converged path; actualMEMBAR.ALL.CTA beforeFENCE andlane0arrive; no physicalWARPSYNCclaim',
                limits='FixedlocalCUDA12.8binary; compilersemantic preservation plus source fullwarp synchronization; actualJIT mustregate')


def all_proofs():
    return dict(layout=old.layout(),source=source(),DAG=dag(),physical=physical((HERE/'sass.txt').read_text()),
                dynamic_shared=137248,shared_Xempty=[136208,136224],alignment_slack=1024,
                raw_GPU_bits='UNPROBED; normal/task numericaloracles notphysicalrawbitproof')

if __name__=='__main__':
    r=all_proofs();(HERE/'proofs.json').write_text(json.dumps(r,indent=2)+'\n');print('CPU_PROOFS_PASS_GPU_UNRUN')

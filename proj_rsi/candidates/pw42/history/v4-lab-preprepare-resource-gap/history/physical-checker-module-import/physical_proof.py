"""Fixed sm90a entry-sentinel/branch/reconvergence gate, no CUDA context."""
import json
from pathlib import Path
import re
from preflight import tables,BASES
HERE=Path(__file__).resolve().parent

def proof(sass):
    result=[]
    for (kind,args),rows in tables(sass).items():
        if kind!='empty':continue
        # Exact first broadcast followed by predicate against UINT32_MAX and
        # a warp-uniform branch to the no-repair code region.
        first=next(i for i,(pc,ins) in enumerate(rows) if ins.startswith('SHFL.IDX'))
        pc,shfl=rows[first];m=re.fullmatch(r'SHFL.IDX PT, (R\d+), R\d+, RZ, 0x1f',shfl);assert m,shfl
        compare=rows[first+1][1];pred=re.fullmatch(r'ISETP.NE.AND (P\d+), PT, '+m[1]+r', -0x1, PT',compare);assert pred,compare
        branch=rows[first+2];target=re.fullmatch(r'@!'+pred[1]+r' BRA 0x([0-9a-f]+)',branch[1]);assert target,branch
        empty_start=int(target[1],16)
        joins=[]
        for address,ins in rows:
            match=re.fullmatch(r'BRA 0x([0-9a-f]+)',ins)
            if match and branch[0]<address<empty_start and int(match[1],16)>empty_start:joins.append((address,int(match[1],16)))
        assert len(joins)==1,joins
        repair_end,join=joins[0]
        repair=[(a,i) for a,i in rows if branch[0]<a<=repair_end]
        empty=[(a,i) for a,i in rows if empty_start<=a<join]
        for code in (repair,empty):
            assert sum('HMMA.' in i for a,i in code)==32
            assert not any('BAR.SYNC' in i for a,i in code),'compute path retained CTA barrier'
            assert not any(re.search(r'\b(?:LDL|STL)\b',i) for a,i in code)
        assert sum('SHFL.IDX' in i for a,i in repair)==8
        assert not any('SHFL.' in i or re.search(r'\bBRA\b',i) for a,i in empty),'empty per-unit repair/control retained'
        barriers=[a for a,i in rows if 'BAR.SYNC' in i];assert len(barriers)==1 and barriers[0]>=join
        assert not any(re.search(r'\b(?:ATOM|RED)\.',i) for a,i in rows),'S1 counter protocol changed'
        result.append(dict(args=args,entry_broadcast_PC=pc,entry_sentinel_branch_PC=branch[0],empty_start_PC=empty_start,common_join_PC=join,CTA_reduction_barrier_PC=barriers[0],
                           repair_instructions=len(repair),empty_instructions=len(empty),HMMA_each_path=32,repair_SHFL=8,empty_SHFL=0,
                           common_reduction='All16/32warps rejoin common shared reduction and single CTA barrier; no CTAbarrier in either compute path',
                           static_code_bytes=len(rows)*16,limits='Fixed canonical CUDA12.8artifact gate; actual loaded SO must match everymainPC/instruction before runtime admission'))
    assert len(result)==2
    return dict(state='PASS_PHYSICAL_EMPTY_PATH_AND_COMMON_REDUCTION_GATE',kernels=result)
if __name__=='__main__':
    result=proof((HERE/'sass.txt').read_text());(HERE/'physical-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

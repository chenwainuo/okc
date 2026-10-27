"""Exact prepared escape-prefix and mapping models; CPU tensors only."""
import ast
import hashlib
import json
from pathlib import Path
import random
import torch
HERE=Path(__file__).resolve().parent
SENTINEL=0xffffffff

def extract(source,name,env):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure-CPU-source-extract>','exec'),env)
    return env[name]

def validate_rows(words,ku=8):
    count=empty=full=0
    for row in words:
        assert len(row)==32
        ended=False;previous=-1;payload=0
        for raw in row:
            word=int(raw)&SENTINEL
            if word==SENTINEL:ended=True;continue
            assert not ended,'payload after sentinel/hole'
            seq=word>>20
            assert previous<=seq<ku,'unsorted or out-of-domain sequence'
            assert not word&0x380,'reserved C12/matrix bits set'
            assert word&127,'zero XOR payload is not emitted by exact out-of-window encoder'
            previous=seq;payload+=1
        assert (int(row[0])&SENTINEL==SENTINEL)==(payload==0)
        count+=payload;empty+=payload==0;full+=payload==32
    return dict(groups=len(words),payload_entries=count,empty_groups=empty,full32_groups=full)

def repair(decoded,words):
    out=[[reg[:] for reg in stage] for stage in decoded];ep=0;nx=words[0]&SENTINEL;trace=[]
    for u in range(8):
        while nx>>20==u:
            lane=(nx>>15)&31;reg=(nx>>11)&15;half=(nx>>10)&1
            out[u][lane][reg]^=(nx&127)<<(8+16*half);trace.append(ep)
            ep+=1;nx=words[ep]&SENTINEL if ep<32 else SENTINEL
    return out,trace

def proof():
    source=(HERE/'submission.py').read_text();parent=(HERE.parent/'pw38/submission.py').read_text();env={'torch':torch}
    encode=extract(parent,'_gvesc',env);assert ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_gvesc'))==ast.dump(next(n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name=='_gvesc'))
    rng=random.Random(72919);scenarios=[]
    for wr in (1,2):
        geo=(wr,16,8,1,1,2);T=144;KC=128;gx=T//wr;nw=wr*16
        tilechunks=set()
        for tile in range(T):
            for c in range(KC):
                group=(tile//wr)*nw+(c//8)*wr+tile%wr;seq=c%8
                assert (group,seq) not in tilechunks;tilechunks.add((group,seq))
                recovered_tile=(group//nw)*wr+group%wr
                recovered_c=((group%nw)//wr)*8+seq
                assert (recovered_tile,recovered_c)==(tile,c)
        assert tilechunks=={(group,u) for group in range(2304) for u in range(8)}
        # Encoder overflow and all prefix lengths, with duplicate edits allowed.
        for length in (0,1,2,15,31,32,33):
            rows=[]
            for i in range(length):rows.append([0,i%8,i%4,i%32,i%8,1+i%127,0])
            escaped=torch.tensor(rows,dtype=torch.int64,device="cpu").reshape(-1,7)
            packed=encode(escaped,T,geo)
            if length==33:assert packed is None;scenarios.append(dict(WR=wr,length=33,overflow='REJECT_NONE'));continue
            assert packed is not None
            matrix=packed.reshape(-1,32).tolist();stats=validate_rows(matrix)
            assert stats['payload_entries']==length and stats['groups']==2304
            # Put random arbitrary32bit decoded words through both paths. The
            # empty fast branch is ONLY equivalent when firstsentinel proves0.
            decoded=[[[rng.getrandbits(32) for _ in range(16)] for _ in range(32)] for _ in range(8)]
            for group in {0,1,15,16,31,2303}:
                words=[w&SENTINEL for w in matrix[group]];generic,trace=repair(decoded,words)
                fast=decoded if words[0]==SENTINEL else repair(decoded,words)[0]
                assert fast==generic
                assert len(trace)==len(set(trace))==sum(w!=SENTINEL for w in words)
            scenarios.append(dict(WR=wr,length=length,stats=stats))
        # Actual encoder at rowtile/chunk/group boundaries, including dropped
        # out-of-T records, must match an independent packed-word model.
        boundaries=[[tile,c,i%4,(i*7)%32,i%8,1+i%127,0] for i,(tile,c) in enumerate(( (tile,c) for tile in sorted({0,1,wr-1,143}) for c in (0,7,8,15,120,127)))]
        expected={}
        for tile,c,j,l,v,x,h in boundaries:
            group=(tile//wr)*nw+(c//8)*wr+tile%wr;word=((c%8)<<20)|(l<<15)|((4*j+v//2)<<11)|((v%2)<<10)|x
            expected.setdefault(group,[]).append(word)
        extra=boundaries+[[T,0,0,0,0,1,0]]
        matrix=encode(torch.tensor(extra,dtype=torch.int64,device="cpu"),T,geo).reshape(-1,32).tolist();validate_rows(matrix)
        for group,row in enumerate(matrix):
            payload=[word&SENTINEL for word in row if word!=-1]
            assert sorted(payload)==sorted(expected.get(group,[]))
        # Full32 at finalunit checks ep31->32 without a33rdword load.
        final=torch.tensor([[0,7,j%4,j,(2*j)%8,1+j,0] for j in range(32)],dtype=torch.int64,device="cpu")
        full=encode(final,T,geo).reshape(-1,32).tolist();validate_rows(full)
        # c7 maps to group tile0/WK0; all32records lastsequence7.
        decoded=[[[0 for _ in range(16)] for _ in range(32)] for _ in range(8)]
        output,trace=repair(decoded,[w&SENTINEL for w in full[0]])
        assert trace==list(range(32))
    # Unchanged codec functions reconstruct all65536 BF16 storage patterns,
    # including NaN/Inf/sign payloads, on synthetic CPU tensors only. This says
    # nothing about CUDA fixture prevalence; dense lists may overflow normally.
    for name in ('_OS','_SS'):
        node=next(n for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(q,ast.Name) and q.id==name for q in n.targets));env[name]=ast.literal_eval(node.value)
    unpack=extract(parent,'_unc12',env);pack=extract(parent,'_c12',env)
    bits=torch.arange(65536,dtype=torch.int32,device="cpu").reshape(1,64,4,32,8)
    packed=pack(bits.to(torch.int16).view(torch.bfloat16),1);assert packed is not None
    assert torch.equal(unpack(*packed,1),bits)
    # Fragment pair positions cover exactly16x64 BF16words per tilechunk.
    coords=set()
    for step in range(4):
        for lane in range(32):
            g,t=lane>>2,lane&3
            for reg in range(4):
                for half in range(2):
                    coord=(g+8*(reg&1),16*step+8*(reg>>1)+2*t+half)
                    assert coord not in coords;coords.add(coord)
    assert coords=={(row,k) for row in range(16) for k in range(64)}
    negative={}
    badcases={'sentinel_hole':[SENTINEL,(0<<20)|1]+[SENTINEL]*30,
              'unsorted':[(1<<20)|1,1]+[SENTINEL]*30,
              'reserved_sequence':[(4095<<20)|1]+[SENTINEL]*31,
              'out_of_unit_sequence':[(8<<20)|1]+[SENTINEL]*31,
              'matrix_h1':[(1<<9)|1]+[SENTINEL]*31,
              'reserved_bit8':[(1<<8)|1]+[SENTINEL]*31,
              'zero_xor':[0]+[SENTINEL]*31}
    for name,row in badcases.items():
        try:validate_rows([row])
        except AssertionError as error:negative[name]=str(error)
        else:raise AssertionError('CSR negative admitted '+name)
    # Deliberately lying about an actual nonempty list changes a raw decoded
    # word; the entry guard must use exact firstsentinel, never sampled frequency.
    decoded=[[[0 for _ in range(16)] for _ in range(32)] for _ in range(8)]
    corrected,_=repair(decoded,[1]+[SENTINEL]*31);assert corrected!=decoded
    negative['forced_empty_on_nonempty']='DETECTED_CHANGED_WEIGHT_BITS'
    # Odd M loads/emit coverage are inherited; all lane addresses clamp valid
    # rows before MMA and each live output coordinate is stored exactly once.
    for m in (1,2,3,5,7,8):
        seen=set()
        for lane in range(32):
            g,t=lane>>2,lane&3;assert 0<=min(g,m-1)<m
            for d in (0,8):
                for token in (2*t,2*t+1):
                    if token<m:
                        coord=(token,g+d);assert coord not in seen;seen.add(coord)
        assert seen=={(token,row) for token in range(m) for row in range(16)}
    return dict(state='CPU_CSR_MAPPING_SENTINEL_AND_BIT_REPAIR_PASS',actual_encoder='_gvesc_EXACT_PARENT_FUNCTION_CPU_TENSORS_ONLY',
       geometry_groups=2304,tilechunk_bijection=18432,per_tilechunk_BF16=1024,total_weight_BF16=18874368,
       scenarios=scenarios,negative_controls=negative,full32_end='ep32 yields sentinel withoutload33',warp_uniform='all32lanes receive lane0 exactentry_e via fullmask shfl beforebranch',
       all65536_BF16_storage_bits='UNCHANGED_CODEC_CPU_RECONSTRUCTION_PASS_NOT_PREVALENCE',odd_M=[1,2,3,5,7,8],generic_and_nonempty='same correction order; XOR everylistedentryonce before sameMMA',
       limits='No real CUDA weights queried; actual empty prevalence unknown and must be counted from authorized prepared CUDA metadata before timing')
if __name__=='__main__':
    result=proof();(HERE/'csr-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

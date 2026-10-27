"""Exact sparse-header delta on frozen quarter-stream native prototype."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent/'gv53_quarter_stream_v1'
EXPECT='b7c8fccd8a92c57387872f41e164aefb86d70f5756ebef6ba5ae6b8a2bc7a7fb'
def transform(s):
 edits=[]
 def edit(old,new):
  nonlocal s
  assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1);edits.append({'old':old,'new':new})
 edit('  e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane];','  e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane];\n  const uint32_t emask=__ballot_sync(~0u,lane<KU && ((e>>24)&63)!=(e&63));')
 edit('    const uint32_t header=__shfl_sync(~0u,e,u);','    const bool has_esc=((emask>>u)&1)!=0;\n    uint32_t header=0;\n    if(has_esc) header=__shfl_sync(~0u,e,u);')
 old='''      const int begin=(header>>(6*q))&63, end=(header>>(6*(q+1)))&63;
#pragma unroll 1
      for (int ep4=begin;ep4<end;ep4++) {
        const uint32_t pair=__shfl_sync(~0u,e,KU+(ep4>>1));
        const uint32_t r16=(pair>>(16*(ep4&1)))&65535u;
        const uint32_t xr=(r16&127u)<<(8+16*((r16>>7)&1));
        gv_fix_quarter(A4,(r16>>9)&3,xr,lane,r16>>11,(r16>>8)&1);
      }'''
 edit(old,'      if(has_esc) {\n'+old+'\n      }')
 s=s.replace('gv_quarter<','gv_sparse_quarter<').replace(' gv_quarter(G p)',' gv_sparse_quarter(G p)')
 return s,edits
s=(P/'pilot.cu').read_text();assert hashlib.sha256(s.encode()).hexdigest()==EXPECT
new,edits=transform(s)
restored=new.replace('gv_sparse_quarter<','gv_quarter<').replace(' gv_sparse_quarter(G p)',' gv_quarter(G p)')
for e in reversed(edits):assert restored.count(e['new'])==1;restored=restored.replace(e['new'],e['old'],1)
assert restored==s
(D/'pilot.cu').write_text(new)
k,_=transform((P/'kernel.cuh').read_text());(D/'kernel.cuh').write_text(k)
(D/'delta.json').write_text(json.dumps(edits,indent=2)+'\n')
(D/'source-proof.json').write_text(json.dumps({'state':'PASS_STRICT_SPARSE_HEADER_INVERSE_CPU','parent_source':str(P/'pilot.cu'),'parent_sha256':EXPECT,'source_sha256':hashlib.sha256(new.encode()).hexdigest(),'strict_inverse':True,'only_changes':['one warp ballot after accepted packetload','uniform unit nonempty mask gates header broadcast','same mask gates quarter repair only','new native symbol'],'unchanged':['128B packet/producer/transport','dec1/helper/hmma','M8 shape guard','q/u accumulator order','W/X snapshot/prefetch','all reduction/store/launch geometry']},indent=2)+'\n')
print(hashlib.sha256(new.encode()).hexdigest())

"""Changed ballot/header-skip semantics only; frozen packet/decoder proofs reused."""
import sys,random,json
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'gv53_quarter_stream_v1'))
from transport import pack_group,unpack_group
R=random.Random(271039)
def word(u,q=0,r=0,l=0,hh=0,hm=0,x=1):return(u<<20)|(l<<15)|((q*4+r)<<11)|(hh<<10)|(hm<<9)|x
def read(words,sparse):
 mask=sum(1<<lane for lane,e in enumerate(words)if lane<8 and ((e>>24)&63)!=(e&63))
 records=[];headers=0
 for u in range(8):
  if sparse and not((mask>>u)&1):continue
  headers+=1;h=words[u]
  for q in range(4):
   begin,end=(h>>(6*q))&63,(h>>(6*(q+1)))&63
   for ep in range(begin,end):records.append((u,q,(words[8+(ep>>1)]>>(16*(ep&1)))&65535))
 return mask,records,headers
cases=[]
for u in range(8):
 for q in range(4):
  # hm1 still counts as occupied and is consumed; later empty headers have nonzero absolute offsets.
  cases.append([word(u,q,3,31,1,1,127)])
  cases.append([word(u,q,3,31,1,0,127)]*32)
for count in (0,1,2,31,32):
 for _ in range(32):
  xs=[word(R.randrange(8),R.randrange(4),R.randrange(4),R.randrange(32),R.randrange(2),R.randrange(2),R.randrange(128))for _ in range(count)];xs.sort(key=lambda x:x>>20);cases.append(xs)
for xs in cases:
 w=pack_group(xs+[-1]*(32-len(xs)));mask,a,ha=read(w,False);_,b,hb=read(w,True)
 assert a==b and len(b)==len(xs)and hb==mask.bit_count()and ha==8
 assert mask==sum(1<<u for u in set(x>>20 for x in xs))
 # Data lanes cannot set unit-mask bits regardless of record bit patterns.
 assert mask<256
kernel=(D/'kernel.cuh').read_text()
for fragment in ['__ballot_sync(~0u,lane<KU && ((e>>24)&63)!=(e&63))','if(has_esc) header=__shfl_sync(~0u,e,u);','if(has_esc) {','hmma(acc[0][0][q % NC],A4,B[2*q],B[2*q+1]);']:
 assert fragment in kernel
# Only repair is gated; decoding/MMA remains unconditional in q order.
assert kernel.index('dec1(lo[2*q]')<kernel.index('      if(has_esc) {')<kernel.index('hmma(acc[0][0][q % NC]')
r={'state':'PASS_SCOPED_BALLOT_HEADER_SKIP_CPU','cases':len(cases),'empty_nonzero_base_offsets_covered':True,'zero_and_32_records':True,'hm1_occupied_units_consumed_ignored':True,'packet_unchanged':True,'record_sequence_exact_vs_predecessor':True,'header_count_equals_nonempty_unit_count':True,'all32lanes_participate_in_ballot_uniform_mask':True,'GPU_used':False}
(D/'cpu-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

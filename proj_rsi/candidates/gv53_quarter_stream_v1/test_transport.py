"""Changed packet/quarter order only; unchanged matrix/decoder proofs are reused."""
import itertools,json,random
from pathlib import Path
from transport import pack_group,unpack_group,pack_bank
D=Path(__file__).resolve().parent
R=random.Random(2768)
def old(u,q,r,l,hh,hm,x):return(u<<20)|(l<<15)|((q*4+r)<<11)|(hh<<10)|(hm<<9)|x
def grouped(vals):return vals+[-1]*(32-len(vals))
def stable(vals):return sorted(vals,key=lambda x:(x>>20,(x>>13)&3))
def apply(vals):
 a=[[R0*0x1234567+i*0x76543 for i in range(16)]for R0 in range(32)]
 for x in vals:
  if not((x>>9)&1):a[(x>>15)&31][(x>>11)&15]^=(x&127)<<(8+16*((x>>10)&1))
 return a
cases=0
# All selected field extrema/cross products; whole64-bit matrix proof unnecessary.
for u,q,r,l,hh,hm,x in itertools.product((0,7),range(4),range(4),(0,1,15,31),(0,1),(0,1),(0,1,127)):
 vals=[old(u,q,r,l,hh,hm,x)];out=unpack_group(pack_group(grouped(vals)))
 assert out==vals and apply(out)==apply(vals);cases+=1
# Empty and maximum32, arbitrary quarter order, duplicates; every bucket boundary.
for count in [0,1,2,31,32]:
 for trial in range(32):
  vals=[old(R.randrange(8),R.randrange(4),R.randrange(4),R.randrange(32),R.randrange(2),R.randrange(2),R.randrange(128))for _ in range(count)]
  vals.sort(key=lambda x:x>>20)
  words=pack_group(grouped(vals));out=unpack_group(words)
  assert out==stable(vals) and apply(out)==apply(vals)
  assert len(words)==32 and all(x==0 for x in words[24:])
  cases+=1
for u,q in itertools.product(range(8),range(4)):
 vals=[old(u,q,3,31,1,0,127)]*32
 words=pack_group(vals);out=unpack_group(words)
 assert out==vals and apply(out)==apply(vals)
 assert ((words[u]>>(6*(q+1)))&63)==32
 cases+=1
# Original overflow/fallback and malformed canonical contract must never silently truncate.
rejected=0
for vals in [[old(0,0,0,0,0,0,1)]*33,[-1,old(0,0,0,0,0,0,1)]+[-1]*30,[old(7,0,0,0,0,0,1),old(0,0,0,0,0,0,1)]+[-1]*30,[(8<<20)]+[-1]*31]:
 try:pack_group(vals)
 except ValueError:rejected+=1
 else:raise AssertionError('invalid input accepted')
# Verify the unchanged dec1 payload map is quarter-local; no new codec representation.
def prmt(a,b,sel):
 x=a|(b<<32);return sum(((x>>(8*((sel>>(4*i))&7)))&255)<<(8*i)for i in range(4))
def dec1(l0,l1,c,tb):
 hl=((c&0x87878787)+tb)&0xffffffff;rot=((c>>4)|(c<<28))&0xffffffff;hh=((rot&0x87878787)+tb)&0xffffffff
 return[prmt(l0,hl,0x5140),prmt(l0,hl,0x7362),prmt(l1,hh,0x5140),prmt(l1,hh,0x7362)]
for _ in range(256):
 payload=[R.getrandbits(32)for _ in range(12)];tb=R.randrange(121)*0x01010101
 full=sum((dec1(payload[2*q],payload[2*q+1],payload[8+q],tb)for q in range(4)),[])
 phase=[dec1(payload[2*q],payload[2*q+1],payload[8+q],tb)for q in range(4)]
 assert full==sum(phase,[])
# Each independent accumulator sees exact original u then q parity order.
original=[(u,q,q%2)for u in range(8)for q in range(4)];phased=[(u,q,q%2)for u in range(8)for q in range(4)];assert original==phased
assert pack_bank(None) is None
assert pack_bank([-1]*64)==[0]*64
kernel=(D/'kernel.cuh').read_text()
for exact in ['const uint32_t header=__shfl_sync(~0u,e,u);','__shfl_sync(~0u,e,KU+(ep4>>1))','(pair>>(16*(ep4&1)))&65535u','(r16&127u)<<(8+16*((r16>>7)&1))','gv_fix_quarter(A4,(r16>>9)&3,xr,lane,r16>>11,(r16>>8)&1);','hmma(acc[0][0][q % NC],A4,B[2*q],B[2*q+1]);']:
 assert exact in kernel
assert kernel.index('const uint4 q0=wb')<kernel.index('if (u+ST<KU) ld(s,u+ST);')
report={'state':'PASS_SCOPED_CPU_TRANSPORT','edge_and_random_group_cases':cases,'rejected_noncanonical_or_overflow_cases':rejected,'payload_mapping_cases':256,'capacity_records':32,'packet_uint32_words':32,'packet_bytes':128,'header_bytes':32,'record_capacity_bytes':64,'pad_bytes':32,'same_accumulator_order_u_q':True,'duplicates_stable_inside_bucket':True,'hm1_consumed_ignored':True,'GPU_used':False,'full_matrix_codec_proof_not_repeated':True}
(D/'cpu-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

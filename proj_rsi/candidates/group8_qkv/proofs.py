"""Pure Python exhaustive address/ownership and symbolic schedule proofs."""
from pathlib import Path
import json,importlib.util,re
H=Path(__file__).resolve().parent
# CPU-import only checker has no torch/CUDA dependency.
spec=importlib.util.spec_from_file_location('g8physical',H.parent/'operand_reverse_group8/liveness.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
def proof():
 seen=bytearray(524288)
 for idx in range(131072):
  v=idx&3;lane=(idx>>2)&31;chunk=(idx>>7)&3;warp=(idx>>9)&7;s=idx>>12
  row=64*(warp>>2)+16*(warp&3)+(lane>>2)+8*(v&1)
  k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
  for u in range(4):
   src=row*4096+k//2+1024*u;dst=idx+131072*u
   assert 0<=src<524288 and not seen[src];seen[src]=1
   stage=dst//4096;off=dst%4096
   # Consumer native A word for this stage/warp/lane/chunk/value.
   assert stage==s+32*u
   assert off==warp*512+chunk*128+lane*4+v
 assert all(seen)
 # Output native-D ownership and partial slice coverage.
 out=bytearray(128*2304)
 for cb in range(0,2304,64):
  for warp in range(8):
   wg,w=divmod(warp,4)
   for lane in range(32):
    g,t=divmod(lane,4)
    for j in range(8):
     row=wg*64+w*16+g;col=cb+8*j+2*t
     for dr in (0,8):
      for dc in (0,1):
       i=(row+dr)*2304+col+dc;assert not out[i];out[i]=1
 assert all(out)
 deps={}
 def edge(n,d):deps.setdefault(n,set()).add(d);deps.setdefault(d,set())
 for s in range(32):
  deps.setdefault(f'P{s}',set())
  if s>=2:edge(f'P{s}',f'P{s-2}')
  if s>=6:edge(f'P{s}',f'P{s-6}')
  if s>=8:edge(f'P{s}',f'R{s-8}')
  edge(f'R{s}',f'G{min(s//2+1,15)}')
 for g in range(16):
  edge(f'G{g}',f'P{2*g}');edge(f'G{g}',f'P{2*g+1}')
  if g:edge(f'G{g}',f'G{g-1}')
 done=set()
 while len(done)<len(deps):
  ready={n for n,d in deps.items() if n not in done and d<=done};assert ready;done|=ready
 # Entry fields remain disjoint for all local stages, lanes, regs and halves.
 for seq in range(32):
  for lane in range(32):
   for reg in range(16):
    for half in range(2):
     ent=(seq<<17)|(lane<<12)|(reg<<8)|(half<<7)|127
     assert ent>>17==seq and (ent>>12)&31==lane and (ent>>8)&15==reg and (ent>>7)&1==half
 # The two independent producers restart at local seq0; odd owner skips0.
 # At most32*32 values per lane/stage? Exact per-warp32stages*32lanes*32BF16values.
 maxentries=32*32*32;cap=((maxentries+1+31)//32)*32
 assert cap==32800
 sass=(H/'sass.txt').read_text();tables=re.split(r'\n\s*Function : ',sass)[1:]
 table=next(t for t in tables if 'rq_qkv_g8_sk4' in t.splitlines()[0])
 physical=L.check_one(table)
 return {'status':'CPU_PROOFS_PASS_GPU_UNRUN','XP_u32_words':len(seen),'XP_bytes':2097152,'output_values':len(out),'partial_bytes':4718592,'slice_stages':32,'global_stages':128,'CSR_max_local_sequence':31,'CSR_sentinel_sequence':32767,'CSR_universal_CAP_bound':cap,'CSR_actual_CAP':'weight dependent, must use _rqesc; no fixed capacity assumption','schedule_DAG_nodes':len(done),'physical':physical,'limits':'Device-only resource prototype. No runtime host wrapper, actual JIT, numerical, graph or performance admission.'}
if __name__=='__main__':
 r=proof();(H/'proofs.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='physical'},indent=2))

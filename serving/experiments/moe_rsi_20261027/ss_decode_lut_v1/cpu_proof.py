"""Exact pure-helper scalar check, unchanged-pipeline check and bounded bank model."""
import hashlib,json,re,struct,subprocess,tempfile,random,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else BASE;P=BASE.parent/'ss_pipeline_v2'
assert not torch.cuda._initialized
src=(D/'kernel.cuh').read_text();parent=(P/'kernel.cuh').read_text()
def helper(source,name):
 start=source.index('__host__ __device__ __forceinline__ uint16_t '+name)
 end=source.index('\n}',start)+2
 return source[start:end].replace('__host__ __device__ __forceinline__','inline')
table=json.loads((D/'magnitude-lut.json').read_text());stride=16 if len(table)==4096 else 8
cpp='#include <cstdint>\n#include <iostream>\n'+helper(src,'lookup_bits')+'\nuint16_t lut[]={'+','.join(map(str,table))+'};\nint main(){for(int q=0;q<16;q++)for(int s=0;s<256;s++)std::cout<<lookup_bits(q,s,lut)<<" ";}'
with tempfile.TemporaryDirectory(prefix='ss-lut-proof-') as tmp:
 p=Path(tmp);(p/'proof.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-O2',str(p/'proof.cpp'),'-o',str(p/'proof')],check=True)
 bits=list(map(int,subprocess.check_output([str(p/'proof')],text=True).split()))
got=torch.tensor([v if v<32768 else v-65536 for v in bits],dtype=torch.int16).reshape(16,256)
q=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.bfloat16)
s=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
ref=(q[:,None]*s[None,:]);finite=torch.isfinite(s)
assert torch.equal(got[:,finite],ref.view(torch.int16)[:,finite])
assert bool(torch.isnan(got.view(torch.bfloat16)[:,~finite]).all())
# Exact finite check includes 62 signed/unsigned zero products and 7 subnormals/sign.
# Matrix descriptors, instructions, math waits, gather, global epilogue unchanged.
assert src[src.index('template<int T> __device__ __forceinline__ void mma'):src.index('template<int T,int DIV> __global__')]==parent[parent.index('template<int T> __device__ __forceinline__ void mma'):parent.index('template<int T,int DIV> __global__')]
for a,b in [('   for(int linear=lane;linear<BW;linear+=128){','constexpr int shared_bytes')]:
 assert src[src.index(a):src.index(b)]==parent[parent.index(a):parent.index(b)]
assert (D/'serving_both.py').read_bytes()==(P/'serving_both.py').read_bytes()
# Show a finite global-folding counterexample under the SAME BF16 boundaries.
a=torch.tensor([1.,1.],dtype=torch.float32)
w=torch.tensor([1.,.00390625],dtype=torch.bfloat16);g=torch.tensor(1.5,dtype=torch.bfloat16)
post=(a*w.float()).sum().to(torch.bfloat16)*g
fold=(a*(w*g).float()).sum().to(torch.bfloat16)
assert post.view(torch.int16)!=fold.view(torch.int16)
# Shared-memory load instruction conflict model: service rounds >= max distinct
# 32-bit words per bank. Same-word/halfword broadcasts collapse. Not a timing model.
def conflicts(scales,codes,v):
 banks=[set() for _ in range(32)]
 for lane in range(32):
  word=((scales[lane//4]*stride+codes[lane])//2)
  banks[word%32].add(word)
 return max(map(len,banks))
rng=random.Random(7);samples=[]
for _ in range(1000):
 sf=[rng.randrange(127) for _ in range(8)];qs=[rng.randrange(stride) for _ in range(32)]
 samples.append(conflicts(sf,qs,0))
# Original actual SASS proof parser works for the two exact changed functions.
import importlib.util
spec=importlib.util.spec_from_file_location('parent_cpu_proof',P/'cpu_proof.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
inherited={'layout':m.layout(),'protocol':m.protocol(),'routing':m.routes()}
assert '(0,8,4,12,16,24,20,28)' in (P/'cpu_proof.py').read_text()
assert 'offsets[8]={0,8,4,12,16,24,20,28}' in src
sass=(D/'sass.txt').read_text();functions=[]
for function in sass.split('Function : ')[1:]:
 name=function.splitlines()[0];rows=[(int(x[1],16),x[2].strip()) for x in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',function)]
 pending=set();issues=groups=handoffs=0
 for pc,ins in rows:
  if 'HGMMA.' in ins:
   base=int(re.search(r'\bR(\d+),',ins)[1]);target=set(range(base,base+4));assert not pending or pending==target;pending=target;issues+=1;continue
  if 'WARPGROUP.DEPBAR.LE' in ins and '0x0' in ins:
   assert pending and issues==4;groups+=1;pending=set();issues=0;continue
  if pending:
   dest=re.match(r'[A-Z0-9_.]+\s+R(\d+)(?:,|\s)',re.sub(r'^@!?P\d+\s+','',ins))
   if dest:assert int(dest[1])not in pending
  if 'BAR.SYNC' in ins and '0x3, 0x80' in ins:handoffs+=1
 assert not pending and groups and handoffs==groups
 count=lambda term:sum(term in ins for _,ins in rows)
 functions.append({'function':name,'instructions':len(rows),'LOP3':count('LOP3'),'LDS':count('LDS'),'LDS_U16':count('LDS.U16'),'HGMMA':count('HGMMA'),'wait0_groups':groups,'reader_barriers':handoffs,'total_LDG_U16_constant':count('LDG.E.U16.CONSTANT')})
assert len(functions)==2
host_equal=None
if (D/'extension_sass.txt').exists():
 def streams(raw):
  return {f.splitlines()[0].strip():re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f) for f in raw.split('Function : ')[1:] if '7projectILi' in f.splitlines()[0]}
 assert streams(sass)==streams((D/'extension_sass.txt').read_text())
 host_equal=True
log=(D/'resource.log').read_text();assert not re.search(r'[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log)
proof={'state':'CPU_ONLY_GPU_UNRUN','finite_exact_product_bit_cases':4064,'signed_zero_included':True,'fp8_subnormal_signs_included':True,'NaN_classification_cases':32,'NaN_sign_payload_equality':'not asserted: invalid finite-numeric input','global_fold_counterexample':{'a':a.tolist(),'w':w.tolist(),'global':float(g),'canonical_postglobal':float(post),'folded_global':float(fold)},'source_unchanged':['mma descriptors/instructions','input gather through math loop and global epilogue','complete serving method'],'inherited_CPU_proofs':inherited,'SASS':functions,'registers':list(map(int,re.findall(r'Used (\d+) registers',log))),'full_host_wrapper_matches_device_SASS':host_equal,'spill_stack_bytes':0,'shared_T8_bytes':2*(4096+8*64)*2+32+len(table)*2+1024,'extra_shared_bytes':len(table)*2,'lut_global_init_bytes_per_CTA':len(table)*2,'extra_LDS_payload_bytes_per_full_K4096_CTA':64*4096*2,'bank_service_rounds_seed7_1000_random_samples':{'mean':sum(samples)/len(samples),'max':max(samples),'histogram':{v:samples.count(v)for v in sorted(set(samples))},'same_scale':conflicts([1]*8,[i%stride for i in range(32)],0),'adversarial_scales_stride8_same_q':conflicts(list(range(0,64,8)),[0]*32,0)},'CUDA_context_initialized':torch.cuda._initialized}
(D/'cpu-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))

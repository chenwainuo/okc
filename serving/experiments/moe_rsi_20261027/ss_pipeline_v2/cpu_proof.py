"""CPU layout/pack/ownership/protocol and actual cubin gates. Never loads CUDA."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module():
 spec=importlib.util.spec_from_file_location('ss_cpu_adapter',HERE/'adapter.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def packing():
 A=module();nibbles=0
 for e,n,k in [(1,64,16),(2,128,64),(1,64,4096)]:
  raw=(torch.arange(e*n*k//2,dtype=torch.int64)%256).to(torch.uint8).reshape(e,n,k//2);pack=A.pack_weight(raw).reshape(-1).tolist();values=raw.reshape(-1).tolist();seen=set()
  for ex in range(e):
   for tile in range(n//64):
    for q in range(k//16):
     for lane in range(128):
      word=pack[((ex*(n//64)+tile)*(k//16)+q)*128+lane]&0xffffffff
      for v,off in enumerate((0,8,4,12,16,24,20,28)):
       row=(lane//4)%8+16*(lane//32)+8*((v//2)%2);kk=2*(lane%4)+(v%2)+8*(v//4);i=(ex*n+tile*64+row)*k+q*16+kk
       assert i not in seen;seen.add(i);assert (word>>off)&15==(values[i//2]>>(4*(i%2)))&15;nibbles+=1
  assert len(seen)==e*n*k
 # Compile and execute the exact PURE host/device integer helper bodies with
 # g++, without cudart or a CUDA context. Torch FP8 conversion is independent.
 import subprocess,tempfile
 kernel=(HERE/'kernel.cuh').read_text()
 def helper(name):
  start=kernel.index('__host__ __device__ __forceinline__ uint16_t '+name);brace=kernel.index('{',start);level=1;end=brace+1
  while level:
   level+=(kernel[end]=='{')-(kernel[end]=='}');end+=1
  return kernel[start:end].replace('__host__ __device__ __forceinline__','inline')
 cpp='#include <cstdint>\n#include <iostream>\n'+helper('e2m1_bits')+'\n'+helper('raw_scale_bits')+'\nint main(){for(int q=0;q<16;q++)std::cout<<e2m1_bits(q)<<" ";for(int s=0;s<256;s++)std::cout<<raw_scale_bits(s)<<" ";}\n'
 with tempfile.TemporaryDirectory(prefix='ss-v2-numeric-')as tmp:
  c=Path(tmp)/'check.cpp';exe=Path(tmp)/'check';c.write_text(cpp);subprocess.run(['g++','-std=c++17','-O2',str(c),'-o',str(exe)],check=True,capture_output=True)
  bits=list(map(int,subprocess.check_output([str(exe)],text=True).split()))
 assert len(bits)==272
 def bf(values):return torch.tensor([v if v<32768 else v-65536 for v in values],dtype=torch.int16).view(torch.bfloat16)
 qb,sb=bf(bits[:16]),bf(bits[16:])
 nv=(ROOT/'muse_moe_w4a16/nvfp4.py').read_text();grid=next(ast.literal_eval(n.value)for n in ast.parse(nv).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_E2M1_GRID'for t in n.targets))
 qref=torch.tensor([(-1 if q&8 else 1)*grid[q&7]for q in range(16)],dtype=torch.float32)
 sf=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).float();finite=torch.isfinite(sf)
 assert int(finite.sum())==254
 assert torch.equal(qb.view(torch.int16),qref.to(torch.bfloat16).view(torch.int16))
 assert torch.equal(sb[finite].view(torch.int16),sf[finite].to(torch.bfloat16).view(torch.int16))
 assert bool(torch.isnan(sb[~finite]).all()) and bits[16+127]==0x7fc0 and bits[16+255]==0xffc0
 got=(qb[:,None]*sb[None,:]);want=(qref[:,None]*sf[None,:]).to(torch.bfloat16)
 assert torch.equal(got[:,finite].view(torch.int16),want[:,finite].view(torch.int16))
 assert bool(torch.isnan(got[:,~finite]).all())
 nonzero=got[:,finite].float().abs();nonzero=nonzero[nonzero>0]
 assert float(nonzero.min())==2**-10 and float(nonzero.max())==2688 and not bool(((nonzero>0)&(nonzero<2**-126)).any())
 # Positive normal+zero source-domain correspondence to inherited bias, not
 # general v1 identity. Subnormal raw bytes are explicit counterexamples.
 positive=[0]+list(range(8,127));normal=sf[positive];half=(normal.to(torch.float16)*128).view(torch.int16).to(torch.int32);processed=((half<<1)>>8)&255
 oldsf_bits=(((processed&128)<<7)|((processed&127)<<4)).tolist();oldsf=bf(oldsf_bits)
 oldq=bf([((q&8)<<12)|((q&7)<<6)for q in range(16)])
 oldproduct=oldq[:,None]*oldsf[None,:];scaled=(qref[:,None]*normal[None,:]*(2.**-119)).to(torch.bfloat16)
 assert torch.equal(oldproduct.view(torch.int16),scaled.view(torch.int16))
 assert bits[17]==0x3b00
 # Exact conditional epilogue power2 check, with intermediates kept normal.
 ds=torch.tensor([.015625,.125,1.,13.25,1024.,1048576.,-13.25],dtype=torch.float32)
 gs=torch.tensor([2.**-16,.03125,1.],dtype=torch.bfloat16)
 old=(ds[:,None]*(2.**-119)).to(torch.bfloat16)*(gs[None,:]*(2.**119))
 new=ds[:,None].to(torch.bfloat16)*gs[None,:]
 assert torch.equal(old.view(torch.int16),new.view(torch.int16))
 # Known non-equivalence when old representation underflows/overflows.
 assert (torch.tensor(2.**-16)*(2.**-119)).to(torch.bfloat16)==0
 assert torch.tensor(2.**-16).to(torch.bfloat16)!=0
 assert torch.isinf(torch.tensor(4096.,dtype=torch.bfloat16)*(2.**119))
 parent=(HERE.parent/'ss_pipeline_v1/adapter.py').read_text();newsrc=(HERE/'adapter.py').read_text()
 def fn(src,name):return ast.dump(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
 for f in ['pack_weight','pack_scale']:assert fn(parent,f)==fn(newsrc,f)
 assert 'processed=raw_scale.view(torch.uint8).contiguous()'in newsrc and 'global_processed=raw_global.to(torch.bfloat16).contiguous()'in newsrc
 return {'lossless_raw_nibbles':nibbles,'exact_CPP_helper_values':272,'finite_raw_E4M3_scales':254,'finite_FP4_times_scale_bit_cases':4064,'both_raw_FP8_NaNs':'sign-preserving quiet BF16 NaNs; invalid finite-numeric inputs','signed_zero_bit_checks':True,'finite_nonzero_BF16_weight_range':[2.**-10,2688.],'BF16_weight_subnormals':False,'old_positive_normal_and_zero_biased_product_cases':len(positive)*16,'conditional_epilogue_power2_cases':21,'global_and_weight_representation':'canonical unbiased; not blanket v1 identity','known_v1_limits':['rawscale01 lossy HALF-byte decode','old biased BF16 partial underflow forD=2^-16','old compensated global overflows forrawgs4096'],'raw_pack_layout_exact_v1':True,'hardware_mover_and_onehot_math':'UNPROBED; stage/fulltask GPU mandatory'}

def layout():
 result=[]
 for t in (8,16,32,64):
  sw=lambda r,k:r*64+(((k//8)^(r&7))*8)+(k&7)
  aw={sw(r,k) for r in range(64)for k in range(64)};bw={sw(r,k)for r in range(t)for k in range(64)}
  assert aw==set(range(4096))and bw==set(range(t*64))
  cells=set()
  for tid in range(128):
   for v in range(t//2):
    row=(tid//4)%8+16*(tid//32)+8*((v//2)%2);col=2*(tid%4)+(v%2)+8*(v//4);assert (row,col)not in cells;cells.add((row,col))
  assert cells=={(r,c)for r in range(64)for c in range(t)}
  stage=(4096+t*64)*2;full=2*stage;empty=full+16;shared=empty+16+1024
  regions=[(0,stage),(stage,2*stage),(full,full+16),(empty,empty+16)]
  assert all(b<=c or d<=a for i,(a,b)in enumerate(regions)for j,(c,d)in enumerate(regions)if i!=j)
  result.append({'tokens':t,'output_scalars':len(cells),'shared_dynamic':shared,'stage_bytes':stage,'full_count':1,'empty_count':1,'math_release_threads_rendezvous':128,'producer_publication_threads':128})
 return result

def protocol(early=False):
 # Exhaust all legal interleavings of producer/math stages, with two ring slots.
 count=64;todo=[(0,0)];seen=set();complete=False
 while todo:
  p,m=todo.pop()
  if(p,m)in seen:continue
  seen.add((p,m));assert 0<=m<=p<=count
  if p==m==count:complete=True;continue
  nexts=[]
  if p<count and p-m<(3 if early else 2):nexts.append((p+1,m))
  if m<p:nexts.append((p,m+1))
  assert nexts
  for pp,mm in nexts:
   if pp>p and p>=2:
    # Slot s%2 can only be overwritten after wait0+bar3 for stage s-2.
    assert m>p-2
   todo.append((pp,mm))
 assert complete
 # Source phase mapping at all64 stages: full arrival s//2 and empty release
 # consumed at next same-slot stage, producer phase(s//2-1), firsttwo skip.
 for s in range(count):
  if s>=2:assert ((s//2)-1)&1==((s-2)//2)&1
 return {'complete_stages':count,'reachable_interleavings':len(seen),'no_dead_states':True,'init_slots':[0,1],'last_reads':[62,63],'producer_final_return':'maththreads retain CTA shared lifetime until finalwait0','assumption':'finite warp/asynchronous-engine progress; not arbitrary scheduler or numeric proof'}

def routes():
 counts=0
 for m in [1,8,16,32,64,128,256,257,384,512,513,753,3012,16384]:
  slots=m*8
  for bm in [8,16,32,48,64]:
   t=next(t for t in [64,32,16,8]if bm%t==0)
   for pattern in ['balanced','skew','empty','one']:
    groups=[[]for _ in range(32)]
    for r in range(slots):
     ex=(r%256)if pattern=='balanced'else 0 if pattern=='skew'else -1 if pattern=='empty'else 0 if r==slots-1 else -1
     if 0<=ex<32:groups[ex].append(r)
    sorted_ids=[];experts=[];writes=set()
    for ex,g in enumerate(groups):
     if not g:continue
     padded=g+[slots]*((-len(g))%bm);sorted_ids+=padded;experts += [ex]*(len(padded)//bm)
    post=len(sorted_ids);capacity=max(bm,((slots+257*(bm-1)+3)//4)*4)
    assert post<=capacity
    for pos in range(0,post,t):
     eb=pos//bm;assert (pos+t-1)//bm==eb
     for tok in range(t):
      route=sorted_ids[pos+tok]
      if route<slots:
       assert route not in writes;writes.add(route);assert 0<=route//8<m and 0<=route<slots
    assert writes==set(r for g in groups for r in g);counts+=1
 return {'shape_block_routing_cases':counts,'up_input':'slot//8 bounds originalM','down_input':'slot bounds M*8','nonlocal_and_empty':'no kernel write; explicit output zero each call','capacity_tail':'ceil grid with per-lane capacity guards; no block-alignment allocation assumption'}

def physical(sass=None):
 log=(HERE/'resource.log').read_text();assert not re.search(r'C75|serializ|[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log,re.I)
 regs=list(map(int,re.findall(r'Used (\d+) registers',log)));assert len(regs)==8 and max(regs)<=64
 proof=[]
 for table in ((HERE/'sass.txt').read_text() if sass is None else sass).split('Function : ')[1:]:
  name=table.splitlines()[0];t,div=map(int,re.search(r'ILi(\d+)ELi(\d+)E',name).groups());rows=[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
  pending=set();issues=groups=0;handoff=0
  for pc,ins in rows:
   plain=re.sub(r'^@!?P\d+\s+','',ins)
   if 'HGMMA.'in ins:
    base=int(re.search(r'\bR(\d+),',ins)[1]);target=set(range(base,base+t//2));assert not pending or pending==target;pending=target;issues+=1;continue
   if 'WARPGROUP.DEPBAR.LE'in ins and '0x0'in ins:
    assert pending;assert issues==4;groups+=1;pending=set();issues=0;continue
   if pending:
    dest=re.match(r'[A-Z0-9_.]+\s+R(\d+)(?:,|\s)',plain)
    if dest:assert int(dest[1])not in pending,('generic Doverwrite',name,pc,ins)
   if 'BAR.SYNC'in ins and '0x3, 0x80'in ins:handoff+=1
  assert not pending and groups>0 and handoff==groups
  assert '0x2, 0x80'in table and 'FENCE.VIEW.ASYNC.S'in table and 'SYNCS.ARRIVE.'in table
  proof.append({'tokens':t,'gather_divisor':div,'static_K64_groups':groups,'four_HGMMA_per_group':True,'actual_D_no_generic_write_before_wait0':True,'math_128thread_barriers_after_wait0':handoff})
 assert len(proof)==8
 return {'actual_cubin_functions':proof,'registers':regs,'spills_stack_local':0,'limit':'fixed offline CUDA12.8 streams and register destinations; future actual JIT canonical regate mandatory'}

def main():
 assert not torch.cuda._initialized
 p={'state':'CPU_ONLY_CANDIDATE_GPU_UNRUN_REVIEW_REQUIRED','packing':packing(),'layout':layout(),'protocol':protocol(),'routing':routes(),'physical':physical()}
 try:protocol(True)
 except AssertionError:p['early_release_negative']='REJECTED'
 else:raise AssertionError('unsafe release admitted')
 source=(ROOT/'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py').read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MuseSpark1xFp4Experts');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_marlin_expert_outputs')
 import textwrap
 body=textwrap.dedent("\n".join(source.splitlines()[fn.lineno-1:fn.end_lineno])).strip();assert hashlib.sha256(body.encode()).hexdigest()=='781aecba1270deb918366284e4f8245ac48cce5e99f77f1a1ea95025aeaef266'
 negatives=[]
 sass=(HERE/'sass.txt').read_text()
 for name,bad in [('missing_async_wait',sass.replace('WARPGROUP.DEPBAR.LE','NO_DRAIN',1)),('missing_reader_rendezvous',sass.replace('BAR.SYNC.DEFER_BLOCKING 0x3, 0x80','NO_READER_BARRIER',1))]:
  try:physical(bad)
  except AssertionError:negatives.append(name)
  else:raise AssertionError('physical gate accepted '+name)
 p['physical_negatives_rejected']=negatives
 def streams(raw):return {t.splitlines()[0].strip():[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)]for t in raw.split('Function : ')[1:]if '7projectILi' in t.splitlines()[0]}
 assert streams(sass)==streams((HERE/'extension_sass.txt').read_text())
 p['full_host_wrapper_object_matches_device_canonical']=True
 import copy
 candidate=ast.parse((HERE/'serving_both.py').read_text());cc=next(n for n in candidate.body if isinstance(n,ast.ClassDef));assert [n.name for n in cc.body if isinstance(n,ast.FunctionDef)]==['_marlin_expert_outputs']
 cf=copy.deepcopy(next(n for n in cc.body if isinstance(n,ast.FunctionDef)))
 up=next(n for n in fn.body if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='moe_wna16_marlin_gemm')
 ci=next(i for i,n in enumerate(cf.body)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='project'and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='cache1')
 cf.body[ci]=copy.deepcopy(up);assert isinstance(cf.body[-2],ast.Assign)and cf.body[-2].targets[0].id=='projected';cf.body[-2:]=[copy.deepcopy(fn.body[-1])]
 assert ast.dump(cf,include_attributes=False)==ast.dump(fn,include_attributes=False)
 p['both_projection_adapter_inverse_to_entire_parent_method_AST']=True
 p['serving_parent_method_exact']=True;p['inherited_finalizer']='unmodified inherited _forward_marlin: norm each expert before router sum';p['CUDA_context_initialized']=torch.cuda._initialized
 (HERE/'cpu-proof.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:v for k,v in p.items()if k not in ['packing','layout','physical']},indent=2))
if __name__=='__main__':main()

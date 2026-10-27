"""Root-only bounded both-GEMM SS correctness screen; no full-MoE speed claim.
--cpu-check uses stdlib only. Production/torch imports occur only in run().
"""
import argparse,ast,hashlib,importlib.util,json,math,re,statistics,subprocess,sys,time,traceback,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2'
HERE=ROOT/'tools/moe_ss_screen_v3'
OFFLINE=ROOT/'tools/moe_ss_screen/offline-v2'
IMAGE='pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel'
TOKENS=(8,16,32,64);DIVISORS=(8,1)
VALUES=(0.,.5,1.,1.5,2.,3.,4.,6.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def verify():
 m=json.loads((HERE/'manifest.json').read_text())
 for rel,pin in m['files'].items():assert sha(ROOT/rel)==pin,('frozen input drift',rel)
 return m

def tile(bm):return next(t for t in reversed(TOKENS) if bm%t==0)
def metadata(t):return (str(t.device),str(t.dtype),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
def valid_scale_code(q):return 0<=q<=255 and (q&127)!=127
def numeric_scale(q):
 code=q&127;exponent=(code>>3)&15;mantissa=code&7
 value=mantissa*2.**-9 if exponent==0 else (1+mantissa/8)*2.**(exponent-7)
 return (-1. if q&128 else 1.)*value if code!=127 else float('nan')
def bf16_bits(value):
 raw=struct.unpack('<Q',struct.pack('<d',value))[0];sign=(raw>>48)&32768;absolute=abs(value)
 if math.isnan(value):return sign|32704
 if absolute==0:return sign
 if absolute<2.**-126:return sign|round(absolute*2.**133)
 rounded=((raw&0x7fffffffffffffff)+((1<<44)-1)+((raw>>45)&1))>>45
 return sign|min(32640,max(0,rounded-(896<<7)))
def numeric_weight_bits(q,sf):return bf16_bits(decode_nibble(q)*numeric_scale(sf))
def decode_nibble(q):return (-1. if q&8 else 1.)*VALUES[q&7]
def validate_routes(sorted_ids,experts,post,bm,slots,local=32,defensive=False):
 assert bm>=8 and bm%8==0 and len(sorted_ids)>=bm and len(experts)>=(len(sorted_ids)+bm-1)//bm
 assert 0<=post<=len(sorted_ids) and (defensive or post%bm==0)
 seen=set();assign={}
 for pos in range(post):
  route=sorted_ids[pos];expert=experts[pos//bm]
  assert -1<=expert<local
  if 0<=route<slots and expert>=0:
   assert route not in seen,'duplicate output slot would race';seen.add(route);assign[route]=expert
 return assign

def route_capacity(slots,bm):return ((slots*bm if slots<257 else slots+257*(bm-1))+3)//4*4

def resource_tables(raw):
 result={}
 for block in raw.split('Function ')[1:]:
  match=re.search(r'projectILi(\d+)ELi(\d+)E',block)
  if match:result[tuple(map(int,match.groups()))]={name:int(re.search(r'\b'+name+r':(\d+)',block)[1]) for name in ('REG','STACK','SHARED','LOCAL')}
 return result

def resource_gate(row,canonical,required):
 # cuobjdump SHARED includes extern-shared alignment padding in the frozen
 # binary; CUDA driver attributes can exclude that padding. The v2 actual
 # failure showed 0 versus canonical1024 with identical production resources.
 # Permit only those two representations, while the separate exact actual
 # binary REG/STACK/SHARED/LOCAL gate remains mandatory and unchanged.
 assert canonical['SHARED']==1024 and canonical['STACK']==canonical['LOCAL']==0
 assert row['registers']==canonical['REG'] and row['local']==0 and row['static_shared'] in (0,canonical['SHARED']) and row['max_threads']>=256 and row['max_dynamic']>=required and row['blocks_per_SM']>=1,('actual resource gate',row,canonical,required)
 return True

def production_sass(actual,canonical):
 def selected(raw):
  return [(table.splitlines()[0].strip(),table)for table in raw.split('Function : ')[1:]if '7projectILi' in table.splitlines()[0]]
 expected=selected(canonical);found=selected(actual)
 assert len(expected)==len(found)==8 and len({name for name,_ in found})==8
 assert {name for name,_ in found}=={name for name,_ in expected},'actual production symbol set differs'
 return '\n'.join('Function : '+table for name,table in found)

def route_fixture(slots,bm,kind):
 """CPU routing independent of SGLang. Global expert IDs map local0..31, others-1."""
 groups={}
 for slot in range(slots):
  expert=(slot%32 if kind=='balanced' else (31 if kind=='skew' else (slot%2 if kind=='duplicates' else 0)))
  if kind=='empty':expert=-1
  if kind=='one' and slot!=slots-1:expert=-1
  if kind=='mixed' and slot%3==0:expert=-1
  groups.setdefault(expert,[]).append(slot)
 ids=[];ex=[]
 for expert,group in sorted(groups.items()):
  group=group+[slots]*((-len(group))%bm);ids+=group;ex.extend([expert]*(len(group)//bm))
 post=len(ids)
 # Actual aligner upper allocation is four-rounded, not expert-block aligned.
 capacity=route_capacity(slots,bm)
 assert post<=capacity
 ids += [slots]*(capacity-post);ex += [-1]*(((capacity+bm-1)//bm)-len(ex))
 if kind=='tail':
  capacity=bm+4;ids=list(range(min(slots,capacity)))+[slots]*max(0,capacity-slots);ex=[0]*((capacity+bm-1)//bm);post=capacity
 assign=validate_routes(ids,ex,post,bm,slots,defensive=kind=='tail')
 return ids,ex,post,assign

def own_terminal(x,out,fresh,meta,equal,oracle):
 assert metadata(x)==meta['x'] and metadata(out)==meta['out'],'retained graph storage/view substituted'
 assert equal(x,fresh),'own current input mismatch'
 return oracle(out,fresh)

def module_from(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

# Appended diagnostic kernel reuses exact codec primitives and roundtrips through
# the same shared swizzle. Independent numeric oracle does not call these helpers.
NAMESPACE='moe_ss_v2'
DIAGNOSTIC=r'''
namespace moe_ss=moe_ss_v2;
namespace ss_screen {
template<int T,int D> __global__ void mover(const uint32_t* P,const uint8_t* S,const moe_ss::B* X,const int32_t* R,moe_ss::B* A,moe_ss::B* B,int N,int K,int rows,int slots,int capacity,int position,int stage,int expert,int nt){
 __shared__ moe_ss::B a[4096];__shared__ moe_ss::B b[T*64];int lane=threadIdx.x;
 P+=(int64_t(expert)*(N/64)+nt)*(K/16)*128;S+=(int64_t(expert)*(N/64)+nt)*(K/16)*64;
 for(int step=0;step<4;++step){uint32_t payload=P[(stage*4+step)*128+lane];int row0=(lane/4)%8+16*(lane/32);uint32_t s0=S[(stage*4+step)*64+row0],s1=S[(stage*4+step)*64+row0+8];uint32_t sc=s0|(s0<<16)|(s1<<8)|(s1<<24);__nv_bfloat162 fs[2],f[4];moe_ss::scale(sc,fs);
 for(int h=0;h<2;++h){uint32_t hw=payload>>(16*h);uint32_t q=((hw&255)<<8)|((hw&0xff00)<<16);moe_ss::dequant(q,f+2*h);f[2*h]=__hmul2(f[2*h],fs[0]);f[2*h+1]=__hmul2(f[2*h+1],fs[1]);}
 auto v=reinterpret_cast<moe_ss::B*>(f);for(int i=0;i<8;++i){int row=row0+8*((i/2)%2),k=2*(lane%4)+(i%2)+8*(i/4);a[moe_ss::sw(row,step*16+k)]=v[i];}}
 for(int i=lane;i<T*64;i+=128){int r=i/64,k=i%64,route=position+r<capacity?R[position+r]:slots;b[moe_ss::sw(r,k)]=(route>=0&&route<slots&&route/D<rows)?X[int64_t(route/D)*K+stage*64+k]:__float2bfloat16(0.f);}
 __syncthreads();for(int i=lane;i<4096;i+=128)A[i]=a[moe_ss::sw(i/64,i%64)];for(int i=lane;i<T*64;i+=128)B[i]=b[moe_ss::sw(i/64,i%64)];
}
template<int T,int D> int64_t info(int field){cudaFuncAttributes a;auto k=moe_ss::project<T,D>;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;if(field==1)return a.numRegs;if(field==2)return a.localSizeBytes;if(field==3)return a.sharedSizeBytes;if(field==4)return a.maxThreadsPerBlock;if(field==5)return a.maxDynamicSharedSizeBytes;if(field==6){int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,256,moe_ss::shared_bytes(T))!=cudaSuccess)return -2;return n;}return -3;}
}
int64_t ss_info(int64_t t,int64_t d,int64_t f){
 #define CASE(T,D) if(t==T&&d==D)return ss_screen::info<T,D>(f);
 CASE(8,1) CASE(16,1) CASE(32,1) CASE(64,1) CASE(8,8) CASE(16,8) CASE(32,8) CASE(64,8)
 #undef CASE
 return -9;
}
void ss_probe(torch::Tensor p,torch::Tensor sf,torch::Tensor x,torch::Tensor r,torch::Tensor a,torch::Tensor b,int64_t t,int64_t d,int64_t pos,int64_t stage,int64_t expert,int64_t nt,int64_t slots,int64_t stream_value){
 auto stream=reinterpret_cast<cudaStream_t>(stream_value);
 #define CASE(T,D) if(t==T&&d==D){ss_screen::mover<T,D><<<1,128,0,stream>>>((uint32_t*)p.data_ptr(),(uint8_t*)sf.data_ptr(),(moe_ss::B*)x.data_ptr(),(int32_t*)r.data_ptr(),(moe_ss::B*)a.data_ptr(),(moe_ss::B*)b.data_ptr(),p.size(1)*64,p.size(2)*16,x.size(0),slots,r.numel(),pos,stage,expert,nt);TORCH_CHECK(cudaGetLastError()==cudaSuccess,"mover probe launch");return;}
 CASE(8,1) CASE(16,1) CASE(32,1) CASE(64,1) CASE(8,8) CASE(16,8) CASE(32,8) CASE(64,8)
 #undef CASE
 TORCH_CHECK(false,"probe dispatch");
}
'''
DECL='\nint64_t ss_info(int64_t,int64_t,int64_t);\nvoid ss_probe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n'

def cpu_check():
 verify();checks=0
 for div in DIVISORS:
  for bm in (8,16,32,48,64):
   for m in (1,5,7,8,9,17,33,64):
    for kind in ('balanced','skew','duplicates','empty','one','mixed','tail'):
     ids,e,p,assign=route_fixture(m*div,bm,kind);assert all(0<=r//div<m for r in assign);checks+=1
 assert tile(48)==16 and len({(tile(t),d) for t in TOKENS for d in DIVISORS})==8
 rejected=[]
 for name,args in [('duplicate',([0,0]+[8]*6,[0],8,8,8)),('post',([8]*8,[0],9,8,8)),('expert',([0]+[8]*7,[32],8,8,8))]:
  try:validate_routes(*args)
  except AssertionError:rejected.append(name)
  else:raise AssertionError('route negative accepted')
 assert decode_nibble(1)==.5 and decode_nibble(15)==-6 and decode_nibble(8)==0
 assert '__global__' in DIAGNOSTIC and 'moe_ss::project<T,D>' in DIAGNOSTIC
 pairs=sum(1 for q in range(16)for sf in range(256)if valid_scale_code(sf)and math.isfinite(decode_nibble(q)*numeric_scale(sf)))
 assert pairs==4064
 return {'state':'CPU_PASS_GPU_UNRUN','numeric_finite_q_scale_pairs':pairs,'route_cases':checks,'route_negatives':rejected,'instantiations':8,'torch_imported':'torch' in sys.modules,'CUDA_calls':0,'scope':'isolated two-projection screen; no full-MoE speed or3x claim'}

def run(args):
 run_entry=time.time()
 if args.output.exists():raise FileExistsError('refuse existing attempt output: '+str(args.output))
 manifest=verify();assert args.review and sha(args.review)==args.review_sha256,'exact independent screen review required'
 import torch
 assert torch.__version__.split('+')[0].startswith('2.8.') and torch.version.cuda=='12.8','requires reviewed PyTorch2.8 CUDA12.8 image; no automatic installs'
 props=torch.cuda.get_device_properties(torch.cuda.current_device())
 assert (props.major,props.minor)==(9,0) and 'H100' in props.name.upper() and props.total_memory>=70_000_000_000,'bounded screen requires H100 SM90 >=70GB'
 torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 args.output.mkdir(parents=True,exist_ok=False)
 result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','inputs':{'screen_manifest_sha256':sha(HERE/'manifest.json'),'source_manifest_sha256':manifest['source_manifest_sha256'],'review_sha256':args.review_sha256},'scope':'isolated both-projection prerequisite; NOT full MoE/stock comparison/3x/official','image':IMAGE,'torch':torch.__version__,'cuda':torch.version.cuda,'device':{'name':props.name,'SMs':props.multi_processor_count,'memory':props.total_memory,'UUID':str(getattr(props,'uuid','UNAVAILABLE'))},'started_at':run_entry,'startup_preflight_before_result_seconds':time.time()-run_entry,'external_wall_unknown':'interpreter/module import, container/lock/transfer outside run-entry; root outer timestamps are charged wall','checks':[],'timings':[],'raw':[],'gates':{},'memory':{},'cost':{'compile_seconds':None,'preparation_seconds':None,'transfer_seconds':None,'transfer_note':'inputs generated on device; image/checkout transfer external and unknown, not zero'},'graph_replays':0}
 live=[];immutable=[]
 def save():dump(args.output/'result.json',result)
 def phase(name):result['phase']=name;save();print(name,flush=True)
 def bind(name,t):
  raw=t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0).view(torch.uint8)
  immutable.append((name,t,metadata(t),raw.clone()))
 def immutables():
  for name,t,meta,raw in immutable:
   assert metadata(t)==meta,('immutable storage/view changed',name)
   actual=t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0).view(torch.uint8)
   assert torch.equal(actual,raw),('immutable bytes changed',name)
 def check(actual,expected,exact=False,raw_bits=False):
  assert actual.shape==expected.shape and actual.dtype==torch.bfloat16 and torch.isfinite(actual).all()
  a,b=actual.float(),expected.float();delta=(a-b).abs();bound=.001+.02*b.abs()
  if raw_bits:assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),('raw BF16 mover byte mismatch',int((actual.view(torch.uint8)!=expected.view(torch.uint8)).sum()))
  if exact:assert torch.equal(actual,expected),('exact basis/mover mismatch',float(delta.max()))
  else:assert bool((delta<=bound).all()),('predeclared screen numeric bound failed',float(delta.max()),int((delta>bound).sum()))
  return {'max_abs':float(delta.max()),'max_relative_nonzero':float((delta/(b.abs()+1e-9)).max()),'exact':exact,'raw_BF16_bytes':raw_bits,'bound':'exact' if exact else 'abs<=0.001+0.02*abs(reference); screen-only, no full-task admission'}
 try:
  version=subprocess.check_output(['nvcc','--version'],text=True,timeout=10)
  assert 'release 12.8' in version,'requires frozen CUDA12.8 compiler'
  result['nvcc']=version
  driver_path=Path('/proc/driver/nvidia/version')
  result['driver']=driver_path.read_text() if driver_path.is_file() else 'UNAVAILABLE'
  phase('actual_JIT_and_resources')
  adapter=module_from(SOURCE/'adapter.py','ss_screen_actual_adapter')
  gate=module_from(SOURCE/'actual_sass_gate.py','ss_screen_actual_gate')
  original_build=None
  import torch.utils.cpp_extension as cpp
  original_build=cpp.load_inline
  def checked_build(*a,**kw):
   original=kw['cuda_sources'];assert original==(SOURCE/'extension_source.cu').read_text(),'actual production extension source drift'
   assert kw['functions']==['run']
   augmented=original+DIAGNOSTIC
   assert augmented[:-len(DIAGNOSTIC)]==original
   kw=dict(kw);name='moe_ss_screen_'+hashlib.sha256(augmented.encode()).hexdigest()[:16]
   if a:a=(name,)+a[1:]
   else:kw['name']=name
   kw['cuda_sources']=augmented;kw['cpp_sources']+=DECL;kw['functions']=['run','ss_info','ss_probe']
   started=time.monotonic();ext=original_build(*a,**kw);result['cost']['compile_seconds']=time.monotonic()-started
   sass=subprocess.check_output(['cuobjdump','--dump-sass',str(ext.__file__)],text=True,timeout=60)
   resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(ext.__file__)],text=True,timeout=60)
   (args.output/'actual.cu').write_text(augmented);(args.output/'actual.sass').write_text(sass);(args.output/'actual.resources').write_text(resources)
   # Import physical checker only now (GPU-authorized runtime); no context creation by CPU-check.
   main_sass=production_sass(sass,(SOURCE/'sass.txt').read_text())
   (args.output/'actual.production.sass').write_text(main_sass)
   result['gates']['actual_eight_PC_streams']=gate.check(main_sass)
   assert resource_tables(resources)==resource_tables((OFFLINE/'canonical.resources').read_text()),'actual main REG/STACK/SHARED/LOCAL drift'
   result['gates']['JIT']={'production_sha256':hashlib.sha256(original.encode()).hexdigest(),'augmented_sha256':hashlib.sha256(augmented.encode()).hexdigest(),'so_sha256':sha(ext.__file__),'sass_sha256':sha(args.output/'actual.sass'),'production_sass_sha256':sha(args.output/'actual.production.sass'),'production_inverse_exact':True,'before_any_project_launch':True}
   return ext
  cpp.load_inline=checked_build
  try:ext=adapter.module()
  finally:cpp.load_inline=original_build
  canonical_resources=resource_tables((OFFLINE/'canonical.resources').read_text())
  assert len(canonical_resources)==8
  attrs=[]
  # Persist all runtime attributes before admission so a failed attempt remains
  # useful even if the first row fails. No project kernel has launched yet.
  for t in TOKENS:
   for div in DIVISORS:
    row={name:int(ext.ss_info(t,div,code)) for name,code in [('registers',1),('local',2),('static_shared',3),('max_threads',4),('max_dynamic',5),('blocks_per_SM',6)]}
    required=2*(4096+t*64)*2+32+1024
    attrs.append(dict(T=t,DIV=div,dynamic_shared=required,canonical=canonical_resources[t,div],**row))
  result['gates']['actual_resources']=attrs;save()
  for row in attrs:
   resource_gate(row,canonical_resources[row['T'],row['DIV']],row['dynamic_shared'])
  phase('input_independent_banks')
  result['memory']['before_banks']=torch.cuda.memory_allocated();start=time.monotonic();gen=torch.Generator(device='cuda').manual_seed(827112)
  banks={};raws={};decoded={}
  for div,n in [(8,8192),(1,4096)]:
   raw=torch.randint(0,256,(32,n,2048),dtype=torch.uint8,device='cuda',generator=gen)
   # Every finite nonnegative E4M3 code, INCLUDING subnormal1..7.
   # v2 uses canonical BF16 weights rather than inherited tiny re-expression.
   scale_codes=torch.tensor(list(range(127)),device='cuda',dtype=torch.uint8)
   sf_bytes=scale_codes[torch.randint(0,len(scale_codes),(32,n,256),device='cuda',generator=gen)]
   # Deterministically exhaust16q×254finite signed scales in first N64/K64
   # tile. Every group16 has nibble0..15; first254groups cover all finite sf.
   finite_codes=list(range(127))+list(range(128,255))
   sf_bytes[0,:64,:4]=torch.tensor(finite_codes+[0,0],device='cuda',dtype=torch.uint8).reshape(64,4)
   raw[0,:64,:32]=torch.tensor([16,50,84,118,152,186,220,254]*4,device='cuda',dtype=torch.uint8)
   sf=sf_bytes.view(torch.float8_e4m3fn)
   assert bool(torch.isfinite(sf.float()).all()),'invalid NaN rawscale domain'
   gs=torch.linspace(.0005,.002,32,device='cuda',dtype=torch.float32)
   bank=adapter.prepare(raw,sf,gs)
   banks[div]=bank;raws[div]=(raw,sf,gs)
   assert torch.equal(bank.globals,gs.to(torch.bfloat16)),'independent global-scale math mismatch'
   for key,tensor in [('raw',raw),('sf',sf),('gs',gs),('packed',bank.packed),('processed_scales',bank.scales),('globals',bank.globals)]:bind(str(div)+'/'+key,tensor)
  torch.cuda.synchronize();result['cost']['preparation_seconds']=time.monotonic()-start;result['memory']['after_banks_and_immutable_copies']=torch.cuda.memory_allocated();save()
  # CPUdouble→integer BF16 lookup uploaded as bits: no GPU FTZ can erase
  # a tiny expected value or signed zero before the hardware mover comparison.
  bits=[[numeric_weight_bits(q,sf) if valid_scale_code(sf) else 32704 for sf in range(256)] for q in range(16)]
  bit_lut=torch.tensor([[v if v<32768 else v-65536 for v in row] for row in bits],device='cuda',dtype=torch.int16)
  def weight(div,expert):
   key=(div,expert)
   if key not in decoded:
    raw,sf,gs=raws[div];lo=raw[expert]&15;hi=raw[expert]>>4
    codes=torch.stack((lo,hi),dim=-1).reshape(raw.shape[1],4096).long()
    # Numeric E2M1 and E4M3 conversion, not packed-lane integer codec.
    scale_codes=sf[expert].view(torch.uint8).repeat_interleave(16,dim=-1).long()
    decoded[key]=bit_lut[codes,scale_codes].contiguous().view(torch.bfloat16)
   return decoded[key]
  def oracle(x,div,assign,slots):
   n=8192 if div==8 else 4096;out=torch.zeros((slots,n),device='cuda',dtype=torch.bfloat16)
   for expert in sorted(set(assign.values())):
    routes=[r for r,e in assign.items() if e==expert];ix=torch.tensor(routes,device='cuda',dtype=torch.long)
    a=x[ix//div].float();w=weight(div,expert).float()
    partial=(a@w.T).to(torch.bfloat16)
    out[ix]=(partial*banks[div].globals[expert]).to(torch.bfloat16)
   return out
  def case(div,bm,m,kind,seed):
   slots=m*div;ids,e,p,assign=route_fixture(slots,bm,kind)
   case_gen=torch.Generator(device='cuda').manual_seed(seed)
   x=(torch.randn((m,4096),generator=case_gen,device='cuda')*.125).to(torch.bfloat16)
   r=torch.tensor(ids,device='cuda',dtype=torch.int32);ex=torch.tensor(e,device='cuda',dtype=torch.int32);post=torch.tensor([p],device='cuda',dtype=torch.int32)
   out=torch.empty((slots,8192 if div==8 else 4096),device='cuda',dtype=torch.bfloat16)
   return {'div':div,'bm':bm,'m':m,'kind':kind,'seed':seed,'slots':slots,'assign':assign,'x':x,'routes':r,'experts':ex,'post':post,'out':out,'route_CPU':ids,'expert_CPU':e}
  def project(c):return adapter.project(c['x'],banks[c['div']],c['routes'],c['experts'],c['post'],c['bm'],c['out'],c['div'])
  def inactive_check(c):
   inactive=torch.ones(c['slots'],device='cuda',dtype=torch.bool)
   if c['assign']:inactive[torch.tensor(list(c['assign']),device='cuda',dtype=torch.long)]=False
   values=c['out'][inactive]
   assert not int(torch.count_nonzero(values.view(torch.uint8))),'nonlocal/unrouted slots must remain explicit +zero bytes'
  def ref(c,fresh):
   inactive_check(c)
   return oracle(fresh,c['div'],c['assign'],c['slots'])
  def route_snapshot(c):return [(key,metadata(c[key]),c[key].clone()) for key in ('routes','experts','post')]
  def route_check(c,snap):
   for key,meta,original in snap:assert metadata(c[key])==meta and torch.equal(c[key],original),('route/counter-metadata or values changed',key)
  phase('mover_probe_all_K_slots')
  # Exercise every K64 stage; endpoints/middle output-channel tiles and experts0/31.
  for div in DIVISORS:
   for t in TOKENS:
    c=case(div,t,max(9,t), 'mixed',8400+t+div);bank=banks[div]
    for stage in range(64):
     expert=31 if stage%2 else 0;nt=[0,(bank.packed.shape[1]-1)//2,bank.packed.shape[1]-1][stage%3]
     a=torch.empty((64,64),device='cuda',dtype=torch.bfloat16);b=torch.empty((t,64),device='cuda',dtype=torch.bfloat16)
     ext.ss_probe(bank.packed,bank.scales,c['x'],c['routes'],a,b,t,div,0,stage,expert,nt,c['slots'],torch.cuda.current_stream().cuda_stream)
     torch.cuda.synchronize();check(a,weight(div,expert)[nt*64:(nt+1)*64,stage*64:(stage+1)*64],True,True)
     expected=torch.zeros_like(b)
     for token in range(t):
      r=c['route_CPU'][token]
      if 0<=r<c['slots']:expected[token]=c['x'][r//div,stage*64:(stage+1)*64]
     check(b,expected,True,True)
    result['checks'].append({'kind':'hardware_dequant_shared_gather','DIV':div,'T':t,'all64_K64stages':True,'experts':[0,31],'Ntiles':'first/middle/last','deterministic_first_tile_q_scale_pairs':4064,'exact_raw_BF16_bytes_including_signed_zero':True});save()
  phase('edge_cases_all_instantiations')
  for div in DIVISORS:
   for bm in (8,16,32,48,64):
    for kind,m in [('balanced',9),('skew',17),('duplicates',7),('mixed',5),('one',1),('empty',9),('tail',17)]:
     c=case(div,bm,m,kind,8417);initial=c['x'].clone();xm=metadata(c['x']);routes=route_snapshot(c)
     c['out'].fill_(float('nan'));project(c);torch.cuda.synchronize();metrics=check(c['out'],ref(c,initial))
     assert metadata(c['x'])==xm and torch.equal(c['x'],initial);route_check(c,routes);immutables()
     if kind=='empty':assert not int(torch.count_nonzero(c['out']))
     # Stream capture must retain current stream, route pointers, output storage.
     g=torch.cuda.CUDAGraph();live.append(g)
     with torch.cuda.graph(g):project(c)
     torch.cuda.synchronize();role={'x':metadata(c['x']),'out':metadata(c['out'])}
     mutation_seed=91000+div*1000+bm*10+m
     mutation_gen=torch.Generator(device='cuda').manual_seed(mutation_seed)
     changed=(torch.randn(c['x'].shape,generator=mutation_gen,device='cuda')*.125).to(torch.bfloat16)
     outliers=changed.clone();outliers[:,::127]=(outliers[:,::127].float()*30).to(torch.bfloat16)
     mutations=[]
     for label,fresh in [('current_X',changed),('zero',torch.zeros_like(initial)),('outliers',outliers)]:
      c['x'].copy_(fresh);c['out'].fill_(float('nan'));g.replay();torch.cuda.synchronize();result['graph_replays']+=1
      mutations.append({'kind':label,'oracle':own_terminal(c['x'],c['out'],fresh,role,torch.equal,lambda out,x:check(out,ref(c,x)))})
      route_check(c,routes);immutables()
     # Explicit valid→empty via existing captured pointer, never recapture/repair.
     if c['assign']:
      c['post'].zero_();c['out'].fill_(float('nan'));g.replay();torch.cuda.synchronize();result['graph_replays']+=1
      assert not int(torch.count_nonzero(c['out'].view(torch.uint8))),'valid→empty retained graph stale output'
      assert torch.equal(c['post'],torch.zeros_like(c['post']))
      c['post'].copy_(routes[-1][2]);route_check(c,routes)
     torch.cuda.synchronize();g.reset();live.remove(g)
     result['checks'].append({'kind':'edge','DIV':div,'T':tile(bm),'block_m':bm,'m':m,'routes':kind,'oracle':metrics,'graph_mutations':mutations,'mutation_seed':mutation_seed,'valid_to_empty':bool(c['assign']),'counter_state':'no global progress counters; routes/experts/post immutable except declared post mutation','drain':'PASS'});save()
  phase('all_K_basis_all_instantiations')
  for div in DIVISORS:
   for bm in TOKENS:
    c=case(div,bm,64,'skew',9801);c['experts'].fill_(0);c['assign']={slot:0 for slot in range(c['slots'])}
    # Preserve padded capacities, but all real route slots now use expert0.
    project(c);torch.cuda.synchronize();routes=route_snapshot(c);role={'x':metadata(c['x']),'out':metadata(c['out'])}
    g=torch.cuda.CUDAGraph();live.append(g)
    with torch.cuda.graph(g):project(c)
    torch.cuda.synchronize()
    for start in range(0,4096,64):
     fresh=torch.zeros_like(c['x']);rows=torch.arange(64,device='cuda');cols=rows+start;fresh[rows,cols]=8
     # Single-term expected result: independent raw numeric dequant, BF16 output
     # boundary then BF16 global multiply. No GEMM reduction-order uncertainty.
     single=(weight(div,0)[:,cols].T.float()*8).to(torch.bfloat16)
     single=(single*banks[div].globals[0]).to(torch.bfloat16)
     expected=single.repeat_interleave(div,dim=0)
     c['x'].copy_(fresh);c['out'].fill_(float('nan'));g.replay();torch.cuda.synchronize();result['graph_replays']+=1
     own_terminal(c['x'],c['out'],fresh,role,torch.equal,lambda out,x:check(out,expected,True));route_check(c,routes);immutables()
    torch.cuda.synchronize();g.reset();live.remove(g)
    result['checks'].append({'kind':'all_K_basis','DIV':div,'T':bm,'all4096K':True,'replays':64,'input_rows_per_replay':64,'output_channels':8192 if div==8 else 4096,'reference':'numeric single-term BF16 boundaries; exact including every real route slot'});save()
  phase('untimed_shared_cache13_dependency_chain')
  # This is an adapter-lifetime diagnostic, not the full expert/norm/router task.
  for bm in (8,16,32,48,64):
   up=case(8,bm,5,'skew',12000+bm);slots=up['slots'];cache13=torch.empty(slots*8192,device='cuda',dtype=torch.bfloat16)
   up['out']=cache13.view(slots,8192);middle=torch.empty((slots,4096),device='cuda',dtype=torch.bfloat16)
   down=dict(up);down.update(div=1,m=slots,x=middle,out=cache13[:slots*4096].view(slots,4096))
   assert up['out'].untyped_storage().data_ptr()==down['out'].untyped_storage().data_ptr()
   assert middle.untyped_storage().data_ptr()!=cache13.untyped_storage().data_ptr()
   def activation(value):
    fp=value.float();return (fp[:,:4096]*torch.sigmoid(fp[:,:4096])*fp[:,4096:]).to(torch.bfloat16)
   def chain():
    project(up);middle.copy_(activation(up['out']));project(down)
   fresh0=up['x'].clone();project(up);torch.cuda.synchronize();check(up['out'],ref(up,fresh0))
   chain();torch.cuda.synchronize();snaps=route_snapshot(up)
   role={'x':metadata(up['x']),'out':metadata(down['out'])};midmeta=metadata(middle);cachemeta=metadata(cache13)
   def chain_oracle(out,fresh):
    expected_up=oracle(fresh,8,up['assign'],slots)
    expected_down=oracle(activation(expected_up),1,up['assign'],slots)
    return check(out,expected_down)
   chain_oracle(down['out'],fresh0)
   g=torch.cuda.CUDAGraph();live.append(g)
   with torch.cuda.graph(g):chain()
   torch.cuda.synchronize();checks=[]
   for kind,fresh in [('zero',torch.zeros_like(fresh0)),('new_X',fresh0.roll(1,1))]:
    up['x'].copy_(fresh);cache13.fill_(float('nan'));middle.fill_(float('nan'));g.replay();torch.cuda.synchronize();result['graph_replays']+=1
    metrics=own_terminal(up['x'],down['out'],fresh,role,torch.equal,chain_oracle)
    if kind=='zero':assert not int(torch.count_nonzero(down['out']))
    assert metadata(middle)==midmeta and metadata(cache13)==cachemeta
    route_check(up,snaps);immutables();checks.append({'kind':kind,'oracle':metrics})
   assert int(torch.count_nonzero(down['out']))>0,'empty transition requires discriminating preceding valid nonzero output'
   up['post'].zero_();cache13.fill_(float('nan'));g.replay();torch.cuda.synchronize();result['graph_replays']+=1
   assert not int(torch.count_nonzero(down['out'].view(torch.uint8))),'empty shared-cache chain stale output'
   up['post'].copy_(snaps[-1][2]);route_check(up,snaps)
   torch.cuda.synchronize();g.reset();live.remove(g)
   result['checks'].append({'kind':'shared_cache13_stream_dependency','block_m':bm,'T':tile(bm),'m':5,'output':'down only; up workspace deliberately consumed then overwritten','activation':'FP32 pointwise diagnostic SwiGLU→BF16, not serving activation oracle','checks':checks,'valid_to_empty':True,'same_stream_alias_drain':'PASS'});save()
  phase('separate_stage_timings_NOT_full_MoE')
  flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
  # Equivalent original-token64/skew fixture: up64rows,down512expanded rows. No reference
  # kernel speed ratio, stock benchmark proxy, winner-only shape selection or ROI.
  for div in DIVISORS:
   input_rows=64 if div==8 else 512
   cells={};graphs={};fixtures={}
   for bm in TOKENS:
    c=case(div,bm,input_rows,'skew',11000+bm);project(c);torch.cuda.synchronize();g=torch.cuda.CUDAGraph();live.append(g)
    with torch.cuda.graph(g):project(c)
    torch.cuda.synchronize();cells[bm]=c;graphs[bm]=g
   live_storage=[t.untyped_storage().data_ptr() for c in cells.values()for t in (c['x'],c['out'])]
   assert len(set(live_storage))==8,'timed X/output storages alias across template arms'
   # All arms use same seed fixture per paired round, distinct retained storage.
   for seed in range(11000,11064):
    fixture_gen=torch.Generator(device='cuda').manual_seed(seed)
    fixtures[seed]=(torch.randn((input_rows,4096),device='cuda',generator=fixture_gen)*.125).to(torch.bfloat16)
   roles={bm:{'x':metadata(c['x']),'out':metadata(c['out'])} for bm,c in cells.items()};snaps={bm:route_snapshot(c) for bm,c in cells.items()};last={};raw=[]
   def one(bm,seed,phase_name):
    c=cells[bm];c['x'].copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[bm].replay();b.record();torch.cuda.synchronize();result['graph_replays']+=1
    us=a.elapsed_time(b)*1000;assert 0<us<math.inf
    row={'DIV':div,'T':bm,'input_rows':input_rows,'original_tokens':64,'phase':phase_name,'seed':seed,'us':us,'zero_included':True};raw.append(row);result['raw'].append(row)
    with (args.output/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    last[bm]=seed;return us
   def terminal(label):
    rows=[]
    # Exact last timed outputs are read before any replay/Xcopy/poison/profile.
    for bm,c in cells.items():
     metrics=own_terminal(c['x'],c['out'],fixtures[last[bm]],roles[bm],torch.equal,lambda out,x:check(out,ref(c,x)))
     route_check(c,snaps[bm]);immutables();rows.append({'T':bm,'phase':label,'seed':last[bm],'oracle':metrics})
    return rows
   f1={}
   for i,bm in enumerate(TOKENS):f1[str(bm)+'_before']=[one(bm,11000+11*i+j,'F1_before') for j in range(11)]
   first_terminal=terminal('F1_before')
   # Eight balanced permutations, four rotations and their reversals;20 rounds
   # gives each position5times with paired-order lead exactly balanced.
   orders=[(0,1,2,3),(3,2,1,0),(1,3,0,2),(2,0,3,1)]
   paired=[]
   for round_index in range(20):
    paired.append({str(TOKENS[i]):one(TOKENS[i],11044+round_index,'paired') for i in orders[round_index%4]})
   paired_terminal=terminal('paired')
   for i,bm in reversed(list(enumerate(TOKENS))):f1[str(bm)+'_after']=[one(bm,11000+11*i+j,'F1_after') for j in range(11)]
   final_terminal=terminal('F1_after')
   profiles=[]
   for bm,c in cells.items():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     graphs[bm].replay();torch.cuda.synchronize();result['graph_replays']+=1
    kernels=[event.name for event in prof.events() if event.device_type==torch.autograd.DeviceType.CUDA]
    project_nodes=[name for name in kernels if re.search(re.escape(NAMESPACE)+r'::project<\s*'+str(bm)+r'\s*,\s*'+str(div)+r'\s*>',name)]
    assert len(project_nodes)==1,('actual graph project template/count mismatch',bm,div,kernels)
    zero_nodes=[name for name in kernels if NAMESPACE+'::project<' not in name]
    assert len(zero_nodes)==1 and all(re.search(r'memset|fill|elementwise',name,re.I) for name in zero_nodes),('timed explicit zero node missing/extra graph compute',kernels)
    trace=args.output/('DIV'+str(div)+'_T'+str(bm)+'.trace.json');prof.export_chrome_trace(str(trace))
    metrics=own_terminal(c['x'],c['out'],fixtures[last[bm]],roles[bm],torch.equal,lambda out,x:check(out,ref(c,x)));route_check(c,snaps[bm]);immutables()
    profiles.append({'T':bm,'GPU_nodes':kernels,'explicit_zero_nodes':zero_nodes,'production_project_nodes':project_nodes,'trace':trace.name,'counted_postterminal_replays':1,'oracle':metrics})
   result['timings'].append({'DIV':div,'meaning':'stage-only throughput/latency; no fullMoE/stock/official/3x gain','input_rows':input_rows,'original_tokens':64,'expanded_slots':512,'N':8192 if div==8 else 4096,'K':4096,'routing':'alllocal expert31,duplicates across distinct route slots','paired':paired,'F1_medians_us':{k:statistics.median(v) for k,v in f1.items()},'terminal':first_terminal+paired_terminal+final_terminal,'profiles':profiles})
   for bm,g in graphs.items():torch.cuda.synchronize();g.reset();live.remove(g)
   save()
  immutables();result['state']='ISOLATED_SCREEN_VALID_ROOT_REVIEW_REQUIRED_FULL_MOE_UNVALIDATED';result['memory']['peak_allocated']=torch.cuda.max_memory_allocated();result['official_or_3x_claim']=False;phase('complete')
 except BaseException:
  result['state']='INVALID_SCREEN_NO_PERFORMANCE_ADMISSION';result['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAIN_RESET')
   except BaseException as error:cleanup.append(repr(error));result['state']='INVALID_CLEANUP'
  result['cleanup']=cleanup;result['finished_at']=time.time();result['elapsed_seconds']=result['finished_at']-result['started_at'];save()

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--review-sha256');args=p.parse_args()
 if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
 assert args.output;run(args)
if __name__=='__main__':main()

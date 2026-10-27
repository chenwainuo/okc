"""One real M8 MoE serving case, source/JIT gate, task correctness then small timings.
No imports from project/GPU runtime occur until run(); no broad suite or promotion.
"""
import argparse,hashlib,importlib,importlib.util,json,re,statistics,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True  # preserve extracted immutable bundle during root imports
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
IMAGE='lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INFO=r'''
namespace ss_pilot {
template<int T,int V> int64_t info(int f){cudaFuncAttributes a;auto k=moe_ss_v2::project<T,V>;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;if(f==1)return a.numRegs;if(f==2)return a.localSizeBytes;if(f==3)return a.sharedSizeBytes;if(f==4)return a.maxThreadsPerBlock;if(f==5)return a.maxDynamicSharedSizeBytes;if(f==6){int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,256,moe_ss_v2::shared_bytes(T))!=cudaSuccess)return -2;return n;}return -3;}
}
int64_t pilot_info(int64_t t,int64_t v,int64_t f){
#define CASE(T,V) if(t==T&&v==V)return ss_pilot::info<T,V>(f);
CASE(8,1) CASE(8,8) CASE(16,1) CASE(16,8) CASE(32,1) CASE(32,8) CASE(64,1) CASE(64,8)
#undef CASE
return -9;}
'''
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def resources(raw):
 out={}
 for block in raw.split('Function ')[1:]:
  m=re.search(r'projectILi(\d+)ELi(\d+)E',block)
  if m:out[tuple(map(int,m.groups()))]={k:int(v)for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',block)}
 return out
def run(args):
 if args.output.exists():raise FileExistsError('refuse existing pilot output')
 assert args.review and sha(args.review)==args.review_sha256
 spec=json.loads((D/'manifest.json').read_text())
 for rel,h in spec['files'].items():assert sha(ROOT/rel)==h,('frozen pilot input drift',rel)
 args.output.mkdir(parents=True);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','image':IMAGE,'scope':'one M8 normal actual unchanged full-expert serving method, original aligner/activation/finalizer; same-process descriptive pilot, not14suite/holdouts/3x/model/official','started_at':time.time(),'source_manifest_sha256':sha(D/'manifest.json'),'review_sha256':args.review_sha256,'jit':[],'correctness':[],'raw':[],'graph_replays':0};live=[]
 def save():(args.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):r['phase']=name;save();print(name,flush=True)
 try:
  phase('materialize_verified_candidate_bundle')
  bundle=load(ROOT/'benchmarks/rsi_bundle.py','ss_pilot_bundle')
  tree=args.output/'candidate-tree';bm=bundle.materialize(D/'candidate.tar.gz',D/'candidate.tar.gz.manifest.json',tree)
  # Use existing frozen SGLang namespace isolation, not an ambient editable tree.
  worker=load(ROOT/'benchmarks/rsi_worker.py','ss_pilot_import_paths');worker.setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  r['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'nvcc':subprocess.check_output(['nvcc','--version'],text=True,timeout=10)}
  props=torch.cuda.get_device_properties(0);r['device']={'name':props.name,'SMs':props.multi_processor_count,'memory':props.total_memory,'uuid':str(getattr(props,'uuid','UNAVAILABLE'))};save()
  assert torch.version.cuda=='13.0' and (props.major,props.minor)==(9,0) and props.total_memory>=70_000_000_000
  reference=importlib.import_module('reference');base=importlib.import_module('serving_moe_base');candidate=importlib.import_module('serving_moe');full=sys.modules['ss_exact_serving_both'];engine=full._engine
  source=tree/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2';assert sha(source/'serving_both.py')==spec['SS_serving_source_sha256']
  import torch.utils.cpp_extension as cpp
  original_build=cpp.load_inline
  def checked_build(*a,**kw):
   name=kw.get('name',a[0]if a else '')
   if not str(name).startswith('moe_ss_both_v2_'):return original_build(*a,**kw)
   original=kw['cuda_sources'];assert original==(source/'extension_source.cu').read_text() and kw['functions']==['run']
   augmented=original+INFO;kw=dict(kw);kw['cuda_sources']=augmented;kw['cpp_sources']+='\nint64_t pilot_info(int64_t,int64_t,int64_t);\n';kw['functions']=['run','pilot_info'];newname='ss_serving_cu130_'+hashlib.sha256(augmented.encode()).hexdigest()[:20]
   if a:a=(newname,)+a[1:]
   else:kw['name']=newname
   start=time.monotonic();ext=original_build(*a,**kw)
   sass=subprocess.check_output(['cuobjdump','--dump-sass',ext.__file__],text=True,timeout=60);res=subprocess.check_output(['cuobjdump','--dump-resource-usage',ext.__file__],text=True,timeout=60)
   (args.output/'actual_ss.cu').write_text(augmented);(args.output/'actual_ss.sass').write_text(sass);(args.output/'actual_ss.resources').write_text(res)
   import shutil
   shutil.copy2(ext.__file__,args.output/'actual_ss.so')
   tables=['Function : '+t for t in sass.split('Function : ')[1:]if '7projectILi'in t.splitlines()[0]];assert len(tables)==8
   production='\n'.join(tables);(args.output/'actual_ss.production.sass').write_text(production)
   # Same source protocol checked on actual CUDA13 instructions. Deliberately
   # no equality requirement against CUDA12.8 PCs/register allocation.
   physical=load(source/'cpu_proof.py','ss_pilot_actual_physical').physical(production)
   rr=resources(res);assert set(rr)=={(t,v)for t in (8,16,32,64)for v in (1,8)}
   attrs=[]
   for (t,v),row in sorted(rr.items()):
    assert row['STACK']==row['LOCAL']==0 and 0<row['REG']<=255 and ((row['REG']+7)//8)*8*256<=65536
    arow={k:int(ext.pilot_info(t,v,f))for k,f in [('registers',1),('local',2),('user_static',3),('max_threads',4),('max_dynamic',5),('occupancy',6)]}
    required=2*(4096+t*64)*2+32+1024
    assert arow['registers']==row['REG'] and arow['local']==0 and arow['user_static']in(0,row['SHARED']) and arow['max_threads']>=256 and arow['max_dynamic']>=required and arow['occupancy']>=1
    attrs.append(dict(T=t,DIV=v,dynamic_shared=required,binary_resources=row,**arow))
   r['jit'].append({'production_source_exact':True,'production_sha256':hashlib.sha256(original.encode()).hexdigest(),'augmented_sha256':hashlib.sha256(augmented.encode()).hexdigest(),'so_sha256':sha(args.output/'actual_ss.so'),'sass_sha256':sha(args.output/'actual_ss.sass'),'compile_seconds':time.monotonic()-start,'physical_functions':physical['actual_cubin_functions'],'actual_resources':attrs,'before_first_SS_math':True,'CUDA12_8_PC_equality_required':False});save();return ext
  cpp.load_inline=checked_build
  phase('actual_weights_and_two_prepares')
  weights=reference.generate_weights(5252);frozen=[(t,t.clone())for t in weights.values()]
  try:
   states={}
   for label,module in [('anchor',base),('SS',candidate)]:
    start=time.monotonic();states[label]=module.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize();r.setdefault('prepare_seconds',{})[label]=time.monotonic()-start;save()
   # Force compile/source/physical/resource gate before any candidate math.
   engine.module();assert len(r['jit'])==1
  finally:cpp.load_inline=original_build
  for layer in states.values():
   for attr in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_scale_2','w2_weight_scale_2','_post_expert_norm_weight'):
    tensor=getattr(layer,attr);frozen.append((tensor,tensor.clone()))
   for attr in ('_ss_up_bank','_ss_down_bank'):
    if hasattr(layer,attr):
     bank=getattr(layer,attr)
     for tensor in (bank.packed,bank.scales,bank.globals):frozen.append((tensor,tensor.clone()))
  assert sha(tree/'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py')=='05ec9a8ad7ed86913b05667725f20c35a446d68af6f94f2ecc4900c5ac1228d5'
  r['identities']={'baseline_archive_sha256':spec['baseline_archive_sha256'],'candidate_archive_sha256':bm['archive_sha256'],'anchor_source_sha256':'05ec9a8ad7ed86913b05667725f20c35a446d68af6f94f2ecc4900c5ac1228d5','baseline_adapter_sha256':sha(tree/'benchmarks/serving_moe_base.py'),'candidate_source_sha256':sha(source/'serving_both.py'),'actual_class':str(type(states['SS'])),'pipeline':'original actual aligner + actual silu_and_mul + inherited forward/fused finalizer, no arithmetic adapters'}
  for name in ('sglang.srt.layers.muse_spark_fp4_moe','sglang.kernels.ops.activation.activation','sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton'):
   mod=importlib.import_module(name);r['identities'][name]={'path':mod.__file__,'sha256':sha(mod.__file__)}
  fixture=reference.generate_input(8,'ep8',7777);work={};modules={'anchor':base,'SS':candidate}
  def immutable():
   for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  def check(label,kind,data,out):
   ok,msg=reference.check_implementation(data,out,weights);r['correctness'].append({'label':label,'kind':kind,'pass':ok,'message':msg});save();assert ok,(label,kind,msg);immutable()
  phase('one_actual_serving_case_before_timing')
  for label,module in modules.items():
   data={k:(v.clone()if isinstance(v,torch.Tensor)else dict(v))for k,v in fixture.items()};before={k:v.clone()for k,v in data.items()if isinstance(v,torch.Tensor)}
   out=module.kernel_fn(data,states[label]);torch.cuda.synchronize();check(label,'eager_M8_normal',data,out)
   assert all(torch.equal(data[k],v)for k,v in before.items())
   for _ in range(2):module.kernel_fn(data,states[label])
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph();live.append(g)
   with torch.cuda.graph(g):out=module.kernel_fn(data,states[label])
   torch.cuda.synchronize();work[label]=(data,out,g)
   for kind,seed in [('current_X',7777),('fresh_X_and_routing',7778)]:
    fresh=reference.generate_input(8,'ep8',seed)
    for k in before:data[k].copy_(fresh[k])
    out.fill_(float('nan'));g.replay();torch.cuda.synchronize();r['graph_replays']+=1;check(label,kind,data,out)
    assert all(torch.equal(data[k],fresh[k])for k in before)
   assert int(torch.count_nonzero(out))>0
   data['topk_ids'].copy_(torch.arange(32,40,device='cuda',dtype=torch.int32).expand(8,8));out.fill_(float('nan'));g.replay();torch.cuda.synchronize();r['graph_replays']+=1
   assert not int(torch.count_nonzero(out.view(torch.uint8)));r['correctness'].append({'label':label,'kind':'valid_to_empty_current_captured_ids','exact_zero':True});save()
  phase('small_same_process_F1_and_ABBA_NOT_full_benchmark')
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record();events[1].record();events[1].synchronize();last={}
  def one(label,seed,kind):
   data,out,g=work[label];fresh=reference.generate_input(8,'ep8',seed)
   for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
   flush.zero_();events[0].record();g.replay();events[1].record();events[1].synchronize();r['graph_replays']+=1;us=events[0].elapsed_time(events[1])*1000;assert 0<us<float('inf')
   r['raw'].append({'label':label,'seed':seed,'kind':kind,'us':us});last[label]=(seed,fresh);return us
  def terminal(kind):
   for label,(seed,fresh)in last.items():
    data,out,g=work[label];assert all(torch.equal(data[key],fresh[key])for key in ('hidden_states','topk_ids','topk_weights'));check(label,'own_terminal_'+kind+'_seed'+str(seed),data,out)
  f1={}
  for block,label in enumerate(('anchor','SS','anchor')):f1[label+('_after'if block==2 else'_before'if block==0 else'')]=[one(label,7800+block*10+i,'F1')for i in range(5)]
  terminal('F1');r['F1_raw_us']=f1;r['F1_medians_us']={k:statistics.median(v)for k,v in f1.items()};save()
  pairs=[]
  for i in range(8):pairs.append({label:one(label,7900+i,'paired')for label in (('anchor','SS')if i%2==0 else('SS','anchor'))})
  terminal('paired');r['paired_raw_us']=pairs;r['paired_anchor_minus_SS_us']=statistics.median(row['anchor']-row['SS']for row in pairs);r['paired_anchor_over_SS_ratio']=statistics.median(row['anchor']/row['SS']for row in pairs)
  for _,_,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  immutable();bundle.verify_tree(tree,bm);r['state']='ONE_TRUE_SERVING_CASE_VALID_SMALL_TIMINGS_ROOT_REVIEW_REQUIRED';r['threefold_or_full14_or_official_claim']=False;phase('complete')
 except BaseException:r['state']='INVALID_ACTUAL_SERVING_PILOT_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAIN_RESET')
   except BaseException as e:cleanup.append(repr(e));r['state']='INVALID_CLEANUP'
  r['cleanup']=cleanup;r['finished_at']=time.time();r['elapsed_seconds']=r['finished_at']-r['started_at'];save()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);args=p.parse_args();run(args)
if __name__=='__main__':main()

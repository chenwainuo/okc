"""Bounded warm M8 real-serving stage screen: exact SS versus full signed decode LUT.
No GPU/runtime import before main. No queue, remote, setup or source mutation.
"""
import argparse,ast,hashlib,importlib,importlib.util,inspect,json,re,statistics,subprocess,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def physical(actual,parent_gate):
 # Reuse the reviewed v2 accumulator lifetime/barrier checker; only expected
 # function cardinality changes from eight to the bounded two T8 functions.
 fn=inspect.getsource(parent_gate.physical)
 assert fn.count('assert len(proof)==8')==1
 ns=dict(parent_gate.__dict__);exec(fn.replace('assert len(proof)==8','assert len(proof)==2'),ns)
 return ns['physical'](actual)['actual_cubin_functions']

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);p.add_argument('--profile',action='store_true');args=p.parse_args()
 if args.output.exists():raise FileExistsError('refuse existing LUT diagnostic output')
 assert sha(args.review)==args.review_sha256
 pins=json.loads((D/'manifest.json').read_text())
 for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,('frozen LUT screen input drift',rel)
 assert sha(args.prior/'result.json')==pins['prior_result_sha256'] and sha(args.prior/'actual_ss.so')==pins['prior_SO_sha256']
 prior=json.loads((args.prior/'result.json').read_text());assert prior['state']=='ONE_TRUE_SERVING_CASE_VALID_SMALL_TIMINGS_ROOT_REVIEW_REQUIRED'
 args.output.mkdir(parents=True);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','started_at':time.time(),'scope':'M8normal, separate actual-serving up/down stage inputs, exact SS original capacity-grid and project zeroing, no full-serving speedup claim','manifest_sha256':sha(D/'manifest.json'),'review_sha256':args.review_sha256,'correctness':[],'raw':[],'phases':[],'jit':[],'profiles':{}};live=[]
 def save():(args.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):r['phases'].append({'phase':name,'elapsed_seconds':time.time()-r['started_at']});r['phase']=name;save();print(name,flush=True)
 try:
  phase('verify_reuse_tree_and_runtime')
  tree=args.prior/'candidate-tree';bundle=load(ROOT/'benchmarks/rsi_bundle.py','ss_lut_bundle');meta=json.loads((tree/'.rsi-bundle.json').read_text());bundle.verify_tree(tree,meta)
  assert meta['archive_sha256']==pins['prior_archive_sha256']
  worker=load(ROOT/'benchmarks/rsi_worker.py','ss_lut_paths');worker.setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  props=torch.cuda.get_device_properties(0);assert torch.version.cuda=='13.0' and(props.major,props.minor)==(9,0)
  r['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':props.name};save()
  reference=importlib.import_module('reference');serving=importlib.import_module('serving_moe');original=sys.modules['ss_exact_serving_both']._engine
  source=tree/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2'
  phase('reuse_exact_original_binary')
  ext=load(args.prior/'actual_ss.so',pins['prior_module_name'])
  import torch.utils.cpp_extension as cpp
  builder=cpp.load_inline
  def reuse(*a,**kw):
   assert kw['cuda_sources']==(source/'extension_source.cu').read_text() and kw['functions']==['run']
   return ext
  cpp.load_inline=reuse
  try:assert original.module()is ext
  finally:cpp.load_inline=builder
  # Importing an isolated adapter creates no extension or CUDA work.
  candidate=load(ROOT/pins['candidate_adapter'],'ss_lut_full_stage_engine')
  support=load(ROOT/'tools/moe_ss_serving_pilot_v1/pilot.py','ss_lut_original_support')
  info=support.INFO.replace('moe_ss_v2','moe_ss_lut_full_v1')
  cases='CASE(8,1) CASE(8,8) CASE(16,1) CASE(16,8) CASE(32,1) CASE(32,8) CASE(64,1) CASE(64,8)'
  assert cases in info;info=info.replace(cases,'CASE(8,1) CASE(8,8)')
  gate=load(source/'cpu_proof.py','ss_lut_parent_lifetime_gate')
  phase('compile_and_gate_two_actual_candidate_functions')
  def checked(*a,**kw):
   production=kw['cuda_sources'];assert kw['functions']==['run']
   assert production.startswith((ROOT/pins['candidate_kernel']).read_text())
   kw=dict(kw);kw['cuda_sources']=production+info;kw['cpp_sources']+='\nint64_t pilot_info(int64_t,int64_t,int64_t);\n';kw['functions']=['run','pilot_info'];name='ss_lut_stage_cu130_'+hashlib.sha256(kw['cuda_sources'].encode()).hexdigest()[:20]
   if a:a=(name,)+a[1:]
   else:kw['name']=name
   start=time.monotonic();module=builder(*a,**kw)
   sass=subprocess.check_output(['cuobjdump','--dump-sass',module.__file__],text=True,timeout=45);res=subprocess.check_output(['cuobjdump','--dump-resource-usage',module.__file__],text=True,timeout=45)
   (args.output/'actual_lut.cu').write_text(kw['cuda_sources']);(args.output/'actual_lut.sass').write_text(sass);(args.output/'actual_lut.resources').write_text(res)
   import shutil
   shutil.copy2(module.__file__,args.output/'actual_lut.so')
   tables=['Function : '+f for f in sass.split('Function : ')[1:]if '7projectILi'in f.splitlines()[0]];assert len(tables)==2
   functions=physical('\n'.join(tables),gate);resources=support.resources(res);assert set(resources)=={(8,1),(8,8)}
   attrs=[]
   for(t,v),row in sorted(resources.items()):
    assert row['STACK']==row['LOCAL']==0 and 0<row['REG']<=255 and((row['REG']+7)//8)*8*256<=65536
    arow={k:int(module.pilot_info(t,v,f))for k,f in [('registers',1),('local',2),('user_static',3),('max_threads',4),('max_dynamic',5),('occupancy',6)]}
    assert arow['registers']==row['REG']and arow['local']==0 and arow['max_threads']>=256 and arow['max_dynamic']>=27680 and arow['occupancy']>=1
    attrs.append(dict(T=t,DIV=v,dynamic_shared=27680,binary_resources=row,**arow))
   r['jit'].append({'compile_seconds':time.monotonic()-start,'production_sha256':hashlib.sha256(production.encode()).hexdigest(),'query_only_augmentation':True,'so_sha256':sha(args.output/'actual_lut.so'),'functions':functions,'resources':attrs,'before_first_candidate_math':True});save();return module
  cpp.load_inline=checked
  try:candidate.module();assert len(r['jit'])==1
  finally:cpp.load_inline=builder
  phase('one_original_prepare_shared_immutable_banks')
  weights=reference.generate_weights(5252);frozen=[(t,t.clone())for t in weights.values()];state=serving.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize()
  for attr in ('_ss_up_bank','_ss_down_bank'):
   bank=getattr(state,attr)
   for t in(bank.packed,bank.scales,bank.globals):frozen.append((t,t.clone()))
  for attr in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_scale_2','w2_weight_scale_2','_post_expert_norm_weight'):
   t=getattr(state,attr);frozen.append((t,t.clone()))
  def immutable():
   for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  # The same exact method code object gets a private copy of its globals for
  # recording. Original class/module globals and factory cache remain untouched.
  captured=[]
  def record(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor):
   before={'x':x.clone(),'bank':bank,'sorted_ids':sorted_ids.clone(),'experts':experts.clone(),'post':post.clone(),'block_m':block_m,'gather_divisor':gather_divisor}
   answer=original.project(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor)
   before['expected']=answer.clone();captured.append(before);return answer
  method=type(state)._marlin_expert_outputs;own_globals=dict(method.__globals__);own_globals['_engine']=types.SimpleNamespace(project=record)
  recorded_method=types.FunctionType(method.__code__,own_globals,method.__name__,method.__defaults__,method.__closure__)
  assert recorded_method.__code__ is method.__code__ and recorded_method.__globals__ is not method.__globals__
  capture_type=type('StageCaptureExperts',(type(state),),{'_marlin_expert_outputs':recorded_method})
  capture_state=capture_type.__new__(capture_type);capture_state.__dict__=state.__dict__.copy()
  def real_inputs(seed,empty=False):
   captured.clear();data=reference.generate_input(8,'ep8',seed)
   if empty:data['topk_ids'].copy_(torch.arange(32,40,device='cuda',dtype=torch.int32).expand(8,8))
   before={k:data[k].clone()for k in('hidden_states','topk_ids','topk_weights')}
   out=serving.kernel_fn(data,capture_state);torch.cuda.synchronize()
   if empty:ok=not int(torch.count_nonzero(out.view(torch.uint8)));msg='all nonlocal exact zero'
   else:ok,msg=reference.check_implementation(data,out,weights)
   r['correctness'].append({'kind':'actual_serving_capture','seed':seed,'empty':empty,'pass':ok,'message':msg});save();assert ok,msg
   assert all(torch.equal(data[k],v)for k,v in before.items());immutable();assert len(captured)==2
   assert[c['gather_divisor']for c in captured]==[8,1]and all(c['block_m']==8 for c in captured)
   return dict(zip(('up','down'),captured.copy()))
  phase('capture_real_stage_inputs_and_graphs')
  initial=real_inputs(7777);work={};engines={'original':original,'full_lut':candidate}
  for stage,rec in initial.items():
   for arm,engine in engines.items():
    args0={k:(v.clone()if isinstance(v,torch.Tensor)else v)for k,v in rec.items()if k!='expected'};out=torch.empty_like(rec['expected']);args0['out']=out
    engine.project(**args0);torch.cuda.synchronize();g=torch.cuda.CUDAGraph();live.append(g)
    with torch.cuda.graph(g):engine.project(**args0)
    torch.cuda.synchronize();work[stage,arm]=(args0,g)
  def install(stage,arm,rec):
   values,g=work[stage,arm]
   for k in('x','sorted_ids','experts','post'):assert values[k].shape==rec[k].shape;values[k].copy_(rec[k])
   return values,g
  def validate(stage,arm,rec,kind):
   values,_=work[stage,arm];out=values['out'];expected=rec['expected'];torch.cuda.synchronize()
   equal=torch.equal(out.view(torch.uint8),expected.view(torch.uint8));report={'stage':stage,'arm':arm,'kind':kind,'bitwise_equal':equal}
   if not equal:report.update(mismatched_scalars=int((out.view(torch.int16)!=expected.view(torch.int16)).sum()),max_abs=float((out.float()-expected.float()).abs().max()))
   r['correctness'].append(report);save();assert equal,report
   for k in('x','sorted_ids','experts','post'):assert torch.equal(values[k].view(torch.uint8),rec[k].view(torch.uint8)),(stage,arm,k)
   immutable()
  for kind,seed,empty in [('fresh_current_input_poison',7778,False),('valid_to_nonlocal_zero',7779,True)]:
   records=real_inputs(seed,empty)
   for stage,rec in records.items():
    for arm in engines:
     values,g=install(stage,arm,rec);values['out'].fill_(float('nan'));g.replay();validate(stage,arm,rec,kind)
  phase('separate_up_down_matched_stage_timings')
  events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record();events[1].record();events[1].synchronize();flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);last={}
  for stage in('up','down'):
   # Warm with the current real captured nonempty input, outside measurements.
   warm=real_inputs(7899)[stage]
   for arm in engines:
    _,g=install(stage,arm,warm)
    for _ in range(3):g.replay()
   torch.cuda.synchronize()
   pairs=[]
   for i in range(8):
    seed=7900+i;rec=real_inputs(seed)[stage];pair={}
    for arm in (('original','full_lut')if i%2==0 else('full_lut','original')):
     values,g=install(stage,arm,rec);flush.zero_();events[0].record();g.replay();events[1].record();events[1].synchronize();us=events[0].elapsed_time(events[1])*1000;assert 0<us<float('inf')
     # Persist direct signed evidence before numeric gates or later profiling.
     r['raw'].append({'stage':stage,'arm':arm,'seed':seed,'round':i,'us':us,'project_zeroing_included':True});save();pair[arm]=us
     validate(stage,arm,rec,'timed_current_input_seed'+str(seed));last[stage,arm]=rec
    pairs.append(pair)
   for arm in engines:validate(stage,arm,last[stage,arm],'own_terminal_before_later_stage')
   r.setdefault('paired',{})[stage]={'raw':pairs,'median_original_minus_full_lut_us':statistics.median(x['original']-x['full_lut']for x in pairs),'median_original_over_full_lut':statistics.median(x['original']/x['full_lut']for x in pairs),'full_lut_wins':sum(x['full_lut']<x['original']for x in pairs)};save()
  phase('direct_terminal_source_and_immutable_checks')
  immutable();bundle.verify_tree(tree,meta)
  for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,rel
  r['validated_direct_timings_completed']=True;r['state']='VALID_SEPARATE_STAGE_TIMINGS_SAVED_PROFILE_OPTIONAL_ROOT_REVIEW';save()
  if args.profile:
   # Timings and own terminals above are complete before any optional profiler.
   for stage in('up','down'):
    for arm in engines:
     phase('optional_profile_'+stage+'_'+arm);rec=last[stage,arm];values,g=install(stage,arm,rec);values['out'].fill_(float('nan'));attempted=False;entry={'optional':True}
     try:
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:attempted=True;g.replay();torch.cuda.synchronize()
      path=args.output/(stage+'_'+arm+'.trace.json');prof.export_chrome_trace(str(path));events0=json.loads(path.read_text())['traceEvents'];kernels=[{'name':e['name'],'duration_us':e.get('dur'),'args':e.get('args',{})}for e in events0 if e.get('cat')=='kernel']
      if not kernels:raise RuntimeError('optional profiler returned no GPU kernel events')
      entry.update(status='COMPLETE',kernels=kernels,trace_sha256=sha(path))
     except Exception:entry.update(status='FAILED_OPTIONAL',error=traceback.format_exc())
     r['profiles'][stage+'_'+arm]=entry;save()
     if attempted:validate(stage,arm,rec,'post_optional_profile_current_input')
  for _,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  immutable();bundle.verify_tree(tree,meta);r['state']='VALID_M8_SEPARATE_DECODE_LUT_STAGE_SCREEN_ROOT_REVIEW_REQUIRED';r['threefold_or_full_serving_gain_claim']=False;phase('complete')
 except BaseException:r['state']='INVALID_LUT_STAGE_SCREEN_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAIN_RESET')
   except BaseException as e:cleanup.append(repr(e));r['state']='INVALID_CLEANUP'
  r['cleanup']=cleanup;r['finished_at']=time.time();r['elapsed_seconds']=r['finished_at']-r['started_at'];save()
if __name__=='__main__':main()

"""Warm actual M8 whole-serving anchor/SS/LUT/LUT+grid comparison; no compilation."""
import argparse,ast,hashlib,importlib,importlib.util,json,statistics,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
LABELS=('anchor','SS','LUT','LUT_grid')
ORDERS=tuple(tuple((i+j)%4 for j in range(4))for i in range(4))
ORDERS=ORDERS+tuple(tuple(reversed(x))for x in ORDERS)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--lut-prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);p.add_argument('--profile',action='store_true');args=p.parse_args()
 if args.output.exists():raise FileExistsError('refuse existing whole-serving output')
 assert sha(args.review)==args.review_sha256
 manifest=json.loads((D/'manifest.json').read_text())
 for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
 for directory,key,filename in [(args.prior,'original_result_sha256','result.json'),(args.prior,'original_SO_sha256','actual_ss.so'),(args.lut_prior,'LUT_result_sha256','result.json'),(args.lut_prior,'LUT_SO_sha256','actual_lut.so')]:assert sha(directory/filename)==manifest[key]
 original_receipt=json.loads((args.prior/'result.json').read_text());lut_receipt=json.loads((args.lut_prior/'result.json').read_text());assert original_receipt['state']=='ONE_TRUE_SERVING_CASE_VALID_SMALL_TIMINGS_ROOT_REVIEW_REQUIRED'and lut_receipt['state']=='VALID_M8_SEPARATE_DECODE_LUT_STAGE_SCREEN_ROOT_REVIEW_REQUIRED'
 args.output.mkdir(parents=True);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','scope':'M8 actual full-serving same-context4arm comparison, sameweights/banks, originaltaskoracle; no14/holdouts/3x/model/official','started_at':time.time(),'manifest_sha256':sha(D/'manifest.json'),'review_sha256':args.review_sha256,'phases':[],'correctness':[],'raw':[],'profiles':{},'reuse':{'new_compilation':False,'new_tree':False,'original_SO_sha256':manifest['original_SO_sha256'],'LUT_SO_sha256':manifest['LUT_SO_sha256']}};live=[]
 def save():(args.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):
  now=time.time();r['phase']=name;r['phases'].append({'phase':name,'timestamp':now,'elapsed_seconds':now-r['started_at']});save();print(name,flush=True)
 try:
  phase('verify_existing_immutable_tree')
  tree=args.prior/'candidate-tree';bundle=load(ROOT/'benchmarks/rsi_bundle.py','ss_whole_bundle');meta=json.loads((tree/'.rsi-bundle.json').read_text());bundle.verify_tree(tree,meta);assert meta['archive_sha256']==manifest['prior_archive_sha256']
  phase('runtime_and_exact_serving_imports')
  paths=load(ROOT/'benchmarks/rsi_worker.py','ss_whole_paths');paths.setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  props=torch.cuda.get_device_properties(0);assert torch.version.cuda=='13.0'and(props.major,props.minor)==(9,0)
  r['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':props.name,'SMs':props.multi_processor_count};save()
  reference=importlib.import_module('reference');serving=importlib.import_module('serving_moe');base=importlib.import_module('serving_moe_base');ss_full=sys.modules['ss_exact_serving_both'];ss=ss_full._engine
  phase('reuse_both_exact_actual_binaries')
  original_ext=load(args.prior/'actual_ss.so',manifest['original_module_name']);lut_ext=load(args.lut_prior/'actual_lut.so',manifest['LUT_module_name'])
  lut_full=load(ROOT/manifest['LUT_serving_source'],'ss_whole_full_lut');lut=lut_full._engine
  import torch.utils.cpp_extension as cpp
  builder=cpp.load_inline
  def prime(engine,extension,expected_source):
   def reused(*a,**kw):assert kw['cuda_sources']==expected_source and kw['functions']==['run'];return extension
   cpp.load_inline=reused
   try:assert engine.module()is extension
   finally:cpp.load_inline=builder
  prime(ss,original_ext,(tree/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2/extension_source.cu').read_text())
  prime(lut,lut_ext,(ROOT/manifest['LUT_extension_source']).read_text())
  # Only host metadata changes for the stacked arm. Use the exact already-proved
  # bound function and actual full-LUT method code with private globals.
  bounds=ast.parse((ROOT/manifest['grid_adapter']).read_text());node=next(n for n in bounds.body if isinstance(n,ast.FunctionDef)and n.name=='mapped_local_row_bound');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact-mapped-local-row-bound>','exec'),ns)
  def grid_project(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor):
   limit=ns['mapped_local_row_bound'](out.shape[0],block_m,sorted_ids.numel());return lut.project(x,bank,sorted_ids[:limit],experts,post,block_m,out,gather_divisor)
  method=lut_full.BothSSExperts._marlin_expert_outputs;private=dict(method.__globals__);private['_engine']=types.SimpleNamespace(project=grid_project);grid_method=types.FunctionType(method.__code__,private,method.__name__,method.__defaults__,method.__closure__)
  grid_class=type('LUTMappedGridExperts',(lut_full.BothSSExperts,),{'_marlin_expert_outputs':grid_method})
  assert grid_method.__code__ is method.__code__ and grid_method.__globals__ is not method.__globals__
  phase('weights_and_readonly_reference_copies')
  weights=reference.generate_weights(5252);frozen=[(t,t.clone())for t in weights.values()]
  phase('one_actual_prepare_shared_immutable_banks')
  ss_state=serving.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize()
  def clone_state(cls):
   layer=cls.__new__(cls);layer.__dict__=ss_state.__dict__.copy();return layer
  states={'anchor':clone_state(base.MuseSpark1xFp4Experts),'SS':ss_state,'LUT':clone_state(lut_full.BothSSExperts),'LUT_grid':clone_state(grid_class)}
  for attr in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_scale_2','w2_weight_scale_2','_post_expert_norm_weight'):
   t=getattr(ss_state,attr);frozen.append((t,t.clone()));assert all(getattr(layer,attr)is t for layer in states.values())
  for attr in ('_ss_up_bank','_ss_down_bank'):
   bank=getattr(ss_state,attr);assert all(getattr(layer,attr)is bank for layer in states.values())
   for t in (bank.packed,bank.scales,bank.globals):frozen.append((t,t.clone()))
  r['state_classes']={label:str(type(layer))for label,layer in states.items()};r['bank_addresses']={'Marlin_w13':ss_state.w13_weight.data_ptr(),'Marlin_w2':ss_state.w2_weight.data_ptr(),'SS_up':ss_state._ss_up_bank.packed.data_ptr(),'SS_down':ss_state._ss_down_bank.packed.data_ptr()};save()
  def immutable():
   for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  def check(label,kind,data,out):
   ok,msg=reference.check_implementation(data,out,weights);r['correctness'].append({'arm':label,'kind':kind,'pass':ok,'message':msg});save();assert ok,(label,kind,msg);immutable()
  def routing(data):
   counts={i:0 for i in range(-1,32)}
   for expert in data['topk_ids'].cpu().flatten().tolist():counts[expert if expert<32 else-1]+=1
   blocks={e:(n+7)//8 for e,n in counts.items()if n};return {'counts':counts,'post':sum(blocks.values())*8,'active_local_blocks':sum(n for e,n in blocks.items()if e>=0),'grid_rows':{'SS':64,'LUT':64,'LUT_grid':36}}
  work={}
  for label in LABELS:
   phase('eager_'+label);data=reference.generate_input(8,'ep8',7777);before={k:data[k].clone()for k in ('hidden_states','topk_ids','topk_weights')};out=serving.kernel_fn(data,states[label]);torch.cuda.synchronize();check(label,'eager_original_task',data,out);assert all(torch.equal(data[k],v)for k,v in before.items())
   phase('capture_'+label);g=torch.cuda.CUDAGraph();live.append(g)
   with torch.cuda.graph(g):out=serving.kernel_fn(data,states[label])
   torch.cuda.synchronize();work[label]=(data,out,g)
   fresh=reference.generate_input(8,'ep8',7778)
   for k in before:data[k].copy_(fresh[k])
   out.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(label,'captured_fresh_X_and_routing',data,out);assert all(torch.equal(data[k],fresh[k])for k in before)
   assert int(torch.count_nonzero(out))>0
   data['topk_ids'].copy_(torch.arange(32,40,device='cuda',dtype=torch.int32).expand(8,8));out.fill_(float('nan'));g.replay();torch.cuda.synchronize();assert not int(torch.count_nonzero(out.view(torch.uint8)));r['correctness'].append({'arm':label,'kind':'valid_to_nonlocal_captured_ids','exact_zero':True});immutable();save()
  assert len({work[l][0]['hidden_states'].data_ptr()for l in LABELS})==len({work[l][1].data_ptr()for l in LABELS})==4
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();end.record();end.synchronize();last={}
  def one(label,seed,kind):
   data,out,g=work[label];fresh=reference.generate_input(8,'ep8',seed)
   for k in ('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
   hist=routing(data);flush.zero_();start.record();g.replay();end.record();end.synchronize();us=start.elapsed_time(end)*1000;assert 0<us<float('inf')
   r['raw'].append({'arm':label,'seed':seed,'kind':kind,'us':us,'routing':hist});last[label]=(seed,fresh);save();return us
  def terminal(label,kind):
   seed,fresh=last[label];data,out,g=work[label];assert all(torch.equal(data[k],fresh[k])for k in ('hidden_states','topk_ids','topk_weights'));check(label,kind+'_seed'+str(seed),data,out)
  phase('direct_F1_brackets')
  f1={};order=('anchor','SS','LUT','LUT_grid','LUT','SS','anchor')
  for block,label in enumerate(order):
   f1[label+str(block)]=[one(label,7800+block*10+i,'F1')for i in range(5)];terminal(label,'own_F1_block'+str(block))
  r['F1_raw_us']=f1;r['F1_medians_us']={k:statistics.median(v)for k,v in f1.items()};save()
  phase('direct_balanced_four_arm_pairs')
  pairs=[]
  for i,order in enumerate(ORDERS):pairs.append({LABELS[j]:one(LABELS[j],7900+i,'paired')for j in order})
  for label in LABELS:terminal(label,'own_paired_terminal')
  r['paired']=pairs;r['pair_orders']=[[LABELS[j]for j in order]for order in ORDERS]
  r['comparisons']={a+'_minus_'+b:{'median_us':statistics.median(row[a]-row[b]for row in pairs),'median_ratio':statistics.median(row[a]/row[b]for row in pairs),'b_wins':sum(row[b]<row[a]for row in pairs)}for a,b in [('anchor','SS'),('anchor','LUT'),('anchor','LUT_grid'),('SS','LUT'),('LUT','LUT_grid')]}
  r['last_outputs_bitwise']={'SS_vs_LUT':torch.equal(work['SS'][1].view(torch.uint8),work['LUT'][1].view(torch.uint8)),'LUT_vs_LUT_grid':torch.equal(work['LUT'][1].view(torch.uint8),work['LUT_grid'][1].view(torch.uint8))};save()
  phase('validated_direct_terminal_source_immutable_completion')
  immutable();bundle.verify_tree(tree,meta)
  for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
  r['validated_direct_timings_completed']=True;r['state']='VALID_WHOLE_SERVING_DIRECT_TIMINGS_COMPLETED_PROFILE_OPTIONAL_ROOT_REVIEW';save()
  if args.profile:
   for label,(data,out,g)in work.items():
    phase('optional_profile_'+label);fresh=reference.generate_input(8,'ep8',7907)
    for k in ('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
    out.fill_(float('nan'));entry={'optional':True};attempted=False
    try:
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:attempted=True;g.replay();torch.cuda.synchronize()
     path=args.output/(label+'.trace.json');prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents'];kernels=[{'name':e['name'],'duration_us':e.get('dur'),'args':e.get('args',{})}for e in events if e.get('cat')=='kernel']
     if not kernels:raise RuntimeError('optional trace has no GPU kernel events')
     entry.update(status='COMPLETE',kernels=kernels,sha256=sha(path))
    except Exception:entry.update(status='FAILED_OPTIONAL',error=traceback.format_exc())
    r['profiles'][label]=entry;save()
    if attempted:
     assert all(torch.equal(data[k],fresh[k])for k in ('hidden_states','topk_ids','topk_weights'));check(label,'post_profile_current_output',data,out)
  phase('final_clean_reset_and_source_checks')
  for _,_,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  immutable();bundle.verify_tree(tree,meta);r['state']='VALID_M8_WHOLE_SERVING_FOUR_ARM_COMPARISON_ROOT_REVIEW_REQUIRED';r['full14_or_threefold_or_official_claim']=False;save()
 except BaseException:r['state']='INVALID_WHOLE_SERVING_COMPARISON_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAIN_RESET')
   except BaseException as e:cleanup.append(repr(e));r['state']='INVALID_CLEANUP'
  r['cleanup']=cleanup;r['finished_at']=time.time();r['elapsed_seconds']=r['finished_at']-r['started_at'];save()
if __name__=='__main__':main()

"""One original M3012EP8 whole-serving anchor versus guarded T64 LUT; no broad suite."""
import argparse,ast,hashlib,importlib,importlib.util,json,math,re,statistics,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
M=3012;LABELS=('anchor','LUT_T64')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def physical(sass):
 proof=[]
 for table in sass.split('Function : ')[1:]:
  if '7projectILi'not in table.splitlines()[0]:continue
  name=table.splitlines()[0];t,div=map(int,re.search(r'ILi(\d+)ELi(\d+)E',name).groups());assert t==64 and div in(1,8)
  pending=set();issues=groups=handoff=0
  for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table):
   pc=int(m[1],16);ins=m[2].strip();plain=re.sub(r'^@!?P\d+\s+','',ins)
   if 'HGMMA.'in ins:
    base=int(re.search(r'\bR(\d+),',ins)[1]);target=set(range(base,base+32));assert not pending or pending==target;pending=target;issues+=1;continue
   if 'WARPGROUP.DEPBAR.LE'in ins and '0x0'in ins:
    assert pending and issues==4;groups+=1;pending=set();issues=0;continue
   if pending:
    dest=re.match(r'[A-Z0-9_.]+\s+R(\d+)(?:,|\s)',plain)
    if dest:assert int(dest[1])not in pending,('generic D overwrite',name,pc,ins)
   if 'BAR.SYNC'in ins and '0x3, 0x80'in ins:handoff+=1
  assert not pending and groups>0 and handoff==groups
  assert '0x2, 0x80'in table and 'FENCE.VIEW.ASYNC.S'in table and 'SYNCS.ARRIVE.'in table
  proof.append(dict(T=t,DIV=div,static_K64_groups=groups,four_HGMMA_per_group=True,accumulator_lifetime_pass=True,math_reader_barriers=handoff))
 assert len(proof)==2 and{x['DIV']for x in proof}=={1,8}
 return proof

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);a=p.parse_args()
 assert sha(a.review)==a.review_sha256
 pins=json.loads((D/'manifest.json').read_text())
 for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,rel
 if a.output.exists():raise FileExistsError(a.output)
 for name,key in [('result.json','original_result_sha256'),('actual_ss.so','original_SO_sha256')]:assert sha(a.prior/name)==pins[key]
 a.output.mkdir(parents=True);r={'state':'RUNNING_NO_ADMISSION','started_at':time.time(),'scope':'Only original M3012normalEP8; actualwhole-serving exact05ec9a8anchor versus T64fullLUT, weights5252; noM8/full14/holdout/3x/model/officialclaim','correctness':[],'raw':[],'phases':[],'jit':[],'routing':{},'review_sha256':a.review_sha256,'manifest_sha256':sha(D/'manifest.json')};live=[]
 def save():(a.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):r['phase']=name;r['phases'].append(dict(phase=name,elapsed_seconds=time.time()-r['started_at']));save();print(name,flush=True)
 try:
  phase('verify_existing_tree_and_frozen_anchor')
  tree=a.prior/'candidate-tree';bundle=load(ROOT/'benchmarks/rsi_bundle.py','t64_bundle');meta=json.loads((tree/'.rsi-bundle.json').read_text());bundle.verify_tree(tree,meta);assert meta['archive_sha256']==pins['prior_archive_sha256']
  assert sha(tree/'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py')==pins['anchor_source_sha256']
  load(ROOT/'benchmarks/rsi_worker.py','t64_paths').setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  props=torch.cuda.get_device_properties(0);assert torch.version.cuda=='13.0'and(props.major,props.minor)==(9,0)and props.total_memory>=70_000_000_000
  free,total=torch.cuda.mem_get_info();assert free>=32*(1<<30),('memory preflight',free)
  r['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':props.name,'SMs':props.multi_processor_count,'total_bytes':total,'initial_free_bytes':free,'device_shared_memory_per_SM':getattr(props,'shared_memory_per_multiprocessor',None),'active_shared_config':'notobserved; CUDAfunctionpreferredcarveoutandoccupancyqueried, noNCUclaim'};save()
  reference=importlib.import_module('reference');serving=importlib.import_module('serving_moe');base=importlib.import_module('serving_moe_base');original=sys.modules['ss_exact_serving_both']._engine
  import torch.utils.cpp_extension as cpp
  builder=cpp.load_inline;ext=load(a.prior/'actual_ss.so',pins['original_module_name'])
  def reused(*args,**kw):assert kw['cuda_sources']==(tree/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2/extension_source.cu').read_text()and kw['functions']==['run'];return ext
  cpp.load_inline=reused
  try:assert original.module()is ext
  finally:cpp.load_inline=builder
  full=load(ROOT/pins['candidate_serving'],'t64_full_serving');engine=full._engine
  support=load(ROOT/'tools/moe_ss_serving_pilot_v1/pilot.py','t64_gate_support')
  info=support.INFO.replace('moe_ss_v2','moe_ss_lut_t64_m3012_v1');old='CASE(8,1) CASE(8,8) CASE(16,1) CASE(16,8) CASE(32,1) CASE(32,8) CASE(64,1) CASE(64,8)';assert old in info;info=info.replace(old,'CASE(64,1) CASE(64,8)');assert info.count('if(f==6){')==1;info=info.replace('if(f==6){','if(f==7)return a.preferredShmemCarveout;if(f==6){')
  phase('one_actual_T64_compile_and_resource_lifetime_gate')
  def checked(*args,**kw):
   production=kw['cuda_sources'];assert production==(ROOT/pins['candidate_extension_source']).read_text()and kw['functions']==['run']
   kw=dict(kw);kw['cuda_sources']=production+info;kw['cpp_sources']+='\nint64_t pilot_info(int64_t,int64_t,int64_t);\n';kw['functions']=['run','pilot_info'];name='ss_lut_t64_m3012_cu130_'+hashlib.sha256(kw['cuda_sources'].encode()).hexdigest()[:20]
   if args:args=(name,)+args[1:]
   else:kw['name']=name
   start=time.monotonic();module=builder(*args,**kw)
   sass=subprocess.check_output(['cuobjdump','--dump-sass',module.__file__],text=True,timeout=30);res=subprocess.check_output(['cuobjdump','--dump-resource-usage',module.__file__],text=True,timeout=30)
   (a.output/'actual_T64.cu').write_text(kw['cuda_sources']);(a.output/'actual_T64.sass').write_text(sass);(a.output/'actual_T64.resources').write_text(res)
   import shutil
   shutil.copy2(module.__file__,a.output/'actual_T64.so');functions=physical(sass);rr=support.resources(res);assert set(rr)=={(64,1),(64,8)};attrs=[]
   for(t,v),row in sorted(rr.items()):
    assert row['STACK']==row['LOCAL']==0 and 0<row['REG']<=128 and((row['REG']+7)//8)*8*256<=65536
    q={k:int(module.pilot_info(t,v,f))for k,f in [('registers',1),('local',2),('user_static',3),('max_threads',4),('max_dynamic',5),('occupancy',6),('preferred_shared_carveout',7)]}
    assert q['registers']==row['REG']and q['local']==0 and q['max_threads']>=256 and q['max_dynamic']>=42016 and q['occupancy']>=1
    attrs.append(dict(T=t,DIV=v,dynamic_shared=42016,binary_resources=row,**q))
   r['jit'].append({'compile_seconds':time.monotonic()-start,'production_sha256':hashlib.sha256(production.encode()).hexdigest(),'SO_sha256':sha(a.output/'actual_T64.so'),'resources':attrs,'functions':functions,'gate_before_first_candidate_math':True});save();return module
  cpp.load_inline=checked
  try:engine.module();assert len(r['jit'])==1
  finally:cpp.load_inline=builder
  phase('weights_one_original_prepare_shared_banks')
  weights=reference.generate_weights(5252);frozen=[(t,t.clone())for t in weights.values()];ss_state=serving.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize()
  def clone(cls):layer=cls.__new__(cls);layer.__dict__=ss_state.__dict__.copy();return layer
  states={'anchor':clone(base.MuseSpark1xFp4Experts),'LUT_T64':clone(full.BothSSExperts)}
  for attr in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_scale_2','w2_weight_scale_2','_post_expert_norm_weight'):
   t=getattr(ss_state,attr);assert all(getattr(s,attr)is t for s in states.values());frozen.append((t,t.clone()))
  for attr in ('_ss_up_bank','_ss_down_bank'):
   bank=getattr(ss_state,attr);assert all(getattr(s,attr)is bank for s in states.values())
   for t in(bank.packed,bank.scales,bank.globals):frozen.append((t,t.clone()))
  def immutable():
   for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  def fingerprint(data):return{k:hashlib.sha256(data[k].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()for k in('hidden_states','topk_ids','topk_weights')}
  def routing(data,seed):
   ids=data['topk_ids'].flatten();count=torch.bincount(ids.to(torch.int64),minlength=256).cpu().tolist();local=count[:32];nonlocal_count=sum(count[32:]);localblocks=sum((n+63)//64 for n in local);post=(localblocks+(nonlocal_count+63)//64)*64
   h={'seed':seed,'global_counts':count,'local_counts':local,'local_routes':sum(local),'mapped_local_average':sum(local)/32,'active_local_blocks':localblocks,'nonlocal_routes':nonlocal_count,'expected_post':post,'source_alignment_capacity':40288,'block_m':64,'T':64,'heuristic_unmapped_average':M*8/32,'local_token_column_utilization':sum(local)/(localblocks*64)if localblocks else 0};r['routing'][str(seed)]=h;return h
  def check(label,kind,data,out,empty=False):
   if empty:ok=not int(torch.count_nonzero(out.view(torch.uint8)));msg='allnonlocal exactzero; originalchecker intentionally inapplicable'
   else:ok,msg=reference.check_implementation(data,out,weights)
   r['correctness'].append(dict(arm=label,kind=kind,pass_=bool(ok),message=msg));save();assert ok,(label,kind,msg);immutable()
  phase('whole_serving_eager_capture_current_fresh_empty')
  initial=reference.generate_input(M,'ep8',7777);work={}
  # One exact canonical original-SS T64 output is a codec/layout correctness control, untimed.
  old_output=serving.kernel_fn(initial,ss_state).clone();torch.cuda.synchronize()
  for label in LABELS:
   data={k:(v.clone()if isinstance(v,torch.Tensor)else dict(v))for k,v in initial.items()};before=fingerprint(data);out=serving.kernel_fn(data,states[label]);torch.cuda.synchronize();check(label,'eager_original_task',data,out);assert fingerprint(data)==before
   if label=='LUT_T64':equal=torch.equal(out.view(torch.uint8),old_output.view(torch.uint8));r['correctness'].append(dict(arm=label,kind='canonical_original_SS_T64_codec_control',bitwise_equal=equal));save();assert equal
   for _ in range(2):serving.kernel_fn(data,states[label])
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph();live.append(g)
   with torch.cuda.graph(g):out=serving.kernel_fn(data,states[label])
   torch.cuda.synchronize();work[label]=(data,out,g)
   for kind,seed in [('captured_current_X_routing',7777),('captured_fresh_X_routing',7778)]:
    fresh=reference.generate_input(M,'ep8',seed)
    for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
    out.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(label,kind,data,out);assert fingerprint(data)==fingerprint(fresh)
   data['topk_ids'].fill_(32);out.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(label,'valid_to_empty_captured_ids',data,out,True)
  del old_output,initial
  free,total=torch.cuda.mem_get_info();r['memory_after_capture']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'free':free,'peak_allocated':torch.cuda.max_memory_allocated()};save();assert free>=8*(1<<30)and torch.cuda.memory_allocated()<48*(1<<30)
  assert work['anchor'][0]['hidden_states'].data_ptr()!=work['LUT_T64'][0]['hidden_states'].data_ptr()and work['anchor'][1].data_ptr()!=work['LUT_T64'][1].data_ptr()
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();end.record();end.synchronize();last={}
  def one(label,fixture,seed,kind,block=None):
   data,out,g=work[label]
   for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fixture[k])
   h=routing(data,seed);before=fingerprint(data);flush.zero_();start.record();g.replay();end.record();end.synchronize();us=start.elapsed_time(end)*1000;assert 0<us<float('inf')
   r['raw'].append(dict(arm=label,seed=seed,kind=kind,block=block,us=us,routing=h));last[label]=(fixture,seed);save();assert fingerprint(data)==before;check(label,'timed_current_output_'+kind,data,out);return us
  def terminal(label,kind):
   fixture,seed=last[label];data,out,g=work[label];assert fingerprint(data)==fingerprint(fixture);check(label,kind+'_seed'+str(seed),data,out)
  phase('fixed_same_fixture_F1_and_8_ABBA_pairs')
  fixed=reference.generate_input(M,'ep8',7904);r['F1_fixed_fixture']={'seed':7904,'tensor_sha256':fingerprint(fixed),'routing':routing(fixed,7904)};save();f1={}
  for block,label in enumerate(('anchor','LUT_T64','anchor')):
   f1[label+str(block)]=[one(label,fixed,7904,'F1_fixed_fixture',block)for _ in range(3)];terminal(label,'own_F1_block'+str(block))
  r['F1_raw_us']=f1;r['F1_medians_us']={k:statistics.median(v)for k,v in f1.items()};save();pairs=[]
  for i in range(8):
   fixture=reference.generate_input(M,'ep8',7900+i);order=LABELS if i%2==0 else tuple(reversed(LABELS));pair={label:one(label,fixture,7900+i,'paired')for label in order};pairs.append(pair)
  for label in LABELS:terminal(label,'own_paired_terminal')
  r['paired']={'raw':pairs,'median_anchor_minus_LUT_us':statistics.median(p['anchor']-p['LUT_T64']for p in pairs),'median_anchor_over_LUT':statistics.median(p['anchor']/p['LUT_T64']for p in pairs),'LUT_wins':sum(p['LUT_T64']<p['anchor']for p in pairs)};save()
  phase('terminal_sources_immutability_and_graph_cleanup')
  immutable();bundle.verify_tree(tree,meta)
  for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,rel
  for _,_,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  r['validated_direct_timings_completed']=True;r['state']='VALID_M3012_T64_WHOLE_SERVING_TWO_ARM_ROOT_REVIEW_REQUIRED';r['full14_or_threefold_or_official_claim']=False;save()
 except BaseException:r['state']='INVALID_M3012_T64_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAIN_RESET')
   except BaseException as e:cleanup.append(repr(e));r['state']='INVALID_CLEANUP'
  r['cleanup']=cleanup;r['elapsed_seconds']=time.time()-r['started_at'];save()
if __name__=='__main__':main()

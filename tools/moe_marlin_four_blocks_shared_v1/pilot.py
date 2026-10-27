"""Warm same-binary M8 original/default/4blocks; shared normal/skewed fixture bank across blocked/interleaved schedules."""
import argparse,hashlib,importlib,importlib.util,json,shutil,statistics,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
LABELS=('anchor','default_shim','contrast');ORDERS=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(0,2,1),(1,0,2))
SEEDS=tuple(range(7800,7805))+tuple(range(7900,7912));DISTS=('normal','skewed')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def decode(code):
 return {'blocks_per_sm':code//1000000000,'threads':code//1000000%1000,'thread_n':code//1000%1000,'thread_k':code%1000}
def opposite(default):
 d=decode(default);key=(d['thread_k'],d['thread_n'],d['threads']);assert key in((128,128,256),(64,128,128)),d
 return 2 if key==(128,128,256)else 1

def run(args):
 assert not args.output.exists(),'refuse existing output';assert sha(args.review)==args.review_sha256
 manifest=json.loads((D/'manifest.json').read_text())
 for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
 args.output.mkdir(parents=True);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','started_at':time.time(),'scope':'M8 EP8 original/default/4blocks; stored17seeds x normal/skewed shared by blocked and six-order interleaved schedules; no14/model/3x/official','manifest_sha256':sha(D/'manifest.json'),'review_sha256':args.review_sha256,'correctness':[],'raw':[],'profiles':{},'jit':[]};live=[]
 def save():(args.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):r['phase']=name;save();print(name,flush=True)
 try:
  phase('verify_existing_immutable_pilot_tree')
  tree=args.prior/'candidate-tree';bundle=load(ROOT/'benchmarks/rsi_bundle.py','anchor_tiles_bundle');meta=json.loads((tree/'.rsi-bundle.json').read_text());bundle.verify_tree(tree,meta);assert meta['archive_sha256']==manifest['prior_archive_sha256']
  paths=load(ROOT/'benchmarks/rsi_worker.py','anchor_tiles_paths');paths.setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  props=torch.cuda.get_device_properties(0);assert torch.version.cuda=='13.0'and(props.major,props.minor)==(9,0)
  r['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':props.name,'SMs':props.multi_processor_count};save()
  reference=importlib.import_module('reference');base=importlib.import_module('serving_moe_base')
  assert sha(tree/'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py')==manifest['anchor_source_sha256']
  # Original-only preparation: no custom SS import, bank creation or SS compilation.
  phase('original_anchor_weights_and_prepare')
  weights=reference.generate_weights(5252);frozen=[(t,t.clone())for t in weights.values()];start=time.monotonic();anchor=base.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize();r['prepare_seconds']=time.monotonic()-start
  states={'anchor':anchor}
  for label in LABELS[1:]:
   layer=type(anchor).__new__(type(anchor));layer.__dict__=anchor.__dict__.copy();states[label]=layer
  for attr in('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_scale_2','w2_weight_scale_2','_post_expert_norm_weight','_expert_map'):
   t=getattr(anchor,attr);frozen.append((t,t.clone()));assert all(getattr(x,attr)is t for x in states.values())
  parent=ROOT/manifest['parent_header'];inc=parent.parent
  for p in(parent,inc/'kernel.h',inc/'marlin_template.h'):
   rel=p.relative_to(ROOT);assert sha(tree/rel)==sha(p),('native parent-tree drift',str(rel))
  phase('verify_and_load_exact_collected_fourblock_SO_no_compilation')
  prior=json.loads((args.native_prior/'result.json').read_text())
  assert sha(args.native_prior/'result.json')==manifest['native_prior_result_sha256']
  assert prior['state']=='VALID_M8_ORIGINAL_MARLIN_FOUR_BLOCK_SCREEN_ROOT_REVIEW_REQUIRED'
  assert sha(args.native_prior/'two_tiles.so')==manifest['native_SO_sha256']==prior['jit'][0]['so_sha256']
  assert prior['manifest_sha256']==manifest['native_prior_manifest_sha256']
  api=importlib.import_module('sglang.kernels.ops.moe.moe_wna16_marlin')
  jitloader=importlib.import_module('sglang.kernels.jit.utils.compile.loader');start=time.monotonic()
  variant=jitloader._load(args.native_prior/'two_tiles.so')
  shutil.copy2(args.native_prior/'two_tiles.so',args.output/'two_tiles.so')
  r['native_load_wall_seconds']=time.monotonic()-start;r['compile_wall_seconds']=0
  r['jit']=[{'so_sha256':sha(args.output/'two_tiles.so'),'reused_exact_prior_SO':True,'before_candidate_math':True}]
  r['native_reuse']={'prior_result_sha256':manifest['native_prior_result_sha256'],'native_SO_sha256':manifest['native_SO_sha256'],'no_native_compile':True};save()
  phase('actual_cuda_four_block_feasibility_before_force_or_timing')
  fields=('legal','registers_per_thread','static_shared_B','local_B','max_threads_per_block','device_registers_per_SM',
          'device_shared_B_per_SM','device_optin_shared_B_per_block','device_threads_per_SM','device_blocks_per_SM',
          'dynamic_shared_3_B','dynamic_shared_4_B','kernel_min_cache_B','actual_occupancy_3','actual_occupancy_4',
          'four_blocks_register_count','four_blocks_total_shared_B','launch_reserve_B')
  feasibility={key:int(variant.anchor_block_info(i))for i,key in enumerate(fields)}
  r['four_block_feasibility']={'queries':feasibility,'before_any_forced_launch':True,'before_GPU_timing':True,
       'query_has_no_CUDA_kernel_launch':True,'mechanism':'3to4blocks+smaller_dynamic_shared_and_Acache; not pure occupancy'};save()
  if feasibility['legal']!=1 or any(v<0 for v in feasibility.values()):
   r['state']='VALID_FOUR_BLOCK_UNSUPPORTED_NO_TIMING_ROOT_REVIEW_REQUIRED';r['reason']='actualCUDAdevice/kernel/occupancy query does not admit4';r['GPU_timing_calls']=0;save();return
  assert feasibility['four_blocks_register_count']<=feasibility['device_registers_per_SM']
  assert feasibility['four_blocks_total_shared_B']<=feasibility['device_shared_B_per_SM']
  assert feasibility['actual_occupancy_4']>=4
  original_gemm=api.moe_wna16_marlin_gemm;original_factory=api._jit_moe_wna16_marlin_module
  active=[None];host_up_calls={label:0 for label in LABELS}
  def routed_gemm(*a,**kw):
   label=active[0];target=(kw.get('size_m'),kw.get('size_n'),kw.get('size_k'),kw.get('top_k'))==(8,8192,4096,8)
   if target:host_up_calls[label]+=1
   if label in LABELS[1:]and target:
    assert kw['is_ep']and not kw['use_atomic_add']and kw['use_fp32_reduce']and kw['is_k_full']and not kw['is_zp_float']and kw['moe_block_size']==8 and not kw['mul_topk_weights']
    assert a[0].dtype==torch.bfloat16 and a[3]is None and a[4].size(1)==256 and a[5]is not None
    def factory(dtype,ep,bias):assert dtype==torch.bfloat16 and ep and not bias;return variant
    api._jit_moe_wna16_marlin_module=factory
    try:return original_gemm(*a,**kw)
    finally:api._jit_moe_wna16_marlin_module=original_factory
   return original_gemm(*a,**kw)
  api.moe_wna16_marlin_gemm=routed_gemm
  def call(label,data):
   active[0]=label
   try:return base.kernel_fn(data,states[label])
   finally:active[0]=None
  def immutable():
   for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  def check(label,kind,data,out):
   ok,msg=reference.check_implementation(data,out,weights);r['correctness'].append({'arm':label,'kind':kind,'pass':ok,'message':msg});save();assert ok,(label,kind,msg);immutable()
  # A real default-path execution records the chooser's actual tile before any force.
  phase('default_receipt_before_contrast')
  probe=reference.generate_input(8,'ep8',7776);before={k:probe[k].clone()for k in('hidden_states','topk_ids','topk_weights')};out0=call('anchor',probe);torch.cuda.synchronize();assert variant.anchor_tile_control(0)==0;out1=call('default_shim',probe);torch.cuda.synchronize();assert int(variant.anchor_tile_control(-3))==1
  default=int(variant.anchor_tile_control(-1));assert default==int(variant.anchor_tile_control(-2))
  if decode(default)!=dict(blocks_per_sm=3,threads=128,thread_n=128,thread_k=64):
   r['state']='VALID_FOUR_BLOCK_UNSUPPORTED_NO_TIMING_ROOT_REVIEW_REQUIRED';r['reason']='actual default differs from intended current3blockK64tile';r['actual_default']=decode(default);r['GPU_timing_calls']=0;save();return
  mode=3
  r['default_receipt']={'encoded':default,'decoded':decode(default),'contrast_mode':mode,'before_any_forced_launch':True,'source_identical_default_chooser_with_actual_variant_attrs':True,'original_anchor_config_profile_pending':True,'default_output_bitwise_equal_original':torch.equal(out0.view(torch.uint8),out1.view(torch.uint8))};assert r['default_receipt']['default_output_bitwise_equal_original'];check('anchor','default_probe',probe,out0);check('default_shim','default_probe',probe,out1);assert all(torch.equal(probe[k],v)for k,v in before.items());save()
  work={};fixture=reference.generate_input(8,'ep8',7777)
  for label in LABELS:
   phase('capture_'+label)
   if label!='anchor':assert variant.anchor_tile_control(mode if label=='contrast'else 0)==(mode if label=='contrast'else 0)
   data={k:(v.clone()if isinstance(v,torch.Tensor)else dict(v))for k,v in fixture.items()};before={k:data[k].clone()for k in('hidden_states','topk_ids','topk_weights')}
   out=call(label,data);torch.cuda.synchronize();check(label,'eager_original_task',data,out);assert all(torch.equal(data[k],v)for k,v in before.items())
   for _ in range(2):call(label,data)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph();live.append(g)
   with torch.cuda.graph(g):out=call(label,data)
   torch.cuda.synchronize();work[label]=(data,out,g)
   if label!='anchor':
    receipt={'default':decode(int(variant.anchor_tile_control(-1))),'selected':decode(int(variant.anchor_tile_control(-2))),'host_target_calls':int(variant.anchor_tile_control(-3))};r.setdefault('capture_receipts',{})[label]=receipt;assert receipt['host_target_calls']>0
    assert (receipt['selected']==receipt['default'])==(label=='default_shim')
   fresh=reference.generate_input(8,'ep8',7778)
   for k in before:data[k].copy_(fresh[k])
   out.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(label,'captured_fresh_X_and_routing',data,out);assert all(torch.equal(data[k],fresh[k])for k in before);assert int(torch.count_nonzero(out))>0
   data['topk_ids'].copy_(torch.arange(32,40,device='cuda',dtype=torch.int32).expand(8,8));out.fill_(float('nan'));g.replay();torch.cuda.synchronize();assert not int(torch.count_nonzero(out.view(torch.uint8)));r['correctness'].append({'arm':label,'kind':'valid_to_nonlocal_captured_ids','exact_zero':True});immutable();save()
  assert len({v[0]['hidden_states'].data_ptr()for v in work.values()})==len({v[1].data_ptr()for v in work.values()})==3
  phase('store_shared_normal_skewed_fixture_bank')
  bank={};fixture_meta={};fixture_originals={}
  def tensor_hash(t):return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  def routing(data):
   counts={i:0 for i in range(-1,32)}
   for expert in data['topk_ids'].cpu().flatten().tolist():counts[expert if expert<32 else-1]+=1
   return counts
  for dist in DISTS:
   for seed in SEEDS:
    key=(dist,seed);fresh=reference.generate_input(8,'ep8',seed,dist=dist);bank[key]=fresh
    fixture_originals[key]={k:fresh[k].clone()for k in('hidden_states','topk_ids','topk_weights')}
    fixture_meta[key]={'dist':dist,'seed':seed,'routing':routing(fresh),'tensor_sha256':{k:tensor_hash(fresh[k])for k in fixture_originals[key]}}
  r['shared_fixtures']=[fixture_meta[(dist,seed)]for dist in DISTS for seed in SEEDS]
  r['schedule_contract']={'same_exact_stored_fixture_objects_for_both_schedules':True,'seeds':list(SEEDS),'distributions':list(DISTS),'interleaved_orders':list(ORDERS),'blocked_order':['anchor','default_shim','contrast','default_shim','anchor'],'timed_terminal_scope':'Only exact own terminal outputs at each block/sweep; not every timing sample'};save()
  # Check the new outlier/routing distribution on each actual retained graph before timing.
  for label,(data,out,g)in work.items():
   fresh=bank[('skewed',SEEDS[0])]
   for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
   out.fill_(float('nan'));g.replay();torch.cuda.synchronize();assert all(torch.equal(data[k],fresh[k])for k in('hidden_states','topk_ids','topk_weights'));check(label,'retained_skewed_before_timing',data,out)
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();end.record();end.synchronize();last={}
  journal=(args.output/'raw.jsonl').open('a')
  def one(label,dist,seed,kind,order_index,position):
   data,out,g=work[label];fresh=bank[(dist,seed)]
   for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
   flush.zero_();start.record();g.replay();end.record();end.synchronize();us=start.elapsed_time(end)*1000;assert 0<us<float('inf')
   row={'arm':label,'dist':dist,'seed':seed,'kind':kind,'order_index':order_index,'position':position,'us':us,'routing':fixture_meta[(dist,seed)]['routing']}
   r['raw'].append(row);journal.write(json.dumps(row)+'\n');journal.flush();last[label]=(dist,seed,fresh);return us
  def terminal(label,kind):
   dist,seed,fresh=last[label];data,out,g=work[label]
   assert all(torch.equal(data[k],fresh[k])for k in('hidden_states','topk_ids','topk_weights'))
   assert all(torch.equal(fresh[k],v)for k,v in fixture_originals[(dist,seed)].items())
   check(label,kind+'_'+dist+'_seed'+str(seed),data,out)
   r['correctness'][-1].update(dist=dist,seed=seed,terminal_scope='only own exact retained terminal output; no claim all timing samples checked')
  r['by_distribution']={}
  for dist in DISTS:
   phase('blocked_shared_bank_'+dist);blocked={}
   for block,label in enumerate(('anchor','default_shim','contrast','default_shim','anchor')):
    blocked[label+str(block)]=[one(label,dist,seed,'blocked',block,-1)for seed in SEEDS];terminal(label,'own_blocked_block'+str(block));save()
   bmed={k:statistics.median(v)for k,v in blocked.items()}
   r['by_distribution'][dist]={'blocked_raw_us':blocked,'blocked_medians_us':bmed,'blocked_control_minus_4block_us':{'anchor_before':bmed['anchor0']-bmed['contrast2'],'anchor_after':bmed['anchor4']-bmed['contrast2'],'default_before':bmed['default_shim1']-bmed['contrast2'],'default_after':bmed['default_shim3']-bmed['contrast2']}}
   phase('six_order_interleaved_shared_bank_'+dist);triplets=[]
   for order_index,order in enumerate(ORDERS):
    for seed in SEEDS:
     times={LABELS[index]:one(LABELS[index],dist,seed,'interleaved',order_index,position)for position,index in enumerate(order)}
     triplets.append({'seed':seed,'order_index':order_index,'times':times})
    for label in LABELS:terminal(label,'own_interleaved_sweep'+str(order_index))
    save()
   contrasts={}
   for a,b in(('anchor','default_shim'),('anchor','contrast'),('default_shim','contrast')):
    deltas=[x['times'][a]-x['times'][b]for x in triplets]
    per_seed={str(seed):statistics.median(x['times'][a]-x['times'][b]for x in triplets if x['seed']==seed)for seed in SEEDS}
    contrasts[a+'_minus_'+b]={'all102_triplets_median_us':statistics.median(deltas),'median_ratio':statistics.median(x['times'][a]/x['times'][b]for x in triplets),'win_tie_loss':[sum(x>0 for x in deltas),sum(x==0 for x in deltas),sum(x<0 for x in deltas)],'per_seed_balanced_median_us':per_seed,'median_of17_seed_medians_us':statistics.median(per_seed.values())}
   r['by_distribution'][dist].update(interleaved_triplets=triplets,interleaved_contrasts=contrasts);save()
  journal.close();r['host_up_calls']=host_up_calls
  for key,fresh in bank.items():assert all(torch.equal(fresh[k],v)for k,v in fixture_originals[key].items())
  immutable();bundle.verify_tree(tree,meta)
  for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
  assert len(r['raw'])==782
  r['validated_direct_timings_completed']=True;r['profiling']='NOT_REQUESTED_REUSED_EXACT_PROFILED_SOURCE_AND_SO';r['state']='VALID_SHARED_FIXTURE_DIRECT_TIMINGS_COMPLETE_ROOT_REVIEW';save()
  for _,_,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  immutable();bundle.verify_tree(tree,meta);r['state']='VALID_M8_SHARED_NORMAL_SKEWED_FOUR_BLOCK_SCREEN_ROOT_REVIEW_REQUIRED';r['full14_or_threefold_or_official_claim']=False;phase('complete')
 except BaseException:r['state']='INVALID_SHARED_FIXTURE_SCREEN_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  if 'original_gemm'in locals():api.moe_wna16_marlin_gemm=original_gemm;api._jit_moe_wna16_marlin_module=original_factory
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset()
   except Exception:cleanup.append(traceback.format_exc())
  r['cleanup']=cleanup;r['finished_at']=time.time();r['elapsed_seconds']=r['finished_at']-r['started_at'];save()
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--native-prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);run(p.parse_args())
if __name__=='__main__':main()

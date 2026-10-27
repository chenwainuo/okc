"""Exact original M8 Marlin: anchor/default-shim/opposite existing up tile; root-only GPU."""
import argparse,hashlib,importlib,importlib.util,json,shutil,statistics,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
LABELS=('anchor','default_shim','contrast');ORDERS=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(0,2,1),(1,0,2))*2
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
 args.output.mkdir(parents=True);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','started_at':time.time(),'scope':'one M8 normal EP8 original whole expert, original math; three arms/two existing gate-up tiles; no14/model/3x/official','manifest_sha256':sha(D/'manifest.json'),'review_sha256':args.review_sha256,'correctness':[],'raw':[],'profiles':{},'jit':[]};live=[]
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
  phase('compile_exact_two_existing_native_tiles')
  from sglang.kernels.jit.utils import load_jit,make_cpp_args
  api=importlib.import_module('sglang.kernels.ops.moe.moe_wna16_marlin');jitloader=importlib.import_module(load_jit.__module__);old_load=jitloader._load
  def collect_load(path):
   out=old_load(path);shutil.copy2(path,args.output/'two_tiles.so');r['jit'].append({'library_source_path':str(path),'so_sha256':sha(path),'before_candidate_math':True});save();return out
  jitloader._load=collect_load;start=time.monotonic()
  try:
   cppargs=make_cpp_args(torch.bfloat16,True,False)
   variant=load_jit('moe_anchor_two_tiles',manifest['variant_sha256'][:12],*cppargs,cuda_files=[str(ROOT/manifest['variant_header'])],cuda_wrappers=[('moe_wna16_marlin_gemm',f'moe_wna16_marlin_gemm<{cppargs}>'),('anchor_tile_control','anchor_tile_control')],extra_include_paths=[str(inc)])
  finally:jitloader._load=old_load
  r['compile_wall_seconds']=time.monotonic()-start;assert len(r['jit'])==1;save()
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
  default=int(variant.anchor_tile_control(-1));assert default==int(variant.anchor_tile_control(-2));mode=opposite(default)
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
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();end.record();end.synchronize();last={}
  def routing(data):
   counts={i:0 for i in range(-1,32)}
   for expert in data['topk_ids'].cpu().flatten().tolist():counts[expert if expert<32 else-1]+=1
   return counts
  def one(label,seed,kind):
   data,out,g=work[label];fresh=reference.generate_input(8,'ep8',seed)
   for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
   hist=routing(data);flush.zero_();start.record();g.replay();end.record();end.synchronize();us=start.elapsed_time(end)*1000;assert 0<us<float('inf');r['raw'].append({'arm':label,'seed':seed,'kind':kind,'us':us,'routing':hist});last[label]=(seed,fresh);save();return us
  def terminal(label,kind):
   seed,fresh=last[label];data,out,g=work[label];assert all(torch.equal(data[k],fresh[k])for k in('hidden_states','topk_ids','topk_weights'));check(label,kind+'_seed'+str(seed),data,out)
  phase('direct_F1_brackets');f1={}
  for block,label in enumerate(('anchor','default_shim','contrast','default_shim','anchor')):
   f1[label+str(block)]=[one(label,7800+i,'F1')for i in range(5)];terminal(label,'own_F1_block'+str(block))
  r['F1_raw_us']=f1;r['F1_medians_us']={k:statistics.median(v)for k,v in f1.items()};save()
  phase('direct_balanced_three_arm_pairs');pairs=[]
  for i,order in enumerate(ORDERS):pairs.append({LABELS[j]:one(LABELS[j],7900+i,'paired')for j in order})
  for label in LABELS:terminal(label,'own_paired_terminal')
  r['paired']=pairs;r['pair_orders']=[[LABELS[j]for j in order]for order in ORDERS];r['comparisons']={a+'_minus_'+b:{'median_us':statistics.median(row[a]-row[b]for row in pairs),'median_ratio':statistics.median(row[a]/row[b]for row in pairs),'b_wins':sum(row[b]<row[a]for row in pairs)}for a,b in(('anchor','default_shim'),('anchor','contrast'),('default_shim','contrast'))};r['host_up_calls']=host_up_calls
  immutable();bundle.verify_tree(tree,meta)
  for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
  r['validated_direct_timings_completed']=True;r['state']='VALID_ANCHOR_TILE_DIRECT_TIMINGS_PROFILE_OPTIONAL_ROOT_REVIEW';save()
  if args.profile:
   for label,(data,out,g)in work.items():
    phase('optional_profile_'+label);fresh=reference.generate_input(8,'ep8',7911)
    for k in('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
    out.fill_(float('nan'));entry={'optional':True};attempted=False
    try:
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:attempted=True;g.replay();torch.cuda.synchronize()
     path=args.output/(label+'.trace.json');prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents'];kernels=[{'name':e['name'],'duration_us':e.get('dur'),'args':e.get('args',{})}for e in events if e.get('cat')=='kernel'];assert kernels
     entry.update(status='COMPLETE',kernels=kernels,sha256=sha(path))
    except Exception:entry.update(status='FAILED_OPTIONAL',error=traceback.format_exc())
    r['profiles'][label]=entry;save()
    if attempted:assert all(torch.equal(data[k],fresh[k])for k in('hidden_states','topk_ids','topk_weights'));check(label,'post_profile_current_output',data,out)
  for _,_,g in work.values():torch.cuda.synchronize();g.reset();live.remove(g)
  immutable();bundle.verify_tree(tree,meta);r['state']='VALID_M8_ORIGINAL_MARLIN_TILE_SCREEN_ROOT_REVIEW_REQUIRED';r['full14_or_threefold_or_official_claim']=False;phase('complete')
 except BaseException:r['state']='INVALID_ANCHOR_TILE_SCREEN_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  if 'original_gemm'in locals():api.moe_wna16_marlin_gemm=original_gemm;api._jit_moe_wna16_marlin_module=original_factory
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset()
   except Exception:cleanup.append(traceback.format_exc())
  r['cleanup']=cleanup;r['finished_at']=time.time();r['elapsed_seconds']=r['finished_at']-r['started_at'];save()
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);p.add_argument('--profile',action='store_true');run(p.parse_args())
if __name__=='__main__':main()

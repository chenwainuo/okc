"""One exact incumbent RQ pooler32 graph; no compilation or fresh selection."""
from pathlib import Path
import argparse,ast,gc,hashlib,importlib.util,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
RQ=ROOT/'results/proj-rsi/scheduler/artifacts/research.rq.descriptor.offset.fixed.01/collection-1793137716124289144/logs/scheduler/rq-descriptor-offset-fixed-01/run'
HELPER=ROOT/'results/proj-rsi/scheduler/artifacts/research.gv53.quarter.sparse.fixed.01/collection-1793141738325901851/logs/scheduler/gv53-quarter-sparse-01/run'
RQ_SHA='ab6c3ad71701098713f7c3435439f905c4d482ae66a995948bfcda254494a32f'
HELPER_SHA='2eb511c0af3c4ce5bd3990e45d376c64d12016b876a2109ac0f263eafc8c17a7'
FUNCTION='_ZN10rq_control2rqILi32ELi6ELi1ELi0ELi0ELi0ELi35EEEvNS_3RqPENS_6RqTmapES2_'
CLEANUP=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def verify():
 m=json.loads((D/'runtime-manifest.json').read_text())
 for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
 return m
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cpu_check():
 verify();assert sha(RQ/'actual.so')==RQ_SHA and sha(HELPER/'actual.so')==HELPER_SHA
 tree=ast.parse(Path(__file__).read_text())
 assert not any(isinstance(n,ast.ImportFrom)and n.module=='torch.utils.cpp_extension'for n in ast.walk(tree))
 return {'state':'CPU_EXACT_RQ_CAPTURE_PASS_RUNTIME_UNRUN','JIT_calls':0,'profile_launches':1,'warmup_launches':5,'fixed_X_seed':91100,'actual_weight_seed':'reference.WEIGHTS_SEED','SO_pins':'PASS','helper':'clear_blas_cache only; no GV calls'}
def run(a):
 global CLEANUP
 verify();assert a.review.is_file() and os.environ.get('RQ_COUNTER_TOKEN')==json.loads((a.output/'session.json').read_text())['capture_token']
 started=time.monotonic();r={'state':'RUNNING_NUMERICS_UNVALIDATED','review_sha256':sha(a.review),'phases':[],'profile_launches':1,'JIT_calls':0};graphs=[]
 def save():dump(a.output/'result.json',r)
 def phase(s):r['phases'].append(s);save();assert time.monotonic()-started<80,'capture80s bound'
 try:
  phase('load_exact_SO_no_JIT')
  import torch
  sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
  torch.set_grad_enabled(False);props=torch.cuda.get_device_properties(0);assert(props.major,props.minor)==(9,0)and props.multi_processor_count>=128
  ext=load('rq_offset_fixed_b9ce4596f79c85acdc',RQ/'actual.so')
  helper=load('gv53_sparse_0e17b3c566c46a7b6981',HELPER/'actual.so');CLEANUP=helper.clear_blas_cache
  r['build']={'SO_sha256':sha(RQ/'actual.so'),'CUDA_sha256':sha(RQ/'actual.cu'),'native_arm':0,'cleanup_helper_SO_sha256':sha(HELPER/'actual.so'),'helper_use':'Only clear_blas_cache, no math/profile launches','JIT':False}
  fields=('registers','local','static','max_dynamic','max_threads','occupancy');r['attributes']={k:int(ext.attrs(0,i+1))for i,k in enumerate(fields)};save()
  at=r['attributes'];assert at['registers']==168 and at['local']==0 and at['static']==0 and at['max_dynamic']>=99488 and at['max_threads']>=384 and at['occupancy']>=1
  r['hardware']={'name':props.name,'SMs':props.multi_processor_count};assert props.multi_processor_count*at['occupancy']>=128
  phase('original_task_weights_codec')
  weights=reference.generate_weights(reference.WEIGHTS_SEED);w=weights['pooler'];env={'torch':torch}
  exec(compile((ROOT/'proj_rsi/candidates/rq_descriptor_offset_trial_v1/codec.py').read_text(),'<originalRQcodec>','exec'),env)
  lay,tab,esc6=env['_rqpack'](w);esc,_=env['_rqesc'](esc6,32,128,4,w.device)
  r['fixture']={'W_seed':reference.WEIGHTS_SEED,'X_seed':91100,'m':32,'N':4096,'K':8192,'KS':4,'packed_pointer_mod32':lay.data_ptr()%32}
  immutable=[(t,t.clone())for t in(w,lay,esc)]
  xb=torch.full((32*8192+32,),11.5,dtype=torch.bfloat16,device='cuda');x=xb[16:-16].view(32,8192)
  ob=torch.full((32*4096+32,),17.25,dtype=torch.bfloat16,device='cuda');out=ob[16:-16].view(32,4096)
  cb=torch.full((2*32+16,),0x19a75d3,dtype=torch.int32,device='cuda');cb[8:-8].zero_();sink=torch.full((4,),0x4735,dtype=torch.int32,device='cuda');prof=torch.empty(0,dtype=torch.int64,device='cuda')
  config={'layer':'pooler','m':32,'n':4096,'k':8192}
  def invoke():assert int(ext.run(0,lay,x,out,esc,cb[8:-8],4,tab,prof,sink))==0
  def integrity():
   for t,b in immutable:assert torch.equal(t.view(torch.uint8),b.view(torch.uint8)),'immutable weight/pack/escape'
   for t,v in [(xb,11.5),(ob,17.25)]:assert bool((t[:16]==v).all())and bool((t[-16:]==v).all())
   assert bool((cb[8:-8]==0).all())and bool((cb[:8]==0x19a75d3).all())and bool((cb[-8:]==0x19a75d3).all())and bool((sink==0x4735).all())
  def oracle(fresh):
   assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8)),'current X changed'
   ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),msg
   integrity();return {'task_FP64':msg,'current_X':'EXACT','RQ_counters_reset0':True,'physical_guards_sink_immutable':'PASS'}
  normal=reference.generate_input(32,'pooler',91100)['x'];zero=torch.zeros_like(normal)
  columns=[0,1,31,63,64,319,320,383,384,511,512,767,768,1984,2047,2048,2111,2368,2431,2432,2495,4031,4095,4096,4159,4416,4479,4480,4543,6144,6207,8191]
  boundary=torch.zeros_like(normal);boundary[torch.arange(32,device=x.device),torch.tensor(columns,device=x.device)]=8
  r['validation']=[]
  for kind,fresh in [('normal',normal),('outliers',reference.generate_input(32,'pooler',91303,dist='outliers')['x']),('zero',zero),('ring_split_boundary',boundary)]:
   x.copy_(fresh);out.fill_(float('nan'));invoke();torch.cuda.synchronize();record={'kind':kind,'oracle':oracle(fresh)}
   if kind=='zero':assert not int(torch.count_nonzero(out))
   if kind=='ring_split_boundary':assert torch.equal(out.float(),(w[:,columns].T*8).float())
   r['validation'].append(record)
  phase('capture_exact_incumbent_graph')
  x.copy_(normal);g=torch.cuda.CUDAGraph(keep_graph=True);graphs.append(g)
  with torch.cuda.graph(g):invoke()
  metadata=load('rq_node_metadata',ROOT/'proj_rsi/audits/hotpath-fresh-counter-v2/selected_graph_metadata.py');nodes=metadata.Inspector().inspect(g.raw_cuda_graph());r['actual_graph_nodes']=nodes;save()
  assert len(nodes)==1 and nodes[0]['node_type']==0;node=nodes[0]
  assert node['function_mangled']==FUNCTION and node['grid']==[32,4,1]and node['block']==[384,1,1]and node['dynamic_shared_bytes']==99488
  assert node['cluster_query_returncode']==0 and node['cluster_dimension']in([0,0,0],[1,1,1])and node['attributes']['local_bytes']==0
  g.instantiate();flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');flush_sink=torch.empty((),device='cuda')
  r['raw']=[]
  for i in range(6):
   x.copy_(normal);out.fill_(float('nan'));torch.sum(flush,0,out=flush_sink);torch.cuda.synchronize()
   if i==5:torch.cuda.nvtx.range_push('profile')
   try:g.replay();torch.cuda.synchronize()
   finally:
    if i==5:torch.cuda.nvtx.range_pop()
   r['raw'].append({'mode':'profile'if i==5 else'warmup','seed':91100})
   if i==5:r['own_terminal']={'seed':91100,'before_any_later_replay_or_mutation':True,'oracle':oracle(normal)}
  phase('profile_own_terminal_validated')
  r['mutations']=[]
  for kind,fresh in [('zero',zero),('normal',reference.generate_input(32,'pooler',91401)['x']),('outliers',reference.generate_input(32,'pooler',91403,dist='outliers')['x']),('boundary',boundary),('fresh_normal',reference.generate_input(32,'pooler',91407)['x'])]:
   x.copy_(fresh);out.fill_(float('nan'));g.replay();torch.cuda.synchronize();record={'kind':kind,'oracle':oracle(fresh),'NaN_output_overwritten':True}
   if kind=='zero':assert not int(torch.count_nonzero(out))
   r['mutations'].append(record)
  torch.cuda.synchronize();g.reset();graphs.remove(g);verify();r['state']='VALID_EXACT_CURRENT_POOLER32_RQ_PROFILE_OUTPUTS';phase('complete')
 except BaseException:r['state']='INVALID_NUMERICAL_OR_CAPTURE_NO_COUNTER_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  r['graph_cleanup']=[]
  for graph in graphs:
   try:torch.cuda.synchronize();graph.reset();r['graph_cleanup'].append('DRAINED_RESET')
   except BaseException as e:r['graph_cleanup'].append(repr(e));r['state']='INVALID_GRAPH_CLEANUP'
  r['cost']={'elapsed_seconds':time.monotonic()-started};save()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2));return
 error=None
 try:run(a)
 except BaseException as e:error=repr(e)
 gc.collect();torch=sys.modules.get('torch')
 if torch is not None and torch.cuda.is_initialized():
  torch.cuda.synchronize();evidence=None;clear_error=None
  try:evidence=list(map(int,CLEANUP()))if CLEANUP else None
  except BaseException as e:clear_error=repr(e)
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  cleanup={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'blas_after':evidence,'clear_error':clear_error,'after_run_scope_and_drain':True}
  r=json.loads((a.output/'result.json').read_text());r['post_scope_cleanup']=cleanup
  if cleanup['allocated']!=0 or evidence is None or evidence[2]!=0:r['state']='INVALID_POST_SCOPE_CLEANUP'
  dump(a.output/'result.json',r);assert cleanup['allocated']==0 and evidence is not None and evidence[2]==0,cleanup
 if error:raise RuntimeError(error)
if __name__=='__main__':main()

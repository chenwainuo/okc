"""Two scored-selected inc501d RQ32 cells, narrow descriptor-offset contrast.
No full prepare/full30/profiler; original codec and RQ host launch preserved.
"""
import argparse,ast,hashlib,json,statistics,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2];LABELS=('control','offset')
CELLS={'pooler':(4096,8192,4),'unpooler':(8192,4096,2)}
DECL='int64_t attrs(int64_t,int64_t);\nint64_t run(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,torch::Tensor);'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def verify():
 m=json.loads((D/'manifest.json').read_text())
 for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
 return m

def schedule():
 rows=[]
 for arm in LABELS:rows += [dict(arm=arm,seed=91000,mode='warmup')for _ in range(3)]
 for block,arm in enumerate(('control','offset','control')):rows += [dict(arm=arm,seed=91100,mode='F1',block=block,sample=i)for i in range(7)]
 for i in range(16):rows += [dict(arm=arm,seed=91200+i,mode='paired',pair=i)for arm in(LABELS if i%2==0 else LABELS[::-1])]
 assert len(rows)==59;return rows

def boundary_columns(k,ks):
 local=k//64//ks
 cols=[]
 for split in range(ks):
  for stage in(0,5,6,7,11,12,local-1):
   for q in(0,1,31,63):
    x=(split*local+stage)*64+q
    if 0<=x<k and x not in cols:cols.append(x)
 # One 32-row invocation; distribute selected split/ring points rather than a full-K basis.
 selected=[cols[(i*len(cols))//32]for i in range(32)]
 assert len(set(selected))==32 and any(x//64%local==6 for x in selected)
 return selected

def cpu_check():
 verify();rows=schedule();assert sum(x['mode']=='F1'for x in rows)==21 and sum(x['mode']=='paired'for x in rows)==32
 assert len({x['seed']for x in rows if x['mode']=='F1'})==1
 pairs=[rows[27+2*i:29+2*i]for i in range(16)];assert sum(p[0]['arm']=='control'for p in pairs)==8
 assert all(len({x['seed']for x in p})==1 for p in pairs)
 parent=ast.parse((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text());codec=ast.parse((D/'codec.py').read_text())
 named=lambda n:n.name if isinstance(n,ast.FunctionDef)else n.targets[0].id
 original={named(n):ast.dump(n,include_attributes=False)for n in parent.body if isinstance(n,(ast.FunctionDef,ast.Assign))and(isinstance(n,ast.FunctionDef)or isinstance(n.targets[0],ast.Name))}
 assert all(original[named(n)]==ast.dump(n,include_attributes=False)for n in codec.body)
 assert all(len(boundary_columns(k,ks))==32 for n,k,ks in CELLS.values())
 return {'state':'CPU_FOCUSED_HARNESS_PASS_GPU_UNRUN','cells':2,'timed_calls':106,'warmup_calls':12,'fixed_F1_fixture':True,'pair_order':[8,8],'boundary_probe_rows_per_arm_cell':32,'codec_AST_nodes':9,'full_prepare':False,'full_basis':False,'profiler':False,'selected_scope':'scored9916geometry only, notfreshprepareselection'}

def run(a):
 started=time.monotonic();verify();assert sha(a.review)==a.review_sha256
 a.output.mkdir(parents=True,exist_ok=False);r={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','started_at':time.time(),'review_sha256':a.review_sha256,'scope':'Scored-selected inc501d RQ32ST6FL35 pooler/unpooler fixed source contrast, no freshpolicy/full30/officialclaim','cells':{},'cost':{},'phases':[]};live=[]
 def save():dump(a.output/'result.json',r)
 def budget():assert time.monotonic()-started<120,'inner120s bound'
 def phase(x):r['phases'].append({'phase':x,'elapsed_seconds':time.monotonic()-started});save();print(x,flush=True);budget()
 try:
  phase('torch_and_exact_two_function_compile')
  import torch
  from torch.utils.cpp_extension import load_inline
  sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
  torch.set_grad_enabled(False);props=torch.cuda.get_device_properties(0)
  assert(props.major,props.minor)==(9,0)and props.multi_processor_count>=128
  r['device']={'name':props.name,'SMs':props.multi_processor_count,'CUDA':torch.version.cuda}
  cu=(D/'trial.cu').read_text();begin=time.monotonic();ext=load_inline('rq_offset_fixed_'+hashlib.sha256(cu.encode()).hexdigest()[:18],cpp_sources=DECL,cuda_sources=cu,functions=['run','attrs'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr'],verbose=True)
  r['cost']['compile_load_seconds']=time.monotonic()-begin;__import__('shutil').copy2(ext.__file__,a.output/'actual.so');(a.output/'actual.cu').write_text(cu)
  r['build']={'CUDA_sha256':hashlib.sha256(cu.encode()).hexdigest(),'SO_sha256':sha(ext.__file__),'native_type':[32,6,1,0,0,0,35],'private_namespaces':['rq_control','rq_offset'],'source_binding':'exact frozen source bodies; scoredselected.json; no added device kernel'}
  fields=('registers','local','static','max_dynamic','max_threads','occupancy')
  r['attributes']={arm:{key:int(ext.attrs(i,j+1))for j,key in enumerate(fields)}for i,arm in enumerate(LABELS)};save()
  for arm,x in r['attributes'].items():assert 0<x['registers']<=168 and x['local']==0 and x['static']==0 and x['max_dynamic']>=99488 and x['max_threads']>=384 and x['occupancy']>=1,(arm,x)
  phase('task_weights_and_original_codec')
  weights=reference.generate_weights(reference.WEIGHTS_SEED);env={'torch':torch};exec(compile((D/'codec.py').read_text(),'<exactcodecAST>','exec'),env)
  flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');flush_sink=torch.empty((),device='cuda',dtype=torch.float32)
  events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record();events[1].record();events[1].synchronize()
  for layer,(n,k,ks)in CELLS.items():
   phase('pack_and_validate_'+layer);w=weights[layer];lay,tab,esc6=env['_rqpack'](w);esc,_=env['_rqesc'](esc6,n//128,k//64,ks,w.device)
   immutable=[(t,t.clone())for t in(w,lay,esc)];entry=r['cells'][layer]={'config':{'layer':layer,'m':32,'n':n,'k':k},'KS':ks,'KCS':32,'geometry':[32,6,1,0,0,0,35],'raw':[],'validation':[],'terminals':[],'mutations':[]};work={}
   fixtures={seed:reference.generate_input(32,layer,seed)['x']for seed in{row['seed']for row in schedule()}}
   for arm in LABELS:
    xb=torch.full((32*k+32,),11.5,dtype=torch.bfloat16,device='cuda');ob=torch.full((32*n+32,),17.25,dtype=torch.bfloat16,device='cuda');cb=torch.full((2*(n//128)+16,),0x19a75d3,dtype=torch.int32,device='cuda');cb[8:-8].zero_();sink=torch.full((4,),0x4735,dtype=torch.int32,device='cuda');prof=torch.empty(0,dtype=torch.int64,device='cuda')
    work[arm]={'x':xb[16:-16].view(32,k),'out':ob[16:-16].view(32,n),'xb':xb,'ob':ob,'cb':cb,'sink':sink,'prof':prof,'calls':0}
   def integrity():
    for t,before in immutable:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8)),'weight/packed/escape mutated'
    for arm,v in work.items():
     assert bool((v['xb'][:16]==11.5).all())and bool((v['xb'][-16:]==11.5).all())and bool((v['ob'][:16]==17.25).all())and bool((v['ob'][-16:]==17.25).all()),('guard',arm)
     assert bool((v['cb'][8:-8]==0).all())and bool((v['cb'][:8]==0x19a75d3).all())and bool((v['cb'][-8:]==0x19a75d3).all())and bool((v['sink']==0x4735).all()),('RQactive_reset0/physicalguards/sink',arm)
   def invoke(arm):
    v=work[arm];code=int(ext.run(LABELS.index(arm),lay,v['x'],v['out'],esc,v['cb'][8:-8],ks,tab,v['prof'],v['sink']));assert code==0,('launch',arm,code)
   def check(arm,fresh,kind):
    v=work[arm];assert torch.equal(v['x'].view(torch.uint8),fresh.view(torch.uint8)),'currentXchanged'
    ok,msg=reference.check_implementation({'x':fresh,'config':entry['config']},v['out'],weights);assert ok and bool(torch.isfinite(v['out']).all()),(layer,arm,kind,msg)
    integrity();return {'arm':arm,'kind':kind,'task_FP64_oracle':msg,'currentX_bytes':'UNCHANGED','RQ_active_counters_reset0':True,'native_calls':v['calls']}
   columns=boundary_columns(k,ks);boundary=torch.zeros_like(fixtures[91000]);boundary[torch.arange(32,device=w.device),torch.tensor(columns,device=w.device)]=8
   entry['boundary_columns']=columns
   for arm in LABELS:
    for kind,fresh in [('normal',fixtures[91000]),('outliers',reference.generate_input(32,layer,91303,dist='outliers')['x']),('zero',torch.zeros_like(boundary)),('descriptor_ring_boundary',boundary)]:
     v=work[arm];v['x'].copy_(fresh);v['out'].fill_(float('nan'));invoke(arm);v['calls']+=1;torch.cuda.synchronize();entry['validation'].append(check(arm,fresh,kind))
     if kind=='descriptor_ring_boundary':assert torch.equal(v['out'].float(),(w[:,columns].T*8).float()),('boundarybasis',arm)
     if kind=='zero':assert not int(torch.count_nonzero(v['out']))
   phase('capture_and_measure_'+layer)
   for arm in LABELS:
    v=work[arm];v['x'].copy_(fixtures[91000]);g=torch.cuda.CUDAGraph();live.append(g)
    with torch.cuda.graph(g):invoke(arm)
    v['graph']=g
   last={};previous_block=None
   def own_terminal(arm,kind):
    assert arm in last;rec=check(arm,fixtures[last[arm]],kind);rec.update(seed=last[arm],before_any_later_replay=True);entry['terminals'].append(rec);save()
   for row in schedule():
    arm=row['arm'];v=work[arm]
    if row['mode']=='F1':
     block=row['block']
     if previous_block is not None and block!=previous_block:own_terminal(entry['raw'][-1]['arm'],'own_F1_block'+str(previous_block))
     previous_block=block
    elif row['mode']=='paired'and previous_block is not None:
     own_terminal(entry['raw'][-1]['arm'],'own_F1_block'+str(previous_block));previous_block=None
    v['x'].copy_(fixtures[row['seed']]);torch.sum(flush,0,out=flush_sink);events[0].record();v['graph'].replay();events[1].record();events[1].synchronize();v['calls']+=1;us=events[0].elapsed_time(events[1])*1000;assert 0<us<float('inf')
    rec=dict(row,us=us);entry['raw'].append(rec);last[arm]=row['seed']
    with(a.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(rec,layer=layer))+'\n');f.flush()
    if row['mode']!='warmup':check(arm,fixtures[row['seed']],'timed_'+row['mode'])
    save();budget()
   for arm in LABELS:own_terminal(arm,'own_paired_terminal')
   F=[x for x in entry['raw']if x['mode']=='F1'];med=[statistics.median(x['us']for x in F if x['block']==b)for b in range(3)];pairs=[{x['arm']:x['us']for x in entry['raw'][27+2*i:29+2*i]}for i in range(16)]
   entry['F1_medians_us']=dict(zip(('control_before','offset','control_after'),med));entry['paired']=pairs;entry['contrasts']={'median_control_minus_offset_us':statistics.median(x['control']-x['offset']for x in pairs),'median_control_over_offset':statistics.median(x['control']/x['offset']for x in pairs),'offset_wins':sum(x['control']>x['offset']for x in pairs),'F1_endpoint_average_control_minus_offset_us':(med[0]+med[2])/2-med[1]};save()
   for kind,fresh in [('zero',torch.zeros_like(boundary)),('fresh_normal',reference.generate_input(32,layer,91407)['x']),('boundary',boundary)]:
    for arm,v in work.items():
     v['x'].copy_(fresh);v['out'].fill_(float('nan'));v['graph'].replay();torch.cuda.synchronize();v['calls']+=1;entry['mutations'].append(check(arm,fresh,kind))
     if kind=='zero':assert not int(torch.count_nonzero(v['out']))
   for v in work.values():torch.cuda.synchronize();v['graph'].reset();live.remove(v['graph'])
   entry['state']='VALID_SCORED_SELECTED_GEOMETRY_FIXED_CELL';entry['cleanup']='DRAINED_RESET';save()
  verify();r['state']='VALID_TWO_RQ32_SCORED_SELECTED_DESCRIPTOR_OFFSET_CELLS_ROOT_REVIEW';phase('complete')
 except BaseException:r['state']='INVALID_NO_PERFORMANCE_ADMISSION';r['error']=traceback.format_exc();save();raise
 finally:
  cleanup=[]
  for g in live:
   try:torch.cuda.synchronize();g.reset();cleanup.append('DRAINED_RESET')
   except BaseException as e:cleanup.append(repr(e));r['state']='INVALID_CLEANUP'
  r['cleanup']=cleanup;r['finished_at']=time.time();r['cost']['run_elapsed_seconds']=time.monotonic()-started;save()

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--review-sha256');a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2));return
 assert a.output and a.review and a.review_sha256;run(a)
if __name__=='__main__':main()

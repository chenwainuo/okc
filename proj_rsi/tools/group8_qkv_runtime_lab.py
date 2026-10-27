"""Root-only actual-policy QKV128 split4 comparison. Import/--cpu-check are stdlib only.

Kernel/host wrapper remains the frozen group8_qkv_trial implementation. This new
attempt driver owns measurement; it never rewrites its parents or forces a bake.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,itertools,json,math,re,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'proj_rsi';TRIAL=BASE/'candidates/group8_qkv_runtime_trial'
LABELS=('pw36','rs32','new')
PINS={'pw36':'2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf',
      'rs32':'a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'}
CONFIG={'layer':'qkv_gate','m':128,'n':2304,'k':8192}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def save(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def verify():
 manifest=json.loads((TRIAL/'manifest.json').read_text())
 for path,digest in manifest['files'].items():assert sha(ROOT/path)==digest,('frozen dependency drift',path)
 for label,digest in PINS.items():assert sha(BASE/'candidates'/label/'submission.py')==digest
 assert manifest['driver_sha256']==sha(__file__)
 return manifest

def measurement_plan():
 plan=[]
 def emit(label,seed,mode):plan.append({'label':label,'input_seed':seed,'mode':mode})
 for label in LABELS:
  for _ in range(5):emit(label,719999,'warmup')
 for block,label in enumerate(('pw36','rs32','new','rs32','pw36')):
  for i in range(21):emit(label,720000+21*block+i,'F1')
 orders=list(itertools.permutations(LABELS))
 for rep in range(30):
  for label in orders[rep%6]:emit(label,730000+rep,'paired')
 return plan

def terminal_record_contract(last,raw):
 assert set(last)==set(LABELS) and raw
 for label in LABELS:
  expected=next(r['input_seed'] for r in reversed(raw) if r['label']==label and r['mode']!='warmup')
  assert last[label]==expected,('wrong final fixture',label)
 return True

def sass_body(text,needle):
 tables=[x for x in re.split(r'\n\s*Function : ',text)[1:] if needle in x.splitlines()[0]]
 assert len(tables)==1,('missing/ambiguous actual kernel',needle)
 return tables[0]

def instructions(body):
 return [(int(match[1],16),' '.join(match[2].split())) for match in re.finditer(r'/\*([0-9a-fA-F]+)\*/\s*([^\n]*?;)',body)]

def cpu_check():
 manifest=verify();plan=measurement_plan()
 assert len(plan)==210 and sum(r['mode']=='F1' for r in plan)==105 and sum(r['mode']=='paired' for r in plan)==90
 pairs=[plan[120+i*3:123+i*3] for i in range(30)]
 assert all(len({r['input_seed'] for r in row})==1 for row in pairs)
 assert {tuple(r['label'] for r in row) for row in pairs}==set(itertools.permutations(LABELS))
 assert all(sum(tuple(r['label'] for r in row)==order for row in pairs)==5 for order in itertools.permutations(LABELS))
 terminal={label:next(r['input_seed'] for r in reversed(plan) if r['label']==label) for label in LABELS}
 terminal_record_contract(terminal,plan)
 rejects=0
 for corrupt in ({'pw36':terminal['pw36']},dict(terminal,new=-1)):
  try:terminal_record_contract(corrupt,plan)
  except AssertionError:rejects+=1
 assert rejects==2
 # Import only CPU proof modules; never torch, a CUDA extension, or a prepare.
 proof=load(BASE/'candidates/group8_qkv/proofs.py','qkv_runtime_cpu_proof').proof()
 counters=load(BASE/'tools/smallm_increment_lab_v3.py','qkv_counter_cpu_support').counter_cpu_proof()
 tree=ast.parse(Path(__file__).read_text())
 terminal_fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='terminal_gate')
 forbidden={n.func.attr for n in ast.walk(terminal_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}&{'replay','copy_','fill_','zero_','reset'}
 assert not forbidden,('terminal gate writes/replays',forbidden)
 R=load(BASE/'candidates/group8_qkv_trial/runtime.py','qkv_cpu_runtime_identity')
 assert hashlib.sha256(R.source().encode()).hexdigest()==manifest['extension_source_sha256']
 offline=(BASE/'candidates/group8_qkv_trial/host.sass').read_text()
 counts={name:len(instructions(sass_body(offline,name))) for name in ('rq_qkv_g8_sk4','rx_pack','qkv_reduce')}
 assert min(counts.values())>0
 return {'status':'CPU_PASS_GPU_UNRUN','GPU_calls':0,'torch_imported':False,'timing_calls':len(plan),'F1_calls':105,'paired_calls':90,'warmup_calls':15,'paired_orders':'all six permutations exactly five times, same fixture within pair','negative_terminal_fixture_controls':rejects,'parent_proof':proof,'source_bound_counter_proof':counters,'offline_instruction_counts':counts,'full_runtime_gates':'IMPLEMENTED_NOT_RUN','extension_identity':R.identity()}

def run(args):
 manifest=verify();assert args.root_review_token,'independent review artifact required'
 review=(ROOT/args.root_review_token).resolve();assert review.is_file() and review.is_relative_to(ROOT)
 import torch
 sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
 assert torch.cuda.is_available()
 args.output.mkdir(parents=True,exist_ok=False)
 result={'status':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(review),'manifest_sha256':sha(TRIAL/'manifest.json'),'scope':CONFIG,'started_at':time.time(),'phases':[],'raw_calls':[],'JIT':{},'validations':[],'counter_checks':[]}
 output=args.output/'result.json';graphs={};pipelines=[]
 def persist():save(output,result)
 def phase(name):result['phases'].append({'phase':name,'at':time.time()});persist();print(name,flush=True)
 R=load(BASE/'candidates/group8_qkv_trial/runtime.py','qkv_frozen_runtime')
 C=load(BASE/'tools/smallm_increment_lab_v3.py','qkv_frozen_counter_support')
 try:
  phase('candidate_JIT_before_any_candidate_math')
  ext=R.extension();binary=Path(ext.__file__)
  actual=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True,timeout=40)
  resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(binary)],text=True,timeout=40)
  (args.output/'candidate.jit.sass').write_text(actual);(args.output/'candidate.resources').write_text(resources)
  shutil.copy2(binary,args.output/'candidate.actual.so')
  offline=(BASE/'candidates/group8_qkv_trial/host.sass').read_text()
  physical=load(BASE/'candidates/operand_reverse_group8/liveness.py','qkv_runtime_physical_gate')
  for name in ('rq_qkv_g8_sk4','rx_pack','qkv_reduce'):
   got=sass_body(actual,name);want=sass_body(offline,name)
   assert instructions(got)==instructions(want),('actual/offline SASS mismatch',name)
   if name=='rq_qkv_g8_sk4':physical.check_one(got)
  attrs={}
  for which,name,regs,threads in ((0,'pack',22,256),(1,'main',122,512),(2,'reduce',16,256)):
   a={str(field):int(ext.qkv_info(which,field)) for field in range(5)}
   assert a['0']==regs and a['1']==0 and a['3']>=threads,(name,a)
   attrs[name]=a
  occupancy=int(ext.qkv_info(1,5));assert occupancy>=1 and int(ext.qkv_info(1,4))>=103600
  assert ((attrs['main']['0']+7)//8)*8*512==65536
  result['JIT']['new']={'source_sha256':hashlib.sha256(R.source().encode()).hexdigest(),'so_sha256':sha(binary),'sass_sha256':sha(args.output/'candidate.jit.sass'),'attrs':attrs,'occupancy_after_optin':occupancy,'physical_A_D_fence_gate':'PASS','actual_instruction_identity':'all three kernels equal frozen offline host SASS'};persist()
  phase('real_task_weights_and_full_actual_prepares')
  weights=reference.generate_weights(reference.WEIGHTS_SEED);weight_snapshots={name:t.clone() for name,t in weights.items()}
  w=weights['qkv_gate'];modules={};states={};picked={};registry={}
  for label in LABELS[:2]:
   module=modules[label]=load(BASE/'candidates'/label/'submission.py','qkv_actual_'+label)
   # No clock reset or source/dispatch edits: original full prepare semantics.
   states[label]=module.prepare(weights);torch.cuda.synchronize()
   assert all(torch.equal(t,weight_snapshots[name]) for name,t in weights.items()),'prepare modified original task weights'
   assert 'cpu' not in states[label]
   state=states[label];rp=state.get('rq',{}).get(('qkv_gate',128))
   contract,counter=C.select_counter_contract(module,state,'qkv_gate',128,2304)
   picked[label]={'counter_contract':contract,'selected_rq':{k:rp[k] for k in ('i','KS','tab','n')} if rp else None}
   if rp is not None:
    assert rp['n']==2304
    original=module._rqcall
    def guarded(st,p,x,out,original=original,rp=rp,label=label):
     assert p is rp,('actual selected plan changed',label)
     rc=original(st,p,x,out);assert rc==0,('actual selected plan declined',label,rc);return rc
    module._rqcall=guarded
    selected_ext=rp.get('mod',module._rqmod[0])
    assert selected_ext is module._rqmod[0],('HOLD unbound alternate RQ module',label)
    so=Path(selected_ext.__file__)
    selected_sass=subprocess.check_output(['cuobjdump','--dump-sass',str(so)],text=True,timeout=40)
    selected_resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(so)],text=True,timeout=40)
    (args.output/(label+'.selected.sass')).write_text(selected_sass);(args.output/(label+'.selected.resources')).write_text(selected_resources)
    generated=module._rqmod[1];assert isinstance(generated,str)
    assert module._rqsrc([module.RQV[module.RQI[n]] for n in module.RQPROD])==generated
    (args.output/(label+'.selected.cu')).write_text(generated)
    tag=hashlib.sha256(generated.encode()).hexdigest();name=selected_ext.__name__
    assert name not in registry or registry[name]==tag,'extension-name collision';registry[name]=tag
    result['JIT'][label]={'so_sha256':sha(so),'source_sha256':tag,'sass_sha256':sha(args.output/(label+'.selected.sass')),'attrs':C.selected_attributes(selected_ext,rp,torch.empty((128,8192),dtype=torch.bfloat16,device=w.device),torch.empty((128,2304),dtype=torch.bfloat16,device=w.device),state),'gate_scope':'actual prepared RQ binary/resources recorded; production prepare internally baked before this external gate'}
    assert result['JIT'][label]['attrs'][0]==168 and result['JIT'][label]['attrs'][1]==0
   else:result['JIT'][label]={'gate_scope':'actual original non-RQ policy; no forced RQ path; actual GPU profile recorded after terminal checks'}
   persist()
  result['actual_prepared_picks']=picked
  phase('actual_task_W_lossless_pack_CSR_and_pipeline_ownership')
  pipe=R.Pipeline(w);pipelines.append(pipe)
  assert pipe.occupancy>=1
  result['actual_W_codec']={'exact_BF16_bits':'PASS','CSR_CAP':pipe.cap,'XP_bytes':pipe.xp.numel()*pipe.xp.element_size(),'partial_bytes':pipe.partial.numel()*pipe.partial.element_size(),'input_derived_cache':False}
  immutable=[];counters=[];active={};seen=set()
  for label,state in states.items():
   _,selected_counter=C.select_counter_contract(modules[label],state,'qkv_gate',128,2304)
   active[label]={C.tensor_identity(selected_counter)} if selected_counter is not None else set()
   for name,t in C.tensors(state):
    if not t.numel():continue
    if 'cnt' in name:counters.append((label,name,t,C.counter_snapshot(label+name,t)));continue
    if 'ws' in name or 'sink' in name:continue
    key=(C.tensor_identity(t),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
    if key not in seen:seen.add(key);immutable.append((label+name,t,t.clone()))
  immutable.extend(('original_weight_'+name,t,weight_snapshots[name]) for name,t in weights.items())
  immutable.extend((name,t,t.clone()) for name,t in [('candidate_W',pipe.W),('candidate_esc',pipe.esc)])
  result['counter_prepare_baselines']=[{k:v for k,v in initial.items() if k!='values'} for _,_,_,initial in counters]
  def invariants(context):
   assert all(torch.equal(t,before) for _,t,before in immutable),'immutable W/packed/metadata changed'
   records=[]
   for label,name,t,initial in counters:
    final=C.counter_snapshot(label+name,t);contract=C.counter_storage_contract(t,active[label])
    if not C.counter_verdict(contract,initial['values'],final['values']):
     result['counter_failure']={'context':context,'name':label+name,'contract':contract,'initial':initial,'final':final};persist();raise AssertionError('counter invariant failure')
    records.append({'name':label+name,'contract':contract,'final_digest':final['digest'],'final_nonzero':final['nonzero']})
   result['counter_checks'].append({'context':context,'records':records})
  def oracle(label,x,out):
   assert bool(torch.isfinite(out).all()),(label,'nonfinite')
   ok,message=reference.check_implementation({'x':x,'config':CONFIG},out,{'qkv_gate':w});assert ok,(label,message)
   return message
  # Shape-only XP oracle. No input-derived tensor is passed as a cache to a kernel.
  idx=torch.arange(131072,dtype=torch.int64,device=w.device);v=idx&3;lane=(idx>>2)&31;chunk=(idx>>7)&3;warp=(idx>>9)&7;s=idx>>12
  row=64*(warp>>2)+16*(warp&3)+(lane>>2)+8*(v&1);k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
  pack_index=torch.cat([row*4096+k//2+1024*u for u in range(4)])
  def workspace_oracle(x):
   assert torch.equal(pipe.xp,x.view(torch.int32).reshape(-1)[pack_index]),'XP incomplete/stale'
   assert bool(torch.isfinite(pipe.partial).all()),'partial incomplete/nonfinite'
   reconstructed=((pipe.partial[0]+pipe.partial[1])+pipe.partial[2])+pipe.partial[3]
   return reconstructed.to(torch.bfloat16)
  def eager(label,x,out):
   if label=='new':assert pipe(x,out)==0
   else:out.copy_(modules[label].kernel_fn({'x':x,'config':CONFIG},states[label]))
  phase('all8192_basis_and_FP64_normal_outliers_all_three_policies')
  for label in LABELS:
   x=torch.zeros((128,8192),dtype=torch.bfloat16,device=w.device);out=torch.empty((128,2304),dtype=torch.bfloat16,device=w.device)
   rows=torch.arange(128,device=w.device)
   for start in range(0,8192,128):
    x.zero_();x[rows,start+rows]=8;eager(label,x,out)
    assert torch.equal(out,(w[:,start:start+128].t()*8)),(label,'basis',start)
   for dist in ('normal','outliers'):
    data=reference.generate_input(128,'qkv_gate',74123,dist=dist);before=data['x'].clone();eager(label,data['x'],out)
    message=oracle(label,data['x'],out);exact=data['x'].double()@w.double().t();diff=out.double()-exact
    rel=float(diff.norm()/exact.norm());rowerr=float((diff.norm(dim=-1)/exact.norm(dim=-1).clamp(min=1e-30)).max())
    assert rel<=4e-3 and rowerr<=2e-2 and torch.equal(data['x'],before),(label,dist,rel,rowerr)
    result['validations'].append({'label':label,'all8192_basis_exact':True,'dist':dist,'relative_L2':rel,'max_row':rowerr,'task':message})
   invariants('basis_FP64_'+label);persist()
  phase('fixed_actual_serving_graphs')
  x=reference.generate_input(128,'qkv_gate',719999)['x'].clone();outputs={}
  for label in LABELS:
   if label=='new':
    out=torch.empty((128,2304),dtype=torch.bfloat16,device=w.device);pipe(x,out);pipe(x,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):assert pipe(x,out)==0
    outputs[label]=out;pipe.retained.append((x,out))
   else:
    module=modules[label];state=states[label];module.kernel_fn({'x':x,'config':CONFIG},state);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):outputs[label]=module.kernel_fn({'x':x,'config':CONFIG},state)
   graphs[label]=graph
  assert len({out.data_ptr() for out in outputs.values()})==3
  def mutations(context):
   original=x.clone();probes=torch.zeros_like(x);rr=torch.arange(128,device=w.device);probes[rr,(rr*37+19)%8192]=8
   fixtures=[('original',original),('zero',torch.zeros_like(x)),('fresh_normal',reference.generate_input(128,'qkv_gate',74567)['x']),('fresh_outliers',reference.generate_input(128,'qkv_gate',74569,dist='outliers')['x']),('allrows_probe',probes)]
   for name,fresh in fixtures:
    x.copy_(fresh)
    for label,graph in graphs.items():
     outputs[label].fill_(float('nan'))
     if label=='new':pipe.xp.fill_(0x5a39c7e1);pipe.partial.fill_(float('nan'))
     graph.replay();torch.cuda.synchronize();assert torch.equal(x,fresh)
     message=oracle(label,fresh,outputs[label])
     if name=='zero':assert not int(torch.count_nonzero(outputs[label]))
     if label=='new':assert torch.equal(outputs[label],workspace_oracle(fresh))
     invariants(context+'_'+label+'_'+name)
     result['validations'].append({'context':context,'label':label,'mutation':name,'NaN_poison':'PASS','current_input':'PASS','task':message})
   x.copy_(original);persist()
  mutations('pre_timing')
  # All BF16 storage patterns through the ACTUAL fresh pack. Nonfinite arithmetic
  # here is intentionally not a numerical trial or a performance observation.
  storage_original=x.clone()
  bits=(torch.arange(x.numel(),dtype=torch.int32,device=w.device)&65535).to(torch.int16).view(torch.bfloat16).reshape_as(x)
  x.copy_(bits);graphs['new'].replay();torch.cuda.synchronize()
  assert torch.equal(pipe.xp,x.view(torch.int32).reshape(-1)[pack_index])
  assert torch.equal(x.view(torch.int16),bits.view(torch.int16)),'storage diagnostic input modified'
  result['all65536_BF16_storage_pack']='PASS_STORAGE_ONLY_NOT_NUMERICAL'
  x.copy_(storage_original);mutations('after_storage_bits')
  fresh=R.Pipeline(w);pipelines.append(fresh);freshout=torch.full((128,2304),float('nan'),dtype=torch.bfloat16,device=w.device)
  immutable.extend((name,t,t.clone()) for name,t in [('fresh_candidate_W',fresh.W),('fresh_candidate_esc',fresh.esc)])
  fresh.xp.fill_(0x5a39c7e1);fresh.partial.fill_(float('nan'));fresh(x,freshout);torch.cuda.synchronize();oracle('fresh_workspace',x,freshout)
  assert torch.equal(fresh.xp,x.view(torch.int32).reshape(-1)[pack_index]) and bool(torch.isfinite(fresh.partial).all())
  fresh_reduced=((fresh.partial[0]+fresh.partial[1])+fresh.partial[2])+fresh.partial[3]
  assert torch.equal(freshout,fresh_reduced.to(torch.bfloat16))
  result['fresh_pipeline_workspace_output']='PASS';invariants('before_timing')
  plan=measurement_plan();fixtures={seed:reference.generate_input(128,'qkv_gate',seed)['x'] for seed in {r['input_seed'] for r in plan}}
  flush=torch.ones(64<<20,dtype=torch.float32,device=w.device);sink=torch.empty((),dtype=torch.float32,device=w.device);last={}
  def terminal_gate(context):
   # READ ONLY. Never replay, fill/copy, reset, profile or repair an actual result.
   terminal_record_contract(last,result['raw_calls'])
   assert torch.equal(x,fixtures[result['raw_calls'][-1]['input_seed']]),'last timed X changed'
   checks=[]
   for label in LABELS:
    fixture=fixtures[last[label]];message=oracle(label,fixture,outputs[label])
    if label=='new':assert torch.equal(outputs[label],workspace_oracle(fixture))
    checks.append({'label':label,'input_seed':last[label],'actual_terminal_before_any_replay':'PASS','task':message})
   invariants('terminal_'+context)
   result.setdefault('strict_terminal_checks',{})[context]=checks;persist()
  phase('full_pipeline_F1_and_balanced_pairs')
  raw_path=args.output/'raw.jsonl'
  with raw_path.open('x') as raw:
   for index,record in enumerate(plan):
    if index==120:terminal_gate('F1')
    x.copy_(fixtures[record['input_seed']]);torch.sum(flush,0,out=sink)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();graphs[record['label']].replay();end.record();torch.cuda.synchronize()
    us=start.elapsed_time(end)*1000;assert math.isfinite(us) and us>0
    row=dict(record,us=us,index=index);result['raw_calls'].append(row);raw.write(json.dumps(row)+'\n');raw.flush()
    if record['mode']!='warmup':last[record['label']]=record['input_seed']
  terminal_gate('paired')
  calls=result['raw_calls'];f1=calls[15:120];blocks=[f1[i*21:(i+1)*21] for i in range(5)];med=[statistics.median(r['us'] for r in b) for b in blocks]
  pairs=[{r['label']:r['us'] for r in calls[120+3*i:123+3*i]} for i in range(30)]
  result['timing']={'F1_block_medians_us':dict(zip(('pw36_before','rs32_before','new','rs32_after','pw36_after'),med)),
   'F1_new_gain_us':{'pw36':min(med[0],med[4])-med[2],'rs32':min(med[1],med[3])-med[2]},
   'paired_rows':pairs,'paired_new_gain_us':{label:statistics.median(r[label]-r['new'] for r in pairs) for label in LABELS[:2]},
   'paired_control_over_new_ratio':{label:statistics.median(r[label]/r['new'] for r in pairs) for label in LABELS[:2]},'unfiltered_calls':len(calls),'timed_graph_new':'fresh pack +split4main +FP32reduce, allthree nodes'};persist()
  phase('profiles_only_after_both_strict_terminal_gates')
  from torch.profiler import profile,ProfilerActivity
  profiles={}
  for label,graph in graphs.items():
   with profile(activities=[ProfilerActivity.CUDA]) as prof:graph.replay();torch.cuda.synchronize()
   kernels=[e.name() for e in prof.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]
   profiles[label]=kernels;oracle(label,x,outputs[label]);invariants('after_profile_'+label)
   if label=='new':
    assert len(kernels)==3 and all(sum(needle in name for name in kernels)==1 for needle in ('rx_pack','rq_qkv_g8_sk4','qkv_reduce')),kernels
    assert torch.equal(outputs[label],workspace_oracle(x))
  result['actual_profile_kernels']=profiles;mutations('post_timing')
  result['status']='VALID_LOCAL_COMPARISON_REVIEW_REQUIRED_NO_OFFICIAL_GAIN';result['official_score']='NOT_MEASURED'
 except BaseException:
  result['status']='INVALID_NO_PERFORMANCE_ADMISSION';result['failure']=traceback.format_exc();raise
 finally:
  try:
   torch.cuda.synchronize()
   for graph in graphs.values():graph.reset()
   torch.cuda.synchronize();result['graph_cleanup']='ALL_CAPTURED_GRAPHS_DRAINED_RESET'
  except BaseException:result['graph_cleanup']='FAILED_ROOT_CLEANUP_REQUIRED';result['cleanup_failure']=traceback.format_exc()
  result['ended_at']=time.time();result['root_cleanup_required']='Verify exclusive guard/container/compute PID absence and device memory released after process exits';persist()

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--output',type=Path);parser.add_argument('--root-review-token');args=parser.parse_args(argv)
 if args.cpu_check:
  print(json.dumps(cpu_check(),indent=2));return
 assert args.output is not None,'--output is required for runtime';run(args)
if __name__=='__main__':main()

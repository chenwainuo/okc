"""Optional exact collected LUT-v1 up-kernel counters, real M8 G6/G7 fixtures.

No imports creating CUDA work before the bounded profiler capability query.
This is a diagnostic, not a timing/admission gate. No compilation or rewrite.
"""
import argparse,csv,hashlib,importlib,importlib.util,json,math,os,re,signal,subprocess,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
NCU='/usr/local/cuda/bin/ncu'
METRICS=('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def bounded_capture(cmd,log):
 child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,shell=False)
 try:return child.wait(timeout=45)
 except BaseException:
  # NCU and its target share this fresh process group; drain both on timeout.
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=3)
  except subprocess.TimeoutExpired:pass
  # The profiler may exit before its target; kill surviving group members too.
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=3)
  raise
def export_contract(path):
 lines=Path(path).read_text().splitlines();start=next(i for i,line in enumerate(lines)if next(csv.reader([line]),[''])[0]=='ID')
 rows=list(csv.DictReader(lines[start:]));groups={};launches={}
 if rows and 'Metric Name'not in rows[0]:
  units=next((row for row in rows if not row['ID']),{})
  for row in rows:
   if not row['ID']:continue
   assert row['ID']not in groups
   assert 'moe_ss_lut_full_v1'in row['Kernel Name']and'projectILi8ELi8EE'in row['Kernel Name']
   assert all(row.get(m)not in(None,'','n/a')for m in METRICS)
   assert all(math.isfinite(float(row[m].replace(',','')))for m in METRICS)
   assert float(row[METRICS[0]].replace(',',''))>0
   groups[row['ID']]={m:{'value':row[m],'unit':units.get(m)}for m in METRICS}
   launches[row['ID']]={k:v for k,v in row.items()if k in('Kernel Name','Grid Size','Block Size')or'replay_passes'in k or k.startswith(('launch__register','launch__shared_mem','launch__occupancy'))}
  assert len(groups)==2,('missing/extra captures',groups)
  return [{'kernel_id':key,'seed':seed,'metrics':group,'launch_and_replay':launches[key]}for(key,group),seed in zip(sorted(groups.items(),key=lambda x:int(x[0])),(7901,7903))]
 for row in rows:
  name=row.get('Metric Name','')
  if name not in METRICS:continue
  assert 'moe_ss_lut_full_v1' in row['Kernel Name']and 'projectILi8ELi8EE' in row['Kernel Name']
  group=groups.setdefault(row['ID'],{})
  assert name not in group,('duplicate metric',row['ID'],name)
  value=float(row['Metric Value'].replace(',',''));assert math.isfinite(value)and(value>0 if name==METRICS[0]else True)
  group[name]={'value':row['Metric Value'],'unit':row['Metric Unit']}
 assert len(groups)==2 and all(set(g)==set(METRICS)for g in groups.values()),('missing/extra captures',groups)
 return [{'kernel_id':key,'seed':seed,'metrics':group}for(key,group),seed in zip(sorted(groups.items(),key=lambda x:int(x[0])),(7901,7903))]
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',required=True,type=Path);p.add_argument('--lut-prior',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--review-sha256',required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
 assert sha(a.review)==a.review_sha256
 pins=json.loads((D/'manifest.json').read_text())
 for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,rel
 if not a.child:
  if a.output.exists():raise FileExistsError(a.output)
  a.output.mkdir(parents=True);r={'state':'CAPABILITY_QUERY_BEFORE_PREPARE','started_at':time.time(),'diagnostic_only':True,'scoring_admission':False,'metrics':METRICS}
  def save():(a.output/'session.json').write_text(json.dumps(r,indent=2)+'\n')
  save()
  try:
   v=subprocess.run([NCU,'--version'],capture_output=True,text=True,timeout=5);r['version']=v.stdout+v.stderr;save();assert v.returncode==0 and re.search(r'\b2025\.3\.1(?:\.0)?\b',r['version'])and'build 36398880'in r['version']
   qargv=[NCU,'--chips','gh100','--query-metrics-mode','suffix','--query-metrics-collection','profiling','--metrics',','.join(m.split('.')[0]for m in METRICS)]
   q=subprocess.run(qargv,capture_output=True,text=True,timeout=10);(a.output/'metric-query.stdout').write_text(q.stdout);(a.output/'metric-query.stderr').write_text(q.stderr);r.update(query_argv=qargv,query_returncode=q.returncode,ncu_sha256=sha(NCU));save()
   names=set(re.findall(r'\b[a-z][a-z0-9_]*__[a-zA-Z0-9_.]+',q.stdout));assert q.returncode==0 and set(METRICS)<=names,('unavailable metrics',set(METRICS)-names)
   cmd=[NCU,'--target-processes','application-only','--kill','no','--replay-mode','kernel','--cache-control','all','--clock-control','none','--profile-from-start','off','--kernel-name-base','mangled','--kernel-name','regex:.*moe_ss_lut_full_v1.*projectILi8ELi8EE.*','--launch-count','2','--metrics',','.join(METRICS),'--export',str(a.output/'up'),'python3',str(Path(__file__).resolve()),'--child','--prior',str(a.prior),'--lut-prior',str(a.lut_prior),'--output',str(a.output),'--review',str(a.review),'--review-sha256',a.review_sha256]
   r.update(state='OPTIONAL_CAPTURE_RUNNING',capture_argv=cmd);save()
   with(a.output/'capture.log').open('x')as log:r['capture_returncode']=bounded_capture(cmd,log)
   save();assert r['capture_returncode']==0
   child=json.loads((a.output/'result.json').read_text());assert child['state']=='VALID_EXACT_V1_UP_DIAGNOSTIC_OUTPUTS'
   report=a.output/'up.ncu-rep';r['report_sha256']=sha(report)
   with(a.output/'up.csv').open('x')as out, (a.output/'export.stderr').open('x')as err:
    exported=subprocess.run([NCU,'--import',str(report),'--page','raw','--csv','--print-kernel-base','mangled','--print-units','base'],stdout=out,stderr=err,timeout=15)
   assert exported.returncode==0;r['captures']=export_contract(a.output/'up.csv');r['replay_pass_evidence']='capture.log retained verbatim; hidden replay pass count requires report/log review';r['state']='OPTIONAL_COUNTER_CAPTURE_COMPLETE_EXPORT_REQUIRES_REVIEW'
  except Exception:
   child=json.loads((a.output/'result.json').read_text())if(a.output/'result.json').exists()else{}
   invalid=child.get('state')=='INVALID_COUNTER_DIAGNOSTIC_OUTPUTS'
   r.update(state='INVALID_COUNTER_DIAGNOSTIC_OUTPUTS'if invalid else'OPTIONAL_COUNTER_CAPTURE_UNAVAILABLE_OR_FAILED',child_state=child.get('state'),error=traceback.format_exc())
  finally:r['elapsed_seconds']=time.time()-r['started_at'];save()
  return
 # Actual child: existing reviewed source/binary/tree identity, one original prepare.
 r={'state':'RUNNING_DIAGNOSTIC','started_at':time.time(),'phases':[],'correctness':[],'fixtures':[],'reuse':{'new_compile':False,'SO_sha256':pins['LUT_SO_sha256']}}
 def save():(a.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
 def phase(name):r['phases'].append({'phase':name,'elapsed_seconds':time.time()-r['started_at']});save()
 try:
  phase('verify_exact_collected_tree_and_binaries')
  for directory,key,file in [(a.prior,'original_result_sha256','result.json'),(a.prior,'original_SO_sha256','actual_ss.so'),(a.lut_prior,'LUT_result_sha256','result.json'),(a.lut_prior,'LUT_SO_sha256','actual_lut.so')]:assert sha(directory/file)==pins[key]
  tree=a.prior/'candidate-tree';bundle=load(ROOT/'benchmarks/rsi_bundle.py','up_counter_bundle');meta=json.loads((tree/'.rsi-bundle.json').read_text());bundle.verify_tree(tree,meta);assert meta['archive_sha256']==pins['prior_archive_sha256']
  load(ROOT/'benchmarks/rsi_worker.py','up_counter_paths').setup_paths(tree)
  import torch
  torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  props=torch.cuda.get_device_properties(0);assert torch.version.cuda=='13.0'and(props.major,props.minor)==(9,0)and props.multi_processor_count==132
  reference=importlib.import_module('reference');serving=importlib.import_module('serving_moe');original=sys.modules['ss_exact_serving_both']._engine
  lut_full=load(ROOT/pins['LUT_serving_source'],'up_counter_lut');lut=lut_full._engine
  import torch.utils.cpp_extension as cpp
  builder=cpp.load_inline
  for engine,path,name,source in [(original,a.prior/'actual_ss.so',pins['original_module_name'],tree/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2/extension_source.cu'),(lut,a.lut_prior/'actual_lut.so',pins['LUT_module_name'],ROOT/pins['LUT_extension_source'])]:
   ext=load(path,name)
   def reuse(*args,**kw):assert kw['cuda_sources']==source.read_text()and kw['functions']==['run'];return ext
   cpp.load_inline=reuse
   try:assert engine.module()is ext
   finally:cpp.load_inline=builder
  phase('weights_and_one_original_prepare')
  weights=reference.generate_weights(5252);state=serving.prepare({k:v.clone()for k,v in weights.items()});torch.cuda.synchronize()
  frozen=[]
  for attr in ('_ss_up_bank','_ss_down_bank'):
   bank=getattr(state,attr)
   for t in(bank.packed,bank.scales,bank.globals):frozen.append((t,t.clone()))
  captured=[]
  def record(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor):
   args=dict(x=x.clone(),bank=bank,sorted_ids=sorted_ids.clone(),experts=experts.clone(),post=post.clone(),block_m=block_m,gather_divisor=gather_divisor)
   ans=original.project(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor);args['expected']=ans.clone();captured.append(args);return ans
  method=type(state)._marlin_expert_outputs;private=dict(method.__globals__);private['_engine']=types.SimpleNamespace(project=record)
  own=types.FunctionType(method.__code__,private,method.__name__,method.__defaults__,method.__closure__);cls=type('UpCounterCapture',(type(state),),{'_marlin_expert_outputs':own});capture_state=cls.__new__(cls);capture_state.__dict__=state.__dict__.copy()
  work=[]
  phase('real_serving_current_inputs_and_empty')
  for seed,expected_blocks in [(7901,6),(7903,7),(7779,0)]:
   captured.clear();data=reference.generate_input(8,'ep8',seed)
   if expected_blocks==0:data['topk_ids'].fill_(32)
   before={k:data[k].clone()for k in('hidden_states','topk_ids','topk_weights')};answer=serving.kernel_fn(data,capture_state);torch.cuda.synchronize()
   if expected_blocks==0:
    ok=not int(torch.count_nonzero(answer.view(torch.uint8)));msg='all nonlocal actual serving output exact zero'
   else:ok,msg=reference.check_implementation(data,answer,weights)
   assert ok,msg
   assert all(torch.equal(data[k],v)for k,v in before.items())and len(captured)==2
   rec=captured[0];assert rec['gather_divisor']==8 and rec['block_m']==8
   ids=rec['experts'].cpu().tolist();post=int(rec['post'].item());blocks=sum(e>=0 for e in ids[:post//8]);assert blocks==expected_blocks
   expected=rec.pop('expected');rec['out']=torch.empty_like(expected);rec['out'].fill_(float('nan'));lut.project(**rec);torch.cuda.synchronize();assert torch.equal(rec['out'].view(torch.uint8),expected.view(torch.uint8))
   tensor_hashes={k:hashlib.sha256(rec[k].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()for k in('x','sorted_ids','experts','post')}
   r['correctness'].append({'seed':seed,'original_task_pass':ok,'up_bitwise_equal':True,'message':msg});r['fixtures'].append({'seed':seed,'active_local_blocks':blocks,'post':post,'capacity':rec['sorted_ids'].numel(),'projection_grid':[128,rec['sorted_ids'].numel()//8,1],'tensor_sha256':tensor_hashes,'bank_addresses':{k:getattr(rec['bank'],k).data_ptr()for k in('packed','scales','globals')}});save()
   if blocks:work.append((seed,rec,expected,{k:rec[k].clone()for k in('x','sorted_ids','experts','post')}))
  phase('warm_exact_up_then_optional_counter_ranges')
  for seed,rec,expected,before in work:
   for _ in range(3):lut.project(**rec)
   torch.cuda.synchronize();assert torch.equal(rec['out'].view(torch.uint8),expected.view(torch.uint8))
  r['validated_direct_outputs_completed']=True;save()
  for seed,rec,expected,before in work:
   rec['out'].fill_(float('nan'));torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
   try:lut.project(**rec);torch.cuda.synchronize()
   finally:torch.cuda.cudart().cudaProfilerStop()
   assert torch.equal(rec['out'].view(torch.uint8),expected.view(torch.uint8))
   assert all(torch.equal(rec[k].view(torch.uint8),v.view(torch.uint8))for k,v in before.items())
   r['correctness'].append({'seed':seed,'own_post_counter_terminal_bitwise_equal':True,'captured_inputs_immutable':True});save()
  phase('final_immutable_source_checks')
  for t,before in frozen:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
  for rel,h in pins['files'].items():assert sha(ROOT/rel)==h,rel
  bundle.verify_tree(tree,meta);r['state']='VALID_EXACT_V1_UP_DIAGNOSTIC_OUTPUTS'
 except BaseException:r.update(state='INVALID_COUNTER_DIAGNOSTIC_OUTPUTS',error=traceback.format_exc());save();raise
 finally:r['elapsed_seconds']=time.time()-r['started_at'];save()
if __name__=='__main__':main()

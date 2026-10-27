"""Independent CPU-only artifact, binary, schedule and numeric validation."""
import collections,copy,hashlib,importlib.util,json,math,pathlib,re,statistics,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
COL=next((ROOT/'results/proj-rsi/scheduler/artifacts/research.cursor.matchedgeometry.gpu.01').glob('collection-*'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=load(ROOT/'proj_rsi/candidates/smallm_matched_geometry_trial/support.py','cursor_review_support');M=S.driver()
def validate(d,journal):
 assert d['status']=='VALID_MATCHED_CURSOR_MECHANISM_AND_ACTUAL_POLICY_REFERENCES_NO_OFFICIAL_ADMISSION'
 assert d['scope']==list(S.SHAPES) and d['source_pins']=={k:list(v) for k,v in M.PINS.items()}
 allraw=[];answer={};schedule=S.schedule()
 for shape,e in d['shapes'].items():
  assert e['status']=='VALID_FIXED_AND_ACTUAL_POLICY_REFERENCES' and e['graph_cleanup']=='DRAINED_RESET_ALL_SEVEN'
  assert not e.get('counter_failure')
  layer,m,n,k,ks=S.SHAPES[shape];assert e['config']==dict(layer=layer,m=m,n=n,k=k) and e['fixed_KS']==ks
  assert set(e['validation'])==set(S.LABELS) and all(v['full_K_basis_FP64'].startswith('ok ') for v in e['validation'].values())
  for lab in ('parent5','cursor5','parent6','cursor6'):
   a=e['fixed_attrs'][lab];stage=int(lab[-1]);assert a['KS']==ks and a['name']=='t32s%dhze'%stage and a['regs_local_shared']==[168,0,83080 if stage==5 else 99488]
   assert e['graph_nodes'][lab]==['void rq<32, %d, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)'%stage]
  assert set(e['graph_nodes'])==set(S.LABELS) and all(len(v)==1 and 'void rq<' in v[0] for v in e['graph_nodes'].values())
  t=e['timing'];raw=t['raw_calls'];allraw.extend(raw);cursor=0;last={};f1=collections.defaultdict(list);pairs=collections.OrderedDict()
  for mode,label,seed,key in schedule:
   if mode=='terminal':
    terminal=t['strict_terminal_checks'][label];assert len(terminal)==7 and {z['label'] for z in terminal}==set(S.LABELS)
    for z in terminal:assert z['seed']==last[z['label']] and z['actual_final_output_before_any_replay']=='PASS' and 'probe rows exact' in z['oracle']
    continue
   row=raw[cursor];cursor+=1
   assert {z:row[z] for z in ['shape','mode','label','input_seed','block']}==dict(shape=shape,mode=mode,label=label,input_seed=seed,block=key)
   assert math.isfinite(row['us']) and row['us']>0
   if mode!='warmup':last[label]=seed
   if mode=='F1':f1[key].append(row['us'])
   if mode=='paired':pairs.setdefault(seed,{})[label]=row['us']
  assert cursor==len(raw)==462 and dict(f1)==t['F1_samples_us'] and list(pairs.values())==t['paired_samples_us']
  med={k:statistics.median(v) for k,v in f1.items()};assert med==t['F1_medians_us']
  assert len(e['mutations'])==35 and {(x['label'],x['kind']) for x in e['mutations']}=={(a,b) for a in S.LABELS for b in ['normal','zero','fresh_normal','outliers','probes']} and all(x['PASS'] is True for x in e['mutations'])
  answer[shape]={'picks':{role:{'name':p['rq_name'],'KS':p['rq']['KS'],'index':p['rq']['i']} for role,p in e['picks'].items()},'comparisons':{}}
  for name,(parent,new) in S.COMPARISONS.items():
   gains=[row[parent]-row[new] for row in pairs.values()];gain=statistics.median(gains);f1gain=min(med[parent+'_before'],med[parent+'_after'])-med[new];ratio=statistics.median(row[parent]/row[new] for row in pairs.values());wtl=[sum(g>0 for g in gains),sum(g==0 for g in gains),sum(g<0 for g in gains)]
   expect={'control':parent,'candidate':new,'F1_conservative_gain_us':f1gain,'paired_median_gain_us':gain,'paired_median_control_over_candidate_ratio':ratio,'paired_all_signed_gains_us':gains,'paired_win_tie_loss':wtl,'gain_threshold_us':0}
   assert t['comparisons'][name]==expect
   mad=statistics.median(abs(g-gain) for g in gains)
   ordergroups={str(i):statistics.median(gains[j] for j in range(i,28,14)) for i in range(14)}
   answer[shape]['comparisons'][name]=expect|{'MAD_us':mad,'range_us':[min(gains),max(gains)],'first_half_median_us':statistics.median(gains[:14]),'last_half_median_us':statistics.median(gains[14:]),'order_strata_median_us':ordergroups}
 assert allraw==journal and len(allraw)==1386
 return answer

def main():
 inv=json.loads((COL/'inventory.json').read_text());assert not inv['missing'] and inv['terminal_exit']==0
 for rel,h in inv['files'].items():assert sha(COL/rel)==h,rel
 for rel,h in inv['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
 clean=inv['cleanup_observation'];assert not clean['cleanup_failed'] and not clean['container_exists'] and clean['gpu_memory_mib']==0 and not any(clean[k] for k in ['compute_pids','guard_pids','containers'])
 manifest=S.verify();M.verify();assert sha(ROOT/'proj_rsi/tools/smallm_matched_geometry_lab.py')=='c6219b3b2114f34bf23a21d047aefcfa6cc23f46df6d01ae4bea45cfeeb41fc3';assert sha(S.HERE/'manifest.json')=='fe5282b9872db72582bd737ac63fa7dc86da3e0db1399f25ca1e3db321aa23d0'
 p=next(COL.rglob('result.json'));run=p.parent;d=json.loads(p.read_text());journal=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()];comparisons=validate(d,journal)
 jit={}
 for role,record in d['JIT'].items():
  assert record['gate']=='before prepare / first candidate math'
  for filename,key in [(role+'.actual.so','so_sha256'),(role+'.sass','SASS_sha256'),(role+'.rq32.cu','source_sha256')]:assert sha(run/filename)==record[key]
  for flag,suffix in [('sass','sass'),('resource-usage','resources')]:
   actual=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-'+flag,str(run/record['archived_so'])],text=True)
   assert actual==(run/(role+'.'+suffix)).read_text()
  counts=M.ops((run/(role+'.sass')).read_text());offline=M.ops((ROOT/'proj_rsi/candidates/smallm_increment'/('after.sass' if role=='increment' else 'before.sass')).read_text())
  resources=(run/(role+'.resources')).read_text()
  for symbol,expect in offline.items():
   assert counts[symbol]==expect
   block=resources.split('Function '+symbol)[-1].split('Function ')[0]
   if block==resources:block=resources.split(symbol)[-1].split('Function')[0]
   assert re.search(r'REG\s*[:=]\s*168\b',block) and re.search(r'LOCAL\s*[:=]\s*0\b',block)
  jit[role]={'binary_redump_exact':True,'all_opcode_resources_regated':True,'functions':len(offline),'so_sha256':record['so_sha256'],'generated_source_sha256':record['source_sha256']}
 assert d['JIT']['pw36']['so_sha256']==d['JIT']['rs32']['so_sha256']
 assert sha(ROOT/'proj_rsi/scheduling/reviews/cursor-matched-geometry.md')==d['review_sha256']
 negatives=[]
 for kind in ['declared_gain','terminal_seed','raw_call_identity']:
  bad=copy.deepcopy(d)
  if kind=='declared_gain':bad['shapes']['qkv32']['timing']['comparisons']['ST5']['paired_median_gain_us']+=1
  elif kind=='terminal_seed':bad['shapes']['qkv32']['timing']['strict_terminal_checks']['F1'][0]['seed']+=1
  else:bad['shapes']['qkv32']['timing']['raw_calls'][0]['label']='cursor5'
  try:validate(bad,journal)
  except AssertionError:negatives.append(kind)
  else:raise AssertionError('negative accepted '+kind)
 proof=M.contract_source_check();assert proof==d['counter_source_bindings']
 baseline=d['counter_prepare_baseline'];assert len(baseline)==71
 result={'job_id':'research.cursor.matchedgeometry.gpu.01','disposition':'VALID_NO_GAIN','disposition_scope':'Valid mixed matched-component observations; no consistent cursor gain or actual-policy integration admission. Preserve every positive and loss.','source_driver_sha256':sha(ROOT/'proj_rsi/tools/smallm_matched_geometry_lab.py'),'trial_manifest_sha256':sha(S.HERE/'manifest.json'),'collection':str(COL.relative_to(ROOT)),'counts':{'inventory_files':len(inv['files']),'trial_pins':len(manifest['files']),'dispatch_pins':len(inv['dispatch']['source_pins']),'raw_calls':len(journal),'terminal_oracles':42,'mutations':105,'full_K_basis_FP64_arm_shapes':21,'JIT_roles_binary_redumped':3,'counter_baseline_views':len(baseline),'counter_unique_storage':len({tuple(v['storage']) for v in baseline})},'cleanup':clean,'comparisons':comparisons,'JIT':jit,'negative_controls_rejected':negatives,'evidence':{'inventory_sha256':sha(COL/'inventory.json'),'result_sha256':sha(p),'raw_sha256':sha(run/'raw.jsonl')},'limits':['Fixed arms isolate cursor at ST5/6 and common KS; actual policies can change stage/flags.','All fixed F1 point effects are negative while some paired point effects are positive; one batch does not resolve small effects.','Warm-cache prebuild and _T0 reset are explicit local research; not cold deployment preparation.','Counters verified by source-bound runtime assertions with baseline metadata; no independent final full-storage dumps are present.','No PW38/PW39/current scored leader comparison and no H200/official performance claim.']}
 (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'disposition':result['disposition'],'counts':result['counts'],'comparisons':{sh:{name:[v['paired_median_gain_us'],v['F1_conservative_gain_us']] for name,v in e['comparisons'].items()} for sh,e in comparisons.items()}},indent=2))
if __name__=='__main__':main()

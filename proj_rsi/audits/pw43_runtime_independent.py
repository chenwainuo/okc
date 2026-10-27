"""Read-only collected PW43 evidence validator; no production import or GPU."""
import collections,hashlib,importlib.util,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/proj-rsi/scheduler/artifacts/pw43.stack.fullscope.01/collection-1793084970479114265'
OUT=BASE/'logs/scheduler/pw43-stack-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('pure_pw43_lab',ROOT/'proj_rsi/tools/pw43_stack_lab.py');lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
 lab.verify();inventory=json.loads((BASE/'inventory.json').read_text());d=json.loads((OUT/'result.json').read_text())
 for path,pin in inventory['files'].items():assert sha(BASE/path)==pin,path
 assert inventory['missing']==[] and inventory['terminal_exit']==0
 clean=inventory['cleanup_observation'];assert clean['cleanup_failed'] is False and clean['container_exists'] is False and clean['containers']==clean['guard_pids']==clean['compute_pids']==[] and clean['gpu_memory_mib']==0
 assert d['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED' and d['phase']=='complete' and not d['official_score_claim'] and d['graph_cleanup']==[] and d['jit_errors']==[]
 assert d['sources']==lab.PINS and len(d['checks'])==222 and len(d['shapes'])==30 and len(d['raw'])==4350
 labels=set(lab.LABELS);expected={(m,l) for m in lab.MS for l in lab.LAYERS}
 for label in labels:
  checks=[c for c in d['checks'] if c['label']==label];assert len(checks)==74
  assert {(c['case']['m'],c['case']['layer']) for c in checks if not c['official']}==expected
  assert [c['case'] for c in checks if c['official']]==lab.OFFICIAL
  assert all(c['counter_contract']=='PASS' and c['oracle'] for c in checks)
  p=d['prepare'][label];assert 0<p['wall_seconds']<240 and p['import_clock_elapsed']<240 and not p['budget_clock_reset'] and not p['serving_CPU_fallback'] and p['build_threads_joined']==2
  device=d['device'][label];assert device['before_prepare'] is True and device['SM_count']>=128 and device['compute_capability']==[9,0]
  lab.O.gate(d['Lt_prepare'][label]);assert d['Lt_prepare'][label]['added_GPU_calls']==d['Lt_prepare'][label]['added_clock_queries']==0
 # Re-run actual collected JIT gates, not a resource-summary assertion.
 assert len(d['jit'])==15
 rqsets=collections.defaultdict(set)
 for j in d['jit']:
  stem=OUT/(j['label']+'.'+j['name']);cu=Path(str(stem)+'.cu');sass=Path(str(stem)+'.sass');res=Path(str(stem)+'.resources')
  assert sha(cu)==j['source_sha256'] and sha(sass)==j['SASS_sha256'] and j['before_return_to_prepare_math']
  if j['RQ_gate'] is not None:
   gate=lab.actual_rq_gate(sass.read_text(),res.read_text(),j['RQ_gate']['flags']==[2,3,65538,65539]);gate['full_RQ']=lab.full_rq_gate(cu.read_text(),sass.read_text(),res.read_text());assert json.dumps(gate,sort_keys=True)==json.dumps(j['RQ_gate'],sort_keys=True)
   rqsets[j['label']].update(tuple(v) for v in gate['full_RQ']['args'])
 assert {k:len(v) for k,v in rqsets.items()}==d['actual_RQ_functions']=={'rs32':17,'pw38':19,'pw43':19}
 journal=[json.loads(line) for line in (OUT/'raw.jsonl').read_text().splitlines()];assert journal==d['raw']
 per_shape=[];loads=collections.defaultdict(lambda:collections.defaultdict(float));lt=[]
 for name,e in d['shapes'].items():
  assert e['graphs']=='RESET_AFTER_DRAIN' and set(e['picks'])==labels
  assert len(e['terminal_oracles'])==6 and {(o['label'],o['phase'],o['seed']) for o in e['terminal_oracles']}=={(l,p,s) for l in labels for p,s in [('F1',9010),('paired',9049)]}
  assert all(o['oracle'] for o in e['terminal_oracles'])
  assert len(e['mutations'])==12 and {(m['label'],m['kind']) for m in e['mutations']}=={(l,k) for l in labels for k in ['zero','outliers','all_probes','fresh_normal']}
  assert all(m['oracle'] and m['output_poison_overwritten'] for m in e['mutations'])
  raw=[v for v in d['raw'] if v['shape']==name];assert len(raw)==145 and all(math.isfinite(v['us']) and v['us']>0 for v in raw)
  f1expected=[(l,9000+i) for l in ('rs32','pw38','pw43','pw38','rs32') for i in range(11)]
  assert [(r['label'],r['seed']) for r in raw if r['mode']=='F1']==f1expected
  orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
  assert [(r['label'],r['seed']) for r in raw if r['mode']=='paired']==[(lab.LABELS[j],9020+i) for i in range(30) for j in orders[i%6]]
  for l in ('rs32','pw38'):
   samples=[r['us'] for r in raw if r['mode']=='F1' and r['label']==l]
   assert e['F1_medians_us'][l+'_before']==statistics.median(samples[:11])
   assert e['F1_medians_us'][l+'_after']==statistics.median(samples[11:])
  assert e['F1_medians_us']['pw43']==statistics.median(r['us'] for r in raw if r['mode']=='F1' and r['label']=='pw43')
  pairs=e['paired_all_samples'];assert len(pairs)==30
  for i,row in enumerate(pairs):
   actual={v['label']:v['us'] for v in raw if v['mode']=='paired' and v['seed']==9020+i};assert actual==row and set(row)==labels
  before=[v['us'] for v in raw if v['mode']=='F1' and v['label']=='pw38'][:11];after=[v['us'] for v in raw if v['mode']=='F1' and v['label']=='pw38'][11:]
  cand=[v['us'] for v in raw if v['mode']=='F1' and v['label']=='pw43'];assert len(before)==len(after)==len(cand)==11
  f1=(statistics.median(before)+statistics.median(after))/2-statistics.median(cand)
  for l in labels-{'pw43'}:
   assert statistics.median(r[l]-r['pw43'] for r in pairs)==e['paired_control_minus_pw43_us'][l]
   assert statistics.median(r[l]/r['pw43'] for r in pairs)==e['paired_control_over_pw43_ratio'][l]
  diff=e['paired_control_minus_pw43_us']['pw38'];per_shape.append({'shape':name,'paired_pw38_minus_pw43_us':diff,'F1_bracket_pw38_minus_pw43_us':f1,'picks':e['picks']})
  load=name.split('/')[0]
  for l in labels:loads['paired_'+l][load]+=statistics.median(r[l] for r in pairs)
  loads['F1_pw38'][load]+=(statistics.median(before)+statistics.median(after))/2;loads['F1_pw43'][load]+=statistics.median(cand)
 for label,a in d['Lt_prepare'].items():
  for s in a['searches']:
   if label!='pw43':continue
   c=s['confirmations'][0];rec=c['record'];lt.append({'layer':s['layer'],'remaining_seconds':s['deadlines'][0]['remaining_seconds'],'returned':s['returned'],'status':rec['status'],'reason':rec['reason'],'codes':c['codes'],'rows':len(rec['rows']),'deltas_ms':rec['deltas'],'timers':len(s['timers'])})
 # Local estimator only: exact score algebra with frozen same-job SGLang vector.
 basis=json.loads((ROOT/'proj_rsi/audits/pw38-next-roi-opportunities.json').read_text())['authority']['same_job_SGLang_component_sums_us']
 B={k.replace(' ',''):v for k,v in basis.items()}
 def score(c):return 4830.8/math.exp(statistics.mean(math.log(B[k]/c[k]) for k in B))
 scores={k:score(v) for k,v in loads.items()};delta={mode:scores[mode+'_pw38']-scores[mode+'_pw43'] for mode in ['paired','F1']}
 result={'state':'PASS_SCOPED_LOCAL_VALIDATION_REGRESSION_NOT_PROMOTION','inventory_sha256':sha(BASE/'inventory.json'),'result_sha256':sha(OUT/'result.json'),'inventory_files':len(inventory['files']),'normal_checks':195,'task_official_style_checks':27,'terminal_oracles':180,'mutations':360,'raw_samples':4350,'actual_RQ_functions':d['actual_RQ_functions'],'prepare':d['prepare'],'Lt_actual_confirmation':lt,'six_load_component_sums_us':loads,'conditional_local_score_estimates_us':scores,'pw38_minus_pw43_local_score_estimate_us':delta,'all_shape_signed_contrasts':per_shape,'limits':['One full prepare per source, not repeat-prepare selection frequency/regret evidence.','External Lt kernels, graph node counts and SASS explicitly UNPROBED; reviewed normal-serving scope permits this library boundary.','Loaded SO hashes recorded by runtime; SO bytes are not collected, so independent rehash covers CU/SASS/resource artifacts, not SO bytes.','Normal allK/full-basis coverage is exactly the frozen lab checks, not a new exhaustive numerical oracle.','Runtime counter/immutability passes are source-bound assertions in the frozen lab; raw before/after counter arrays are not retained in this result, so this audit cannot separately reconstruct each physical transition.', 'No official H200 score/gain or family rejection inferred.'],'lessons':[{'id':'pw43-weighted-deadline-fallback-actual','scope':'PW43v1 one H100 fullprepare, fourBF16M256','observation':'Only qkv confirmation completed; laterthree searches started with expired weighted deadlines and nofinalists, triggering torch defaults and later poolerRQ selection.','next_test':'Account descriptor initialization/reference fixture and search admission costs within each slice before proposing another selector; compare repeated actual policies on disjoint evaluation.','transfer':'Bounded tuning changes fallback policy as well as winner selection; earlier local wins do not justify forcing a candidate.'}],'GPU_calls_by_this_validator':0}
 out=ROOT/'proj_rsi/audits/pw43-runtime-independent.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('all_shape_signed_contrasts','six_load_component_sums_us','lessons','limits')},indent=2))
if __name__=='__main__':main()

"""Read immutable two-cell RQ receipt. No launch, new gate or source mutation."""
from pathlib import Path
import argparse,hashlib,json,statistics
D=Path(__file__).resolve().parent;ROOT=D.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(path,obj):
 text=json.dumps(obj,indent=2,allow_nan=False)+'\n'
 if path.exists():assert path.read_text()==text,'frozen output differs; use a NEW analysis version/path'
 else:path.write_text(text)
def classify(a,inventory):
 state=a.get('state','UNKNOWN');error=a.get('error','');phase=a.get('phases',[{}])[-1].get('phase')
 if state=='VALID_TWO_RQ32_SCORED_SELECTED_DESCRIPTOR_OFFSET_CELLS_ROOT_REVIEW':return {'kind':'VALID_MEASUREMENT','phase':phase,'causal_limit':'No failure; effects are fixed scored-selected geometry, not fresh selector or official transfer.'}
 if state=='INVALID_CLEANUP' or inventory.get('cleanup_observation',{}).get('cleanup_failed'):return {'kind':'INVALID_CLEANUP','phase':phase,'causal_limit':'Do not admit performance; retain raw results and runtime cleanup evidence.'}
 if any(x in error for x in['pin drift','review','FileNotFoundError','ModuleNotFoundError']):kind='INVALID_SETUP_BINDING'
 elif phase=='torch_and_exact_two_function_compile':kind='INVALID_SETUP_COMPILE_OR_RESOURCE'
 elif 'boundarybasis'in error:kind='INVALID_CHANGED_DESCRIPTOR_BOUNDARY'
 elif any(x in error for x in['RQactive_reset0','physicalguards','weight/packed/escape mutated','currentXchanged','guard']):kind='INVALID_MUTATION_OR_COUNTER_PROTOCOL'
 elif any(x in error for x in['launch','CUDA error','capture']):kind='INVALID_RUNTIME_OR_CAPTURE'
 elif 'inner120s'in error:kind='INCOMPLETE_BUDGET'
 else:kind='INVALID_NUMERICAL_OR_OTHER_RETAIN_EXACT_ERROR'
 return {'kind':kind,'phase':phase,'error':error,'causal_limit':'Conditional classification from frozen phase/error only; not unique kernel-cause attribution or a family rejection.'}
def analyze(result):
 a=json.loads(result.read_text());invpath=next((p/'inventory.json'for p in result.parents if(p/'inventory.json').exists()),None);inv=json.loads(invpath.read_text())if invpath else{}
 report={'observed_state':a.get('state'),'result_path':str(result.relative_to(ROOT)),'result_sha256':sha(result),'failure_classification':classify(a,inv),'scope':a.get('scope'),'cost':a.get('cost'),'device':a.get('device'),'build':a.get('build'),'actual_attributes':a.get('attributes'),'cells':{},'cleanup':a.get('cleanup'),'external_cleanup':inv.get('cleanup_observation'),'terminal_exit':inv.get('terminal_exit'),'source_claim':'scored9916selectedNT32ST6FL35 fixed source contrast only, no freshprepare/policy/full30/officialgain claim','new_gates':False}
 for name,cell in a.get('cells',{}).items():
  raw=cell.get('raw',[]);F=[x for x in raw if x['mode']=='F1'];P=[x for x in raw if x['mode']=='paired'];pairs=[]
  for rep in sorted({x['pair']for x in P}):
   block=[x for x in P if x['pair']==rep];arms={x['arm']:x for x in block}
   if set(arms)=={'control','offset'}:pairs.append({'pair':rep,'seed':arms['control']['seed'],'same_seed':arms['control']['seed']==arms['offset']['seed'],'control_us':arms['control']['us'],'offset_us':arms['offset']['us'],'signed_control_minus_offset_us':arms['control']['us']-arms['offset']['us'],'control_over_offset':arms['control']['us']/arms['offset']['us']})
  f1={str(b):statistics.median(x['us']for x in F if x['block']==b)for b in sorted({x['block']for x in F})}
  c={'state':cell.get('state'),'config':cell.get('config'),'KS':cell.get('KS'),'KCS':cell.get('KCS'),'geometry':cell.get('geometry'),'raw_calls':len(raw),'warmup_calls':sum(x['mode']=='warmup'for x in raw),'F1_calls':len(F),'paired_calls':len(P),'F1_single_fixture_seeds':sorted({x['seed']for x in F}),'F1_block_medians_us':f1,'per_pair':pairs,'own_terminal_records':cell.get('terminals',[]),'validation_records':cell.get('validation',[]),'mutation_records':cell.get('mutations',[]),'boundary_columns':cell.get('boundary_columns'),'cleanup':cell.get('cleanup')}
  if pairs:c['paired']={'complete_pairs':len(pairs),'signed_deltas_us':[x['signed_control_minus_offset_us']for x in pairs],'median_control_minus_offset_us':statistics.median(x['signed_control_minus_offset_us']for x in pairs),'median_control_over_offset':statistics.median(x['control_over_offset']for x in pairs),'offset_wins':sum(x['signed_control_minus_offset_us']>0 for x in pairs),'offset_losses':sum(x['signed_control_minus_offset_us']<0 for x in pairs),'ties':sum(x['signed_control_minus_offset_us']==0 for x in pairs),'control_first':sum(next(x for x in P if x['pair']==p['pair'])['arm']=='control'for p in pairs),'offset_first':sum(next(x for x in P if x['pair']==p['pair'])['arm']=='offset'for p in pairs)}
  if set(f1)=={'0','1','2'}:c['F1']={'control_before_minus_offset_us':f1['0']-f1['1'],'control_after_minus_offset_us':f1['2']-f1['1'],'endpoint_average_control_minus_offset_us':(f1['0']+f1['2'])/2-f1['1']}
  report['cells'][name]=c
 journal=result.parent/'raw.jsonl'
 if journal.exists():
  actual=[json.loads(x)for x in journal.read_text().splitlines()if x.strip()];expected=[dict(row,layer=name)for name,cell in a.get('cells',{}).items()for row in cell.get('raw',[])];report['journal']={'sha256':sha(journal),'rows':len(actual),'matches_result':actual==expected}
 report['evidence_files']={str(result.relative_to(ROOT)):sha(result)}
 if invpath:report['evidence_files'][str(invpath.relative_to(ROOT))]=sha(invpath)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,default=D);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);r=analyze(args.result.resolve());freeze(out/'analysis.json',r)
 print(json.dumps({'state':r['observed_state'],'failure':r['failure_classification'],'cells':{k:{x:v.get(x)for x in['state','paired','F1']}for k,v in r['cells'].items()},'cost':r['cost']},indent=2))

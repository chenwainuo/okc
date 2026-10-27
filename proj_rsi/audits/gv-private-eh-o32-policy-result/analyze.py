"""CPU summary using existing full-policy contrast/load-sum estimator definitions."""
import json,math,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 path=next((ROOT/'results/proj-rsi/scheduler/artifacts/gv.privateeh.o32.fullpolicy.01').glob('*/logs/scheduler/*/run/result.json'))
 d=json.loads(path.read_text());collection=path.parents[4];inventory=json.loads((collection/'inventory.json').read_text());c,t=tuple(d['sources'])
 assert d['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED'and d['phase']=='complete'
 assert not d['jit_errors']and not d['graph_cleanup']and d['official_score_claim']is False
 assert len(d['checks'])==148 and len(d['raw'])==2910 and len(d['shapes'])==30
 assert [json.loads(line)for line in path.with_name('raw.jsonl').read_text().splitlines()]==d['raw']
 for rel,pin in inventory['files'].items():assert sha(collection/rel)==pin,rel
 cleanup=inventory['cleanup_observation'];assert inventory['terminal_exit']==0 and cleanup['cleanup_failed']is False
 assert not cleanup['compute_pids']and not cleanup['guard_pids']and not cleanup['container_exists']and cleanup['gpu_memory_mib']==0
 assert all(r['counter_contract']=='PASS'and r['oracle']for r in d['checks'])
 out={'state':'VALID_EH_O32_VS_LOCAL_EH_INCREMENTAL_FULL30_POINT','result':str(path.relative_to(ROOT)),'result_sha256':sha(path),'sources':d['sources'],'shapes':{},'six_load_sums':{m:{a:{}for a in [c,t]}for m in ['paired','F1']},'checks':148,'raw':2910,'journal_exact_match':True,'terminals':0,'mutations':0,'prepare':d['prepare'],'actual_o32_basis':d['actual_o32_basis'],'cleanup':cleanup,'inventory_sha256':sha(collection/'inventory.json'),'collected_files_rehashed':len(inventory['files'])}
 for shape,e in d['shapes'].items():
  rows=[r for r in d['raw']if r['shape']==shape];expected=[]
  for arm in [c,t,c]:expected.extend({'label':arm,'mode':'F1','seed':9000+i}for i in range(11))
  for i in range(32):expected.extend({'label':arm,'mode':'paired','seed':9020+i}for arm in ([c,t]if i%2==0 else[t,c]))
  assert [{k:r[k]for k in ['label','mode','seed']}for r in rows]==expected
  assert all(math.isfinite(r['us'])and r['us']>0 for r in rows)
  pairs=[{r['label']:r['us']for r in rows[33+2*i:35+2*i]}for i in range(32)];assert pairs==e['paired_all_samples']
  fm={c:(statistics.median(r['us']for r in rows[:11])+statistics.median(r['us']for r in rows[22:33]))/2,t:statistics.median(r['us']for r in rows[11:22])}
  differences=[r[c]-r[t]for r in pairs];gain=statistics.median(differences)
  assert gain==e['paired_control_minus_candidate_us'][c]
  assert e['graphs']=='RESET_AFTER_DRAIN'and len(e['terminal_oracles'])==4 and len(e['mutations'])==8
  assert {(r['label'],r['phase'],r['seed'])for r in e['terminal_oracles']}=={(a,phase,seed)for a in[c,t]for phase,seed in[('F1',9010),('paired',9051)]}
  assert all(r['oracle']for r in e['terminal_oracles'])and all(r['oracle']and r['output_poison_overwritten']for r in e['mutations'])
  out['terminals']+=4;out['mutations']+=8
  out['shapes'][shape]={'paired_gain_us':gain,'F1_gain_us':fm[c]-fm[t],'paired_all_signed_differences_us':differences,'paired_wins_ties_losses':[sum(v>0 for v in differences),sum(v==0 for v in differences),sum(v<0 for v in differences)],'F1_arm_us':fm,'picks':e['picks']}
  load=shape.split('/')[0]
  for mode,values in [('F1',fm),('paired',{a:statistics.median(r[a]for r in pairs)for a in[c,t]})]:
   for a,v in values.items():out['six_load_sums'][mode][a][load]=out['six_load_sums'][mode][a].get(load,0)+v
 out['actual_policy_geometric_load_speedup']={m:math.exp(statistics.mean(math.log(v[c][l]/v[t][l])for l in v[c]))for m,v in out['six_load_sums'].items()}
 out['signs']={m:{'wins':sum(e[key]>0 for e in out['shapes'].values()),'ties':sum(e[key]==0 for e in out['shapes'].values()),'losses':sum(e[key]<0 for e in out['shapes'].values())}for m,key in [('paired','paired_gain_us'),('F1','F1_gain_us')]}
 out['selection_changes']=[{'shape':shape,'control':e['picks'][c],'candidate':e['picks'][t]}for shape,e in out['shapes'].items()if e['picks'][c]!=e['picks'][t]]
 out['limits']=['Existing F1 endpoint-average and sums of per-shape paired medians; not official score forecast.','All signed losses and opposing estimators retained; no minimum effect/significance gate.','Fresh prepare selections/native placements differ; no isolated private-epilogue causal package claim.','Source/state/counter assertions are inherited and successful; actual numerical tensor bytes are not archived for offline reconstruction.']
 (HERE/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k]for k in ['state','actual_policy_geometric_load_speedup','signs','collected_files_rehashed','terminals','mutations']},indent=2))
if __name__=='__main__':main()

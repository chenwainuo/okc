"""Read a collected result only; no GPU, queue or source mutation."""
from pathlib import Path
import argparse,hashlib,json,statistics,itertools
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.result.read_text());valid=r['state']=='VALID_ONE_UNPOOLER_COMPACT256_HALF_CAPACITY_SCREEN_ROOT_REVIEW'
o={'state':'ACTUAL_VALID'if valid else'ACTUAL_INVALID_NO_PERFORMANCE_ADMISSION','result_path':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'observed_state':r['state'],'attributes':r.get('attributes'),'arm_eligibility':r.get('arm_eligibility'),'cost':r.get('cost'),'error':r.get('error'),'performance':None}
if valid:
 labels=r['measured_arms'];assert 'control'in labels and 'compact'in labels
 e=r['cells']['unpooler'];rows=e['raw'];timed=[x for x in rows if x['mode']!='warmup'];assert all(x['arm']in labels for x in rows)
 assert len({x['seed']for x in rows if x['mode']=='F1'})==1
 pairs=[{x['arm']:x for x in rows if x['mode']=='paired'and x['pair']==i}for i in range(12)]
 assert all(set(x)==set(labels)and len({v['seed']for v in x.values()})==1 for x in pairs)
 contrasts={}
 for left,right in itertools.combinations(labels,2):
  ds=[x[left]['us']-x[right]['us']for x in pairs]
  lead=sum([x['arm']for x in rows if x['mode']=='paired'and x['pair']==i].index(left)<[x['arm']for x in rows if x['mode']=='paired'and x['pair']==i].index(right)for i in range(12));assert lead==6
  contrasts[left+'_minus_'+right]={'signed_pair_savings_us':ds,'median_saving_us':statistics.median(ds),'right_wins_losses_ties':[sum(d>0 for d in ds),sum(d<0 for d in ds),sum(d==0 for d in ds)],'lead_follow':[lead,12-lead],'F1_saving_us':e['contrasts'][left+'_minus_'+right]['F1_bracket_average_saving_us']}
 for arm,x in r['arm_eligibility'].items():
  if arm not in labels:assert x['performance_if_skipped']is None and x['runtime_resident_capacity']<x['required_resident_CTAs']
 assert e['cleanup']=='DRAINED_RESET'
 o['performance']={'contrasts':contrasts,'F1_blocks':e['F1_medians_us'],'own_terminal_records':len(e['terminals']),'validation_records':len(e['validation']),'mutation_records':len(e['mutations']),'timed_calls':len(timed),'measured_arms':labels,'limits':'Capacity is CUDA occupancy query evidence; no twofold mathWG/speed claim or causal individual-stall attribution. Original HALF skips have unavailable performance.'}
s=json.dumps(o,indent=2)+'\n';assert not a.output.exists()or a.output.read_text()==s;a.output.write_text(s);print(s)

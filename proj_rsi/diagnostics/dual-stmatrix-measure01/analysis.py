"""Independent, reproducible CPU audit of every frozen DP/DS diagnostic record."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as S

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in (HERE,*HERE.parents) if (p/'proj_rsi/tools/dual_stmatrix_measure.py').is_file())
BASE=ROOT/'proj_rsi/candidates'
RAW=HERE/'dual-stmatrix-measure01.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(RAW)=='aa9c14d22deb9110a50e36e12b056658d1ba91b305ab7645ec052e0f47311378'
d=json.loads(RAW.read_text());raw=d['raw'];plan=d['plan']
assert d['status']=='VALID_FIXED_SOURCE_MEASUREMENT_REVIEW_REQUIRED' and d['phase']=='COMPLETE'
assert d['driver_sha256']==sha(ROOT/'proj_rsi/tools/dual_stmatrix_measure.py')=='4812bbcfa2314db3db9ca481dcfa052daabb54e6d332c02c2866705dbc3d4293'
assert sha(HERE/'plan.json')=='2643a6043239623bc6a67cfcf8e59753c2a9faa38afe3a245f071673284eb997'
assert plan==json.loads((HERE/'plan.json').read_text())['plan']
assert len(raw)==350 and sum(not r['warmup'] for r in raw)==320
assert d['graph_cleanup']=='RESET_BOTH_AFTER_DRAIN'
assert d['no_parent_prepare'] and d['no_incumbent_or_official_claim'] and d['weights_pack_escape_immutable_counters_zero']

def canonical(text,token):
 tables=[x for x in re.split(r'\n\s*Function : ',text)[1:] if token in x.splitlines()[0]];assert len(tables)==1
 return [(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',tables[0])]

for label,folder,token in [('dp','operand_reverse_dual_producer','rq_px_dp'),('ds','operand_reverse_dual_stmatrix','rq_px_dp_stmx')]:
 here=BASE/folder;manifest=json.loads((here/'manifest.json').read_text())
 assert sha(here/'manifest.json')==d['frozen'][label]['manifest_sha256']
 assert all(sha(here/name)==expected for name,expected in manifest['files'].items())
 assert d['frozen'][label]['source_sha256']==manifest['source_sha256']
 j=d['actual_JIT'][label];artifact=HERE/Path(j['artifact']).name
 assert sha(artifact)==j['sha256'] and j['exact_frozen_PC_instruction_identity_main_and_pack']
 for t in (token,'rx_pack'):assert canonical(artifact.read_text(),t)==canonical((here/'sass.txt').read_text(),t)
 assert j['A4']['status']=='PASS_ALL_ACTUAL_MAIN_BINARIES'
 gate=j['store_proxy_both_owner_barriers']['binaries'];assert len(gate)==1
 assert gate[0]['STSM']==(4 if label=='ds' else 0) and gate[0]['scalar_STS']==(16 if label=='ds' else 32)
 assert gate[0]['owner_barrier_predicates']['complementary_bar3_bar4']
 a=d['attrs'][label]
 assert a['regs']==122 and a['local']==0 and a['smem']==103600 and a['occupancy']==1 and a['max_threads']>=512
 assert ((a['regs']+7)//8)*8*512==65536 and a['pack_regs']==22 and a['pack_local']==0

# Reconstruct ALL intended calls, including warmups, input seeds and actual
# previous-policy run length across per-mode validators (which finish DS once).
expected=[]
for mi,mode in enumerate(('steady_before','paired','steady_after')):
 warm=('dp','ds') if mi%2==0 else ('ds','dp')
 for label in warm:
  for i in range(5):expected.append(dict(mode=mode,label=label,warmup=True,warmup_index=i))
 if mode=='paired':
  for p in plan['paired_orders']:
   for pos,label in enumerate(p['order']):expected.append(dict(mode=mode,label=label,warmup=False,pair=p['pair'],swapped_block=p['swapped_block'],within_swapped_block=p['within_swapped_block'],position=pos,input_seed=p['input_seed']))
 else:
  seed0=120000 if mode=='steady_before' else 130000
  for b,label in enumerate(plan[mode]):
   for pos in range(32):expected.append(dict(mode=mode,label=label,warmup=False,block=b,position=pos,input_seed=seed0+32*b+pos))
assert len(expected)==350
previous_timed=None;last_mode=None;previous_end=-1;last_seeds={}
for row,e in zip(raw,expected):
 assert all(row[k]==v for k,v in e.items()),(row,e)
 if row['mode']!=last_mode:previous='ds';run=1;last_mode=row['mode']
 assert row['previous_case']==previous and row['previous_timed_case']==previous_timed
 assert row['preceding_same_variant_run_length']==(run if previous==row['label'] else 0)
 assert previous_end<=row['host_start_monotonic_ns']<=row['host_before_start_event_monotonic_ns']<=row['host_end_monotonic_ns']
 assert math.isfinite(row['us']) and row['us']>0
 previous_end=row['host_end_monotonic_ns'];run=run+1 if previous==row['label'] else 1;previous=row['label']
 if not row['warmup']:previous_timed=row['label'];last_seeds[(row['mode'],row['label'])]=row['input_seed']

# Every post-mode final timed output checked before any validation replay.
assert set(d['validation'])=={'initial','steady_before','paired','steady_after'}
assert sum(map(len,d['validation'].values()))==62
for mode,checks in d['validation'].items():
 assert len(checks)==(14 if mode=='initial' else 16)
 for label in ('dp','ds'):
  a=[x for x in checks if x['label']==label]
  bits=[x for x in a if 'storage_all65536bits' in x];assert len(bits)==1 and bits[0]['storage_all65536bits']=='PASS'
  fresh=[x for x in a if 'fresh_workspace_and_output' in x];assert len(fresh)==1 and fresh[0]['fresh_workspace_and_output']=='PASS'
  mutations=[x for x in a if 'input' in x]
  assert {x['input'] for x in mutations}=={'original','zero','fresh_normal','fresh_outliers','all_probes'}
  assert all(x['XP_all1MiB_currentX'] and x['input_immutable'] and x['output_NaN_and_XP_poison_overwritten'] and x['oracle'] for x in mutations)
  if mode!='initial':
   immediate=[x for x in a if 'actual_timed_output_before_any_replay_or_mutation' in x]
   assert len(immediate)==1 and immediate[0]['input_seed']==last_seeds[(mode,label)]
   assert immediate[0]['actual_timed_output_before_any_replay_or_mutation']=='PASS' and immediate[0]['XP_current_recorded_input_exact']
  identity=d['identity_after_validation'][mode][label]
  assert identity['regs']==122 and identity['local']==0 and identity['occupancy']==1 and identity['smem']==103600
  assert identity['same_loaded_binary_sha256'] and identity['same_generated_source_sha256']

paired=[]
for i in range(64):
 a=[r for r in raw if r['mode']=='paired' and not r['warmup'] and r['pair']==i]
 assert len(a)==2 and [r['label'] for r in a]==plan['paired_orders'][i]['order']
 assert a[0]['input_seed']==a[1]['input_seed']==110000+i
 v={r['label']:r['us'] for r in a}
 paired.append(dict(pair=i,order=[r['label'] for r in a],dp_us=v['dp'],ds_us=v['ds'],gain_us=v['dp']-v['ds'],ratio_DP_over_DS=v['dp']/v['ds']))
assert sum(p['order']==['dp','ds'] for p in paired)==32
assert all(paired[2*b]['order']==paired[2*b+1]['order'][::-1] for b in range(32))
assert paired==d['summary']['paired_comparisons']
gains=[p['gain_us'] for p in paired];ratios=[p['ratio_DP_over_DS'] for p in paired]
swapped=[(ratios[2*b]*ratios[2*b+1])**.5 for b in range(32)]
assert S.median(gains)==d['summary']['paired_median_gain_us']
assert sum(g>0 for g in gains)==d['summary']['paired_wins']==48
assert S.median(ratios)==d['summary']['median64_raw_ratios_DP_over_DS']
assert swapped==d['summary']['swapped_block_geometric_ratios'] and S.median(swapped)==d['summary']['median32_swapped_geometric_ratios']
medians={mode:[S.median(r['us'] for r in raw if r['mode']==mode and not r['warmup'] and r['block']==b) for b in range(3)] for mode in ('steady_before','steady_after')}
assert medians==d['summary']['steady_block_medians_us']
a,b=medians['steady_before'],medians['steady_after']
brackets={'before_mean_of_DP_brackets_minus_DS':(a[0]+a[2])/2-a[1],'after_DP_minus_mean_of_DS_brackets':b[1]-(b[0]+b[2])/2}
assert brackets==d['summary']['bracketed_effects_us']

def stats(a):
 q=S.quantiles(a,n=4,method='inclusive')
 return dict(n=len(a),median=S.median(a),mean=S.mean(a),min=min(a),max=max(a),Q1=q[0],Q3=q[2])
result={'raw_sha256':sha(RAW),'status':'INDEPENDENT350RECORD_PLAN_FIXTURE_IDENTITY_OUTPUT_STATISTICS_PASS',
 'records':{'total':350,'timed':320,'warmup_preserved':30,'paired_comparisons':64,'AB':32,'BA':32,'steady_blocks':6,'validation':62,'immediate_actual_timed_outputs':6},
 'paired_gain_us':stats(gains),'wins':48,'losses':16,'median64_raw_ratio_DP_over_DS':S.median(ratios),'median32_local_swapped_geometric_ratio':S.median(swapped),
 'absolute_paired_medians_us':{l:S.median(p[l+'_us'] for p in paired) for l in ('dp','ds')},
 'by_order':{'_'.join(o):stats([p['gain_us'] for p in paired if p['order']==o]) for o in (['dp','ds'],['ds','dp'])},
 'four_sequential16pair_epochs':[{'epoch':i,'gain':stats(gains[16*i:16*(i+1)]),'wins':sum(g>0 for g in gains[16*i:16*(i+1)])} for i in range(4)],
 'steady_block_medians_us':medians,'bracketed_effects_us':brackets,
 'decision':'LOCAL_FIXED_SOURCE_DIAGNOSTIC_FAVORS_DS_OVER_DP_NO_INCUMBENT_OR_OFFICIAL_COMPARISON',
 'previous_mixed_result_retained':'DS01 F1+.832003us,paired-.255998us(10/21wins); no overwritten or pooled samples',
 'limits':['2wayvs4way,kernelmix, freshpercomparisonXvsfixedX, absenceofparentprepare/allocationstate changed together; not isolated causal explanation',
           'local paired estimators not byte-identical server overhead/CI calibration; no official confidence interval',
           'one independent fixed-source run, no incumbent or full-submission preparation; no officialscore projection',
           'postmode validity diagnostics do not assert numerical oracle after every timed sample',
           'separate instrumented profile durations do not decompose unprofiled replaygain or prove store-phase cause']}
def rounded(x):
 if isinstance(x,float):return round(x,9)
 if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
 if isinstance(x,list):return [rounded(v) for v in x]
 return x
(HERE/'analysis.json').write_text(json.dumps(rounded(result),indent=2)+'\n')
print(json.dumps({k:rounded(result[k]) for k in ('status','absolute_paired_medians_us','paired_gain_us','wins','median64_raw_ratio_DP_over_DS','median32_local_swapped_geometric_ratio','bracketed_effects_us')},indent=2))

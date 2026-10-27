import copy,hashlib,json
from pathlib import Path
D=Path(__file__).parent;B=Path('proj_rsi/learnings.json');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=sha(B);assert base=='ecc72424e988fec834e948bb7e101e87d465e3107ece452b4f3d53b546855891'
r=json.loads(B.read_text());old=[x for x in r['lessons'] if x['id']=='gv53-quarter-pipeline-control-cost'][-1];x=copy.deepcopy(old)
x.update(version=old['version']+1,supersedes=old['record_sha256'],record_sha256=None,title='Earlier load issue can outperform earlier decode, but identical-body transfer remains mixed')
def ev(id,p):
 x['evidence'].append({'id':id,'path':p,'sha256':sha(p)});return json.loads(Path(p).read_text())
e=ev('earlyload','proj_rsi/audits/gv53-earlyload-result-v1/analysis.json');p=ev('package','proj_rsi/audits/gv53-selected-earlyload-bodies-result-v1/analysis.json')
def ob(id,k,ptr,msg,outcome='neutral'):
 v={'earlyload':e,'package':p}[k]
 for z in ptr.strip('/').split('/'):v=v[int(z)] if isinstance(v,list) else v[z]
 x['observations'].append(dict(id=id,statement=msg,outcome=outcome,evidence=k,selector={'json_pointer':ptr,'equals':v}))
ob('early-paired','earlyload','/timing/paired_median_us','M8 early next-W/X issue: control-new paired +0.192000531us, versus prior quarter/sparse losses; a changed load dependency, not another repair format.','positive')
ob('early-f1','earlyload','/timing/F1_endpoint_average_us','Same screen F1 endpoint average +0.160000287us; both endpoints and paired order medians positive.','positive')
ob('early-orders','earlyload','/timing/control_first_paired_median_us','Initial control-first paired median +0.128000043us.','positive')
ob('early-reverse-orders','earlyload','/timing/new_first_paired_median_us','Initial new-first paired median +0.271999277us.','positive')
ob('early-resources','earlyload','/actual_resources','Both arms64 registers/occupancy1/local0; this is latency scheduling evidence without an occupancy gain.')
ob('body-identity','package','/three_native_streams_match_frozen_pilot','Production-package extracted parent/keyed/early-load native streams match frozen pilot; a dispatch/body mismatch does not explain the observed fixed-role timing difference.')
for m in ['1','8']:
 for key in ['paired_median_control_minus_option_us','paired_mean_us','F1_average_us','control_first_median_us','option_first_median_us']:
  value=p['cells'][m][key];ob('package-'+m+'-'+key,'package','/cells/'+m+'/'+key,f'Package M{m} {key}: {value:+.9f}us; control-option sign. M1 is keyed, M8 is early-load; effects cannot be added.', 'positive' if value>0 else 'negative')
ob('pack-fixture','package','/pack/W_seed','Package Wseed5344 matches original early-load screen; same weight seed is not an independent-fixture replication.')
ob('package-no-policy','package','/production_prepare_selection','Fixed role execution is proven; actual chooser selection and official score remain unmeasured.')
for path,role in [('proj_rsi/candidates/gv53_earlyload_v1/submission.py','early-load source'),('proj_rsi/candidates/gv53_selected_earlyload_v1/submission.py','production package source'),('proj_rsi/candidates/gv53_earlyload_v1/README.md','realized load-placement and instruction-count scope')]:
 x['context']['sources'].append({'path':path,'sha256':sha(path),'role':role})
x['context'].update(experiment_id='research.gv53.earlyload.fixed.01; research.gv53.selected.earlyload.bodies.01; prior quarter/sparse',comparators=['parent versus quarter/sparse prior trials','parent versus early next-W/X issue M8','package fixed parent/keyed M1 and parent/early-load M8'],disposition='VALID_MIXED')
x['related_queue_ids']+=['research.gv53.earlyload.fixed.01','research.gv53.selected.earlyload.bodies.01']
x['hypotheses'] += [
 {'statement':'Changing incumbent next-W/X issue before current decode can avoid the quarter-stream format/control overhead and improve this M8 cell.','status':'supported_in_scope','observation_ids':['early-paired','early-f1','early-resources']},
 {'statement':'Hardware load-wait reduction causes the early-load gain.','status':'unproven','observation_ids':['early-paired','early-resources','cause']},
 {'statement':'Early-load effect magnitude transfers unchanged to the extracted production package.','status':'unproven','observation_ids':['early-paired','body-identity','package-8-paired_median_control_minus_option_us','package-8-option_first_median_us']}]
x['scope']='Extends quarter/sparse evidence with one H100 M8 early-load screen and same-seed5344 package fixed-body trial. Initial +0.192us paired/+0.160us F1 becomes package +0.031999us paired/+0.176000us F1, but package paired mean -0.033333us and order medians +0.096000/-0.192000us. M1 keyed package separately loses -0.048000us paired/-0.960001us F1. No independent fixture or official transfer claim.'
x['transfer_limits'] += ['Earlier native issue is realized; reduced hardware wait was not measured for this binary. Prior quarter/sparse counters do not attribute early-load cause.','Binary placement, run order, clocks and residual timing variation remain possible transfer factors; none is identified as the cause.','Prior quarter/sparse and early-load jobs differ in weights and event handling; their difference is not a controlled causal contrast.','The old earlier-first-MMA sufficiency refutation is scoped to quarter/sparse, not all load/decode pipelining.']
f='Next implementation hypothesis: move only next-W loads early while leaving next-X issue unchanged, eliminating the X snapshot/live range. Compare parent, current W+X early-load and W-only at the same selected M8 tuple and same stored W/X, retain both pair orders and F1 endpoints. If W-only loses while W+X wins, W-only sufficiency is refuted; if W-only wins with reduced native copy/live-range cost, pursue that narrower schedule. No unchanged rerun or new profiling gate.'
x['falsification']=f
x['next_test']={'question':'Can W-only earlier issue preserve useful overlap while removing X-snapshot copy/live-range overhead?','controls':['Exact parent and current combined W+X early-load bodies','Same M8 tuple, stored W/X, escape bank and timing method; both pair orders','Keep package fixed-body results separate from actual chooser/official CHECK'],'discriminator':f,'stop_rule':'If native W-only scheduling is not distinct, stop that implementation; if valid timing loses, retain it and reconsider X dependence rather than reject load scheduling. Root decides whether this new question earns a run.','required_gates':['Reuse unchanged scope-appropriate correctness/lifetime/native identity checks; no additional broad proof gate.']}
x['cost']={'basis':'Existing measured early-load total5.120794627s and package5.455110821s; proposed W-only implementation/measurement cost unknown, no reservation requested.','cpu_minutes':None,'gpu_seconds':None}
out={'status':'PRIVATE_UNSEALED_SINGLE_LESSON_EXTENSION','base_registry_sha256':base,'base_registry_version':9,'append_lessons':[x],'append_ideas':[],'live_registry_mutated':False,'purpose':'One causal/transfer extension with an actionable W-only issue ablation, not an automatic experiment or new gate.'}
(D/'records.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
checks=0
for q in x['evidence']+x['context']['sources']:assert sha(q['path'])==q['sha256']
for o in x['observations']:
 q=next(a for a in x['evidence'] if a['id']==o['evidence']);s=o['selector'];raw=Path(q['path']).read_text()
 if 'json_pointer' in s:
  v=json.loads(raw)
  for z in s['json_pointer'].strip('/').split('/'):v=v[int(z)] if isinstance(v,list) else v[z]
  assert v==s['equals']
 else:assert s['contains'] in raw
 checks+=1
assert sha(B)==base
(D/'checks.json').write_text(json.dumps({'state':'HASHES_AND_FACT_SELECTORS_PASS_UNSEALED','observations':checks,'new_observations':19,'live_registry_sha256':base,'GPU_API_SSH_queue_git_actions':0},indent=2)+'\n')
print(sha(D/'records.json'),checks)

"""Reproduce append-only registry v4 from frozen v3 and bound local outcomes."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from proj_rsi import learning as L
R=L.ROOT
BASE='proj_rsi/learning_archive/registry-v3-43ede5d0.json'
PW='proj_rsi/audits/pw43-runtime-independent.json'
PROFILE='proj_rsi/audits/pw38-selected-profile-02/analysis.json'
OLD='results/proj-rsi/scheduler/artifacts/pw38.selected.profile.v1.01/collection-1793085702590585040/logs/scheduler/pw38-selected-profile-01/run/capture.json'
def evidence(id,path):return {'id':id,'path':path,'sha256':hashlib.sha256((R/path).read_bytes()).hexdigest()}
def build():
 data=json.loads((R/BASE).read_text());data['registry_version']=4
 template=data['lessons'][0]
 def lesson(id,title,family,experiment,disposition,paths,facts,hypothesis,nexttest,queues):
  row=deepcopy(template);row.update(id=id,version=1,supersedes=None,title=title,tags=[family,'actual-control','validity','local'],scope='Local H100 only; no H200 forecast or official submission inferred.',related_queue_ids=queues)
  row['context']={'experiment_id':experiment,'family':family,'parents':['scored PW38 source29e557ea53f3'],'comparators':['actual independently prepared PW38','preserved earlier attempt where applicable'],'hardware':{'kind':'GPU','model':'H100 local132SM'},'validity':'VALID_LOCAL' if disposition=='VALID_REGRESSION' else 'VALIDATED_DIAGNOSIS','disposition':disposition,'sources':[dict(path=p,sha256=evidence('',p)['sha256'],role='immutable outcome/source-bound independent audit') for _,p in paths]}
  row['evidence']=[evidence(i,p) for i,p in paths];row['observations']=[{'id':i,'statement':s,'outcome':o,'evidence':e,'selector':{'json_pointer':ptr,'equals':value}} for i,s,o,e,ptr,value in facts]
  row['hypotheses']=[{'statement':hypothesis,'status':'unproven','observation_ids':[f[0] for f in facts]}]
  row['confidence']={'observation':'high','causality':'low','transfer':'low'};row['transfer_limits']=['No official run or H200 transfer established.','Do not convert profile samples or conditional local aggregate estimates into saved official time.']
  row['falsification']=nexttest;row['next_test']={'question':title,'controls':['Frozen actual PW38 full prepare and source','Same selected graph nodes and strict validity gates'],'discriminator':nexttest,'stop_rule':'Retain uncertainty on incomplete or conflicting evidence; no forced winner or family rejection.','required_gates':['Source/JIT identity and all numerical/terminal/current-input/counter/cleanup checks','Independent review and root hardware authorization before execution']}
  row['conditional_benefit']={'kind':'conditional_scenario_not_measured','value':None,'unit':'us','conditions':['No measured future gain or hardware transfer is asserted.']};row['cost']={'basis':'Future test not quoted or scheduled by this backfill.','cpu_minutes':None,'gpu_seconds':None};row['record_sha256']=L.digest(row);return row
 pw=json.loads((R/PW).read_text());profile=json.loads((R/PROFILE).read_text());old=json.loads((R/OLD).read_text())
 data['lessons'].append(lesson('pw43-weighted-deadline-fallback-actual','Whole prepare admission must preserve the actual incumbent, not just beat torch','prepare-selection','pw43.stack.fullscope.01','VALID_REGRESSION',[('pw43',PW),('brief','proj_rsi/audits/pw43-refinement-brief.json')],[
 ('paired','PW38 minus PW43 descriptive paired local score is negative; retain the regression.','negative','pw43','/pw38_minus_pw43_local_score_estimate_us/paired',pw['pw38_minus_pw43_local_score_estimate_us']['paired']),
 ('f1','F1 descriptive local aggregate also regresses.','negative','pw43','/pw38_minus_pw43_local_score_estimate_us/F1',pw['pw38_minus_pw43_local_score_estimate_us']['F1']),
 ('scope','4350 raw calls retained across30 shapes.','neutral','pw43','/raw_samples',4350),
 ('expired','O confirmation actually had negative remaining time and returned incomplete.','negative','pw43','/Lt_actual_confirmation/1/status','incomplete')],
 'Weighted admission consumed later-cell time; weak fallback versus torch may contribute, but selected-policy and timing changes are not isolated causal proof.',
 'Use the immutable refinement brief two-by-two admission accounting x confirmation comparator arms, plus separately prepared actual PW38; charge all costly work to the original whole-Lt budget, preserve all signs and proxy-vs-actual differences.', ['pw43.stack.fullscope.01']))
 data['lessons'].append(lesson('selected-profile-context-cluster-failure','A repaired CUDA metadata context does not validate a failed clustered profile','selected-profile','pw38.selected.profile.v1.01 + pw38.selected.profile.v2.02','INVALID_SETUP',[('old',OLD),('profile',PROFILE)],[
 ('old-invalid','Source revision2/jobv1 failed before prepare and admitted no profile.','invalid','old','/status',old['status']),
 ('context','Revision3/jobv2 completed actual unchanged prepare73.931s after explicit context establishment.','positive','profile','/prepare/wall_seconds',profile['prepare']['wall_seconds']),
 ('capture','Capture failed rc9; no counter or latency admission.','invalid','profile','/capture_returncode',9),
 ('terminal','All profile terminal oracle/counter/cleanup fields are missing, including router; raw rows remain diagnostic only.','invalid','profile','/missing_terminal_fields',profile['missing_terminal_fields']),
 ('route','Actual qkv1 is S2/V0 SM90 DSM cluster reduction, not global-counter waiting.','neutral','profile','/source_bound_qkv1/route',profile['source_bound_qkv1']['route'])],
 profile['causality'],profile['next_test']['single_change']+' '+profile['next_test']['discriminator'],['pw38.selected.profile.v1.01','pw38.selected.profile.v2.02']))
 for row in data['lessons'][-2:]:
  idea=deepcopy(data['ideas'][0]);idea.update(id='idea.'+row['id'],version=1,supersedes=None,title=row['title'],tags=row['tags'],lesson_refs=[row['id']+'@1'],existing_roi_ids=row['related_queue_ids'],mechanism=row['hypotheses'][0]['statement'],discriminator=row['next_test']['discriminator'],conditional_benefit=row['conditional_benefit'],cost=row['cost']);idea['record_sha256']=L.digest(idea);data['ideas'].append(idea)
 L.validate(data);return data
if __name__=='__main__':
 raw=(json.dumps(build(),indent=2,allow_nan=False)+'\n').encode();digest=hashlib.sha256(raw).hexdigest();p=R/('proj_rsi/learning_archive/registry-v4-'+digest[:8]+'.json')
 if p.exists() and p.read_bytes()!=raw:raise RuntimeError('immutable archive conflict')
 p.write_bytes(raw);Path(str(p)+'.sha256').write_text(digest+'  '+p.name+'\n');L.DEFAULT.write_bytes(raw);print(digest)

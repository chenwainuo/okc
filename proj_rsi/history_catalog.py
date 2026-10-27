"""Build auditable history from immutable local official/job/measurement evidence.

CPU only. Regenerate: python3 -m proj_rsi.history_catalog
Verify checked-in renderings: python3 -m proj_rsi.history_catalog --check
Add new evidence as a new immutable attempt; never replace a parent's measurements.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRIMARY='results/proj-rsi/takeover/20261027'
INPUTS={
 'live_jobs':f'{PRIMARY}/live-jobs-audit.json',
 'live_board':f'{PRIMARY}/live-leaderboard-audit.json',
 'live_rs32':f'{PRIMARY}/live-rs32-audit.json',
 'live_pw36':f'{PRIMARY}/live-pw36-audit.json',
 'completeness':'proj_rsi/audits/current-leaderboard-gains-completeness.json',
 'official_refs':'proj_rsi/audits/current-leaderboard-gains-official.refs.json',
 'source_map':'proj_rsi/audits/current-leaderboard-stack-source.json',
 'cursor_v3':'proj_rsi/audits/smallm-v3-01/analysis.json',
 'native_stmatrix':'proj_rsi/audits/native-stmatrix-01/analysis.json',
 'native_x4':'proj_rsi/audits/native-x4-01/analysis.json',
 'pw38_local':'proj_rsi/audits/pw38-stack-01/analysis.json',
 'native_combo':'proj_rsi/audits/native-x4-stmatrix-01/analysis.json',
}
# Curated source ancestry, supported by the source-map/official audit references.
# Per-job selected routes are always parsed from that job, not copied from this map.
BASE=['lossless12bit','GV_direct_register','exact_escapes','router_hi_lo','tuned_Lt','shape_bake_fallback']
P1=['P1_selfzero_small_router_unpooler','RR6_router1']
RW=['RW_wgmma_router128']
P2=['P2_short_qkv_chains']
RQ=['RQ128_registerA_splitK']
SMALL=['RQ16_32','HALF_bit4','EARLY_bit5']
RTK=['RTK_TMA_router128_256']
VERSION_STACKS={
 'r6-c12':('prior r5/fable lineage; exact parent revision unknown',BASE,'Lossless12-bit GV + tuned Lt / hi+lo router baseline'),
 'pw23':('r6-c12 / pw22',BASE+['RP_pipelined_router128'],'r6 + pipelined split-K router128'),
 'pw31':('pw23 + pw27h + pw29h',BASE+['RP_pipelined_router128']+P1+RW,'P1 self-zero small paths + RW router128 over pw23'),
 'pw33':('pw31 + pw28h/P2',BASE+['RP_pipelined_router128']+P1+RW+P2,'pw31 + shorter-chain qkv geometry'),
 'pw34':('pw33 + rs128',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ,'pw33 + RQ128 paths; selected only after prepare repair'),
 'pw35':('pw34 + pw29m/RTK',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+RTK,'pw34 + one-node TMA RTK router128/256'),
 'rs32':('pw34 + rq16/32',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL,'pw34 + rq16/32; retains old RW router; no PW35 RTK'),
 'pw36':('pw35 + rs32 small-M branch',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK,'rs32 small-M branch + PW35 RTK router'),
 'pw37':('pw36 + smallm cursor',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['smallm_cursor'],'PW36 + cursor source only; NOT SCORED; cursor v3 has regressions'),
 'pw38':('pw36 + rs32_eh_backport',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor'],'PW36 + optional EH bit65536; excludes cursor; CPU only, NOT SCORED'),
}

def read(root,path):return json.loads((root/path).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(root,path):
 p=root/path
 return {'path':path,'sha256':digest(p),'bytes':p.stat().st_size} if p.is_file() else {'path':path,'missing':True}
def utc(timestamp):return dt.datetime.fromtimestamp(timestamp,dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
def archive_jobs(root):
 result={}
 paths=list(sorted((root/'proj_rsi/candidates').glob('*/profiles/*.json')))
 paths+=list(sorted((root/'results/proj-rsi/releases').glob('*/profiles/*.json')))
 paths+=list(sorted((root/'results/proj-rsi/jobs').glob('*.json')))
 for p in paths:
  try:raw=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if not isinstance(raw,dict) or not isinstance(raw.get('benchmarks'),list) or not raw.get('job_id'):continue
  if raw['job_id'] in result:continue
  candidate=p.parents[1].name if p.is_relative_to(root/'proj_rsi/candidates') else None
  result[raw['job_id']]={'candidate':candidate,'raw':raw,'ref':ref(root,str(p.relative_to(root)))}
 return result

def parse_picks(root,rawref):
 import re
 p=(root/rawref['path']).with_suffix('.log');picks=[]
 if not p.exists():return [],None
 for line_no,line in enumerate(p.read_text().splitlines(),1):
  m=re.match(r'PICK (\w+) m=(\d+) (\S+) ([0-9.]+)',line)
  if m:picks.append({'layer':m[1],'m':int(m[2]),'selected':m[3],'bake_us':float(m[4]),'line':line_no})
 return picks,ref(root,str(p.relative_to(root)))

def resolve_sources(root=ROOT):
 config=read(root,'proj_rsi/history_sources.json')
 if config.get('schema_version')!=1:raise ValueError('unsupported history sources schema')
 snapshots=config.get('jobs_snapshots',[])
 if len({r['id'] for r in snapshots})!=len(snapshots):raise ValueError('duplicate snapshot id')
 for snapshot in snapshots:
  path=Path(snapshot['path'])
  if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe history source path')
  if digest(root/path)!=snapshot['sha256']:raise ValueError('immutable API snapshot changed: '+snapshot['id'])
 matches=[row for row in snapshots if row['id']==config['active_jobs_snapshot']]
 if len(matches)!=1:raise ValueError('active jobs snapshot is unknown')
 paths=dict(INPUTS,live_jobs=matches[0]['path'])
 for association in config.get('job_source_associations',{}).values():
  for key in ('source','attestation'):
   reference=association[key];path=Path(reference['path'])
   if path.is_absolute() or '..' in path.parts or digest(root/path)!=reference['sha256']:raise ValueError('release association changed')
  for reference in association.get('additional_attestations',[]):
   path=Path(reference['path'])
   if path.is_absolute() or '..' in path.parts or digest(root/path)!=reference['sha256']:raise ValueError('release completion attestation changed')
 return paths,config

def merged_live_jobs(root,config):
 # API responses may be paginated to the latest50. A new capture must never
 # delete an older known submission/check. Latest observed row wins by ID.
 merged={};origins={}
 for snapshot in config['jobs_snapshots']:
  rows=read(root,snapshot['path'])
  if not isinstance(rows,list):raise ValueError('jobs snapshot must be a normalized list')
  for row in rows:
   jid=row['job_id']
   if jid in merged and merged[jid]['mode']!=row['mode']:raise ValueError('job mode changed across snapshots')
   merged[jid]=row;origins[jid]=ref(root,snapshot['path'])
  if snapshot['id']==config['active_jobs_snapshot']:break
 return list(merged.values()),origins

def build(root=ROOT):
 paths,source_config=resolve_sources(root)
 inputs={key:read(root,path) for key,path in paths.items()}
 active_snapshot_row_count=len(inputs['live_jobs'])
 live,api_origins=merged_live_jobs(root,source_config)
 archive=archive_jobs(root)
 for version,expected in inputs['source_map']['source_fingerprints'].items():
  source=root/f'proj_rsi/candidates/{version}/submission.py'
  if source.exists() and digest(source)!=expected:raise ValueError(f'Immutable audited source changed: {version}')
 if len({r['job_id'] for r in live})!=len(live):raise ValueError('duplicate official live job id')
 jobs=[]
 for live_row in sorted(live,key=lambda x:(x['submitted_at'],x['job_id'])):
  if live_row.get('problem')!='muse_bf16_proj':raise ValueError('unexpected board in live job snapshot')
  item=archive.get(live_row['job_id']);association=source_config.get('job_source_associations',{}).get(live_row['job_id'])
  candidate=association.get('candidate') if association else item['candidate'] if item else None
  source=association['source']['path'] if association else f'proj_rsi/candidates/{candidate}/submission.py' if candidate else None
  raw=item['raw'] if item else {};summary=raw.get('summary',{})
  if item and live_row['score_us'] is not None and summary.get('score_us')!=live_row['score_us']:
   raise ValueError('Live/archive score mismatch for '+live_row['job_id'])
  picks,logref=parse_picks(root,item['ref']) if item else ([],None)
  parent,features,description=VERSION_STACKS.get(candidate,('UNKNOWN',[],'Historical candidate; see raw/hypothesis; exact feature ancestry not independently established'))
  selected_kernels=[{'case':p.get('case'),'report':p.get('report')} for p in raw.get('profiles',[])]
  selected_features=[]
  if any('rq:' in p['selected'] for p in picks):selected_features.append('RQ actual exact-M picks: '+', '.join(f"{p['layer']}{p['m']}={p['selected']}" for p in picks if 'rq:' in p['selected']))
  if any(p['selected'].startswith('rt') for p in picks):selected_features.append('RTK actual router picks: '+', '.join(f"M{p['m']}={p['selected']}" for p in picks if p['selected'].startswith('rt')))
  if live_row['job_id']=='5a39372456514f20':description='pw33 + RQ source, but RQ bake skipped-budget; no selected RQ gain in this attempt'
  identity='Current local file hash + archived path/git ancestry association; historical uploaded bytes not independently attested' if source else 'UNKNOWN: failed check has no archived source mapping'
  if association:identity=association['association']
  api=inputs.get('live_'+candidate) if candidate in ('rs32','pw36') else None
  row={'API_status_source':api_origins[live_row['job_id']],'job_id':live_row['job_id'],'submitted_at':live_row['submitted_at'],'submitted_utc':utc(live_row['submitted_at']),
   'mode':live_row['mode'],'API_result_status':live_row.get('result_status'),'leaderboard_submission':live_row['mode']=='run','candidate':candidate,'status':live_row['status'],
   'passed':live_row['passed'],'score_us':live_row['score_us'],'tests_passed':summary.get('tests_passed'),'tests':summary.get('tests'),'speedup':summary.get('speedup'),'speedup_ci':summary.get('speedup_ci'),
   'source':ref(root,source) if source else None,'release_attestation':association.get('attestation') if association else None,'additional_release_attestations':association.get('additional_attestations',[]) if association else [],'source_identity_status':identity,'api_content_hash':api.get('content_hash') if api else None,
   'api_content_hash_algorithm':'UNKNOWN; do not equate server content_hash to raw file SHA256',
   'parent':parent,'compiled_or_available_features':features,'stack_description':description,'actual_picks':picks,'actual_selected_features':selected_features,
   'actual_timed_profiles':selected_kernels,'raw':item['ref'] if item else None,'log':logref,
   'archive_status':'ARCHIVED_30_SHAPES' if len(raw.get('benchmarks',[]))==30 else ('ARCHIVED_PARTIAL' if item else 'PENDING_NO_ARCHIVE' if live_row['status'] not in ('done','failed','cancelled') else 'FAILED_NO_ARCHIVE' if live_row['passed'] is False else 'COMPLETED_NO_ARCHIVE_KEEP_UNKNOWN'),
   'causal_delta_vs_other_jobs_us':None,'causal_caution':'Independent official jobs are not paired controls; score changes are observed outcomes, not causal feature increments.'}
  jobs.append(row)
 submissions=[r for r in jobs if r['leaderboard_submission']]
 checks=[r for r in jobs if not r['leaderboard_submission']]
 findings=[]
 for finding in inputs['completeness']['findings']:
  r=dict(finding);r['immutable_id']='improvement.'+hashlib.sha256(r['id'].encode()).hexdigest()[:16]
  r['currently_included']=r['in_rs32'];r['stacked_versions']=[];r['stack_tested_jobs']=[];r['score_tested_jobs']=[]
  r['missing_reason']='See finding/comparator/validity; compiled presence is not selected inclusion.'
  r['next_action']='Preserve evidence; require valid complete actual-selected-baseline comparison before integration.'
  if r['id']=='PW35 one-node rtk router128/256':
   r.update(stacked_versions=['pw35','pw36','pw37'],stack_tested_jobs=['5d880cd0aa504209','e22a4b9d693d4902'],score_tested_jobs=['5d880cd0aa504209','e22a4b9d693d4902'],missing_reason='rs32 is the parallel PW34+small-M branch; RTK absent. Combined PW36 officially tested and aggregate did not beat incumbent.',next_action='Analyze selective stack controls; do not label combined source untested or infer cross-job additive score.')
  elif r['id']=='RS128b early-half epilogue EH':
   r.update(stacked_versions=['rs128b lab only'],missing_reason='EH bit16=65536 absent; HALF bit4=16/EARLY bit5=32 are different. Original remote distributions not locally archived.',next_action='Recover raw historical controls; independently review source and compare current selected full serving path; preserve 0.13–0.16us observation without old killbar filter.')
  elif r['id']=='smallm cursor attempt01':
   r.update(id='smallm cursor attempt01 + valid v3 follow-up',classification='INVALID_ATTEMPT01_THEN_VALID_V3_MIXED_REGRESSION',finding='Attempt01 invalid with zero timings. Latest v3 validates770 raw calls/allfive shapes: tiny qkv16 +0.032us and pooler32 +0.096us vsPW36, but pooler32−0.144us vsmatched RS32 and unpooler32−0.384/−0.352us versusPW36/RS32. No incremental cursor/shipping win; PW37 stack remains NOT SCORED.',verification='Latest independent all-record v3 audit; root owns final disposition',stacked_versions=['smallm_increment','pw37 source only'],missing_reason='Valid local mixed/regressing comparison; not a combined PW36+cursor official result.',next_action='Preserve tiny positives and regressions; diagnose exact selected-path controls before a new immutable implementation.')
   r['evidence']=r['evidence']+[ref(root,'proj_rsi/audits/smallm-v3-01/report.md')]
  if r['classification'].startswith('SHIPPED_INCLUDED'):
   r['stacked_versions']=['rs32','pw36'];r['stack_tested_jobs']=['fb548b5dc6f04696','e22a4b9d693d4902'];r['score_tested_jobs']=list(r['stack_tested_jobs']);r['missing_reason']='Included or superseded by a faster selected path; not a missing additive feature.'
  findings.append(r)
 # Append new immutable local attempts; preserve small parent positives and
 # simultaneous actual-incumbent losses in separate fields, never net them.
 local_observations=[]
 for key,name,version,job in [('native_stmatrix','Native-X STMatrix attempt01','reverse_tma_x_stmatrix','native.stmatrix.v1.01'),('native_x4','Native-X4 attempt01','reverse_tma_x4','native.x4.v1.01')]:
  data=inputs[key];folder='native-stmatrix-01' if key=='native_stmatrix' else 'native-x4-01'
  parent_gain=data['paired_new_median_gain_us']['native'];actual_gain=data['paired_new_median_gain_us']['incumbent']
  finding={'id':name,'immutable_id':'improvement.'+job,'classification':'VALID_LOCAL_ACTUAL_INCUMBENT_REGRESSION_WITH_SMALL_PARENT_PROGRESS_NOT_SHIPPED',
   'in_rs32':'No','currently_included':'No; NOT SHIPPED','scope':'Local H100 unpooler M128; no official H200 score',
   'finding':f"Same-run median paired gain versus native parent +{parent_gain:.6f}us; versus actual selected local incumbent {actual_gain:.6f}us, 0/32 actual-incumbent wins. F1 parent gain +{data['F1_new_gain_us']['native']:.6f}us and actual loss {data['F1_new_gain_us']['incumbent']:.6f}us. Preserve the tiny parent observation without a kill-bar cutoff; it is not a shipping gain, significance claim or H200 forecast.",
   'verification':f"Independent all295-call reconstruction, actual-JIT gates, eight strict terminal oracles,20 mutations and archived cleanup; native raw mover bits remain UNPROBED. Result SHA256 {data['result_sha256']}",
   'evidence':[ref(root,f'proj_rsi/audits/{folder}/report.md'),ref(root,INPUTS[key]),ref(root,f'proj_rsi/scheduling/validations/{folder}.json')],
   'stacked_versions':[version],'stack_tested_jobs':[],'score_tested_jobs':[],
   'local_tested_attempt':job,'local_result_sha256':data['result_sha256'],
   'parent_paired_gain_us':parent_gain,'actual_incumbent_paired_gain_us':actual_gain,'official_score_us':None,
   'missing_reason':'Valid local version loses actual selected incumbent despite parent progress; no integration or official submission.',
   'next_action':'Retain source and small positive observation; diagnose remaining actual-incumbent gap before any new immutable mechanism or composition.'}
  findings.append(finding)
  for control,gain in data['paired_new_median_gain_us'].items():
   local_observations.append({'observation_id':f'paired.{job}.{control}','attempt_id':job,'version':version,'control':control,'gain_us':gain,'F1_gain_us':data['F1_new_gain_us'][control],'wins':data['paired_new_wins'][control],'pairs':32,'hardware':'H100','result_sha256':data['result_sha256'],'evidence':ref(root,INPUTS[key]),'shipping_gain':False})
 for finding in findings:
  if finding['id']=='CPU-only remaining ports/designs':
   finding['finding']='Historical CPU-only snapshot is superseded for native STMatrix and X4 by the separate valid local regression findings below. X4+STMatrix/QKV split4/codec and other newly prepared ports require their own runtime evidence; static instruction/byte counts are not gains.'
 local=inputs['pw38_local']
 pw38_check=next((r for r in jobs if r['job_id']=='a3ed4ce5886a4415'),None)
 check_pending=pw38_check is None or pw38_check['status'] not in ('done','failed','cancelled')
 findings.append({'id':'PW38 full-policy EH stack local validation + official check','immutable_id':'improvement.pw38.stack.fullscope.01','classification':'VALIDATED_LOCAL_FULL_POLICY_MIXED_RESULTS_OFFICIAL_CHECK_'+('PENDING' if check_pending else 'COMPLETED')+'_NOT_SHIPPED','in_rs32':'No','currently_included':'No; PW38 standalone local/release source; NOT leaderboard scored',
  'scope':'H100 all30 shapes; official check a3ed4ce5886a4415 status '+str(pw38_check['status'] if pw38_check else 'unknown')+' in active API snapshot',
  'finding':'4350 raw calls/all30 shapes,222 task checks,180 strict terminal oracles,360 mutations and15 JIT archives. EH selected qkv128/pooler128; pooler128 paired +0.320us versus RS32/+0.224us versus PW36, qkv128 −0.032/+0.080us (F1 positive, paired inconclusive). M128/M256 router gains +1.024/+1.824us versus RS32 belong to merged RTK; each −0.048us versus PW36. Pooler32 prepared-policy regression −0.240/−0.128us. Small wins/ties/losses are retained; modest descriptive local aggregate gain is not an official H200 gain.',
  'verification':'All-record local CPU reconstruction and independent review; source29e557ea53f3... release associated to accepted check with bookkeeping recovery/no resubmission.',
  'evidence':[ref(root,p) for p in [paths['pw38_local'],'proj_rsi/audits/pw38-stack-01/report.md','proj_rsi/audits/pw38-stack-independent.json','proj_rsi/audits/pw38-stack-independent.md','results/proj-rsi/releases/pw38-v1-29e557ea53f3/attestation.json']],
  'stacked_versions':['pw38'],'stack_tested_jobs':[],'score_tested_jobs':[],'local_tested_attempt':local['job_id'],'official_check_jobs':['a3ed4ce5886a4415'],'missing_reason':'Official CHECK is not a scored submission. Check API status: '+str(pw38_check['status'] if pw38_check else 'unknown')+'; archive status: '+str(pw38_check['archive_status'] if pw38_check else 'unknown'),
  'next_action':'Wait existing accepted check; archive actual tests/timing/picks and preparation evidence. Never resubmit due local bookkeeping error or claim promotion from local aggregate.'})
 if pw38_check and pw38_check['raw']:
  record=findings[-1]
  if pw38_check['passed'] is True and pw38_check['archive_status']=='ARCHIVED_30_SHAPES':record['stack_tested_jobs']=['a3ed4ce5886a4415']
  record['official_check_score_us']=pw38_check['score_us'];record['official_check_tests']=[pw38_check['tests_passed'],pw38_check['tests']]
  record['official_check_picks']=pw38_check['actual_picks'];record['official_check_profiles']=pw38_check['actual_timed_profiles']
  record['evidence']+=([pw38_check['raw'],pw38_check['log']]+pw38_check['additional_release_attestations'])
  record['finding']+=' Official CHECK completed9/9 with30timings at '+str(pw38_check['score_us'])+'us; EH selected qkv128/pooler128. Confidence intervals overlap rs32; this CHECK does not update leaderboard best or establish a causal/additive gain.'
  record['next_action']='Independent root review of completed check before any scored release; keep actual picks, all regressions and source attestation. No automatic leaderboard promotion.'
 combo=inputs['native_combo']
 findings.append({'id':'Native-X4 + STMatrix composition attempt01','immutable_id':'improvement.native.x4.stmatrix.v1.01','classification':'VALID_LOCAL_MIXED_PARENT_PROGRESS_ACTUAL_INCUMBENT_REGRESSION_NOT_SHIPPED','in_rs32':'No','currently_included':'No; NOT SHIPPED','scope':'H100 unpooler128, no official score',
  'finding':'414 raw calls; paired +0.239998us versus native, +0.064000us versus X4, +0.128001us versus STMatrix, but −7.088000us versus actual incumbent (0/40wins). F1 gains versus X4/STMatrix are each −0.064000us: retain conflicting methods and tiny positives, do not infer additive benefit from source commutativity.',
  'verification':'Independent all-record reconstruction, four actual-JIT gates,10 terminal/25 mutation oracles; native mover bits remain UNPROBED.',
  'evidence':[ref(root,p) for p in [paths['native_combo'],'proj_rsi/audits/native-x4-stmatrix-01/report.md','proj_rsi/scheduling/validations/native-x4-stmatrix-01.json']],
  'stacked_versions':['reverse_tma_x4_stmatrix'],'stack_tested_jobs':[],'score_tested_jobs':[],'missing_reason':'Valid local composition loses actual selected incumbent; mixed parent estimates, no official submission.',
  'next_action':'Retain all parent and composed versions; diagnose actual whole-pipeline gap before another immutable mechanism.'})
 for control,gain in combo['paired_new_median_gain_us'].items():
  local_observations.append({'observation_id':'paired.native.x4.stmatrix.v1.01.'+control,'attempt_id':'native.x4.stmatrix.v1.01','version':'reverse_tma_x4_stmatrix','control':control,'gain_us':gain,'F1_gain_us':combo['F1_new_gain_us'][control],'wins':combo['paired_new_wins'][control],'pairs':40,'hardware':'H100','result_sha256':combo['result_sha256'],'evidence':ref(root,paths['native_combo']),'shipping_gain':False})
 # Maintain observation IDs independent of row position and keep all tiny records.
 observations=[]
 for r in inputs['completeness']['positive_bake_samples_no_killbar_filter']:
  row=dict(r);row['catalog_cohort']='initial_45_job_audit';token=f"{r['job']}:{r['line']}:{r['mode']}";row['observation_id']='bake.'+hashlib.sha256(token.encode()).hexdigest()[:20]
  row['feature_level']=False;observations.append(row)
 # New archived official jobs add observations without modifying the original
 # 224-observation audit or filtering its tiny positives.
 import re
 known_jobs={r['job'] for r in inputs['completeness']['official_jobs']}
 for official_job in jobs:
  if official_job['job_id'] in known_jobs or official_job['log'] is None:continue
  log_path=root/official_job['log']['path'];bakes=[]
  for line_number,line in enumerate(log_path.read_text().splitlines(),1):
   match=re.match(r'BAKE (\w+) m=(\d+) (\S+) ([0-9.]+)',line)
   if match:bakes.append((match[1],int(match[2]),match[3],float(match[4]),line_number))
  picks={(r['layer'],r['m']):(r['selected'],r['bake_us']) for r in official_job['actual_picks']}
  for layer,m,mode,value,line_number in bakes:
   if mode=='r6':continue
   controls=[r for r in bakes if r[0]==layer and r[1]==m and r[2]=='r6']
   if not controls:continue
   control=min(controls,key=lambda r:abs(r[4]-line_number));gain=control[3]-value
   if gain<=0:continue
   token=f"{official_job['job_id']}:{line_number}:{mode}"
   observations.append({'observation_id':'bake.'+hashlib.sha256(token.encode()).hexdigest()[:20],'job':official_job['job_id'],'candidate':official_job['candidate'],'layer':layer,'m':m,'mode':mode,'candidate_bake_us':value,'logged_r6_control_us':control[3],'positive_bake_difference_us':gain,'below_legacy_0_3us':gain<.3,'below_legacy_0_5us':gain<.5,'actual_final_pick':picks.get((layer,m)),'log':official_job['log']['path'],'line':line_number,'control_line':control[4],'classification':'Positive rounded same-job sample versus named r6 control; not necessarily incremental versus faster existing RQ mode. No threshold filter.','feature_level':False,'catalog_cohort':'live_official_new_archives'})
 selected_observations=[]
 for jid,record in inputs['official_refs']['jobs'].items():
  for observation in record.get('selected_positive_bakes',[]):selected_observations.append(dict(observation,job=jid))
 # Candidate inventory is historical snapshot plus newly created directories, all preserved.
 historical={r['candidate']:r for r in inputs['completeness']['candidate_inventory']}
 candidates=[]
 for d in sorted((root/'proj_rsi/candidates').iterdir()):
  if not d.is_dir():continue
  name=d.name;old=historical.get(name,{})
  candidate_jobs=[j['job_id'] for j in jobs if j['candidate']==name]
  parent,features,description=VERSION_STACKS.get(name,('UNKNOWN',[],'See hypothesis/manifest/README and archived records; no inferred gain'))
  candidate={'candidate':name,'immutable_id':'candidate.'+name,'parent':parent,'source':ref(root,str((d/'submission.py').relative_to(root))) if (d/'submission.py').exists() else None,
   'compiled_or_available_features':features,'description':description,'official_jobs':candidate_jobs,'leaderboard_submitted':any(j['leaderboard_submission'] and j['candidate']==name for j in jobs),
   'historical_summary_paths':old.get('summary_paths',[]),'historical_text_artifacts_scanned':old.get('text_artifacts_scanned'),
   'new_since_completeness_snapshot':name not in historical,'actual_serving_evidence':'Per-job actual picks/profiles; never infer selection from source presence.'}
  if name=='pw37':candidate['status']='CPU_SOURCE_STACK_NOT_SCORED; cursor v3 is not a demonstrated incremental win'
  if name=='pw38':candidate['status']='PW36_PLUS_OPTIONAL_EH_VALIDATED_LOCAL_OFFICIAL_CHECK_'+('PENDING' if check_pending else 'COMPLETED')+'_NOT_SCORED'
  if name=='pw38':candidate['description']='PW36 + optional EH, no cursor; full30-shape local validation complete with mixed observations; accepted official check is separate from leaderboard scoring.'
  candidate['review_and_manifest_refs']=[ref(root,str(p.relative_to(root))) for p in sorted(d.glob('*')) if p.is_file() and p.name in ('manifest.json','PROVENANCE.md','hypothesis.json','RESULTS.md')]
  candidate['named_source_components']=source_nodes(root/d/'submission.py') if (d/'submission.py').exists() else []
  candidates.append(candidate)
 missing=[j['job_id'] for j in jobs if not j['raw']]
 pending=[j['job_id'] for j in jobs if j['archive_status']=='PENDING_NO_ARCHIVE']
 failed_missing=[j['job_id'] for j in jobs if j['archive_status']=='FAILED_NO_ARCHIVE']
 return {'schema_version':1,'scope':'Complete live official job snapshot + preserved local improvements/observations; CPU-only rendering',
  'authority':{'best_job':'fb548b5dc6f04696','score_us':3200.6150084373335,'source_sha256':inputs['source_map']['source_sha256'],'live_board':inputs['live_board'].get('you'),'achievement_claim':False},
  'methodology':['Union preserved API captures by job ID to survive latest50 pagination; newest observed row wins. Mode=run determines leaderboard submissions; checks remain separate including failures/noarchive.','Stable job/candidate/improvement/observation IDs; every input/source/raw/review link carries SHA256 when locally present.','Source compiled/available features are distinct from actual per-job selected policies and timed profiles.','No independent-job score subtraction is reported as a causal feature delta.','Feature findings and repeated positive preparation observations are separate tables; no positive threshold removes small observations.','Keep invalid attempts and superseded parents immutable; repairs get new source or attempt identity.','Review provenance gaps explicitly; server content_hash algorithm unknown and not raw SHA256 attestation.'],
  'history_sources_config':ref(root,'proj_rsi/history_sources.json'),'active_jobs_snapshot':source_config['active_jobs_snapshot'],'preserved_jobs_snapshots':source_config['jobs_snapshots'],'input_provenance':{k:ref(root,v) for k,v in paths.items()},'additional_evidence':[ref(root,p) for p in ['proj_rsi/audits/smallm-v3-01/report.md','proj_rsi/audits/native-stmatrix-01/report.md','proj_rsi/audits/native-x4-01/report.md']],
  'counts':{'active_API_snapshot_rows':active_snapshot_row_count,'all_official_jobs':len(jobs),'leaderboard_submissions':len(submissions),'checks':len(checks),'archived_jobs':len(jobs)-len(missing),'complete_profile_jobs':sum(r['archive_status']=='ARCHIVED_30_SHAPES' for r in jobs),'partial_profile_jobs':sum(r['archive_status']=='ARCHIVED_PARTIAL' for r in jobs),'no_archive_jobs':len(missing),'pending_no_archive_jobs':len(pending),'failed_no_archive_jobs':len(failed_missing),'feature_findings':len(findings),'positive_observations':len(observations),'preserved_original_positive_observations':224,'new_official_positive_observations':len(observations)-224,'local_paired_observations':len(local_observations),'selected_positive_observations':len(selected_observations),'below_0_3us':sum(r['positive_bake_difference_us']<.3 for r in observations),'below_0_5us':sum(r['positive_bake_difference_us']<.5 for r in observations),'candidates':len(candidates)},
  'submissions':submissions,'checks':checks,'all_official_jobs':jobs,'improvements':findings,'positive_observations':observations,'selected_positive_observations':selected_observations,'local_paired_observations':local_observations,'candidates':candidates,
  'pw38_latest_local':local,'native_composition_latest':combo,'cursor_v3_latest':inputs['cursor_v3'],'version_source_ancestry_comparisons':inputs['source_map']['named_AST_comparisons'],'current_exact_source_components':inputs['source_map']['components'],'current_exact_selected_routes':inputs['source_map']['official_selected_routes'],
  'coverage_gaps':inputs['completeness']['coverage']['gaps']+['Failed check jobs without raw/source archive remain UNKNOWN: '+', '.join(failed_missing),'Pending accepted jobs awaiting archive remain separate: '+', '.join(pending),'Older PW34 current file hash does not establish exact historical bytes of each prepare-guard version.','The completeness cursor attempt01 finding is superseded by retained latest valid v3 result; both are preserved.']}

def source_nodes(path):
 import ast
 source=path.read_text();tree=ast.parse(source);lines=source.splitlines(keepends=True);nodes=[]
 for ordinal,node in enumerate(tree.body):
  name=getattr(node,'name',None)
  if name is None and isinstance(node,(ast.Assign,ast.AnnAssign)):
   targets=node.targets if isinstance(node,ast.Assign) else [node.target]
   name=','.join(t.id for t in targets if isinstance(t,ast.Name))
  if not name:continue
  segment=''.join(lines[node.lineno-1:node.end_lineno])
  nodes.append({'name':name,'ordinal':ordinal,'line':node.lineno,'end_line':node.end_lineno,'AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'source_segment_sha256':hashlib.sha256(segment.encode()).hexdigest()})
 return nodes

def csv_text(rows,fields):
 output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for row in rows:
  writer.writerow({key:json.dumps(row.get(key),ensure_ascii=False,sort_keys=True) if isinstance(row.get(key),(dict,list)) else row.get(key) for key in fields})
 return output.getvalue()

def render(catalog):
 jobfields=['job_id','submitted_utc','mode','status','API_result_status','candidate','score_us','passed','tests_passed','tests','archive_status','parent','stack_description','compiled_or_available_features','actual_selected_features','source','source_identity_status','api_content_hash','API_status_source','raw','log','causal_caution']
 files={
 'proj_rsi/history_catalog.json':json.dumps(catalog,indent=2,ensure_ascii=False)+'\n',
 'proj_rsi/tables/submissions.csv':csv_text(catalog['submissions'],jobfields),
 'proj_rsi/tables/checks.csv':csv_text(catalog['checks'],jobfields),
 'proj_rsi/tables/all_official_jobs.csv':csv_text(catalog['all_official_jobs'],jobfields),
 'proj_rsi/tables/improvements.csv':csv_text(catalog['improvements'],['immutable_id','id','classification','currently_included','scope','finding','verification','stacked_versions','stack_tested_jobs','score_tested_jobs','missing_reason','next_action','evidence']),
 'proj_rsi/tables/positive_observations.csv':csv_text(catalog['positive_observations'],['observation_id','job','candidate','layer','m','mode','candidate_bake_us','logged_r6_control_us','positive_bake_difference_us','below_legacy_0_3us','below_legacy_0_5us','actual_final_pick','log','line','control_line','classification','feature_level','catalog_cohort']),
 'proj_rsi/tables/local_observations.csv':csv_text(catalog['local_paired_observations'],['observation_id','attempt_id','version','control','gain_us','F1_gain_us','wins','pairs','hardware','result_sha256','shipping_gain','evidence']),
 'proj_rsi/tables/candidates.csv':csv_text(catalog['candidates'],['immutable_id','candidate','parent','source','compiled_or_available_features','official_jobs','leaderboard_submitted','status','description','historical_summary_paths','new_since_completeness_snapshot']),
 }
 counts=catalog['counts']
 text='''# Projection kernel history

This catalog preserves all official submissions and known improvement evidence. Regenerate with `python3 -m proj_rsi.history_catalog`; check deterministic renderings with `python3 -m proj_rsi.history_catalog --check`. No GPU or service calls occur during generation.

The active versioned API snapshot contains **{all_jobs} official jobs:{runs} leaderboard submissions and{checks} checks**. The best is **rs32/fb548b5dc6f04696:3200.615008us**, local source `a7b6e010…5a35ad`. Baseline R6's scored submission is included. Five older failed checks retain unknown source associations; partial failure metadata is preserved where found and genuinely missing archives remain explicit. PW38 check status and archived official evidence are shown below, separately from scored submissions and older failed/source-unknown checks. Preserved-snapshot union prevents latest50 pagination from dropping baseline R6.

PW35's real router gains are absent best rs32, but their stack was already tested as PW36 (3212.415452us). The omitted historical EH epilogue0.13–0.16us result remains visible with weaker raw provenance. Latest cursorv3 contains tiny wins, ties and regressions; it establishes no incremental shipping gain. PW37 and PW38 source work are not leaderboard rows without their own API run jobs.

## Every leaderboard submission, chronologically

|UTC submitted|Version / job|Score us|What was stacked / actually exercised|
|---|---|---:|---|
'''
 for r in catalog['submissions']:
  text+=f"|{r['submitted_utc']}|{r['candidate'] or 'UNKNOWN'} / `{r['job_id']}`|{r['score_us']:.6f}|{r['stack_description']}. [Raw](../{r['raw']['path']})|\n"
 pw38=next((r for r in catalog['checks'] if r['job_id']=='a3ed4ce5886a4415'),{})
 text=text.replace('{pw38_check_status}',str(pw38.get('status','unknown'))).replace('{all_jobs}',str(counts['all_official_jobs'])).replace('{runs}',str(counts['leaderboard_submissions'])).replace('{checks}',str(counts['checks']))
 text+='''
Each row is an independent official measurement. Scores are outcomes, not causal feature deltas. All10 archived scored jobs passed9/9. Source descriptions state ancestry/availability; per-job picks and actual timed profile reports in the JSON distinguish selection from compiled presence. Initial PW34's RQ phase was skipped; later PW34 repaired it.

## Prospective source stacks, not leaderboard submissions

- PW37 `7913564f…a0330`: PW36 + cursor. CPU source only; latest standalone cursorv3 has mixed/negative evidence, not an accepted improvement.
- PW38: PW36 + optional EHbit65536; cursor excluded. Local full30-shape validation is complete with mixed results. Official CHECK `a3ed4ce5886a4415` status: {pw38_check_status}; it is not a leaderboard submission. See its [manifest](candidates/pw38/manifest.json) and [provenance](candidates/pw38/PROVENANCE.md). Actual local EH selection and mixed performance are preserved separately; official CHECK evidence is preserved independently from leaderboard scoring.

## Every preserved improvement finding

|Finding|Status / currently included|Evidence, tested stack and remaining action|
|---|---|---|
'''
 for r in catalog['improvements']:
  text+=f"|{r['id']}|{r['classification'].replace('_',' ').lower()}; {r['currently_included']}|{r['finding']} Tested official stack: {', '.join(dict.fromkeys(r['stack_tested_jobs']+r['score_tested_jobs'])) or 'not established for this finding'}. Leaderboard-scored jobs: {', '.join(r['score_tested_jobs']) or 'none'}. {r['next_action']}|\n"
 text+=f'''\nThe **{counts['feature_findings']} feature findings** are distinct from **{counts['positive_observations']} positive rounded preparation observations**, including **{counts['below_0_3us']} below0.3us / {counts['below_0_5us']} below0.5us**. All are preserved without legacy kill-bar filtering; the original224 observations (76below0.3/107below0.5) remain immutable, with new official-job observations appended separately. Separate local paired observations retain the latest STMatrix parent+0.175999us/X4 parent+0.352001us alongside actual-incumbent−7.455999us/−7.472000us; both are NOT SHIPPED. The independent official audit additionally identifies {counts['selected_positive_observations']} selected positive observations across its33 ledger jobs; repeated bakes/alternative modes are not independent features or additive gains. Small paired cursor positives from the latest v3 audit are separately retained in that finding, not fabricated as official bake rows.

## Tables and evidence

- [All10 submissions](tables/submissions.csv), [all{counts['checks']} checks](tables/checks.csv), [all{counts['all_official_jobs']} official jobs](tables/all_official_jobs.csv).
- [All improvement findings](tables/improvements.csv), [all positive preparation observations](tables/positive_observations.csv), [local paired observations including losses](tables/local_observations.csv), [all{counts['candidates']} candidate directories](tables/candidates.csv).
- [Machine catalog](history_catalog.json): exact source18-component map/all30 serving routes, source hashes, raw/log/review links, primary input inventory and missing evidence.
- [Official component history](audits/current-leaderboard-gains-official.md), [source ancestry](audits/current-leaderboard-stack-source.md), [corpus completeness](audits/current-leaderboard-gains-completeness.md), [latest cursorv3](audits/smallm-v3-01/report.md).

## Preservation and update method

1. Append each new API snapshot path/SHA to `history_sources.json`, advance its active snapshot/revision, and preserve all older captures. Archive each official/API response and raw log/profile under its immutable job ID. Keep failures/noarchive/unknownsource rows. Mode`run` identifies a leaderboard submission; local H100 experiments and CPU-only source stacks remain separate.
2. Freeze each implementation and review, recording SHA256, parent and attempt. Attach actual prepare picks and timed graph/profile evidence. Never infer selection from a variant's presence in source.
3. Add a feature-level finding or a new immutable observation linked to its comparator, hardware, numerical/graph validity and uncertainty. Retain negative/conflicting estimators and every small positive; no threshold deletes evidence.
4. Record stack integration and actual tested/scored job IDs. A measured gain against a failed parent, mathematical scenario or CPU instruction reduction is not an official incumbent gain.
5. Run the generator and CPU history tests; review changed rows and input hashes. Keep historical raw/source parents unchanged. Updating an audit changes its hash and requires explicit regeneration.

Server`content_hash` has an undocumented algorithm: live rs32`cd14084a…` differs from local fileSHA256`a7b6e010…`. Path/git/source comparisons establish association; they do not attest downloaded historical uploaded payload bytes. Earlier PW34 variants likewise need revision-specific payload provenance. Remote historical lab distributions missing locally remain explicit gaps. No benchmark achievement or additive cross-job score claim follows from this catalog.
'''
 text=text.replace('{pw38_check_status}',str(pw38.get('status','unknown')))
 if pw38.get('score_us') is not None:
  text+='\nPW38 CHECK: **'+str(pw38['score_us'])+'us**, '+str(pw38.get('tests_passed'))+'/'+str(pw38.get('tests'))+' tests; all30 shapes archived. This is not a leaderboard submission or a new leaderboard best.\n'
 files['proj_rsi/HISTORY.md']=text
 return files

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args(argv)
 catalog=build();outputs=render(catalog);different=[]
 for name,text in outputs.items():
  path=ROOT/name
  if args.check:
   if not path.exists() or path.read_text()!=text:different.append(name)
  else:
   import os,tempfile
   path.parent.mkdir(parents=True,exist_ok=True)
   fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
   try:
    with os.fdopen(fd,'w') as handle:handle.write(text);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
   finally:
    if os.path.exists(temporary):os.unlink(temporary)
 if different:raise SystemExit('History outputs stale/missing: '+', '.join(different))
 print(json.dumps(catalog['counts'],sort_keys=True))
if __name__=='__main__':main()

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
 'pw39_local':'proj_rsi/audits/pw39-stack-01/analysis.json',
 'pw40_local':'proj_rsi/audits/pw40-stack-01/analysis.json',
 'pw43_local':'proj_rsi/audits/pw43-runtime-independent.json',
 'pw45_local':'proj_rsi/audits/pw45-runtime-independent.json',
 'pw46_local':'proj_rsi/audits/pw46-runtime-independent.json',
 'pw43_brief':'proj_rsi/audits/pw43-refinement-brief.json',
 'complete_stack_local':'proj_rsi/audits/complete-gv-rq-eh-o32-policy-result/analysis.json',
 'eh_o32_local':'proj_rsi/audits/gv-private-eh-o32-policy-result/analysis.json',
 'submission_stack':'proj_rsi/audits/submission-stack-20261027/analysis.json',
 'lut_whole_local':'tools/moe_ss_lut_whole_v1/actual-analysis/analysis.json',
 'lut_whole_review':'tools/moe_ss_lut_whole_v1/actual-analysis/done-evidence.json',
 'm1_policy_local':'proj_rsi/audits/gv53-splitcta-m1-policy-result/analysis.json',
 'unpooler256_local':'proj_rsi/audits/gv-lt-unpooler256-balanced-focused-result/analysis.json',
 'pw38_profile_failure':'proj_rsi/audits/pw38-selected-profile-02/analysis.json',
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
 'pw39':('pw38 + bounded prepare confirmation',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor','bounded_prepare_confirmation'],'PW38 + bounded host prepare confirmation; kernel bodies unchanged; local paired/F1 conflict and official CHECK regression, REFINE not scored'),
 'pw40':('pw38 + narrow QKV W-ring contender',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor','optional_QKV_wring_x6w4hz'],'PW38 + optional QKV128 X6/W4 contender, no PW39 confirmation; W-ring validated locally but selected nowhere; official CHECK is separate, not scored'),
 'pw45':('pw38 + optional o32 HALF fullK contenders',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor','optional_o32_HALF52_fullK_ST4_ST5'],'PW38 + optional o32 HALF52 fullK ST4/ST5 contenders; original controls retained; actual per-job selections recorded separately'),
 'gv_predxor_pw45_v2':('pw45 + standalone GV predicated escape repair',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor','optional_o32_HALF52_fullK_ST4_ST5','optional_GV_predicated_escape_xor'],'PW45 + standalone GV predicated escape repair alternatives with original controls and actual prepare selection retained'),
 'rq32_predxor':('pw45 + RQ32 predicated escape repair',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor','optional_o32_HALF52_fullK_ST4_ST5','RQ32_predicated_escape_xor'],'PW45 + RQ32 predicated escape repair; CHECK and SCORED are distinct observations'),
 'pw38':('pw36 + rs32_eh_backport',BASE+['RP_pipelined_router128']+P1+RW+P2+RQ+SMALL+RTK+['optional_EH_bit16_no_cursor'],'PW36 + optional EH bit65536; excludes cursor; completed CHECK and separate scored run'),
}

# These source compositions are frozen release relationships, not causal deltas.
GV_FEATURES=VERSION_STACKS['gv_predxor_pw45_v2'][1]
VERSION_STACKS.update({
 'gv_small_original_choice_v2':('gv_predxor_pw45_v2 + original-kernel alternative',GV_FEATURES+['optional_original_GV_choice'],'Independent GV-only diagnostic branch; local paired/F1 conflict, own official CHECK'),
 'gv_lt_qkv256_balanced_v1':('gv_predxor_pw45_v2 + balanced QKV256 Lt selector',GV_FEATURES+['balanced_QKV256_Lt_confirmation'],'Independent GV-only diagnostic branch; final RQ supersedes Lt, own official CHECK'),
 'gv53_splitcta_selection_v1':('gv_predxor_pw45_v2 + optional split-CTA selector',GV_FEATURES+['optional_GV53_splitCTA'],'Independent GV-only diagnostic branch; source availability and actual selection remain distinct'),
 'gv53_splitcta_m1_selection_v1':('gv_predxor_pw45_v2 + M1-only split-CTA selector',GV_FEATURES+['optional_GV53_M1_splitCTA'],'Independent local M1-only optional selector; mixed shape observations and aggregate regression'),
 'gv_lt_unpooler256_balanced_v1':('gv_predxor_pw45_v2 + balanced unpooler256 Lt selector',GV_FEATURES+['balanced_unpooler256_Lt_confirmation'],'Independent local focused selector; identical complete selected Lt configuration and measured regression'),
 'gv_predxor_keepchoice_v1':('gv_predxor_pw45_v2 + alternative keepchoice selector',GV_FEATURES+['alternative_GV_keepchoice_policy'],'GV-only parent plus alternative keepchoice; parallel branch, not part of compatible full stack'),
 'rq32_gv_predxor_pw45_v1':('gv_predxor_pw45_v2 + RQ predxor',GV_FEATURES+['RQ32_predicated_escape_xor'],'GV + RQ predxor; parallel branch, own scored outcome'),
 'gv_private_eh_screen_v1':('gv_predxor_pw45_v2 + private EH',GV_FEATURES+['private_EH_free_QKV_pooler128'],'GV + private EH; parallel branch, own scored outcome'),
 'gv_private_eh_o32_v1':('gv_private_eh_screen_v1 + O32 factor1.0',GV_FEATURES+['private_EH_free_QKV_pooler128','O32_positive_factor1.0'],'GV + private EH + O32 factor1.0; locally measured versus EH parent only; not officially scored'),
 'complete_gv_rq_eh_o32_v1':('gv_predxor_pw45_v2 + RQ predxor + private EH + O32 factor1.0',GV_FEATURES+['RQ32_predicated_escape_xor','private_EH_free_QKV_pooler128','O32_positive_factor1.0'],'Complete compatible GV + RQ + private EH + O32 source, locally measured versus currentGV and officially scored; keepchoice excluded; factor configuration does not prove binding selection or isolated gain'),
})

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
 board_snapshots=config.get('board_snapshots',[])
 for snapshot in board_snapshots:
  path=Path(snapshot['path'])
  if path.is_absolute() or '..' in path.parts or digest(root/path)!=snapshot['sha256']:raise ValueError('immutable board snapshot changed')
 if board_snapshots:
  boards=[r for r in board_snapshots if r['id']==config['active_board_snapshot']]
  if len(boards)!=1:raise ValueError('active board snapshot unknown')
  paths['live_board']=boards[0]['path']
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
 for entry in config.get('completed_receipt_jobs',[]):
  for key in ('receipt','result'):
   reference=entry[key];path=Path(reference['path'])
   if path.is_absolute() or '..' in path.parts or digest(root/path)!=reference['sha256']:raise ValueError('completed receipt evidence changed')
  receipt=read(root,entry['receipt']['path']);result=read(root,entry['result']['path']);jid=entry['job_id'];summary=result['summary']
  if receipt['job']!=jid or result['job_id']!=jid:raise ValueError('completed receipt job mismatch')
  association=config['job_source_associations'].get(jid)
  if not association or receipt['source_sha256']!=association['source']['sha256']:raise ValueError('completed receipt source mismatch')
  if receipt['kind'] not in ('check','run') or summary.get('status') not in ('ok','error'):raise ValueError('completed receipt outcome not terminal')
  if jid in merged:continue
  merged[jid]={'job_id':jid,'problem':receipt['problem'],'mode':receipt['kind'],'submitted_at':dt.datetime.fromisoformat(receipt['submitted_at'].replace('Z','+00:00')).timestamp(),'status':'done','result_status':summary['status'],'passed':summary['passed'],'score_us':summary['score_us'],'status_evidence_kind':'accepted_receipt_and_own_completed_official_result_not_API_capture'}
  origins[jid]=entry['result']
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
  row={'status_evidence_kind':live_row.get('status_evidence_kind','preserved_API_jobs_capture'),'status_evidence_source':api_origins[live_row['job_id']],'API_status_source':None if live_row.get('status_evidence_kind') else api_origins[live_row['job_id']],'job_id':live_row['job_id'],'submitted_at':live_row['submitted_at'],'submitted_utc':utc(live_row['submitted_at']),
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
 pw38_runs=[r for r in submissions if r['candidate']=='pw38']
 pw38_run=pw38_runs[-1] if pw38_runs else None
 team=inputs['live_board'].get('you',{}).get('team')
 own_board_rows=[r for r in inputs['live_board'].get('rows',[]) if r.get('team')==team]
 board_row=own_board_rows[0] if len(own_board_rows)==1 else None
 best=next((r for r in submissions if board_row and r['job_id']==board_row.get('job')),None)
 if best is None:best=next(r for r in submissions if r['job_id']=='fb548b5dc6f04696')
 if best['passed'] is not True or best['score_us'] is None:raise ValueError('board best lacks completed successful official run')
 previous_runs=[r for r in submissions if r['submitted_at']<best['submitted_at'] and r['passed'] is True and r['score_us'] is not None]
 previous_best=min(previous_runs,key=lambda r:r['score_us']) if previous_runs else best
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
 if pw38_check and pw38_check['passed'] is True:
  for feature in findings:
   if feature['id']=='RS128b early-half epilogue EH':
    feature['stacked_versions']=['rs128b historical lab','rs32_eh_backport','pw38']
    feature['stack_tested_jobs']=['a3ed4ce5886a4415'];feature['official_check_jobs']=['a3ed4ce5886a4415']
    feature['missing_reason']='Absent current best rs32, but now explicitly backported into PW38 and official-CHECK tested with selected qkv128/pooler128 EH. Historical raw weakness remains attached to original observation; new evidence is linked separately.'
    feature['next_action']='Retain historical tiny observation and later PW38 evidence; await the already accepted scored PW38 outcome, without another significance threshold.'
   elif feature['id']=='PW35 one-node rtk router128/256':
    if 'pw38' not in feature['stacked_versions']:feature['stacked_versions'].append('pw38')
    if 'a3ed4ce5886a4415' not in feature['stack_tested_jobs']:feature['stack_tested_jobs'].append('a3ed4ce5886a4415')
 if pw38_run:
  record=next(r for r in findings if r['id']=='PW38 full-policy EH stack local validation + official check')
  record['official_scored_submission_jobs']=[r['job_id'] for r in pw38_runs]
  record['scored_submission_status']=pw38_run['status'];record['scored_submission_score_us']=pw38_run['score_us']
  record['currently_included']='Not current best rs32; PW38 scored submission '+pw38_run['job_id']+' status '+pw38_run['status']
  record['classification']='PW38_VALIDATED_LOCAL_AND_CHECK_SCORED_SUBMISSION_'+pw38_run['status'].upper()+'_NOT_CURRENT_BEST'
  record['evidence']+=[pw38_run['release_attestation']]+pw38_run['additional_release_attestations']
  record['next_action']='Wait existing scored submission '+pw38_run['job_id']+'; keep its pending score NULL and do not copy CHECK score or resubmit. Continue source-bound stacking research.'
  if pw38_run['passed'] is True and pw38_run['score_us'] is not None:
   record['classification']='PW38_OFFICIAL_STACK_SCORED_AND_BOARD_VERIFIED_CURRENT_BEST' if best['job_id']==pw38_run['job_id'] else 'PW38_OFFICIAL_STACK_SCORED_COMPLETED'
   record['currently_included']='Yes; current best PW38' if best['candidate']=='pw38' else 'PW38 scored, current best is '+str(best['candidate'])
   record['score_tested_jobs']=[pw38_run['job_id']]
   if pw38_run['job_id'] not in record['stack_tested_jobs']:record['stack_tested_jobs'].append(pw38_run['job_id'])
   record['finding']+=' Scored run '+pw38_run['job_id']+' completed9/9,30shapes at '+str(pw38_run['score_us'])+'us; board selection for current best is '+str(best['candidate'])+'. Observed package point difference versus prior best is not isolated EH causality.'
   record['next_action']='Preserve scored source, outcomes, current actual picks and all prior packages; continue evidence-driven stacking per user authorization.'
   for feature in findings:
    if feature['id'] in ('RS128b early-half epilogue EH','PW35 one-node rtk router128/256'):
     feature['currently_included']='Yes; current best PW38 scored stack'
     feature['missing_reason']='Historical rs32 omission is resolved in the current PW38 scored stack; original comparator-specific observations remain preserved, not relabelled causal gains.'
     if pw38_run['job_id'] not in feature['score_tested_jobs']:feature['score_tested_jobs'].append(pw38_run['job_id'])
     if pw38_run['job_id'] not in feature['stack_tested_jobs']:feature['stack_tested_jobs'].append(pw38_run['job_id'])
     feature['next_action']='Retain original observations and later current-best PW38 scored evidence; continue stacking without discarding tiny measured improvements.'
 if best['candidate']=='pw38':
  for feature in findings:
   feature['current_best_context']='PW38 '+best['job_id']+'; original observation/evidence retained with its own historical comparator.'
   if feature['classification'].startswith('SHIPPED_INCLUDED'):
    if 'pw38' not in feature['stacked_versions']:feature['stacked_versions'].append('pw38')
    if best['job_id'] not in feature['score_tested_jobs']:feature['score_tested_jobs'].append(best['job_id'])
    if best['job_id'] not in feature['stack_tested_jobs']:feature['stack_tested_jobs'].append(best['job_id'])
   if feature['id']=='pw22/pw23/pw29h router128 progression':feature['currently_included']='Code retained as controls/fallback; RTK supersedes old selected RW at128/256 in current scored PW38.'
 for version,jid in [('pw39','49d4063f97c14250'),('pw40','6c4eff6d63784f90')]:
  data=inputs[version+'_local'];record=next((r for r in jobs if r['job_id']==jid),None)
  folder=version+'-stack-01';aggregate=data['local_descriptive_geomean_control_over_'+version]['pw38']
  finding={'id':version.upper()+' full-policy local validation + official check','immutable_id':'improvement.'+version+'.stack.fullscope.01','classification':'LOCAL_VALID_WITH_MIXED_SELECTION_EVIDENCE_OFFICIAL_CHECK_'+str(record['status'] if record else 'UNKNOWN').upper()+'_NOT_SCORED',
   'in_rs32':'No','currently_included':'No; current scored best remains PW38','scope':'Local H100 full30-shape comparison plus distinct official CHECK, never a scored submission',
   'finding':f"Local all4350 raw calls/30shapes/222 task checks/180 strict terminal oracles/360 mutations validated. Versus PW38, descriptive paired six-load GM control/candidate={aggregate['paired_marginal_median_sums_us']:.9f}; conservative F1 GM={aggregate['F1_conservative_sum_us']:.9f}. Every small shape win, tie, loss and both estimators are preserved below; ratios are local summaries, not official-score forecasts.",
   'verification':'Source-bound complete local artifact reanalysis/independent review; actual prepare/JIT/counter/current-X/cleanup evidence retained.',
   'evidence':[ref(root,p) for p in [paths[version+'_local'],'proj_rsi/audits/'+folder+'/report.md','proj_rsi/audits/'+version+'-stack-independent.json','proj_rsi/scheduling/validations/'+folder+'.json']],
   'stacked_versions':[version],'stack_tested_jobs':[jid] if record and record['passed'] else [],'score_tested_jobs':[],'local_tested_attempt':data['job_id'],'official_check_jobs':[jid],
   'official_check_status':record['status'] if record else None,'official_check_score_us':record['score_us'] if record else None,'missing_reason':'Official CHECK is distinct from scored release; pending/completed status does not establish isolated mechanism benefit.'}
  if version=='pw39':
   finding['classification']='VALID_LOCAL_PAIRED_GAIN_F1_LOSS_OFFICIAL_CHECK_REGRESSION_REFINE_NOT_SCORED'
   finding['finding']+=' Bounded confirmation actually ran, but no serving-kernel source changed. Official CHECK49d4063f97c14250 passes9/9 at3212.6810592521706us, above PW38 best3198.6175079817267us. Root decision REFINE/not scored; this version is not a family kill.'
   finding['next_action']='Retain opposite F1 result and official regression; diagnose selection/confirmation context with a new immutable version, do not score the losing check or retire the family.'
  else:
   finding['finding']+=' W-ring received70 successful host calls and runtime/JIT validation, but was SELECTED NOWHERE. QKV128 screen17.09us beats old nonEH17.15 by0.06 but loses actual EH16.83 by0.26. Whole-policy positive point estimates cannot be attributed to serving W-ring; no PW39 confirmation included.'
   finding['next_action']='Await existing official CHECK6c4eff6d63784f90; do not copy local ratios into H200 score or infer W-ring causal gain. Preserve unselected contender and actual prepared policy.'
   if record and record['status']=='done' and record['score_us'] is not None:
    finding['classification']='VALID_LOCAL_POLICY_POINT_GAINS_WRING_UNSELECTED_OFFICIAL_CHECK_REGRESSION_REFINE_NOT_SCORED'
    log_text=(root/record['log']['path']).read_text() if record['log'] else ''
    import re
    screen={match[1]:float(match[2]) for match in re.finditer(r'BAKE qkv_gate m=128 (rq:[^ ]+) ([0-9.]+)',log_text)}
    done=re.search(r'PREP done in (\d+)s',log_text)
    finding['official_prepare']={'total_seconds':int(done[1]) if done else None,'RQ_budget_skip_present':'skipped-budget' in log_text,'QKV128_screen_us':screen,'actual_QKV128_pick':next((p for p in record['actual_picks'] if p['layer']=='qkv_gate' and p['m']==128),None)}
    finding['finding']+=' Official CHECK6c4eff6d63784f90 completed9/9 at '+str(record['score_us'])+'us, above PW38best3198.6175079817267us. All prepare phases completed; total '+str(finding['official_prepare']['total_seconds'])+'s, no RQ budget skip. H200 QKV128 W-ring17.12us loses old hinted nonEH16.67us; actual selection stays old hinted RQ. Root decision REFINE/not scored, not family retirement.'
    finding['next_action']='Preserve full local positives, unselected W-ring, and official regression. Diagnose actual source/selection/phase costs in a new immutable version; no scoring of this losing check, no family kill.'
  if record:
   finding['evidence']+=[r for r in (record['raw'],record['log'],record['release_attestation']) if r]
   finding['official_check_picks']=record['actual_picks']
  findings.append(finding)
  for shape in data['all_shapes']:
   for control,comparison in shape['comparisons'].items():
    gain=comparison['paired_control_minus_'+version+'_us'];f1=comparison['F1_control_minus_'+version+'_us'];wtl=comparison[version.upper()+'_wins_ties_losses']
    local_observations.append({'observation_id':'paired.'+data['job_id']+'.'+shape['shape']+'.'+control,'attempt_id':data['job_id'],'version':version,'shape':shape['shape'],'control':control,'gain_us':gain,'F1_gain_us':f1,'paired_ratio':comparison['paired_median_control_over_'+version],'wins':wtl[0],'ties':wtl[1],'losses':wtl[2],'pairs':sum(wtl),'same_selected_plan':comparison['same_selected_plan_as_'+version],'hardware':'H100','result_sha256':data['result_sha256'],'evidence':ref(root,paths[version+'_local']),'shipping_gain':False})
 # Local PW43 is not an official job; retain every signed shape contrast.
 pw43=inputs['pw43_local']
 findings.append({'id':'PW43 full-policy weighted Lt local regression','immutable_id':'improvement.pw43.stack.fullscope.01','classification':'VALID_LOCAL_FULL_POLICY_REGRESSION_NOT_SHIPPED','in_rs32':'No','currently_included':'No; NOT SHIPPED','scope':'Actual H100 local fullprepare against PW38; no official submission',
  'finding':'4350 raw samples/30 shapes,222 numerical checks,180 terminal oracles and360 mutations validated. PW38 minus PW43 descriptive conditional local score: paired '+str(pw43['pw38_minus_pw43_local_score_estimate_us']['paired'])+'us; F1 '+str(pw43['pw38_minus_pw43_local_score_estimate_us']['F1'])+'us. Both regress; positive/tiny shape observations remain below. Only qkv256 confirmation completed; later cells expired. Not an official H200 score or isolated source causality.',
  'verification':'Independent complete local validity audit; one prepare per source; repeat-prepare frequency/regret unknown; loaded SO bytes absent from collection.', 'stacked_versions':['pw43'],'stack_tested_jobs':['pw43.stack.fullscope.01'],'score_tested_jobs':[],'missing_reason':'Actual-incumbent fullpolicy regression; no official run invented.','next_action':'Use immutable PW43 refinement brief two-by-two accounting/comparator experiment; preserve actual-incumbent versus proxy distinction, unchanged original budgets and every raw sign.',
  'evidence':[ref(root,paths['pw43_local']),ref(root,paths['pw43_brief']),ref(root,'proj_rsi/audits/pw43-runtime-independent.md')]})
 for shape in pw43['all_shape_signed_contrasts']:
  local_observations.append({'observation_id':'paired.pw43.stack.fullscope.01.'+shape['shape']+'.pw38','attempt_id':'pw43.stack.fullscope.01','version':'pw43','shape':shape['shape'],'control':'pw38','gain_us':shape['paired_pw38_minus_pw43_us'],'F1_gain_us':shape['F1_bracket_pw38_minus_pw43_us'],'paired_ratio':None,'wins':None,'ties':None,'losses':None,'pairs':None,'same_selected_plan':shape['picks']['pw38']==shape['picks']['pw43'],'hardware':'H100','result_sha256':pw43['result_sha256'],'evidence':ref(root,paths['pw43_local']),'shipping_gain':False})
 findings.append({'id':'PW38 selected profile context repair and GV cluster failure','immutable_id':'improvement.pw38.selected.profile.context.01','classification':'INVALID_PROFILE_DIAGNOSIS_NO_COUNTER_OR_LATENCY_ADMISSION','in_rs32':'No','currently_included':'No serving change','scope':'H100 selected profile attempts; informational setup diagnosis only',
  'finding':'Source revision2/jobv1 failed CUDA context metadata before prepare. Revision3/jobv2 repaired actual context and completed unchanged prepare73.931256s, then capture rc9/GV clusterLaunchFailed. Actual QKV1 S2/V0 uses SM90 DSM cluster barriers, not global-counter waiting. Missing postrange terminal/counter/cleanup fields invalidate all profile rows, including completed router rows. No admitted counter, latency gain or actual cause inferred.',
  'verification':'Source-bound raw collection and independent CPU diagnosis; context repair is observed, recipe/replay/instrumentation cause remains hypothesis.','stacked_versions':[],'stack_tested_jobs':['pw38.selected.profile.v1.01','pw38.selected.profile.v2.02'],'score_tested_jobs':[],'missing_reason':'Incomplete strict terminal oracles; all exported counters diagnostic only.','next_action':'Review hardware-only metrics versus broad sections with same five actual graph nodes and all terminal gates; success implicates removed recipe without proving software-patch versus replay mechanism.', 'evidence':[ref(root,paths['pw38_profile_failure'])]})
 # Validated local PW45/PW46 never synthesize official submission IDs.
 for version,attempt in [('pw45','research.pw45.gpu.01'),('pw46','pw46.stack.fullscope.01')]:
  data=inputs[version+'_local'];delta=data['pw38_minus_'+version+'_local_score_estimate_us']
  findings.append({'id':version.upper()+' actual full-policy runtime','immutable_id':'improvement.'+version+'.stack.fullscope.01','classification':'VALID_LOCAL_PAIRED_POINT_GAIN_F1_POINT_LOSS_NOT_OFFICIAL' if version=='pw45' else 'VALID_LOCAL_FULL_POLICY_REGRESSION_NOT_SHIPPED','in_rs32':'No','currently_included':'No; local experiment only','scope':'Actual independently prepared PW38 versus '+version.upper()+' on H100; no H200 forecast',
   'finding':'All222 correctness records/4350 raw samples/180 terminal oracles/360 mutations validate. PW38-minus-'+version.upper()+' local conditional aggregate: paired '+str(delta['paired'])+'us; F1 '+str(delta['F1'])+'us. All30 exact signed effects retained. '+('Positive paired point and opposite F1 remain explicit without new effectsize/significance gate; actual o32 selects GV45, not fixed HALF52.' if version=='pw45' else 'Both aggregate points regress. Proxy preservation does not prove actual-incumbent parity; downstream family bake may supersede Lt confirmation.'),
   'verification':'Independent collected inventory/source/JIT/terminal/raw/counter-source/cleanup audit; one prepare per source, no repeat-prepare frequency evidence.','stacked_versions':[version],'stack_tested_jobs':[attempt],'score_tested_jobs':[],'missing_reason':'No official submission inferred from local job; root owns evaluation decision.','next_action':'Preserve exact source and actual selected plans; root can evaluate measured positive point under official controls. Refine losing versions with a new immutable hypothesis, no forced winner or family kill.','evidence':[ref(root,paths[version+'_local']),ref(root,'proj_rsi/audits/'+version+'-runtime-independent.md')]})
  for row in data['all_shape_signed_contrasts']:
   wtl=row['paired_wins_ties_losses'];local_observations.append({'observation_id':'paired.'+attempt+'.'+row['shape']+'.pw38','attempt_id':attempt,'version':version,'shape':row['shape'],'control':'pw38','gain_us':row['paired_pw38_minus_'+version+'_us'],'F1_gain_us':row['F1_bracket_pw38_minus_'+version+'_us'],'paired_ratio':None,'wins':wtl[0],'ties':wtl[1],'losses':wtl[2],'pairs':sum(wtl),'same_selected_plan':row['picks']['pw38']==row['picks'][version],'hardware':'H100','result_sha256':data['result_sha256'],'evidence':ref(root,paths[version+'_local']),'shipping_gain':False})
 fixed=inputs['pw45_local']['fixed_o32_mechanism']
 findings.append({'id':'PW45 fixed o32 ST5 tiny gain and ST4 loss','immutable_id':'improvement.pw45.fixed.o32.01','classification':'VALID_LOCAL_FIXED_TINY_GAIN_UNSELECTED_NOT_SHIPPED','in_rs32':'No','currently_included':'No; actual PW45 o32 selects GV45','scope':'H100 o32 fullK1024/N8192 direct fixed arms, separate from prepared policy','finding':'Fixed ST5 HALF52 versus old ST5 earlyZI34 gains paired0.063999556us/F1 0.080000143us (17/30 paired wins); ST4 HALF52 loses paired−0.496000051us/F1−0.560000073us. Preserve every tiny/negative sign. Active warpgroups/shared/KS/store differ; no singlefactor or package causality asserted.','verification':'210 raw,6 strict terminal outputs,12 mutations,3 actual single-kernel graph profiles,all1024basis+FP64 and fullphysical counter/sink source guards validate.','stacked_versions':['pw45'],'stack_tested_jobs':['research.pw45.gpu.01'],'score_tested_jobs':[],'missing_reason':'New fixed contenders are not selected in the actual fullpolicy serving path.','next_action':'Retain fixed gain as mechanism evidence; compare future actual selected fullpolicy under independent controls, no hardcoded winner.','evidence':[ref(root,paths['pw45_local'])]})
 for contrast,row in fixed['contrasts'].items():
  local_observations.append({'observation_id':'fixed.research.pw45.gpu.01.'+contrast,'attempt_id':'research.pw45.gpu.01','version':'pw45_fixed_o32','shape':'o32/'+contrast,'control':contrast.split('_minus_')[0],'gain_us':row['paired_median_gain_us'],'F1_gain_us':row['F1_endpoint_average_gain_us'],'paired_ratio':row['paired_median_ratio'],'wins':row['wins'],'ties':None,'losses':None,'pairs':30,'same_selected_plan':False,'hardware':'H100','result_sha256':inputs['pw45_local']['result_sha256'],'evidence':ref(root,paths['pw45_local']),'shipping_gain':False})
 # Completed CHECK and accepted scored intent remain distinct authorities.
 pw45_checks=[row for row in checks if row['candidate']=='pw45']
 pw45_runs=[row for row in submissions if row['candidate']=='pw45']
 for feature in findings:
  if feature['immutable_id'] not in ('improvement.pw45.stack.fullscope.01','improvement.pw45.fixed.o32.01'):continue
  if pw45_checks and feature['immutable_id']=='improvement.pw45.stack.fullscope.01':
   feature['classification']='VALID_LOCAL_PAIRED_POINT_GAIN_F1_POINT_LOSS_OFFICIAL_CHECK_POINT_GAIN_SCORED_PENDING'
   feature['currently_included']='No; current best remains PW38; PW45 official CHECK complete and scored result pending'
  feature['official_check_jobs']=[row['job_id'] for row in pw45_checks]
  feature['pending_scored_submission_jobs']=[row['job_id'] for row in pw45_runs if row['status'] not in ('done','failed','cancelled')]
  for row in pw45_checks:
   feature['stack_tested_jobs'].append(row['job_id'])
   feature['evidence'] += [r for r in (row['raw'],row['log'],row['release_attestation']) if r]
   if row['status']=='done' and row['passed'] is True:
    feature['official_check_score_us']=row['score_us'];feature['official_check_picks']=row['actual_picks']
    feature['finding']+=' Official CHECK'+row['job_id']+' passed9/9 at'+str(row['score_us'])+'us, observed package point'+str(3198.6175079817267-row['score_us'])+'us below scored PW38. CHECK is not a leaderboard submission or isolated kernel causal gain.'
  completed_runs=[row for row in pw45_runs if row['status']=='done' and row['passed'] is True]
  if completed_runs:
   row=completed_runs[-1];feature['score_tested_jobs']=[r['job_id'] for r in completed_runs];feature['stack_tested_jobs'] += [r['job_id'] for r in completed_runs]
   feature['official_scored_score_us']=row['score_us'];feature['official_scored_picks']=row['actual_picks']
   if feature['immutable_id']=='improvement.pw45.fixed.o32.01':feature['currently_included']='No; HALF52 unselected. Local o32 GV45, CHECK r6, SCORED existing RQ34 are separate observations.'
   feature['evidence'] += [r for r in (row['raw'],row['log'],row['release_attestation']) if r]+row['additional_release_attestations']
   if feature['immutable_id']=='improvement.pw45.stack.fullscope.01':
    feature['classification']='PW45_OFFICIAL_STACK_SCORED_AND_BOARD_VERIFIED_CURRENT_BEST' if best['candidate']=='pw45' else 'PW45_OFFICIAL_STACK_SCORED_COMPLETED'
    feature['currently_included']='Yes; current best PW45 package' if best['candidate']=='pw45' else 'PW45 scored; another package is current best'
   feature['finding']+=' Distinct SCORED'+row['job_id']+' completed9/9 at'+str(row['score_us'])+'us; actual o32 selected old t32s5ze-k2 (RQflags34), not new HALF52. Package difference is not isolated new kernel causality.'
   feature['missing_reason']='Scored source/actual routes archived; new HALF52 optional contenders remain unselected.'
   feature['next_action']='Continue current selected hotpath research from this frozen scored stack; preserve PW38 historical and all local/CHECK/scored signs.'
  if feature['pending_scored_submission_jobs']:
   feature['finding']+=' Same frozen source accepted for scored run '+','.join(feature['pending_scored_submission_jobs'])+'; final score and board outcome UNKNOWN/pending. Do not copy CHECK score into that row.'
   feature['next_action']='Await actual scored result and fresh board authority; preserve tiny CHECK point and opposite local F1 without a new effectsize/significance gate.'
 if best['candidate'] in ('pw45','gv_predxor_pw45_v2','rq32_predxor'):
  for feature in findings:
   feature['current_best_context']=best['candidate']+' '+best['job_id']+'; original evidence and comparator retained historically.'
   if feature['classification'].startswith('SHIPPED_INCLUDED'):
    feature['currently_included']='Source retained in '+best['candidate']+'; actual selection is recorded in current scored routes'
    if 'pw45' not in feature['stacked_versions']:feature['stacked_versions'].append('pw45')
    if best['job_id'] not in feature['score_tested_jobs']:feature['score_tested_jobs'].append(best['job_id'])
    if best['job_id'] not in feature['stack_tested_jobs']:feature['stack_tested_jobs'].append(best['job_id'])
   if feature['id']=='RS128b early-half epilogue EH':
    feature['currently_included']='Compiled in '+best['candidate']+'; actual EH selection is recorded in scored routes; historical PW38 evidence retained'
    feature['missing_reason']='Source inclusion/scored package validation established; actual current choices recorded separately. No causal EH gain assigned.'
   if feature['id']=='PW35 one-node rtk router128/256':feature['currently_included']='Source retained; actual RTK selection is recorded in current '+best['candidate']+' scored routes'
 for version,title in [('gv_predxor_pw45_v2','GV predicated escape policy scored current best'),('rq32_predxor','RQ32 predicated escape official checks and scored outcome'),('gv_predxor_keepchoice_v1','GV alternative keepchoice scored outcome'),('rq32_gv_predxor_pw45_v1','GV plus RQ predxor scored outcome'),('gv_private_eh_screen_v1','GV plus private EH scored outcome'),('complete_gv_rq_eh_o32_v1','Complete GV RQ privateEH O32 scored outcome')]:
  runs=[row for row in submissions if row['candidate']==version];version_checks=[row for row in checks if row['candidate']==version]
  if not runs:continue
  row=runs[-1];is_best=row['job_id']==best['job_id']
  findings.append({'id':title,'immutable_id':'improvement.'+version+'.official.01','classification':'OFFICIAL_SCORED_CURRENT_BEST' if is_best else 'OFFICIAL_SCORED_COMPLETED_NOT_BEST','in_rs32':'No','currently_included':'Yes; current scored package' if is_best else 'No; independently scored package, not current best','scope':'Exact frozen release, own official 9/9 and30shape profiles; package point, no isolated mechanism causality','finding':'Scored '+row['job_id']+' completed9/9 at '+str(row['score_us'])+'us. '+('GV standalone predxor selected in four actual scored routes: poolerM1/M8/M16 and unpoolerM16; other route/selection changes retained separately. Direct scored submission followed valid local full-policy evidence; no redundant official CHECK.' if version=='gv_predxor_pw45_v2' else VERSION_STACKS[version][2]+'. No isolated causal or additive gain claimed.' if version!='rq32_predxor' else 'RQ32 CHECK87044866f4494726 at3182.7213us and SCORED7b2da0f017d54e3d at3193.8183us are distinct outcomes. This source is not the parent of current GV-only best.'),'verification':'Own completed official result and preserved API/board captures; current per-route source bindings and scored reports retained.','stacked_versions':[version],'stack_tested_jobs':[r['job_id'] for r in version_checks+runs],'score_tested_jobs':[r['job_id'] for r in runs],'missing_reason':'No isolated source causal delta inferred.','next_action':'Continue measured stacking against actual current scored stack without minimum-gain veto.','evidence':[row['raw'],row['release_attestation']],'official_check_jobs':[r['job_id'] for r in version_checks],'official_scored_score_us':row['score_us'],'official_scored_picks':row['actual_picks']})
 # Preserve both newly completed full30 comparisons with their actual controls.
 for key,version,control,attempt in [('complete_stack_local','complete_gv_rq_eh_o32_v1','gv_predxor_pw45_v2','complete.gv.rq.eh.o32.fullpolicy.01'),('eh_o32_local','gv_private_eh_o32_v1','gv_private_eh_screen_v1','gv.privateeh.o32.fullpolicy.01')]:
  data=inputs[key]
  for shape,point in data['shapes'].items():
   wins,ties,losses=point['paired_wins_ties_losses']
   local_observations.append({'observation_id':'paired.'+attempt+'.'+shape,'attempt_id':attempt,'version':version,'shape':shape,'control':control,'gain_us':point['paired_gain_us'],'F1_gain_us':point['F1_gain_us'],'wins':wins,'ties':ties,'losses':losses,'pairs':32,'hardware':'H100','result_sha256':data['result_sha256'],'shipping_gain':False,'evidence':ref(root,paths[key])})
 data=inputs['eh_o32_local']
 findings.append({'id':'EH plus O32 incremental local full30 versus EH parent','immutable_id':'improvement.gv_private_eh_o32_v1.local.01','classification':'VALID_LOCAL_FULL30_POSITIVE_VS_EH_PARENT_NOT_OFFICIALLY_SCORED','currently_included':'No; not current GV best','scope':'Whole package versus private EH parent only, not currentGV','finding':'Paired '+str(data['actual_policy_geometric_load_speedup']['paired'])+'x; F1 '+str(data['actual_policy_geometric_load_speedup']['F1'])+'x. Signed local losses retained; ratios cannot be added to other comparisons.','verification':data['state'],'stacked_versions':['gv_private_eh_o32_v1'],'stack_tested_jobs':['gv.privateeh.o32.fullpolicy.01'],'score_tested_jobs':[],'missing_reason':'No official scored outcome; no isolated feature causal gain','next_action':'Preserve incremental evidence separately from complete-stack score','evidence':[ref(root,paths['eh_o32_local'])]})
 # New CPU-audited local results remain distinct from official receipt rows.
 for key,version,attempt,title in [('m1_policy_local','gv53_splitcta_m1_selection_v1','research.gv53.splitcta.m1.fullpolicy.01','M1-only splitCTA actual full30 local policy'),('unpooler256_local','gv_lt_unpooler256_balanced_v1','gv.lt.unpooler256.balanced.focused.01','Balanced unpooler256 focused actual local policy')]:
  data=inputs[key]
  summary=('Paired load ratio '+str(data['actual_policy_geometric_load_speedup']['paired'])+'; endpoint F1 '+str(data['actual_policy_geometric_load_speedup']['F1'])+'; conservative F1 '+str(data['actual_policy_geometric_load_speedup']['F1_conservative'])+'. '+str(data['prepare_interpretation'])) if key=='m1_policy_local' else data['reason']
  findings.append({'id':title,'immutable_id':'improvement.'+attempt,'classification':'VALID_LOCAL_ACTUAL_POLICY_REGRESSION_NOT_SCORED','currently_included':'No; independent local branch','scope':'H100 actual selected policy versus current GV source; no official leaderboard claim','finding':summary,'verification':data['state'],'stacked_versions':[version],'stack_tested_jobs':[attempt],'score_tested_jobs':[],'missing_reason':'Local observations do not establish official gain; all signs and source pins retained','next_action':'Preserve observations; diagnose selection mechanism and compare new immutable versions against current source','evidence':[ref(root,paths[key])]})
  points=data['shapes'] if key=='m1_policy_local' else {'verify32/unpooler':{'paired_gain_us':data['timing']['paired_delta_us'],'F1_gain_us':data['timing']['F1_conservative_control_minus_candidate_us'],'paired_wins_ties_losses':data['timing']['paired_win_tie_loss']}}
  for shape,point in points.items():
   wins,ties,losses=point['paired_wins_ties_losses']
   local_observations.append({'observation_id':'paired.'+attempt+'.'+shape,'attempt_id':attempt,'version':version,'shape':shape,'control':'gv_predxor_pw45_v2','gain_us':point['paired_gain_us'],'F1_gain_us':point['F1_gain_us'],'wins':wins,'ties':ties,'losses':losses,'pairs':sum([wins,ties,losses]),'hardware':'H100','shipping_gain':False,'evidence':ref(root,paths[key])})
 for row in checks:
  if row['status_evidence_kind']!='accepted_receipt_and_own_completed_official_result_not_API_capture':continue
  findings.append({'id':row['candidate']+' completed official CHECK','immutable_id':'improvement.official.check.'+row['job_id'],'classification':'OFFICIAL_CHECK_COMPLETED_NOT_SCORED_NOT_BEST','currently_included':'No; independent diagnostic branch','scope':'Own official CHECK, distinct from scored release and board authority','finding':'CHECK '+row['job_id']+' completed9/9 at '+str(row['score_us'])+'us;30 timed profiles. '+row['stack_description']+'.','verification':row['source_identity_status'],'stacked_versions':[row['candidate']],'stack_tested_jobs':[row['job_id']],'score_tested_jobs':[],'missing_reason':'CHECK is not a leaderboard submission or isolated feature causal gain','next_action':'Preserve source, own result and actual routes; refine against current scored GV source','evidence':[row['raw'],row['release_attestation']],'official_check_jobs':[row['job_id']],'official_check_score_us':row['score_us']})
 lut=inputs['lut_whole_local'];attempt='research.moe.ss.lut.whole.01'
 findings.append({'id':'MoE LUT whole serving versus SS and actual anchor','immutable_id':'improvement.'+attempt,'classification':'VALID_LOCAL_PARENT_GAIN_ACTUAL_ANCHOR_REGRESSION_NOT_SHIPPED','currently_included':'No; local MoE research, distinct from projection leaderboard','scope':'Actual four-arm H100 M8 whole serving,8 matched input/routing seeds with balanced pair order','finding':'Actual anchor remains faster: anchor/LUT ratio '+str(lut['comparisons']['anchor_minus_LUT']['median_ratio'])+' and anchor/LUTgrid '+str(lut['comparisons']['anchor_minus_LUT_grid']['median_ratio'])+',0/8 candidate wins each. LUT versus original SS improves '+str(lut['comparisons']['SS_minus_LUT']['median_us'])+'us, ratio '+str(lut['comparisons']['SS_minus_LUT']['median_ratio'])+',8/8; this parent gain is not an anchor win. Grid incremental '+str(lut['comparisons']['LUT_minus_LUT_grid']['median_us'])+'us,6/8, modest/uncertain. F1 blocks use distinct seeds/routing workloads and are noncausal unmatched workload summaries.','verification':str(lut['correctness_count'])+' own numerical checks pass;4 empty exact checks;SS/LUT/grid final outputs bit exact;67 raw calls. Reviewed DONE evidence retains cleanup pointer.','stacked_versions':['actual_SS_LUT_whole','actual_SS_LUT_grid_whole'],'stack_tested_jobs':[attempt],'score_tested_jobs':[],'missing_reason':'Valid SS-parent gain but actual anchor regression; no official projection claim','next_action':'Preserve all raw signed comparisons and confounding; diagnose remaining actual-anchor gap before a new immutable experiment','evidence':[ref(root,paths['lut_whole_local']),ref(root,paths['lut_whole_review']),ref(root,lut['result'])]})
 for name,point in lut['comparisons'].items():
  control,version=name.split('_minus_',1)
  local_observations.append({'observation_id':'paired.'+attempt+'.'+name,'attempt_id':attempt,'version':version,'shape':'MoE.M8.whole_serving','control':control,'gain_us':point['median_us'],'F1_gain_us':None,'F1_limitation':'Unmatched F1 block seeds/routing; no causal contrast','paired_ratio':point['median_ratio'],'wins':point['b_wins'],'ties':0,'losses':8-point['b_wins'],'pairs':8,'hardware':'H100','result_sha256':lut['result_sha256'],'shipping_gain':False,'evidence':ref(root,paths['lut_whole_local'])})
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
   observations.append({'observation_id':'bake.'+hashlib.sha256(token.encode()).hexdigest()[:20],'job':official_job['job_id'],'candidate':official_job['candidate'],'layer':layer,'m':m,'mode':mode,'candidate_bake_us':value,'logged_r6_control_us':control[3],'positive_bake_difference_us':gain,'below_legacy_0_3us':gain<.3,'below_legacy_0_5us':gain<.5,'actual_final_pick':picks.get((layer,m)),'log':official_job['log']['path'],'line':line_number,'control_line':control[4],'classification':'Positive rounded same-job sample versus named r6 control; not necessarily incremental versus faster existing RQ mode. No threshold filter.','feature_level':False,'catalog_cohort':'official_'+official_job['mode']+'_'+official_job['job_id']})
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
  if name in ('gv_predxor_pw45_v2','rq32_predxor','gv_predxor_keepchoice_v1','rq32_gv_predxor_pw45_v1','gv_private_eh_screen_v1','complete_gv_rq_eh_o32_v1'):candidate['status']='SCORED_COMPLETED_BOARD_VERIFIED_CURRENT_BEST' if best['candidate']==name else 'OFFICIAL_SCORED_COMPLETED_NOT_CURRENT_BEST'
  if name in ('gv_small_original_choice_v2','gv_lt_qkv256_balanced_v1','gv53_splitcta_selection_v1'):candidate['status']='OFFICIAL_CHECK_COMPLETED_NOT_SCORED'
  if name in ('gv53_splitcta_m1_selection_v1','gv_lt_unpooler256_balanced_v1'):candidate['status']='VALID_LOCAL_ACTUAL_POLICY_REGRESSION_NOT_SCORED'
  if name=='gv_private_eh_o32_v1':candidate['status']='VALID_LOCAL_FULL30_VS_EH_PARENT_ONLY_NOT_SCORED'
  if name=='pw45':candidate['status']='SCORED_COMPLETED_BOARD_VERIFIED_CURRENT_BEST' if best['candidate']=='pw45' else 'OFFICIAL_CHECK_SCORED_SUBMISSION_RECORDED'
  if name=='pw38':candidate['status']=('PW38_SCORED_SUBMISSION_'+pw38_run['status'].upper()) if pw38_run else 'PW36_PLUS_OPTIONAL_EH_VALIDATED_LOCAL_OFFICIAL_CHECK_'+('PENDING' if check_pending else 'COMPLETED')+'_NOT_SCORED'
  if name=='pw39':candidate['status']='OFFICIAL_CHECK_VALID_REGRESSION_REFINE_NOT_SCORED'
  if name=='pw40':
   official40=next((j for j in jobs if j['job_id']=='6c4eff6d63784f90'),None)
   candidate['status']='OFFICIAL_CHECK_VALID_REGRESSION_REFINE_NOT_SCORED' if official40 and official40['status']=='done' else 'LOCAL_VALID_POLICY_POINT_GAINS_WRING_UNSELECTED_CHECK_PENDING_NOT_SCORED'
  if name=='pw38':candidate['description']='PW36 + optional EH, no cursor; full30-shape local validation complete with mixed observations; accepted official check is separate from leaderboard scoring.'
  candidate['review_and_manifest_refs']=[ref(root,str(p.relative_to(root))) for p in sorted(d.glob('*')) if p.is_file() and p.name in ('manifest.json','PROVENANCE.md','hypothesis.json','RESULTS.md')]
  candidate['named_source_components']=source_nodes(root/d/'submission.py') if (d/'submission.py').exists() else []
  candidates.append(candidate)
 missing=[j['job_id'] for j in jobs if not j['raw']]
 pending=[j['job_id'] for j in jobs if j['archive_status']=='PENDING_NO_ARCHIVE']
 failed_missing=[j['job_id'] for j in jobs if j['archive_status']=='FAILED_NO_ARCHIVE']
 return {'schema_version':1,'scope':'Complete live official job snapshot + preserved local improvements/observations; CPU-only rendering',
  'authority':{'best_job':best['job_id'],'candidate':best['candidate'],'score_us':best['score_us'],'source_sha256':best['source']['sha256'] if best['source'] else None,'live_board':inputs['live_board'].get('you'),'board_row':board_row,'previous_best_job':previous_best['job_id'],'previous_score_us':previous_best['score_us'],'observed_package_point_change_us':previous_best['score_us']-best['score_us'],'point_change_is_isolated_feature_gain':False,'achievement_claim':False},
  'methodology':['Union preserved API captures by job ID to survive latest50 pagination; newest observed row wins. Append explicitly labeled accepted-receipt plus own-result rows where no newer API capture exists; do not invent captures. Mode=run determines leaderboard submissions; checks remain separate including failures/noarchive.','Stable job/candidate/improvement/observation IDs; every input/source/raw/review link carries SHA256 when locally present.','Source compiled/available features are distinct from actual per-job selected policies and timed profiles.','No independent-job score subtraction is reported as a causal feature delta.','Feature findings and repeated positive preparation observations are separate tables; no positive threshold removes small observations.','Keep invalid attempts and superseded parents immutable; repairs get new source or attempt identity.','Review provenance gaps explicitly; server content_hash algorithm unknown and not raw SHA256 attestation.'],
  'history_sources_config':ref(root,'proj_rsi/history_sources.json'),'active_jobs_snapshot':source_config['active_jobs_snapshot'],'preserved_jobs_snapshots':source_config['jobs_snapshots'],'active_board_snapshot':source_config.get('active_board_snapshot'),'preserved_board_snapshots':source_config.get('board_snapshots',[]),'input_provenance':{k:ref(root,v) for k,v in paths.items()},'additional_evidence':[ref(root,p) for p in ['proj_rsi/audits/smallm-v3-01/report.md','proj_rsi/audits/native-stmatrix-01/report.md','proj_rsi/audits/native-x4-01/report.md']],
  'counts':{'active_API_snapshot_rows':active_snapshot_row_count,'all_official_jobs':len(jobs),'leaderboard_submissions':len(submissions),'checks':len(checks),'archived_jobs':len(jobs)-len(missing),'complete_profile_jobs':sum(r['archive_status']=='ARCHIVED_30_SHAPES' for r in jobs),'partial_profile_jobs':sum(r['archive_status']=='ARCHIVED_PARTIAL' for r in jobs),'no_archive_jobs':len(missing),'pending_no_archive_jobs':len(pending),'failed_no_archive_jobs':len(failed_missing),'feature_findings':len(findings),'positive_observations':len(observations),'preserved_original_positive_observations':224,'new_official_positive_observations':len(observations)-224,'new_official_positive_observations_by_job':{j['job_id']:sum(r['job']==j['job_id'] for r in observations) for j in jobs if any(r['job']==j['job_id'] and r['catalog_cohort']!='initial_45_job_audit' for r in observations)},'local_paired_observations':len(local_observations),'pending_scored_submissions':sum(r['status'] not in ('done','failed','cancelled') for r in submissions),'selected_positive_observations':len(selected_observations),'below_0_3us':sum(r['positive_bake_difference_us']<.3 for r in observations),'below_0_5us':sum(r['positive_bake_difference_us']<.5 for r in observations),'candidates':len(candidates)},
  'submissions':submissions,'checks':checks,'all_official_jobs':jobs,'improvements':findings,'positive_observations':observations,'selected_positive_observations':selected_observations,'local_paired_observations':local_observations,'candidates':candidates,
  'submission_stack_reconciliation':inputs['submission_stack'],'complete_stack_latest_local':inputs['complete_stack_local'],'eh_o32_incremental_latest_local':inputs['eh_o32_local'],'lut_whole_latest_local':inputs['lut_whole_local'],'m1_policy_latest_local':inputs['m1_policy_local'],'unpooler256_latest_local':inputs['unpooler256_local'],
  'pw39_latest_local':inputs['pw39_local'],'pw40_latest_local':inputs['pw40_local'],'pw38_latest_local':local,'native_composition_latest':combo,'cursor_v3_latest':inputs['cursor_v3'],'version_source_ancestry_comparisons':inputs['source_map']['named_AST_comparisons'],'historical_rs32_source_map':ref(root,paths['source_map']),'current_stack_view':current_view(root,best,inputs['source_map']),'current_exact_source_components':current_view(root,best,inputs['source_map'])['components'],'current_exact_selected_routes':current_view(root,best,inputs['source_map'])['routes'],
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

def current_view(root,best,historical):
 """Current source view from frozen release AST + this scored job's profiles.
 Old rs32 audits remain byte-identical historical evidence, not relabelled files.
 """
 import re
 source=best['source'];nodes=source_nodes(root/source['path']);byname={n['name']:n for n in nodes}
 def bindings(names):
  return [dict(byname[name],path=source['path'],source_sha256=source['sha256']) for name in names if name in byname]
 components=[]
 for old in historical['components']:
  old_hashes={n['name']:n.get('AST_sha256') for n in old['bindings']}
  bound=bindings(old_hashes)
  for b in bound:b['same_AST_as_historical_rs32']=b['AST_sha256']==old_hashes[b['name']]
  components.append({'id':old['id'],'name':old['name'],'status':'SOURCE_PRESENT; actual applicability is in scored routes','scope':old['scope'],'implementation':old['implementation'],
   'bindings':bound,'isolated_causal_gain_us':None,'historical_parent_component':old['id'],'gain_scope':'Exact source inclusion and scored package validation; no isolated component causal gain inferred.'})
 components += [
  {'id':'rtk_router','name':'PW35 one-node TMA RTK router128/256','status':'SELECTED_IN_SCORED_PROFILE','scope':'router128/256','implementation':'PW35 RTK branch inherited through PW36; old RW alternatives remain source controls.','bindings':bindings(['RWCU','RTV','RSP','_rsbake','_rscall']),'isolated_causal_gain_us':None},
  {'id':'eh_epilogue','name':'Optional EH early-half-slab epilogue bit65536','status':'SELECTED_IN_SCORED_PROFILE','scope':'qkv128/pooler128','implementation':'EHbit16=65536 is distinct from HALFbit4=16 and EARLYbit5=32. Original non-EH controls retained; no cursor patch.','bindings':bindings(['RQCU','RQV','RQPROD','RQBAKE']),'isolated_causal_gain_us':None}]
 routes=[];picks={(p['layer'],p['m']):p for p in best['actual_picks']}
 for index,p in enumerate(best['actual_timed_profiles']):
  case=p['case'];report=p['report'];family='Lt/vendor'
  if 'rq<' in report:family='rq'
  elif 'rtk<' in report:family='rtk'
  elif 'rwk<' in report:family='rw'
  elif 'gv_predxor<' in report:family='gv_predxor'
  elif 'gv<' in report:family='gv'
  elif 'rr<' in report:family='rr'
  gv_match=re.search(r'(?:gv_predxor|gv)<([^>]+)>',report)
  gv_template=[int(v.strip()) for v in gv_match[1].split(',')] if gv_match else None
  match=re.search(r'rq<([^>]+)>',report);flags=None
  if match:
   try:flags=int(match[1].split(',')[-1].strip())
   except ValueError:pass
  names=['kernel_fn']+(['RQCU','_rqcall'] if family=='rq' else ['RWCU','_rscall'] if family in ('rtk','rw') else ['GVCU','_gvcall'] if family in ('gv','gv_predxor','rr') else ['_v12','_ltsearch'])
  routes.append({'index':index,'layer':case['layer'],'m':case['m'],'load':case['load'],'family':family,'RQ_flags':flags,'GV_predxor_selected':family=='gv_predxor','GV_template':gv_template,'EH_selected':flags is not None and bool(flags&65536),'HALF_selected':flags is not None and bool(flags&16),'EARLY_selected':flags is not None and bool(flags&32),'actual_prepare_pick':picks.get((case['layer'],case['m'])),'kernel_report':report,'profile':best['raw'],'bindings':bindings(names)})
 if best['candidate'] in ('pw45','gv_predxor_pw45_v2','rq32_predxor'):
  components.append({'id':'o32_half52_optional','name':'Optional o32 HALF52 fullK ST4/ST5','status':'SELECTED_IN_SCORED_PROFILE' if any(r['layer']=='o' and r['m']==32 and r['RQ_flags']==52 for r in routes) else 'SOURCE_COMPILED_BUT_UNSELECTED_IN_SCORED_PROFILE','scope':'o32 only; fixed mechanism gains are separate from actual selected path','implementation':'Optional h32s4e/h32s5e candidates; original controls retained.','bindings':bindings(['RQCU','RQV','RQSBAKE','RQPROD2','_rqsrc','_rqcall']),'isolated_causal_gain_us':None})
 if best['candidate']=='gv_predxor_pw45_v2':
  components.append({'id':'gv_predicated_escape_xor','name':'Standalone GV predicated escape XOR repair','status':'SELECTED_IN_SCORED_PROFILE' if any(r['GV_predxor_selected'] for r in routes) else 'SOURCE_PRESENT_NOT_SELECTED_IN_SCORED_PROFILE','scope':'Seven reviewed format1 GV templates; per-route selection below','implementation':'Separate gv_predxor kernel retains original gv controls; removes divergent scalar repair branch with predicated XOR. Source ancestry is PW45, not the RQ32 predxor experiment.','bindings':bindings(['GVCU','GVI','_gvcall','_gvsrc','kernel_fn']),'isolated_causal_gain_us':None})
 for component in components:
  if component['id']=='eh_epilogue':component['status']='SELECTED_IN_SCORED_PROFILE' if any(r['EH_selected'] for r in routes) else 'SOURCE_PRESENT_NOT_SELECTED_IN_SCORED_PROFILE'
  if component['id']=='rtk_router':component['status']='SELECTED_IN_SCORED_PROFILE' if any(r['family']=='rtk' for r in routes) else 'SOURCE_PRESENT_NOT_SELECTED_IN_SCORED_PROFILE'
 previous_path='proj_rsi/audits/selected-hotpaths-pw38/current-stack-snapshot.json'
 previous_pw45_path='results/proj-rsi/releases/pw45-v1-835acb90b47b/historical-current-stack-before-gv.json'
 pw45_diffs=[]
 if best['candidate']=='gv_predxor_pw45_v2':
  old=read(root,previous_pw45_path);oldroutes={(r['load'],r['layer'],r['m']):r for r in old['routes']}
  for route in routes:
   previous=oldroutes[(route['load'],route['layer'],route['m'])]
   def kernel_identity(r):
    lines=r['kernel_report'].splitlines();return re.split(r'\s{2,}(?=\d)',lines[1],maxsplit=1)[0].strip() if len(lines)>1 else 'UNKNOWN'
   if kernel_identity(previous)!=kernel_identity(route):pw45_diffs.append({'load':route['load'],'layer':route['layer'],'m':route['m'],'previous_kernel':kernel_identity(previous),'current_kernel':kernel_identity(route),'causal_gain_us':None})
 diffs=[]
 if best['candidate']=='pw45':
  previous=read(root,previous_path);oldroutes={(r['load'],r['layer'],r['m']):r for r in previous['routes']}
  def identity(route):
   lines=route['kernel_report'].splitlines();return re.split(r'\s{2,}(?=\d)',lines[1],maxsplit=1)[0].strip() if len(lines)>1 else 'UNKNOWN'
  for route in routes:
   old=oldroutes[(route['load'],route['layer'],route['m'])];a=old.get('actual_prepare_pick') or {};b=route.get('actual_prepare_pick') or {}
   if identity(old)!=identity(route) or a.get('selected')!=b.get('selected'):
    diffs.append({'load':route['load'],'layer':route['layer'],'m':route['m'],'previous_family':old['family'],'current_family':route['family'],'previous_kernel':identity(old),'current_kernel':identity(route),'previous_pick':a.get('selected'),'current_pick':b.get('selected'),'causal_gain_us':None})
 return {'status':'CURRENT_BOARD_VERIFIED_'+str(best['candidate']).upper()+'_SOURCE_AND_ACTUAL_SCORED_ROUTES','official_job_id':best['job_id'],'score_us':best['score_us'],'source':source,'parent':best['stack_description'],'historical_pw38_view':ref(root,previous_path) if best['candidate']=='pw45' else None,'historical_pw45_view':ref(root,previous_pw45_path) if best['candidate']=='gv_predxor_pw45_v2' else None,'route_differences_from_pw45':pw45_diffs,'route_differences_from_pw38':diffs,'new_HALF52_selected_routes':[{'load':r['load'],'layer':r['layer'],'m':r['m']} for r in routes if r['RQ_flags']==52],'historical_rs32_audit_status':'HISTORICAL_RETAINED_UNCHANGED','historical_rs32_job':historical['official_job_id'],'components':components,'routes':routes,'check_job':None if best['candidate']=='gv_predxor_pw45_v2' else '87044866f4494726' if best['candidate']=='rq32_predxor' else '87db1d6a076c4da4' if best['candidate']=='pw45' else 'a3ed4ce5886a4415','scored_job':best['job_id'],'limitations':['Server content_hash algorithm is unknown; local release attestation does not equate it to raw SHA256.','Package scores/actual routes are observed; individual feature causal deltas are not inferred.','Source presence is separate from route selection; current pooler EH choice may differ from the CHECK choice.']}

def csv_text(rows,fields):
 output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for row in rows:
  writer.writerow({key:json.dumps(row.get(key),ensure_ascii=False,sort_keys=True) if isinstance(row.get(key),(dict,list)) else row.get(key) for key in fields})
 return output.getvalue()

def render(catalog):
 jobfields=['job_id','submitted_utc','mode','status','API_result_status','candidate','score_us','passed','tests_passed','tests','archive_status','parent','stack_description','compiled_or_available_features','actual_selected_features','source','source_identity_status','api_content_hash','API_status_source','status_evidence_kind','status_evidence_source','raw','log','causal_caution']
 files={
 'proj_rsi/history_catalog.json':json.dumps(catalog,indent=2,ensure_ascii=False)+'\n',
 'proj_rsi/tables/submissions.csv':csv_text(catalog['submissions'],jobfields),
 'proj_rsi/tables/checks.csv':csv_text(catalog['checks'],jobfields),
 'proj_rsi/tables/all_official_jobs.csv':csv_text(catalog['all_official_jobs'],jobfields),
 'proj_rsi/tables/improvements.csv':csv_text(catalog['improvements'],['immutable_id','id','classification','currently_included','scope','finding','verification','stacked_versions','stack_tested_jobs','score_tested_jobs','missing_reason','next_action','evidence']),
 'proj_rsi/tables/positive_observations.csv':csv_text(catalog['positive_observations'],['observation_id','job','candidate','layer','m','mode','candidate_bake_us','logged_r6_control_us','positive_bake_difference_us','below_legacy_0_3us','below_legacy_0_5us','actual_final_pick','log','line','control_line','classification','feature_level','catalog_cohort']),
 'proj_rsi/tables/local_observations.csv':csv_text(catalog['local_paired_observations'],['observation_id','attempt_id','version','shape','control','gain_us','F1_gain_us','paired_ratio','wins','ties','losses','pairs','same_selected_plan','hardware','result_sha256','shipping_gain','evidence']),
 'proj_rsi/tables/candidates.csv':csv_text(catalog['candidates'],['immutable_id','candidate','parent','source','compiled_or_available_features','official_jobs','leaderboard_submitted','status','description','historical_summary_paths','new_since_completeness_snapshot']),
 }
 counts=catalog['counts']
 text='''# Projection kernel history

This catalog preserves all official submissions and known improvement evidence. Regenerate with `python3 -m proj_rsi.history_catalog`; check deterministic renderings with `python3 -m proj_rsi.history_catalog --check`. No GPU or service calls occur during generation.

The union of preserved API captures and explicitly labeled accepted-receipt/own-result records contains **{all_jobs} official jobs:{runs} leaderboard submissions and{checks} checks**. The best is **{best_candidate}/{best_job}:{best_score}us**, local source `{best_source}`. Baseline R6's scored submission is included. Five older failed checks retain unknown source associations; partial failure metadata is preserved where found and genuinely missing archives remain explicit. PW38 check status and archived official evidence are shown below, separately from scored submissions and older failed/source-unknown checks. Preserved-snapshot union prevents latest50 pagination from dropping baseline R6.

Historical best rs32 omitted PW35 RTK; PW36 tested that combination (3212.415452us). PW38 includes RTK plus selectable EH; current best is {current_best_name}. PW45 adds optional o32 HALF52 contenders, whose source presence must be distinguished from actual selection. The omitted historical EH epilogue0.13–0.16us result remains visible with weaker raw provenance. Latest cursorv3 contains tiny wins, ties and regressions; it establishes no incremental shipping gain. PW37 remains an unscored source experiment; Each completed mode=run remains distinct from its preceding CHECK; the active board's selected job and matching scored raw result determine current authority.

Complete compatible GV+RQ+privateEH+O32 source d1ca is now measured full30 against currentGV (paired1.0016700108951944x, F11.002913179565869x), and scored job161bdb8de7354344 passed9/9 at3187.14239967648us, losing currentGV. EH+O32 c425 is a separate incremental local comparison versusEH aaf40d only. Keepchoice244e, GV+RQ137e and GV+EHaaf4 are parallel scored branches and also lostGV. Exact composition, receipts and signed evidence: [stack reconciliation](audits/submission-stack-20261027/report.md). Source presence does not establish selected use or isolated causal gain.

## Every leaderboard submission, chronologically

|UTC submitted|Version / job|Score us|What was stacked / actually exercised|
|---|---|---:|---|
'''
 for r in catalog['submissions']:
  score_display=f"{r['score_us']:.6f}" if r['score_us'] is not None else 'Pending — unknown'
  raw_display=f"[Raw](../{r['raw']['path']})" if r['raw'] else 'No result archive yet; frozen source/attestation retained'
  text+=f"|{r['submitted_utc']}|{r['candidate'] or 'UNKNOWN'} / `{r['job_id']}`|{score_display}|{r['stack_description']}. {raw_display}|\n"
 pw38=next((r for r in catalog['checks'] if r['job_id']=='a3ed4ce5886a4415'),{})
 text=text.replace('{best_candidate}',str(catalog['authority']['candidate'])).replace('{best_job}',catalog['authority']['best_job']).replace('{best_score}',f"{catalog['authority']['score_us']:.9f}").replace('{best_source}',str(catalog['authority']['source_sha256']))
 text=text.replace('{current_best_name}',str(catalog['authority']['candidate']))
 text=text.replace('{pw38_check_status}',str(pw38.get('status','unknown'))).replace('{all_jobs}',str(counts['all_official_jobs'])).replace('{runs}',str(counts['leaderboard_submissions'])).replace('{checks}',str(counts['checks']))
 text+='''
Each row is an independent official measurement. Scores are outcomes, not causal feature deltas. Every completed archived scored job passed9/9; accepted pending scored rows have unknown score/tests until their own result completes. Source descriptions state ancestry/availability; per-job picks and actual timed profile reports in the JSON distinguish selection from compiled presence. Initial PW34's RQ phase was skipped; later PW34 repaired it.

## Preserved source stack versions

- PW37 `7913564f…a0330`: PW36 + cursor. CPU source only; latest standalone cursorv3 has mixed/negative evidence, not an accepted improvement.
- PW38: historical board-verified scored stack `a3f0ebf25a754c01`,3198.617507982us. PW36 + optional EHbit65536; cursor excluded. Local full30-shape validation is complete with mixed results. Official CHECK `a3ed4ce5886a4415` status: {pw38_check_status}; it is not a leaderboard submission. See its [manifest](candidates/pw38/manifest.json) and [provenance](candidates/pw38/PROVENANCE.md). Actual local EH selection and mixed performance are preserved separately; official CHECK evidence is preserved independently from leaderboard scoring.

## Every preserved improvement finding

|Finding|Status / currently included|Evidence, tested stack and remaining action|
|---|---|---|
'''
 for r in catalog['improvements']:
  text+=f"|{r['id']}|{r['classification'].replace('_',' ').lower()}; {r['currently_included']}|{r['finding']} Tested stack/attempts: {', '.join(dict.fromkeys(r['stack_tested_jobs']+r['score_tested_jobs'])) or 'not established for this finding'}. Leaderboard-scored jobs: {', '.join(r['score_tested_jobs']) or 'none'}. {r['next_action']}|\n"
 text+=f'''\nThe **{counts['feature_findings']} feature findings** are distinct from **{counts['positive_observations']} positive rounded preparation observations**, including **{counts['below_0_3us']} below0.3us / {counts['below_0_5us']} below0.5us**. All are preserved without legacy kill-bar filtering; the original224 observations (76below0.3/107below0.5) remain immutable, with new official-job observations appended separately. Separate local paired observations retain the latest STMatrix parent+0.175999us/X4 parent+0.352001us alongside actual-incumbent−7.455999us/−7.472000us; both are NOT SHIPPED. The independent official audit additionally identifies {counts['selected_positive_observations']} selected positive observations across its33 ledger jobs; repeated bakes/alternative modes are not independent features or additive gains. Small paired cursor positives from the latest v3 audit are separately retained in that finding, not fabricated as official bake rows.

## Tables and evidence

- [All{counts['leaderboard_submissions']} submissions](tables/submissions.csv), [all{counts['checks']} checks](tables/checks.csv), [all{counts['all_official_jobs']} official jobs](tables/all_official_jobs.csv).
- [All improvement findings](tables/improvements.csv), [all positive preparation observations](tables/positive_observations.csv), [local paired observations including losses](tables/local_observations.csv), [all{counts['candidates']} candidate directories](tables/candidates.csv).
- [Machine catalog](history_catalog.json): exact source component map/all30 serving routes, source hashes, raw/log/review links, primary input inventory and missing evidence.
- [Official component history](audits/current-leaderboard-gains-official.md), [HISTORICAL rs32 source ancestry](audits/current-leaderboard-stack-source.md), [CURRENT scored stack](CURRENT_STACK.md), [corpus completeness](audits/current-leaderboard-gains-completeness.md), [latest cursorv3](audits/smallm-v3-01/report.md).

## Preservation and update method

1. Append each new API snapshot path/SHA to `history_sources.json`, advance its active snapshot/revision, and preserve all older captures. Archive each official/API response and raw log/profile under its immutable job ID. When no newer API capture exists, append `completed_receipt_jobs` references to the accepted source-pinned receipt and own completed official result; retain explicit receipt/result provenance and NULL API status source. Keep failures/noarchive/unknownsource rows. Mode`run` identifies a leaderboard submission; local H100 experiments and CPU-only source stacks remain separate.
2. Freeze each implementation and review, recording SHA256, parent and attempt. Attach actual prepare picks and timed graph/profile evidence. Never infer selection from a variant's presence in source.
3. Add a feature-level finding or a new immutable observation linked to its comparator, hardware, numerical/graph validity and uncertainty. Retain negative/conflicting estimators and every small positive; no threshold deletes evidence.
4. Record stack integration and actual tested/scored job IDs. A measured gain against a failed parent, mathematical scenario or CPU instruction reduction is not an official incumbent gain.
5. Run the generator and CPU history tests; review changed rows and input hashes. Keep historical raw/source parents unchanged. Updating an audit changes its hash and requires explicit regeneration.

Server`content_hash` has an undocumented algorithm: live rs32`cd14084a…` differs from local fileSHA256`a7b6e010…`. Path/git/source comparisons establish association; they do not attest downloaded historical uploaded payload bytes. Earlier PW34 variants likewise need revision-specific payload provenance. Remote historical lab distributions missing locally remain explicit gaps. No benchmark achievement or additive cross-job score claim follows from this catalog.
'''
 text=text.replace('{current_best_name}',str(catalog['authority']['candidate']))
 text=text.replace('{pw38_check_status}',str(pw38.get('status','unknown')))
 if pw38.get('score_us') is not None:
  text+='\nPW38 CHECK: **'+str(pw38['score_us'])+'us**, '+str(pw38.get('tests_passed'))+'/'+str(pw38.get('tests'))+' tests; all30 shapes archived. This CHECK is not a leaderboard submission; the distinct scored run supplies current leaderboard authority.\n'
 view=catalog['current_stack_view']
 files['proj_rsi/current_leaderboard_stack.json']=json.dumps(view,indent=2)+'\n'
 current='# Current leaderboard stack: '+str(catalog['authority']['candidate'])+'\n\n'
 current+='Board-verified scored job `'+view['scored_job']+'`: **'+str(view['score_us'])+'us**,9/9 and30 timed shapes. Source `'+view['source']['sha256']+'`. '+view['parent']+('. No redundant preceding CHECK was required for this scored GV release.' if view['check_job'] is None else '. The preceding CHECK is a distinct job.')+'\n\n'
 current+='The old [rs32 source audit](audits/current-leaderboard-stack-source.md) and [official history](audits/current-leaderboard-gains-official.md) are HISTORICAL snapshots retained unchanged. Current bindings below are regenerated from this frozen current release source and this scored job; they are not copied CHECK selections.\n\n'
 current+='|Component|Source/selection evidence|\n|---|---|\n'
 for component in view['components']:current+='|'+component['name']+'|'+component['status']+'; '+str(len(component['bindings']))+' current AST/source bindings; no isolated causal delta.|\n'
 current+='\n|Load / layer / M|Actual scored family / prepare pick|EH / HALF / EARLY flags|\n|---|---|---|\n'
 for route in view['routes']:
  pick=route['actual_prepare_pick'];name=pick['selected'] if pick else 'See actual kernel profile'
  current+='|'+str(route['load'])+' / '+route['layer']+' / '+str(route['m'])+'|'+route['family']+' / '+name+'|'+str(route['EH_selected'])+' / '+str(route['HALF_selected'])+' / '+str(route['EARLY_selected'])+'|\n'
 current+='\nFull per-route kernel reports, exact current source-node SHA256/line bindings and historical equivalence checks are in [machine view](current_leaderboard_stack.json). Source presence does not imply measured benefit on every route. Server payload hash remains opaque; no component-wise or cross-run additive gain claim is made.\n'
 if view.get('historical_pw38_view'):
  current+='\nPW38 historical source/routes are preserved byte-for-byte at `'+view['historical_pw38_view']['path']+'`. New HALF52 selected routes: **'+str(len(view['new_HALF52_selected_routes']))+'**. Optional EH is compiled but actual selection is evaluated per row.\n'
  current+='\n|Shape|Previous PW38 pick / kernel|Current PW45 pick / kernel|\n|---|---|---|\n'
  for diff in view['route_differences_from_pw38']:current+='|'+diff['load']+'/'+diff['layer']+' M'+str(diff['m'])+'|'+str(diff['previous_pick'])+' / '+diff['previous_kernel']+'|'+str(diff['current_pick'])+' / '+diff['current_kernel']+'|\n'
 if view.get('historical_pw45_view'):
  current+='\nPW45 historical source/routes are preserved byte-for-byte at `'+view['historical_pw45_view']['path']+'`. Selected standalone GV predxor routes: **'+str(sum(r['GV_predxor_selected'] for r in view['routes']))+'**.\n'
  current+='\n|Shape|Previous PW45 kernel|Current scored kernel|\n|---|---|---|\n'
  for diff in view['route_differences_from_pw45']:current+='|'+diff['load']+'/'+diff['layer']+' M'+str(diff['m'])+'|'+diff['previous_kernel']+'|'+diff['current_kernel']+'|\n'
 current+='\nLatest complete compatible GV+RQ+privateEH+O32 source d1ca was measured locally against this GV source and officially scored as `161bdb8de7354344`: **3187.14239967648us**,9/9; it did not replace GV. The local whole-package positive point does not establish official benefit. EH+O32 c425 was measured versusEH parent only. [Exact stacking provenance and all branch outcomes](audits/submission-stack-20261027/report.md). Latest archived board rank: **'+str(catalog['authority']['board_row']['rank'])+'**.\n'
 current+='\nLatest independently archived official CHECKs (receipt + own completed result, no new board capture): '+', '.join('`'+r['job_id']+'` '+str(r['score_us'])+'us' for r in catalog['checks'] if r['status_evidence_kind']=='accepted_receipt_and_own_completed_official_result_not_API_capture')+'. All passed9/9; none is a scored submission. M1-only full30 and focused unpooler256 local regressions remain separate in [history](HISTORY.md). MoE LUT whole improves its SS parent but remains slower than its actual anchor; it is separate local research with no projection leaderboard claim.\n'
 files['proj_rsi/CURRENT_STACK.md']=current
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

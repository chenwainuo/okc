"""Read exact checkpoint-selected receipt; retain signed effects and confounds."""
import hashlib,json,math,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
jid='research.moe.ss.lut.pair2.stage.01'
j=json.loads((ROOT/'results/proj-rsi/scheduler/checkpoint.json').read_text())['jobs'][jid]
receipts=[o['transition']['evidence'] for o in j['observations'] if o.get('transition',{}).get('to')=='VALIDATING']
assert receipts;receipt=receipts[-1];invpath=ROOT/receipt['inventory'];assert sha(invpath)==receipt['sha256'];inv=json.loads(invpath.read_text());collection=invpath.parent
assert not inv['missing'] and inv['terminal_exit']==0
for rel,h in inv['files'].items():assert sha(collection/rel)==h,rel
for rel,h in inv['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
p=collection/'logs/scheduler/moe-ss-lut-pair2-stage-01/result.json';r=json.loads(p.read_text())
assert r['state']=='VALID_M8_PAIRED_STAGE_STAGE_SCREEN_ROOT_REVIEW_REQUIRED'
assert r['validated_direct_timings_completed'] and not r['cleanup']
clean=receipt['cleanup_observation'];assert receipt['cleanup_verified'] and not clean['cleanup_failed'] and clean['terminal_exit']==0 and not clean['compute_pids'] and not clean['guard_pids'] and not clean['container_exists'] and clean['gpu_memory_mib']==0
checks=r['correctness'];assert len(checks)==95 and len(r['raw'])==50
for check in checks:
 for key in ['pass','reference_pass','bitwise_equal','bitwise_equal_v1_full_serving']:
  if key in check:assert check[key]is True,(check,key)
assert len(r['jit'])==1;actual=r['jit'][0]
assert actual['so_sha256']==sha(p.parent/'actual_paired.so')
source=ROOT/'serving/experiments/moe_rsi_20261027/ss_pipeline_pair2_v1/extension_source.cu'
assert actual['production_sha256']==sha(source) and(p.parent/'actual_paired.cu').read_text().startswith(source.read_text())
assert {(x['T'],x['DIV'])for x in actual['resources']}=={(8,1),(8,8)}
for x in actual['resources']:assert x['dynamic_shared']==46112 and x['occupancy']==4 and x['local']==x['binary_resources']['STACK']==0
for f in actual['functions']:assert f['eight_HGMMA_per_group'] and f['actual_D_no_generic_write_before_wait0'] and f['static_K128_groups']==f['math_128thread_barriers_after_wait0']==3
stage={};sm=r['runtime']['SMs'];assert sm==132
for s in ['up','down']:
 raw=[x for x in r['raw']if x['stage']==s];paired=[x for x in raw if x['kind']=='paired'];f1=[x for x in raw if x['kind']=='F1_fixed_B9'];assert len(paired)==16 and len(f1)==9
 fixed=r['F1_frozen_workloads'][s];assert fixed['seed']==7904 and fixed['routing']['active_local_blocks']==9
 assert all(x['seed']==7904 and x['routing']==fixed['routing']for x in f1)
 rows=[]
 for i in range(8):
  q=[x for x in paired if x['round']==i];assert len(q)==2 and q[0]['seed']==q[1]['seed']==7900+i and q[0]['routing']==q[1]['routing']
  a=next(x for x in q if x['arm']=='v1_full_lut');b=next(x for x in q if x['arm']=='v2_pair2');B=a['routing']['active_local_blocks'];n=B*(128 if s=='up'else 64)
  rows.append(dict(seed=a['seed'],active_blocks=B,control_us=a['us'],candidate_us=b['us'],saving_us=a['us']-b['us'],control_over_candidate=a['us']/b['us'],order=[x['arm']for x in q],valid_CTAs_source_model=n,control_residency_wave_model=math.ceil(n/(sm*(6 if s=='up'else 4))),candidate_residency_wave_model=math.ceil(n/(sm*4))))
 saves=[x['saving_us']for x in rows];ratio=[x['control_over_candidate']for x in rows]
 med=r['F1_medians_us'][s];a0=med['v1_full_lut0'];a2=med['v1_full_lut2'];b=med['v2_pair21']
 stage[s]={'pairs':rows,'paired_median_saving_us':statistics.median(saves),'paired_median_control_over_candidate':statistics.median(ratio),'paired_candidate_wins':sum(x>0 for x in saves),'paired_control_median_us':statistics.median(x['control_us']for x in rows),'paired_candidate_median_us':statistics.median(x['candidate_us']for x in rows),'F1_medians_us':med,'F1_control_endpoint_average_us':(a0+a2)/2,'F1_endpoint_average_saving_us':(a0+a2)/2-b,'F1_before_saving_us':a0-b,'F1_after_saving_us':a2-b,'F1_fixed_workload':fixed,'actual_candidate_resources':next(x for x in actual['resources'] if x['DIV']==(8 if s=='up'else 1))}
 assert abs(stage[s]['paired_median_saving_us']-r['paired'][s]['median_v1_minus_v2_us'])<1e-9
 # Every arm's final paired output and each F1 block terminal was checked.
 for arm in ['v1_full_lut','v2_pair2']:
  assert any(c.get('stage')==s and c.get('arm')==arm and c.get('kind')=='own_terminal_before_later_stage'for c in checks)
 for block,arm in enumerate(['v1_full_lut','v2_pair2','v1_full_lut']):
  assert any(c.get('stage')==s and c.get('arm')==arm and c.get('kind')=='own_F1_terminal_before_next_block'+str(block)for c in checks)
assert any(c['kind']=='mixed_own_terminal_after_stage_timings' for c in checks)
analysis={'state':'VALID_PAIR2_BOTH_STAGE_REGRESSION_WITH_RETAINED_B7_UP_GAINS','result':str(p.relative_to(ROOT)),'result_sha256':sha(p),'inventory':str(invpath.relative_to(ROOT)),'inventory_sha256':sha(invpath),'raw_calls':50,'correctness_records':95,'all_recorded_numeric_and_bitwise_checks_pass':True,'fixed_F1_fixture_per_stage':True,'own_terminals_all_arms_and_stages':True,'wall_seconds':r['elapsed_seconds'],'candidate_compile_seconds':actual['compile_seconds'],'actual_candidate_resources':actual['resources'],'stage':stage,'cleanup_verified':True,'terminal_exit':0,'scope':'M8EP8normal;two stage diagnostics vs exactv1LUT control. Mixedfullserving checked numerically only; no fullservingtiming, incumbent comparison, official or stackclaim.','causal_interpretation':['Actual8MMA/groups and32logicalrounds passed; reducing sync rounds did not yield overall stage gain in this package.','Up candidate shared/ring increases reduce residency6to4; source wave model has B6 1to2waves, B7 2to2waves, B9/B11 2to3waves. Both B7 gains retained, model is inference not measured wave scheduling.','Down unchanged4CTA/SM yet all8lose. Residency alone is insufficient; producer-stage batching, scheduling/ILP and pipeline startup/tail are confounded.','No profiler counters or PC attribution collected; current producer-versus-consumer bottleneck remains unknown.'],'next_decision':'Keep pair2 source/results; do not stack whole pair2. Preserve bothB7 up gains without claiming runtime selectable coverage. Prioritize warm exactv1 producer/consumer attribution or lowercost actual-resource scheduling diagnosis before another codec/stage expansion; wait1 remains lowROI with historical no-help. One failed version does not reject grouped stages or SS family.'}
(D/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
evidence={'job_id':jid,'recommended_state':'DONE','recommended_disposition':'VALID_REGRESSION','observed_state':r['state'],'scope':'pair2 fullLUT T8 up/down stages only, M8EP8, not fullserving/anchor/official','up':{k:v for k,v in stage['up'].items() if k not in['pairs','F1_fixed_workload','actual_candidate_resources']},'down':{k:v for k,v in stage['down'].items() if k not in['pairs','F1_fixed_workload','actual_candidate_resources']},'preserved_positive_subcases':[x for x in stage['up']['pairs'] if x['saving_us']>0],'correctness':{'records':95,'all_recorded_checks_pass':True,'all_own_F1_and_paired_terminals_pass':True,'full_chain_reference_and_bitwise_current_fresh_nonlocal_pass':True,'actual_compile_gate_and_sourcepins_pass':True},'measurement':{'raw_calls':50,'F1_fixed_fixture_calls':18,'paired_calls':32,'paired_rounds_per_stage':8,'wall_seconds':r['elapsed_seconds'],'candidate_compile_seconds':actual['compile_seconds']},'terminal_exit':0,'cleanup_verified':True,'cleanup_observation':clean,'missing_artifacts':[],'evidence_files':{str(p.relative_to(ROOT)):sha(p),str(invpath.relative_to(ROOT)):sha(invpath),str((D/'analysis.json').relative_to(ROOT)):sha(D/'analysis.json')},'root_action':'Root may mark DONE/VALID_REGRESSION after independent scoped review; no incumbent/stack/promotion change; preserve B7 positives and all failures.','best_next_decision':analysis['next_decision'],'loop_contract':'profile->measure->hypothesis->optimize/reimplement->test->validate->stack->submit->learn->reprioritize->repeat; idle<=10s; proj_rsi/ITERATION.md'}
(D/'done-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({'disposition':evidence['recommended_disposition'],'analysis_sha256':sha(D/'analysis.json'),'done_evidence_sha256':sha(D/'done-evidence.json'),'stage':{s:{k:stage[s][k]for k in['paired_median_saving_us','paired_median_control_over_candidate','paired_candidate_wins','F1_endpoint_average_saving_us']}for s in stage}},indent=2))

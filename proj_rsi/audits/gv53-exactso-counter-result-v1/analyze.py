"""Offline exact-SO profile validation and aggregate contrasts; no GPU/API/queue calls."""
from pathlib import Path
import hashlib,importlib.util,json,math,csv,io
D=Path(__file__).resolve().parent;R=D.parents[2];C=R/'results/proj-rsi/scheduler/artifacts/research.gv53.exactso.counters.01/collection-1793142336494175357';P=C/'logs/scheduler/gv53-exactso-counters-01/run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 i=json.loads((C/'inventory.json').read_text());assert i['terminal_exit']==0 and not i['missing']
 for rel,h in i['files'].items():assert sha(C/rel)==h,rel
 for rel,h in i['dispatch']['source_pins'].items():assert sha(R/rel)==h,rel
 e=i['cleanup_observation'];assert not e['cleanup_failed']and not e['compute_pids']and not e['containers']and not e['guard_pids']and not e['container_exists']and e['gpu_memory_mib']==0
 spec=importlib.util.spec_from_file_location('profile_recipe',R/'proj_rsi/candidates/gv53_exactso_counters_v1/profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 r=json.loads((P/'result.json').read_text());s=json.loads((P/'session.json').read_text());saved=json.loads((P/'analysis.json').read_text())
 assert s['state']=='COMPLETE_COUNTER_DIAGNOSTIC_ROOT_EXTERNAL_CLEANUP_REQUIRED'and s['capture_returncode']==0 and s['metric_query_returncode']==0 and not s['profile_durations_are_gain_evidence']
 assert '2025.1.1.0'in s['ncu_version']and set(m.METRICS)<=set(s['available_metrics']);m.metric_names((P/'metric-query.stdout').read_text())
 for name,h in s['artifact_hashes'].items():assert sha(P/name)==h,name
 analysis=m.parse((P/'raw.csv').read_text(),r);assert analysis==saved
 assert r['source_sha256']==sha(R/'proj_rsi/candidates/gv53_quarter_sparse_header_v1/pilot.cu') and r['control_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert sha(P/'sparse.so')=='2eb511c0af3c4ce5bd3990e45d376c64d12016b876a2109ac0f263eafc8c17a7'and sha(P/'quarter.so')=='4b85c059c953e91989f6d613e8148269cfbb8e0480ab120585abcac492c3e44c'
 assert sha(P/'sparse.cu')=='0e17b3c566c46a7b6981135a5aadf49a88e454ed1cd3fa1c07e60e56c50d6aa1'and sha(P/'quarter.cu')=='607e6faec9d5b5e706ac3109c9a83af3ab8ba6e7b17e2bbea7a7de1fb7316038'
 cell=r['fixed_cells']['8'];journal=[json.loads(line)for line in(P/'raw.jsonl').read_text().splitlines()];assert journal==cell['raw']and len(journal)==18
 assert len([x for x in journal if x['mode']=='profile'])==3 and all(x['seed']==91600 for x in journal)
 for arm in m.LABELS:
  n=r['actual_graph_nodes'][arm][0];assert sha(P/(arm+'.actual.sass'))==n['SASS_sha256']
  assert n['cluster_query_returncode']==0 and n['cluster_dimension']in([0,0,0],[1,1,1]) and n['theoretical_CTA_occupancy_per_SM']==1
  a=r['raw_attributes'][arm];assert a['registers']==64 and a['local']==0 and a['occupancy']==1
  assert cell['validation'][arm]['basis']=='EXACT'and len(cell['validation'][arm]['basis_columns'])==8 and len(cell['validation'][arm]['task'])==3
  term=cell['terminals'][arm];assert term['current_seed']==91600 and term['before_any_replay_or_mutation']and term['oracle']['current_X']=='EXACT'
 assert len(cell['mutations'])==15 and all(x['NaN_output_overwritten']for x in cell['mutations'])
 assert r['pack']['W_seed']==5344 and r['pack']['actual_W_bit_reconstruction']=='EXACT'and r['transport']['actual_accepted_records_exact_after_stable_bucketing']
 assert r['transport']['record_count']==4759 and r['transport']['parent_bank_sha256']=='5ab99292bd9381ab917d87664e1ea4d9f337fef77c5260762fed4600975a38ac'and r['transport']['quarter_bank_sha256']=='0eaf570801c58ac610d253a4a4bf7be8783c5495b76afe415940228522cb7665'
 clocks=[]
 for x in r['clock_context']:
  assert x['returncode']==0 and not x['stderr'];v=next(csv.reader(io.StringIO(x['stdout'])));assert len(v)==8
  clocks.append(dict(arm=x['arm'],position=x['position'],timestamp=v[0].strip(),uuid=v[2].strip(),pstate=v[3].strip(),SM_MHz=float(v[4]),memory_MHz=float(v[5]),temperature_C=float(v[6]),power_W=float(v[7])))
 assert len(clocks)==6 and all(x['uuid']==r['hardware']['UUID']or x['uuid']=='GPU-'+r['hardware']['UUID']for x in clocks)
 arms={x['arm']:x for x in analysis['arms']};contrasts={}
 for a,b in [('control','quarter'),('control','sparse'),('quarter','sparse')]:
  av=arms[a]['metrics'];bv=arms[b]['metrics'];contrasts[a+'_to_'+b]={k:{'difference':bv[k]['value']-v['value'],'relative_percent':100*(bv[k]['value']/v['value']-1)if v['value']else None,'unit':v['unit']}for k,v in av.items()}
 out={'job_id':i['dispatch'].get('job_id','research.gv53.exactso.counters.01'),'state':'VALID_EXACTSO_AGGREGATE_DIAGNOSTIC','recommended_DONE_disposition':'VALID_DIAGNOSTIC','gain_us':None,'official_score':None,'repair_time_fraction':None,'artifact_hashes_verified':len(i['files']),'runtime_pins_verified':len(i['dispatch']['source_pins']),'arms':analysis['arms'],'contrasts':contrasts,'clock_snapshots':clocks,'actual_resources':r['raw_attributes'],'actual_graph_nodes':r['actual_graph_nodes'],'escape_work_predictions':r['escape_work_predictions'],'cost':r['cost'],'validity':{'rows':18,'profiled_selected_launches':3,'replay_passes_per_selected_launch':[x['replay_passes']for x in analysis['arms']],'profile_own_terminal_checks':3,'initial_task_plus_FP64_oracles':9,'basis_columns_per_arm':8,'mutations':15,'same_current_X_seed':91600,'actual_W_seed':5344,'native_JIT_builds':r['cost']['native_JIT_builds'],'post_scope_cleanup':r['post_scope_cleanup'],'external_cleanup':e},'evidence':{str(p.relative_to(R)):sha(p)for p in[C/'inventory.json',P/'result.json',P/'session.json',P/'raw.csv',P/'analysis.json']},'cause_confidence':{'source_setup_numeric_lifetime':'PASS observed, exact intended threefunctions selected','instructions':'PROVEN quarter+36.65%/sparse+16.37% executedwarpinst vsparent; sparse−14.84% vsquarter','hardware_pressure':'PROVEN measured longscoreboard79.48–82.56%samples/DRAM70.58–73.28%elapsedpeak/issue20.32–27.41%; not walltimefractions or repairPC attribution','clock':'PROVEN sixsnapshotsP0/1980SM/2619mem,35–36C; withinpass or priorcleanrun clocks UNKNOWN','clean_loss_cause':'UNKNOWN: profile-durationranking differsclean; unchangedtraffic and more dynamicinstructions establish tradeoff, not lossallocation'}}
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'pins':out['runtime_pins_verified'],'hashes':out['artifact_hashes_verified'],'cost':out['cost'],'clock':clocks},indent=2))
if __name__=='__main__':main()

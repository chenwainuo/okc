"""Recompute capture validity and static PC bindings locally; never export/SSH/GPU."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'results/proj-rsi/scheduler/artifacts/native.profile.v1.01/collection-1793077258598870796'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 r=json.loads((ARCHIVE/'logs/scheduler/native-profile-01/driver.json').read_text());inventory=json.loads((ARCHIVE/'inventory.json').read_text())
 for rel,pin in inventory['files'].items():assert sha(ARCHIVE/rel)==pin,rel
 spec=importlib.util.spec_from_file_location('native_profile_frozen',ROOT/'proj_rsi/tools/native_x_compare_ncu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert sha(ROOT/'proj_rsi/tools/native_x_compare_ncu.py')==r['driver_sha256']
 support=m.support_module();summaries={}
 for role,v in r['variants'].items():
  sass=(ARCHIVE/'logs/scheduler/native-profile-01'/('driver.'+role+'.jit.sass')).read_text();assert hashlib.sha256(sass.encode()).hexdigest()==v['JIT_sha256']
  _,mod,checker=m.bind(role)
  gate=checker.check(sass);assert gate==v['physical_gate']
  canonical=support.actual_binary_gate('dp' if role=='DP' else 'new',sass);assert json.loads(json.dumps(canonical))==v['canonical_gate']
  mapping=m.pc_map(sass,gate,role);assert json.loads(json.dumps(mapping))==v['PC_map']
  p=mapping[0];parts={key:len(p['roles'][key]) for key in ('math_PCs','producer_PCs','common_setup_unreachable_PCs')}
  sets=[set(p['roles'][key]) for key in parts];assert all(not x&y for i,x in enumerate(sets) for y in sets[i+1:])
  summaries[role]={'canonical_actual_SASS_gate_recomputed':'PASS','physical_gate_recomputed':'PASS','role_PC_map_recomputed':'PASS','static_role_instruction_counts_NOT_SAMPLES':parts,'static_wait_sites_by_class_NOT_SAMPLES':dict(collections.Counter(w['classification'] for w in p['waits'])),'static_site_counts_NOT_SAMPLES':{name:len(value) for name,value in p['sites'].items()},'resources':v['resources'],'post_range_oracles':v['selected_oracles'],'dynamic_sample_denominators':{'math':None,'producer':None,'common':None,'unmapped':None,'total':None},'dynamic_analysis_status':'QUEUED_CPU_EXPORT','PC_map':p}
  assert v['resources']['main_regs']==122 and v['resources']['main_local']==0 and v['resources']['occupancy_after_optin']==1
  assert v['selected_oracles']['cnt']=='ZERO' and v['selected_oracles']['W_pack_esc_X']=='IMMUTABLE' and v['selected_oracles']['poison_overwritten'] is True
 assert r['graph_cleanup']==['RESET_AFTER_DRAIN']*2 and r['status']=='COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED'
 assert r['phase']=='IMMEDIATE_ORACLES_PASSED' and r['profile_nodes_expected']==['DP_rx_pack','rq_px_dp','rq_tma_x']
 assert r['variants']['DP']['selected_oracles']['XP']=='EXACT_ALL_1MiB'
 assert r['variants']['NX']['selected_oracles']['XP']=='NONE_RAW_TMA_LDSM_BITS_UNPROBED'
 clean=inventory['cleanup_observation'];assert clean['container_exists'] is False and clean['guard_pids']==clean['containers']==clean['compute_pids']==[] and clean['gpu_memory_mib']==0 and clean['cleanup_failed'] is False
 host=(ARCHIVE/'logs/scheduler/native.profile.v1.01.host.log').read_text()
 assert [name for name in ('rx_pack','rq_px_dp','rq_tma_x') if 'Profiling "'+name+'"' in host]==['rx_pack','rq_px_dp','rq_tma_x']
 assert host.count('34 passes')==3 and 'TRIAL_EXIT okc-proj-native-profile-01 0' in host
 report=HERE/'input/native-profile-01.ncu-rep';assert sha(report)==sha(ARCHIVE/'logs/scheduler/native-profile-01.ncu-rep')
 result={'status':'CAPTURE_VALIDITY_PASS_COUNTER_ANALYSIS_QUEUED_CPU_EXPORT','job_id':'native.profile.v1.01','driver_sha256':r['driver_sha256'],'archive':str(ARCHIVE.relative_to(ROOT)),'inventory_files_verified':len(inventory['files']),'inventory_sha256':sha(ARCHIVE/'inventory.json'),'report_sha256':sha(report),'profile_nodes_observed':['rx_pack','rq_px_dp','rq_tma_x'],'NCU_replay_passes_per_node':34,'graphs_drained_reset':2,'cleanup_verified':True,'cleanup_observation':clean,'variants':summaries,'local_export_capability':{'ncu_executable':shutil.which('ncu'),'ncu_report_PythonSDK':None,'search':'PATH; /opt,/usr/local,/usr/bin up to6levels; Python importlib spec','status':'QUEUED_CPU_EXPORT'},'unavailable_capture_metrics':['ctc__rx_bytes_data_user.sum','ctc__rx_bytes_data_user.sum.pct_of_peak_sustained_elapsed','ctc__rx_bytes_data_user.sum.per_second','ctc__tx_bytes_data_user.sum','ctc__tx_bytes_data_user.sum.pct_of_peak_sustained_elapsed','ctc__tx_bytes_data_user.sum.per_second'],'interpretation_limits':r['preregistration']['interpretation'],'sample_sufficiency_required':r['preregistration']['sample_sufficiency'],'recommendation':'Keep counter interpretation pending serialized CPU export and source/raw reconciliation; no speed/disposition gain claim. Capture validity passes independently.'}
 (HERE/'capture-validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'inventory_files':result['inventory_files_verified'],'report_sha256':sha(report),'static_counts_NOT_samples':{r:v['static_role_instruction_counts_NOT_SAMPLES'] for r,v in summaries.items()}},indent=2))
if __name__=='__main__':main()

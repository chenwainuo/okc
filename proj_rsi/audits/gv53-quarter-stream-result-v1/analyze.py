"""Reconcile recorded one-cell artifacts and all signed estimators; CPU only."""
import argparse,hashlib,importlib.util,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
BASE=None;RUN=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 global BASE,RUN
 p=argparse.ArgumentParser();p.add_argument('--collection',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
 BASE=args.collection.resolve();RUN=BASE/'logs/scheduler/gv53-quarter-stream-01/run'
 assert not args.output.exists(),'refuse overwrite of result analysis; use a fresh path'
 inv=json.loads((BASE/'inventory.json').read_text());result=json.loads((RUN/'result.json').read_text())
 assert inv['terminal_exit']==0 and not inv['missing']
 for rel,pin in inv['files'].items():assert sha(BASE/rel)==pin,rel
 for rel,pin in inv['dispatch']['source_pins'].items():assert sha(ROOT/rel)==pin,rel
 assert result['state']=='ONE_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'
 assert result['profiler_calls']==result['full_prepare_calls']==0
 assert result['source_sha256']=='b7c8fccd8a92c57387872f41e164aefb86d70f5756ebef6ba5ae6b8a2bc7a7fb'
 assert result['control_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert result['build']['SO_sha256']==sha(RUN/'actual.so')and result['build']['CUDA_sha256']==sha(RUN/'actual.cu')
 expected=(ROOT/'proj_rsi/candidates/gv53_quarter_stream_v1/pilot.cu').read_text()+'\n'+(ROOT/'proj_rsi/candidates/gv53_quarter_stream_trial_v1/binding.cu').read_text()
 assert(RUN/'actual.cu').read_text()==expected
 driver=ROOT/'proj_rsi/candidates/gv53_quarter_stream_trial_v1/driver.py';spec=importlib.util.spec_from_file_location('frozen_keyed_driver_for_schedule',driver);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert mod.cpu_check()['state']=='PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN'
 journal=[json.loads(line)for line in(RUN/'raw.jsonl').read_text().splitlines()];expected_journal=[];rows=[];numerics=0
 for m in(8,):
  entry=result['fixed_cells'][str(m)];raw=entry['raw'];assert len(raw)==91
  for actual,planned in zip(raw,mod.schedule()):
   assert {k:v for k,v in actual.items()if k!='us'}==planned
   assert math.isfinite(actual['us'])and actual['us']>0
   expected_journal.append(dict(actual,m=m))
  assert entry['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and entry['cleanup']=='RESET_AFTER_DRAIN'
  for arm in mod.LABELS:
   v=entry['validation'][arm];assert v['basis']=='EXACT'and v['basis_columns']==mod.basis_columns()
   assert [x['kind']for x in v['task']]==['normal','outliers','zero'];numerics+=3
  def oracle(message):
   assert message['current_X']=='EXACT'and'error'in message['task']
   assert math.isfinite(message['FP64_L2'])and message['FP64_L2']<=.004
   assert math.isfinite(message['worst_row'])and message['worst_row']<=.02
  for valid in entry['validation'].values():
   for check in valid['task']:oracle(check['oracle'])
  for phase,end in(('F1',43),('paired',91)):
   terminals=entry['terminals'][phase];assert len(terminals)==2
   for arm,t in zip(mod.LABELS,terminals):
    assert t['arm']==arm and t['before_any_replay_or_mutation']
    assert t['seed']==next(r['seed']for r in reversed(raw[:end])if r['arm']==arm and r['mode']!='warmup')
    oracle(t['oracle'])
  expected_mutations=[(kind,arm)for kind in('zero','normal','outliers','boundary_probes','fresh_normal')for arm in mod.LABELS]
  assert[(r['kind'],r['arm'])for r in entry['mutations']]==expected_mutations
  for mutation in entry['mutations']:assert mutation['NaN_output_overwritten'];oracle(mutation['oracle'])
  f1=[statistics.median(r['us']for r in raw[10+11*i:21+11*i])for i in range(3)]
  assert dict(zip(('control_before','quarter','control_after'),f1))==entry['F1_medians_us']
  pairs=[{r['arm']:r['us']for r in raw[43+2*i:45+2*i]}for i in range(24)];assert pairs==entry['paired']
  differences=[r['control']-r['quarter']for r in pairs];contrast=entry['contrasts']
  assert statistics.median(differences)==contrast['paired_median_control_minus_new_us']
  assert(f1[0]+f1[2])/2-f1[1]==contrast['F1_endpoint_average_control_minus_new_us']
  assert sum(x>0 for x in differences)==contrast['paired_new_wins']
  marginal={arm:statistics.median(r[arm]for r in pairs)for arm in mod.LABELS}
  rows.append({'m':m,'F1_control_before_us':f1[0],'F1_new_us':f1[1],'F1_control_after_us':f1[2],'F1_endpoint_average_control_minus_new_us':contrast['F1_endpoint_average_control_minus_new_us'],'F1_conservative_min_endpoint_control_minus_new_us':min(f1[0],f1[2])-f1[1],'paired_median_control_minus_new_us':statistics.median(differences),'paired_AB_median_control_minus_new_us':statistics.median(differences[::2]),'paired_BA_median_control_minus_new_us':statistics.median(differences[1::2]),'paired_new_wins':sum(x>0 for x in differences),'paired_control_wins':sum(x<0 for x in differences),'paired_ties':sum(x==0 for x in differences),'paired_arm_medians_us':marginal,'difference_of_marginal_paired_medians_control_minus_new_us':marginal['control']-marginal['quarter'],'paired_differences_all_us':differences})
 assert journal==expected_journal and len(journal)==91
 for i,arm in enumerate(mod.LABELS):
  a=result['raw_attributes'][arm];assert 0<a['registers']<=255 and a['local']>=0 and a['user_static']>=0 and a['max_dynamic']>=17408 and a['max_threads']>=1024 and a['occupancy']>=1 and a['arm_echo']==i
 cleanup=inv['cleanup_observation'];assert not cleanup['cleanup_failed']and not cleanup['container_exists']and not cleanup['compute_pids']and not cleanup['guard_pids']and cleanup['gpu_memory_mib']==0
 assert result['cleanup']==[]
 post=result['post_scope_cleanup'];assert post['allocated']==0 and post['clear_error']is None
 assert post['blas_map_entries_workspace_bytes_after'][-1]==0
 assert post['after_run_scope_and_drain']
 analysis={'state':'VALID_RECORDED_QUARTER_M8_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION','job_id':inv['job_id'],'collection':str(BASE.relative_to(ROOT)),'evidence':{str(p.relative_to(ROOT)):sha(p)for p in(BASE/'inventory.json',RUN/'result.json',RUN/'raw.jsonl',RUN/'actual.cu',RUN/'actual.so',driver)},'artifact_hashes_verified':len(inv['files']),'source_pins_verified':len(inv['dispatch']['source_pins']),'rows':rows,'recorded_validation':{'task_checks':numerics,'basis_columns_per_arm_cell':8,'basis_columns_total':16,'terminal_sites':4,'mutation_replays':10,'raw_observations':91,'timed_observations':81,'warmups':10,'raw_attributes':result['raw_attributes'],'physical_cleanup':{k:v for k,v in cleanup.items()if'log'not in k}},'cost':result['cost'],'limits':['Positive differences favor new; paired matched-difference medians and marginal-median differences are distinct and can disagree.','Inspect each signed estimator independently; sign conflicts and small gains remain evidence.','This validates recorded fixed implementation observations, not producer critical-path cycle latency, selected policy gain or official H200 score.','Output/counter tensors were not archived; numeric/current-X/full-backing assertions are recorded successful frozen execution, not fresh offline GPU recomputation.','Source/control identity and build/source/SO hashes are checked; no new SASS or profiler interpretation is required.','No threshold, automatic rerun, forced selection, promotion or family rejection follows.']}
 analysis['post_scope_cleanup']=result['post_scope_cleanup']
 analysis['parent_selector_distinction']='Fixed implementation compares current official rolled kernel versus quarter-packet phased decode on one prospectiveWseed5344, without prepare/selected-route/official measurement.'
 analysis['next_falsifier']='Realized earlier firstMMA with extra fixed header/loop/shuffle work didnot improve primary paired median. Small next discriminator: same new128B transport and four-quarter repair producer, postpone all4HMMAs until allquarters repaired, versus current phased consumer; holds packet fixed to distinguish producer/transport overhead from quarter-MMA overlap. This is an unmeasured controlled ablation, not an automatic unchanged rerun or family rejection.'
 analysis['transport']=result['transport']
 assert result['transport']['groups']==4096 and result['transport']['bytes_per_warpgroup']==128 and result['transport']['actual_accepted_records_exact_after_stable_bucketing'] and result['transport']['separate_bank_pointers'] and result['transport']['outside_timed_region']
 analysis['signed_summary']={'positive_paired_cells':[r['m']for r in rows if r['paired_median_control_minus_new_us']>0],'negative_paired_cells':[r['m']for r in rows if r['paired_median_control_minus_new_us']<0],'F1_conflict_cells':[r['m']for r in rows if r['paired_median_control_minus_new_us']*r['F1_endpoint_average_control_minus_new_us']<0]}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(analysis,indent=2)+'\n')
 print(json.dumps({'state':analysis['state'],'rows':rows,'cost':analysis['cost']},indent=2))
if __name__=='__main__':main()

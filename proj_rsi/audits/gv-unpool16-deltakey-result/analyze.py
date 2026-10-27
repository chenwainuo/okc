"""CPU-only collected fixed unpoolerM16 analysis; no queue/GPU/API operation."""
from pathlib import Path
import hashlib,json,math,statistics
D=Path(__file__).resolve().parent;R=D.parents[2];JOB='research.gv.unpool16.deltakey.01';CAND=R/'proj_rsi/candidates/gv_unpool16_deltakey_v1';SOURCE='713c46b37e50f71410ef4c7116f7fcb94bea71bd7b64c7f33d2a9e0d93e24f6c';CONTROL='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9';LABELS=('control','deltakey');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.relative_to(R)),'sha256':sha(p)}
def main():
 B=R/'results/proj-rsi/scheduler/artifacts'/JOB
 if not B.exists():print('NOT_COLLECTED');return
 C=sorted(B.glob('collection-*'))[-1];I=json.loads((C/'inventory.json').read_text());P=C/'logs/scheduler/gv-unpool16-deltakey-01/run';r=json.loads((P/'result.json').read_text())if(P/'result.json').exists()else{}
 for rel,h in I['files'].items():assert sha(C/rel)==h,rel
 out={'job_id':JOB,'source_state':r.get('state','NO_RESULT_JSON'),'phase':r.get('phase'),'terminal_exit':I['terminal_exit'],'artifact_hashes_verified':len(I['files']),'evidence':[pin(C/'inventory.json')],'new_GPU_API_queue_git_calls':0}
 for p in(P/'result.json',P/'raw.jsonl',C/'logs/scheduler'/f'{JOB}.host.log'):
  if p.exists():out['evidence'].append(pin(p))
 if r.get('state')!='ONE_FIXED_UNPOOLER16_CELL_COMPLETE_ROOT_REVIEW_REQUIRED':
  out.update(state='SETUP_RUNTIME_OR_CLEANUP_FAILURE_PRESERVED_NO_PERFORMANCE_ADMISSION',error=r.get('error'),attributes=r.get('raw_attributes'),post_scope_cleanup=r.get('post_scope_cleanup'),raw=r.get('fixed_cells',{}))
  host=C/'logs/scheduler'/f'{JOB}.host.log'
  if host.exists():out['host_log_tail']=host.read_text()[-10000:]
 else:
  assert I['terminal_exit']==0 and not I['missing']and r['phase']=='complete'and not r['cleanup']
  for rel,h in I['dispatch']['source_pins'].items():assert sha(R/rel)==h,rel
  assert r['source_sha256']==sha(CAND/'submission.py')==SOURCE and r['control_sha256']==CONTROL
  assert sha(P/'actual.cu')==r['build']['CUDA_sha256']and sha(P/'actual.so')==r['build']['SO_sha256']
  assert(P/'actual.cu').read_text()==(CAND/'pilot.cu').read_text()+'\n'+(CAND/'binding.cu').read_text()
  assert r['profiler_calls']==r['full_prepare_calls']==0
  s=r['fixed_cells']['16'];assert s['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and s['cleanup']=='RESET_AFTER_DRAIN'
  assert s['config']=={'layer':'unpooler','m':16,'n':8192,'k':4096}
  raw=s['raw'];assert len(raw)==91 and sum(q['mode']!='warmup'for q in raw)==81 and all(math.isfinite(q['us'])and q['us']>0 for q in raw)
  journal=[json.loads(line)for line in(P/'raw.jsonl').read_text().splitlines()];assert journal==[dict(q,m=16)for q in raw]
  f1=raw[10:43];med=[statistics.median(q['us']for q in f1[11*i:11*(i+1)])for i in range(3)];assert s['F1_medians_us']==dict(zip(('control_before','deltakey','control_after'),med))
  pairs=[]
  for i in range(24):
   group=raw[43+2*i:45+2*i];assert[q['arm']for q in group]==list(LABELS if i%2==0 else LABELS[::-1])and len({q['seed']for q in group})==1;pairs.append({q['arm']:q['us']for q in group})
  assert pairs==s['paired'];deltas=[q['control']-q['deltakey']for q in pairs];paired=statistics.median(deltas);ratio=statistics.median(q['control']/q['deltakey']for q in pairs)
  assert paired==s['contrasts']['paired_median_control_minus_new_us']and(med[0]+med[2])/2-med[1]==s['contrasts']['F1_endpoint_average_control_minus_new_us']
  assert len(s['mutations'])==10 and all(q['NaN_output_overwritten']for q in s['mutations'])
  for label in LABELS:
   assert len(s['validation'][label]['task'])==3 and s['validation'][label]['basis']=='EXACT'and len(s['validation'][label]['basis_columns'])==16
  for phase,end in(('F1',43),('paired',91)):
   terms=s['terminals'][phase];assert len(terms)==2 and {x['arm']for x in terms}==set(LABELS)
   for q in terms:
    assert q['before_any_replay_or_mutation']and q['seed']==next(z['seed']for z in reversed(raw[:end])if z['arm']==q['arm']and z['mode']!='warmup')and q['oracle']['current_X']=='EXACT'
  cleanup=r['post_scope_cleanup'];assert cleanup['after_run_scope_and_drain']and cleanup['allocated']==0 and cleanup['clear_error']is None and cleanup['blas_map_entries_workspace_bytes_after'][2]==0
  ext=I['cleanup_observation'];assert not ext['cleanup_failed']and not ext['container_exists']and not ext['compute_pids']and not ext['guard_pids']and not ext['containers']and ext['gpu_memory_mib']==0
  out.update(state='CPU_COLLECTED_FIXED_CELL_VALIDATED',disposition='VALID_GAIN'if paired>0 else'VALID_REGRESSION'if paired<0 else'VALID_NO_GAIN',disposition_basis='Primary paired point; signed F1 and every loss preserved, no significance/minimumgain gate.',source_sha256=SOURCE,control_sha256=CONTROL,fixed_template=[1,2,2,1,4,16,1,1,2,1],hardware=r['hardware'],actual_resources=r['raw_attributes'],pack=r['pack'],timing={'sign':'control_minus_deltakey_us,positivefavorsnew','paired_delta_us':paired,'paired_control_over_new_ratio':ratio,'win_tie_loss':[sum(x>0 for x in deltas),sum(x==0 for x in deltas),sum(x<0 for x in deltas)],'delta_range_us':[min(deltas),max(deltas)],'F1_medians_us':s['F1_medians_us'],'F1_before_minus_new_us':med[0]-med[1],'F1_after_minus_new_us':med[2]-med[1],'F1_endpoint_average_minus_new_us':(med[0]+med[2])/2-med[1]},validation={'runtime_source_pins':len(I['dispatch']['source_pins']),'journal_rows':91,'clean_timing_calls':81,'warmups':10,'own_terminal_oracles':4,'retained_mutations':10,'task_FP64_checks_before_timing':6,'basis_columns_per_arm':16,'basis_launches_per_arm':1,'counter_scratch_input_weight_contracts':'Frozen source successful invoked full-backing S1/input/weight assertions; finalindividualbytevalues notarchived.','post_scope_cleanup':cleanup,'external_cleanup':{k:v for k,v in ext.items()if k!='host_log_tail'}},cost=r['cost'],limits=['FixedexactH200selectedgeometry comparison, freshservingprepare may selectanotherpath.','Directeventtimingisnotcalibratedofficialscore orprofiler duration.','Static88instruction/2register reduction isnotdynamicrepairtime fraction orspeedprediction.','No full30/official gain or additivecompositionclaim; sourcecompatiblepackage mustbe measured.'])
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('host_log_tail','raw','validation','evidence')},indent=2))
if __name__=='__main__':main()

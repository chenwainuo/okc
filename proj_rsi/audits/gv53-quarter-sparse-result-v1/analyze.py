"""Read-only three-arm fixed-cell archive reconciliation; root owns DONE."""
import argparse,hashlib,importlib.util,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--collection',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();BASE=args.collection.resolve();RUN=BASE/'logs/scheduler/gv53-quarter-sparse-01/run'
 assert not args.output.exists(),'Use a new output path, never overwrite frozen evidence'
 inv=json.loads((BASE/'inventory.json').read_text());r=json.loads((RUN/'result.json').read_text())
 assert inv['terminal_exit']==0 and not inv['missing']
 for rel,h in inv['files'].items():assert sha(BASE/rel)==h,rel
 for rel,h in inv['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
 assert r['state']=='THREE_ARM_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'
 assert r['source_sha256']=='f1fa67927bf0ea43c64642934ea650da8f42b918e383c8e7b9e17ee52bb855da'
 assert r['control_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert r['profiler_calls']==r['full_prepare_calls']==0
 assert r['build']['SO_sha256']==sha(RUN/'actual.so')and r['build']['CUDA_sha256']==sha(RUN/'actual.cu')
 expected=(ROOT/'proj_rsi/candidates/gv53_quarter_sparse_header_v1/pilot.cu').read_text()+'\n'+(ROOT/'proj_rsi/candidates/gv53_quarter_sparse_trial_v1/binding.cu').read_text();assert(RUN/'actual.cu').read_text()==expected
 driver=ROOT/'proj_rsi/candidates/gv53_quarter_sparse_trial_v1/driver.py';sp=importlib.util.spec_from_file_location('frozen_sparse_schedule',driver);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);assert mod.cpu_check()['state']=='PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN'
 prior=r['predecessor_SO'];assert prior['SO_sha256']==mod.OLD_SO_SHA==sha(ROOT/prior['path']) and prior['PyInit_module_name']==mod.OLD_INIT and prior['source_CUDA_sha256']==mod.OLD_CUDA_SHA and prior['loaded_without_compile'] and prior['native_arm']==1
 expected_bindings={a:{'module':'collected_predecessor'if a=='quarter'else'new_compile','native_arm':0 if a=='control'else 1}for a in mod.LABELS};assert r['arm_bindings']==expected_bindings
 assert set(r['fixed_cells'])=={'8'};e=r['fixed_cells']['8'];raw=e['raw'];planned=mod.schedule();assert len(raw)==len(planned)==129
 for actual,plan in zip(raw,planned):
  assert {k:v for k,v in actual.items()if k!='us'}==plan
  assert math.isfinite(actual['us'])and actual['us']>0
 journal=[json.loads(s)for s in(RUN/'raw.jsonl').read_text().splitlines()];assert journal==[dict(x,m=8)for x in raw]
 assert e['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and e['cleanup']=='RESET_AFTER_DRAIN'
 oracles=[]
 def oracle(o):
  assert o['current_X']=='EXACT'and 'error'in o['task']
  assert math.isfinite(o['FP64_L2'])and o['FP64_L2']<=.004 and math.isfinite(o['worst_row'])and o['worst_row']<=.02;oracles.append(o)
 for arm in mod.LABELS:
  v=e['validation'][arm];assert v['basis']=='EXACT'and v['basis_columns']==mod.basis_columns()
  assert[x['kind']for x in v['task']]==['normal','outliers','zero']
  for x in v['task']:oracle(x['oracle'])
 for phase,end in(('blocked',57),('triples',129)):
  terms=e['terminals'][phase];assert len(terms)==3
  for arm,t in zip(mod.LABELS,terms):
   assert t['arm']==arm and t['before_any_replay_or_mutation'];assert t['seed']==next(x['seed']for x in reversed(raw[:end])if x['arm']==arm and x['mode']!='warmup');oracle(t['oracle'])
 assert[(x['kind'],x['arm'])for x in e['mutations']]==[(k,a)for k in('zero','normal','outliers','boundary_probes','fresh_normal')for a in mod.LABELS]
 for x in e['mutations']:assert x['NaN_output_overwritten'];oracle(x['oracle'])
 blocks=[raw[15+7*i:22+7*i]for i in range(6)];bm=[statistics.median(x['us']for x in b)for b in blocks];names=('control','quarter','sparse','sparse','quarter','control')
 assert e['blocked_medians_us']==[{'arm':a,'us':v,'block':i}for i,(a,v)in enumerate(zip(names,bm))]
 endpoints={a:[bm[i]for i,n in enumerate(names)if n==a]for a in mod.LABELS};av={a:statistics.mean(v)for a,v in endpoints.items()}
 triples=[raw[57+3*i:60+3*i]for i in range(24)];values=[{x['arm']:x['us']for x in t}for t in triples];assert values==e['triples'];comparisons=[]
 for a,b in mod.CONTRASTS:
  differences=[v[a]-v[b]for v in values];order=[t[0]['order']for t in triples];ab=[d for d,o in zip(differences,order)if o.index(a)<o.index(b)];ba=[d for d,o in zip(differences,order)if o.index(b)<o.index(a)];assert len(ab)==len(ba)==12
  recorded=e['contrasts'][a+'_minus_'+b];assert recorded=={'paired_median_us':statistics.median(differences),'blocked_endpoint_average_us':av[a]-av[b],'second_arm_wins':sum(x>0 for x in differences),'first_arm_wins':sum(x<0 for x in differences),'ties':sum(x==0 for x in differences)}
  marg={x:statistics.median(v[x]for v in values)for x in(a,b)}
  comparisons.append({'first_arm':a,'second_arm':b,'positive_favors_second_arm':True,'paired_median_first_minus_second_us':statistics.median(differences),'first_before_second_median_us':statistics.median(ab),'second_before_first_median_us':statistics.median(ba),'second_arm_wins':sum(x>0 for x in differences),'first_arm_wins':sum(x<0 for x in differences),'ties':sum(x==0 for x in differences),'blocked_endpoint_average_first_minus_second_us':av[a]-av[b],'blocked_conservative_minfirst_minus_maxsecond_us':min(endpoints[a])-max(endpoints[b]),'marginal_paired_arm_medians_us':marg,'difference_of_marginal_paired_medians_us':marg[a]-marg[b],'all24signed_effects_us':differences})
 for a in mod.LABELS:
  attrs=r['raw_attributes'][a];assert 0<attrs['registers']<=255 and attrs['local']>=0 and attrs['user_static']>=0 and attrs['max_dynamic']>=17408 and attrs['max_threads']>=1024 and attrs['occupancy']>=1 and attrs['arm_echo']==expected_bindings[a]['native_arm']
 tr=r['transport'];assert tr['groups']==4096 and tr['bytes_per_warpgroup']==128 and tr['record_count']==4759 and tr['actual_accepted_records_exact_after_stable_bucketing']and tr['separate_bank_pointers']and tr['outside_timed_region']
 previous=json.loads((ROOT/'proj_rsi/audits/gv53-quarter-stream-result-v1/analysis.json').read_text())['transport'];assert all(tr[k]==previous[k]for k in['parent_bank_sha256','quarter_bank_sha256','record_count'])
 cleanup=inv['cleanup_observation'];assert not cleanup['cleanup_failed']and not cleanup['container_exists']and not cleanup['compute_pids']and not cleanup['guard_pids']and cleanup['gpu_memory_mib']==0
 post=r['post_scope_cleanup'];assert r['cleanup']==[]and post['allocated']==0 and post['reserved']==0 and post['clear_error']is None and post['blas_map_entries_workspace_bytes_after'][-1]==0 and post['after_run_scope_and_drain']
 analysis={'state':'VALID_RECORDED_THREE_ARM_M8_PRIMARY_REGRESSIONS','job_id':inv['job_id'],'collection':str(BASE.relative_to(ROOT)),'artifact_hashes_verified':len(inv['files']),'dispatch_source_pins_verified':len(inv['dispatch']['source_pins']),'comparisons':comparisons,'blocked_medians_us':e['blocked_medians_us'],'blocked_arm_endpoints_us':endpoints,'blocked_arm_endpoint_averages_us':av,'raw_attributes':r['raw_attributes'],'predecessor_SO':prior,'arm_bindings':r['arm_bindings'],'transport':tr,'recorded_validity':{'normal_outlier_zero_task_checks':9,'basis_columns_total':24,'own_terminal_sites':6,'graph_mutation_replays':15,'oracle_snapshots':len(oracles),'max_FP64_L2':max(x['FP64_L2']for x in oracles),'max_worst_row':max(x['worst_row']for x in oracles),'raw_observations':129,'timed_observations':114,'warmups':15,'all_pairwise_orders':{'first_before_second':12,'reverse':12},'each_arm_position_count':8},'post_scope_cleanup':post,'physical_cleanup':{k:v for k,v in cleanup.items()if'log'not in k},'cost':r['cost'],'cost_note':'compile_and_load_seconds includes separately logged predecessor_SO_load_seconds; do not add them.','evidence':{str(p.relative_to(ROOT)):sha(p)for p in[BASE/'inventory.json',RUN/'result.json',RUN/'raw.jsonl',RUN/'actual.cu',RUN/'actual.so',driver,ROOT/prior['path']]},'limits':['Positive first-minus-second effects favor second; retain all signs/order subsets/blocked differences.','Three pairwise contrasts share the same24triples; do not count them as independent trials or add gains.','Fixed same-W/X native implementation test, no selected policy/fullprepare/official score claim.','Output/bank/counter tensors not archived: offline reconciliation validates recorded frozen runtime assertions, not fresh GPU recomputation.','Queried CTA capacity is not measured achieved residency; source-level shuffle counts are not hardware time fractions.','Primary regression does not retire family or support automatic unchanged retry.']}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps({'state':analysis['state'],'comparisons':comparisons,'cost':analysis['cost']},indent=2))
if __name__=='__main__':main()

"""CPU-only analysis of exact production native-body validation, not prepare selection."""
from pathlib import Path
import hashlib,json,re,statistics,math,subprocess
D=Path(__file__).resolve().parent;R=D.parents[2];JOB='research.gv53.selected.earlyload.bodies.01';Q=R/'proj_rsi/candidates/gv53_selected_earlyload_v1';T=R/'proj_rsi/candidates/gv53_selected_earlyload_trial_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();med=statistics.median

def rows(text):return{c.splitlines()[0].strip():re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',c)for c in text.split('Function : ')[1:]}
def main():
 collections=sorted((R/'results/proj-rsi/scheduler/artifacts'/JOB).glob('collection-*'))
 if not collections:print('WAIT_LOCAL_COLLECTION');return
 C=collections[-1];P=C/'logs/scheduler/gv53-selected-earlyload-bodies-01/run'
 if not(P/'result.json').exists():print('WAIT_RESULT');return
 i=json.loads((C/'inventory.json').read_text());r=json.loads((P/'result.json').read_text())
 for rel,h in i['files'].items():assert sha(C/rel)==h,rel
 for rel,h in i['dispatch']['source_pins'].items():assert sha(R/rel)==h,rel
 assert i['terminal_exit']==0 and not i['missing']and r['phase']=='complete'and not r['cleanup']and r['state']=='TWO_FIXED_PACKAGE_BODY_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED'
 assert r['source_sha256']==sha(Q/'submission.py')=='17d44d1140ab09cdbab1f2835841e2891fc1c7c01c1eb4d9a2833c242d99a3e6'and r['pilot_sha256']==sha(Q/'pilot.cu')
 assert r['control_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert sha(P/'actual.cu')==r['build']['CUDA_sha256']and sha(P/'actual.so')==r['build']['SO_sha256']and(P/'actual.cu').read_text()==(Q/'pilot.cu').read_text()+'\n'+(T/'bridge.cuh').read_text()
 dump=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(P/'actual.so')],text=True);a=rows(dump);expected=rows((Q/'pilot.sass').read_text());saved=rows((P/'actual.sass').read_text())
 for name,ok in r['build']['actual_fullPC_predicate_encoding_equals_pilot'].items():assert ok and a[name]==saved[name]==expected[name]
 assert len(r['build']['actual_fullPC_predicate_encoding_equals_pilot'])==3 and r['build']['CXX_Torch_compile_calls']==0 and r['build']['native_compile_calls']==1 and all(s['returncode']==0 for s in r['build']['stages'])
 for index,role in enumerate(('parent','keyed_M1','earlyload_M8')):
  x=r['raw_attributes'][role];assert x['registers']==64 and x['local']==x['user_static']==0 and x['max_dynamic']==17408 and x['max_threads']==1024 and x['occupancy']==1 and x['arm_echo']==index
 journal=[json.loads(x)for x in(P/'raw.jsonl').read_text().splitlines()];assert journal==[dict(x,m=m)for m in(1,8)for x in r['fixed_cells'][str(m)]['raw']]
 cells={}
 for m in(1,8):
  c=r['fixed_cells'][str(m)];raw=c['raw'];assert len(raw)==91 and sum(x['mode']!='warmup'for x in raw)==81 and all(math.isfinite(x['us'])and x['us']>0 for x in raw)
  assert c['config']=={'layer':'pooler','m':m,'n':4096,'k':8192}and c['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and c['cleanup']=='RESET_AFTER_DRAIN'
  assert len(c['host_journal_spans_ns'])==91
  for x,(b,e)in zip(raw,c['host_journal_spans_ns']):assert x['host_before_start_ns']<=x['host_after_end_enqueue_ns']<=x['host_after_sync_ns']<=b<=e
  for arm in('control','selected_option'):
   v=c['validation'][arm];assert len(v['task'])==3 and v['basis']=='EXACT'and len(v['basis_columns'])==8
  for phase,end in[('F1',43),('paired',91)]:
   ts=c['terminals'][phase];assert len(ts)==2 and {x['arm']for x in ts}=={'control','selected_option'}
   for x in ts:assert x['before_any_replay_or_mutation']and x['oracle']['current_X']=='EXACT'and x['seed']==next(y['seed']for y in reversed(raw[:end])if y['arm']==x['arm']and y['mode']!='warmup')
  assert len(c['mutations'])==10 and all(x['NaN_output_overwritten']for x in c['mutations'])
  f=raw[10:43];fm=[med(x['us']for x in f[j*11:(j+1)*11])for j in range(3)];assert c['F1_medians_us']==dict(zip(('control_before','selected_option','control_after'),fm))
  pairs=[]
  for j in range(24):
   pair=raw[43+2*j:45+2*j];assert[x['arm']for x in pair]==(['control','selected_option']if j%2==0 else['selected_option','control'])and len({x['seed']for x in pair})==1;pairs.append({x['arm']:x['us']for x in pair})
  assert pairs==c['paired'];ds=[x['control']-x['selected_option']for x in pairs];point=med(ds);assert point==c['contrasts']['paired_median_control_minus_new_us']
  cells[str(m)]={'actual_fixed_roles':['parent','keyed'if m==1 else'earlyload'],'scope':'Source-bound fixednativearm routing; no prepare/selector comparison executed','disposition_point':'VALID_GAIN'if point>0 else'VALID_REGRESSION'if point<0 else'VALID_NO_GAIN','paired_median_control_minus_option_us':point,'paired_mean_us':statistics.mean(ds),'paired_ratio':med(x['control']/x['selected_option']for x in pairs),'win_tie_loss':[sum(x>0 for x in ds),sum(x==0 for x in ds),sum(x<0 for x in ds)],'paired_each_delta_us':ds,'control_first_median_us':med(ds[::2]),'option_first_median_us':med(ds[1::2]),'F1_medians_us':c['F1_medians_us'],'F1_before_minus_option_us':fm[0]-fm[1],'F1_after_minus_option_us':fm[2]-fm[1],'F1_average_us':(fm[0]+fm[2])/2-fm[1],'F1_drift_us':fm[2]-fm[0],'F1_same_seed_endpoint_medians_us':[med(f[j]['us']-f[11+j]['us']for j in range(11)),med(f[22+j]['us']-f[11+j]['us']for j in range(11))],'host_submit_median_us':med((x['host_after_end_enqueue_ns']-x['host_before_start_ns'])/1000 for x in raw[10:]),'host_journal_median_us':med((e-b)/1000 for b,e in c['host_journal_spans_ns'])}
 assert r['pack']['actual_W_bit_reconstruction']=='EXACT'and r['escape_contract']['both_arms_same_original_bank']and r['escape_contract']['transport_repack_calls']==0
 cl=r['post_scope_cleanup'];assert cl['allocated']==cl['reserved']==0 and cl['clear_error']is None and cl['after_run_scope_and_drain']and cl['blas_map_entries_workspace_bytes_after'][2]==0
 ex=i['cleanup_observation'];assert not ex['cleanup_failed']and not ex['compute_pids']and not ex['containers']and not ex['guard_pids']and not ex['container_exists']and ex['gpu_memory_mib']==0
 out={'state':'VALID_FIXED_PRODUCTION_BODY_PACKAGE','recommended_DONE_disposition':'VALID_DIAGNOSTIC','job_id':JOB,'source_sha256':r['source_sha256'],'artifact_hashes_verified':len(i['files']),'launch_pins_verified':len(i['dispatch']['source_pins']),'actual_CU_sha256':sha(P/'actual.cu'),'actual_SO_sha256':sha(P/'actual.so'),'three_native_streams_match_frozen_pilot':True,'actual_resources':r['raw_attributes'],'hardware':r['hardware'],'pack':r['pack'],'escape_contract':r['escape_contract'],'cells':cells,'build':r['build'],'cost':r['cost'],'validity':{'journal_rows':182,'clean_calls':162,'warmups':20,'initial_task_plus_FP64_oracles':12,'own_terminal_checks':8,'basis_columns_per_arm_cell':8,'mutations':20,'post_scope_cleanup':cl,'external_cleanup':ex},'production_prepare_selection':'UNMEASURED; sourceGO+CPUchooser/dispatch tests separate; fixedroles are deliberately selected bythisdeviceharness','official_gain_us':None,'evidence':{str(x.relative_to(R)):sha(x)for x in[C/'inventory.json',P/'result.json',P/'raw.jsonl',P/'actual.sass',P/'native_compile.stderr']}}
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'cells':cells,'cost':out['cost'],'pins':out['launch_pins_verified'],'hashes':out['artifact_hashes_verified']},indent=2))
if __name__=='__main__':main()

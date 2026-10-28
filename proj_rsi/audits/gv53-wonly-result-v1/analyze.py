"""Strict three-arm W-only actual-result reconciliation; CPU only."""
from pathlib import Path
import argparse,json,hashlib,re,statistics,math,itertools
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def expected_schedule():
 labels=('parent','W_X','W_only');out=[]
 for arm in labels:out.extend(dict(arm=arm,seed=95000,mode='warmup')for _ in range(5))
 for block,arm in enumerate(('parent','W_X','W_only','parent')):out.extend(dict(arm=arm,seed=95100+i,mode='F1',block=block,terminal=i==10)for i in range(11))
 for seed in range(95200,95208):
  for oi,order in enumerate(itertools.permutations(labels)):
   out.extend(dict(arm=arm,seed=seed,mode='paired',order=oi,position=j,terminal=seed==95207 and oi==5)for j,arm in enumerate(order))
 return out

def sass(p):
 out={}
 for c in p.read_text().split('Function : ')[1:]:out[c.splitlines()[0].strip()]=re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',c)
 return out

def analyze(C):
 I=json.loads((C/'inventory.json').read_text());assert I['job_id']=='research.gv53.earlyload.wonly.fixed.01'and I['terminal_exit']==0 and not I['missing']
 for p,h in I['files'].items():assert sha(C/p)==h,('artifact drift',p)
 for p,h in I['dispatch']['source_pins'].items():assert sha(R/p)==h,('source drift',p)
 P=next(C.glob('logs/scheduler/*/run/result.json'));x=json.loads(P.read_text());assert x['state']=='THREE_ARM_W_ONLY_FALSIFIER_COMPLETE_ROOT_REVIEW_REQUIRED'
 S=R/'proj_rsi/candidates/gv53_earlyload_wonly_v1';assert x['source_sha256']==sha(S/'submission.py')and x['pilot_sha256']==sha(S/'pilot.cu')
 assert (P.parent/'actual.cu').read_text()==(S/'pilot.cu').read_text()+'\n'+(S/'bridge.cuh').read_text();assert sha(P.parent/'actual.so')==x['build']['SO_sha256']
 a,b=sass(P.parent/'actual.sass'),sass(S/'pilot.sass');symbols={'parent':'_Z10gv_predxorILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G','W_X':'_Z12gv_earlyloadILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G','W_only':'_Z18gv_earlyload_wonlyILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G'}
 native={}
 for role,name in symbols.items():assert a[name]and a[name]==b[name];native[role]={'instructions':len(a[name]),'full_PC_predicate_encoding_control_exact_pilot':True}
 assert [native[k]['instructions']for k in symbols]==[360,384,392]
 attrs=x['raw_attributes'];assert set(attrs)==set(symbols)
 for i,role in enumerate(symbols):
  q=attrs[role];assert q['arm_echo']==i and q['registers']==64 and q['local']==q['user_static']==0 and q['max_dynamic']>=17408 and q['max_threads']>=1024 and q['occupancy']>=1
 cell=x['fixed_cells']['8'];rows=cell['raw'];expected=expected_schedule();assert len(rows)==len(expected)==203
 for r,e in zip(rows,expected):
  assert {k:v for k,v in r.items()if k not in ['us','host_before_start_ns','host_after_end_enqueue_ns','host_after_sync_ns']}==e
  assert math.isfinite(r['us'])and r['us']>0 and r['host_before_start_ns']<=r['host_after_end_enqueue_ns']<=r['host_after_sync_ns']
 journal=[json.loads(s)for s in(P.parent/'raw.jsonl').read_text().splitlines()];assert journal==[dict(r,m=8)for r in rows]
 spans=cell['host_journal_spans_ns'];assert len(spans)==203 and all(a<=b for a,b in spans)
 assert cell['cleanup']=='RESET_AFTER_DRAIN'and cell['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'
 oracles=[]
 for role in symbols:
  v=cell['validation'][role];assert [z['kind']for z in v['task']]==['normal','outliers','zero'];assert v['basis']=='EXACT'and v['basis_columns']==[0,1,31,63,64,511,512,8191];oracles.extend(z['oracle']for z in v['task'])
 terms=cell['terminals'];flagged=[r for r in rows if r.get('terminal')];assert len(terms)==len(flagged)==7
 for term,row in zip(terms,flagged):
  q=term['receipt'];assert term['mode']==row['mode']and q['arm']==row['arm']and q['seed']==row['seed']and q['before_any_next_replay_or_mutation'];oracles.append(q['oracle'])
 mutations=cell['mutations'];assert [(q['kind'],q['arm'])for q in mutations]==[(kind,arm)for kind in ['zero','normal','outliers','boundary_probes','fresh_normal']for arm in symbols]
 assert all(q['NaN_output_overwritten']for q in mutations);oracles.extend(q['oracle']for q in mutations)
 for q in oracles:assert q['current_X']=='EXACT'and 0<=q['FP64_L2']<=.004 and 0<=q['worst_row']<=.02
 f1={i:{r['seed']:r['us']for r in rows if r['mode']=='F1'and r['block']==i}for i in range(4)};assert all(set(v)==set(range(95100,95111))for v in f1.values());med=[statistics.median(f1[i].values())for i in range(4)]
 groups=[]
 for seed in range(95200,95208):
  for order in range(6):
   rec=[r for r in rows if r['mode']=='paired'and r['seed']==seed and r['order']==order];assert [r['arm']for r in rec]==list(list(itertools.permutations(symbols))[order]);groups.append({'seed':seed,'order':order,'times_us':{r['arm']:r['us']for r in rec}})
 assert groups==cell['paired'];contrasts={}
 for left,right in [('parent','W_X'),('parent','W_only'),('W_X','W_only')]:
  name=left+'_minus_'+right;vals=[g['times_us'][left]-g['times_us'][right]for g in groups];m=statistics.median(vals);assert m==cell['contrasts'][name]['paired_median_us']
  order_points={str(o):statistics.median(vals[i]for i,g in enumerate(groups)if g['order']==o)for o in range(6)}
  by_seed={str(seed):statistics.median(vals[i]for i,g in enumerate(groups)if g['seed']==seed)for seed in range(95200,95208)}
  orders=list(itertools.permutations(symbols));before=[vals[i]for i,g in enumerate(groups)if orders[g['order']].index(left)<orders[g['order']].index(right)];after=[vals[i]for i,g in enumerate(groups)if orders[g['order']].index(left)>orders[g['order']].index(right)]
  contrasts[name]={'paired_median_us':m,'paired_mean_us':statistics.mean(vals),'wins_ties_losses':[sum(v>0 for v in vals),sum(v==0 for v in vals),sum(v<0 for v in vals)],'left_first_median_us':statistics.median(before),'right_first_median_us':statistics.median(after),'six_order_medians_us':order_points,'eight_seed_medians_us':by_seed,'median_of_eight_seed_medians_us':statistics.median(by_seed.values()),'all_48_signed_us':vals,'interpretation':'Positive favors right arm.48triples share8unique input fixtures; no independence/significance inference.'}
 bracket={}
 for i,arm in[(1,'W_X'),(2,'W_only')]:
  bracket[arm]={'parent_before_minus_arm_us':med[0]-med[i],'parent_after_minus_arm_us':med[3]-med[i],'endpoint_average_minus_arm_us':(med[0]+med[3])/2-med[i],'conservative_min_endpoint_minus_arm_us':min(med[0],med[3])-med[i],'same_fixture_before_median_us':statistics.median(f1[0][seed]-f1[i][seed]for seed in f1[0]),'same_fixture_after_median_us':statistics.median(f1[3][seed]-f1[i][seed]for seed in f1[3])}
  for k,v in cell['F1_contrasts'][arm].items():assert v==bracket[arm][k]
 cleanup=x['post_scope_cleanup'];assert cleanup['allocated']==cleanup['reserved']==0 and cleanup['clear_error']is None and cleanup['blas_map_entries_workspace_bytes_after'][2]==0 and cleanup['after_run_scope_and_drain'];assert x['cleanup']==[]
 ext=I['cleanup_observation'];assert ext['terminal_exit']==0 and ext['cleanup_failed']is False and ext['gpu_memory_mib']==0 and not ext['container_exists']and not ext['compute_pids']and not ext['guard_pids']and not ext['containers']
 p=contrasts['parent_minus_W_only']['paired_median_us'];disposition='VALID_GAIN'if p>0 else'VALID_REGRESSION'if p<0 else'VALID_NO_GAIN'
 report={'state':'VALID_FIXED_THREE_ARM_W_ONLY_RESULT','job_id':I['job_id'],'recommended_disposition':disposition,'primary_comparator':'Actual same-run parent501d native vs W-only; all pair/F1/order signs retained.','hardware':x['hardware'],'native':native,'attributes':attrs,'scope':'M8fixed body only; compilerpartlyhoistsX andW-only392>WX384>parent360slots, notpure fixed-X isolation or guaranteedcheaperstaging.','counts':{'raw':203,'warmup':15,'clean':188,'F1':44,'paired':144,'paired_groups':48,'unique_paired_X_fixtures':8,'normal_zero_outlier_task_checks':9,'basis_sets':3,'basis_columns':24,'own_terminals':7,'mutations':15},'W_seed':x['pack']['W_seed'],'contrasts':contrasts,'F1':{'arm_medians_us':dict(zip(['parent_before','W_X','W_only','parent_after'],med)),'parent_drift_us':med[3]-med[0],'brackets':bracket,'W_X_minus_W_only_block_median_us':med[1]-med[2],'last_contrast_scope':'Different adjacent armblocks on same11fixtures, not a bracketed/paired isolated effect.'},'cost':x['cost'],'build':x['build'],'cleanup':{'internal':cleanup,'external':ext},'validated_artifact_hash_count':len(I['files']),'validated_dispatch_source_pins':len(I['dispatch']['source_pins']),'max_saved_FP64_L2':max(q['FP64_L2']for q in oracles),'max_saved_worst_row':max(q['worst_row']for q in oracles),'claims_limits':['One actualWseed reused from earlier W_Xstudy; not independent weight replication or selection/official gain.','Observed relative motion/codegen is real; unique W/X latency or overlap attribution remains unproven because instructions/control schedules differ.','No output bytes saved for independent CPU GPU-numeric recomputation; source-bound successful own-output/currentX/runtime assertions reviewed.','No minimum-size/significance cutoff; keep positive/mixed/losing observations and select a scoped next discriminator.'],'files':{'inventory':{'path':str((C/'inventory.json').relative_to(R)),'sha256':sha(C/'inventory.json')},'result':{'path':str(P.relative_to(R)),'sha256':sha(P)}},'GPU_API_SSH_queue_git_actions':0,'contract':'proj_rsi/ITERATION.md'}
 (D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 done={'job_id':I['job_id'],'recommended_state':'DONE','recommended_disposition':disposition,'scope':report['scope'],'analysis_path':str((D/'analysis.json').relative_to(R)),'analysis_sha256':sha(D/'analysis.json'),'inventory_sha256':sha(C/'inventory.json'),'result_sha256':sha(P),'paired_positive_favors_right_us':{k:v['paired_median_us']for k,v in contrasts.items()},'F1_parent_minus_variant':bracket,'all_native_current_output_ownership_cleanup_gates_passed':True,'no_policy_or_official_gain_claim':True,'no_queue_mutation':True};(D/'done-evidence.json').write_text(json.dumps(done,indent=2)+'\n');print(json.dumps(done,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--collection',required=True,type=Path);a=p.parse_args();analyze(a.collection.resolve())

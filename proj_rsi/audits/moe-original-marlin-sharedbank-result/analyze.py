"""CPU-only immutable collected shared-bank validation, no queue/GPU/API calls."""
from pathlib import Path
import collections,hashlib,json,statistics
D=Path(__file__).resolve().parent;R=D.parents[2];JOB='research.moe.original.marlin.sharedbank.01';SEEDS=list(range(7800,7805))+list(range(7900,7912));LABELS=('anchor','default_shim','contrast');ORDERS=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(0,2,1),(1,0,2));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.relative_to(R)),'sha256':sha(p)}
def signs(v):return [sum(x>0 for x in v),sum(x==0 for x in v),sum(x<0 for x in v)]
def main():
 C=next((R/'results/proj-rsi/scheduler/artifacts'/JOB).glob('collection-*'));P=C/'logs/scheduler/moe-original-marlin-sharedbank-01';I=json.loads((C/'inventory.json').read_text());r=json.loads((P/'result.json').read_text());raw=[json.loads(x)for x in(P/'raw.jsonl').read_text().splitlines()]
 assert I['terminal_exit']==0 and not I['missing']
 for rel,h in I['files'].items():assert sha(C/rel)==h,rel
 for rel,h in I['dispatch']['source_pins'].items():assert sha(R/rel)==h,rel
 cleanup=I['cleanup_observation'];assert not cleanup['cleanup_failed']and not cleanup['container_exists']and not cleanup['compute_pids']and not cleanup['guard_pids']and not cleanup['containers']and cleanup['gpu_memory_mib']==0
 assert r['state']=='VALID_M8_SHARED_NORMAL_SKEWED_FOUR_BLOCK_SCREEN_ROOT_REVIEW_REQUIRED'and r['phase']=='complete'and r['validated_direct_timings_completed']and not r['cleanup']
 assert raw==r['raw']and len(raw)==782 and len(r['correctness'])==60
 assert all(x.get('pass',x.get('exact_zero',False))for x in r['correctness'])
 terminals=[x for x in r['correctness']if x['kind'].startswith('own_')];assert len(terminals)==46 and all('only own exact retained terminal'in x['terminal_scope']for x in terminals)
 assert r['compile_wall_seconds']==0 and r['native_reuse']['no_native_compile']and r['jit'][0]['reused_exact_prior_SO']and sha(P/'two_tiles.so')==r['native_reuse']['native_SO_sha256']==r['jit'][0]['so_sha256']=='57187d60fb029cf5d7d7ce8c09e438600108b501b4b69e1b3cfba125de1aaba3'
 assert not r['profiles']and r['profiling']=='NOT_REQUESTED_REUSED_EXACT_PROFILED_SOURCE_AND_SO'
 q=r['four_block_feasibility']['queries'];assert q['legal']==1 and q['actual_occupancy_4']>=4 and q['four_blocks_register_count']<=q['device_registers_per_SM']and q['four_blocks_total_shared_B']<=q['device_shared_B_per_SM']
 assert r['capture_receipts']['default_shim']['selected']==dict(blocks_per_sm=3,threads=128,thread_n=128,thread_k=64)
 assert r['capture_receipts']['contrast']['selected']==dict(blocks_per_sm=4,threads=128,thread_n=128,thread_k=64)and r['default_receipt']['default_output_bitwise_equal_original']
 meta={(x['dist'],x['seed']):x for x in r['shared_fixtures']};assert len(meta)==34 and set(meta)=={(dist,seed)for dist in('normal','skewed')for seed in SEEDS}
 assert all(set(x['tensor_sha256'])=={'hidden_states','topk_ids','topk_weights'}for x in meta.values())
 for row in raw:assert row['routing']==meta[(row['dist'],row['seed'])]['routing']and 0<row['us']<float('inf')
 outputs={};per_fixture=[]
 for dist,g in r['by_distribution'].items():
  rows=[x for x in raw if x['dist']==dist];assert len(rows)==391
  blocked=[x for x in rows if x['kind']=='blocked'];interleaved=[x for x in rows if x['kind']=='interleaved'];assert len(blocked)==85 and len(interleaved)==306
  bm={}
  for block,label in enumerate(('anchor','default_shim','contrast','default_shim','anchor')):
   group=blocked[block*17:(block+1)*17];assert[x['seed']for x in group]==SEEDS and all(x['arm']==label and x['order_index']==block and x['position']==-1 for x in group)
   values=[x['us']for x in group];assert values==g['blocked_raw_us'][label+str(block)];bm[label+str(block)]=statistics.median(values)
  assert bm==g['blocked_medians_us']
  endpoints={};matched={}
  for key,a,index in(('anchor_before','anchor',0),('anchor_after','anchor',4),('default_before','default_shim',1),('default_after','default_shim',3)):
   endpoints[key]=bm[a+str(index)]-bm['contrast2'];values=[x-y for x,y in zip(g['blocked_raw_us'][a+str(index)],g['blocked_raw_us']['contrast2'])];matched[key]={'median_same_fixture_delta_us':statistics.median(values),'win_tie_loss':signs(values),'per_seed_us':dict(zip(map(str,SEEDS),values))}
  assert endpoints==g['blocked_control_minus_4block_us']
  trips=g['interleaved_triplets'];assert len(trips)==102
  for oi,order in enumerate(ORDERS):
   for si,seed in enumerate(SEEDS):
    group=interleaved[(oi*17+si)*3:(oi*17+si+1)*3];assert[x['arm']for x in group]==[LABELS[j]for j in order]and all(x['seed']==seed and x['order_index']==oi for x in group)and[x['position']for x in group]==[0,1,2]
    assert trips[oi*17+si]=={'seed':seed,'order_index':oi,'times':{x['arm']:x['us']for x in group}}
  contrasts={}
  for a,b in(('anchor','default_shim'),('anchor','contrast'),('default_shim','contrast')):
   deltas=[x['times'][a]-x['times'][b]for x in trips];per_seed={str(seed):statistics.median(x['times'][a]-x['times'][b]for x in trips if x['seed']==seed)for seed in SEEDS};c={'all102_triplets_median_us':statistics.median(deltas),'median_ratio':statistics.median(x['times'][a]/x['times'][b]for x in trips),'win_tie_loss':signs(deltas),'per_seed_balanced_median_us':per_seed,'median_of17_seed_medians_us':statistics.median(per_seed.values())};assert c==g['interleaved_contrasts'][a+'_minus_'+b];contrasts[a+'_minus_'+b]=dict(c,unique_seed_win_tie_loss=signs(per_seed.values()))
  outputs[dist]={'blocked_arm_medians_us':bm,'blocked_difference_of_arm_medians_us':endpoints,'blocked_estimator_matched_same_seed_differences':matched,'interleaved':contrasts,'primary_point_disposition':'VALID_GAIN'if contrasts['anchor_minus_contrast']['all102_triplets_median_us']>0 else'VALID_REGRESSION'}
  for seed in SEEDS:
   h=meta[(dist,seed)]['routing'];local=[v for k,v in h.items()if k!='-1'];per_fixture.append({'dist':dist,'seed':seed,'local_routes':sum(local),'distinct_local_experts':sum(x>0 for x in local),'max_local_routes':max(local),'interleaved_anchor_minus4_median_us':contrasts['anchor_minus_contrast']['per_seed_balanced_median_us'][str(seed)],'interleaved_default_minus4_median_us':contrasts['default_shim_minus_contrast']['per_seed_balanced_median_us'][str(seed)]})
 out={'job_id':JOB,'state':'CPU_COLLECTED_SHARED_BANK_VALIDATED','disposition':'VALID_GAIN','qualification':'NormalM8pairedpointgain; skewedregression explicitly retained. No benchmark-distribution/seed selector or fullsuite claim.','distributions':outputs,'per_fixture_route_signs_descriptive':per_fixture,'native_reuse':r['native_reuse'],'four_block_feasibility':r['four_block_feasibility'],'capture_receipts':r['capture_receipts'],'runtime':r['runtime'],'wall_seconds':r['elapsed_seconds'],'native_load_wall_seconds':r['native_load_wall_seconds'],'variant_compile_wall_seconds':r['compile_wall_seconds'],'validation':{'artifact_hashes':len(I['files']),'runtime_source_pins':len(I['dispatch']['source_pins']),'unique_stored_fixtures':34,'unique_fixtures_per_dist':17,'ordered_triplets_per_dist':102,'clean_timing_calls':782,'oracle_or_zero_checks':60,'own_timed_terminal_oracles':46,'timing_sample_output_scope':'Only exact retained terminals are numerically checked; not all782outputs.','fixture_identity_scope':'Same stored fixture transport/immutable checks executed by frozen source; per-row route receipt consistency verified from artifacts. Actual GPUfixture bytes not separately archived.','cleanup':{k:v for k,v in cleanup.items()if k!='host_log_tail'}},'integration_direction':'New M8EP8shape/device-legality-only4block gate/up candidate with original3fallback/allotherpaths original; retain skewed losses. No dist/seed/output-based rule. Original14+holdout validation before fullsuiteclaim; no official8H200model claim.','limits':['Samefixture interleaved arm contrasts are matched and eachfixture sees all6orders;102triplets are17unique workloads, not102independentroutes.','Blocked F1 difference-of-medians and interleaved median-of-differences are different nonlinear estimators; all4same-seed blocked endpoints are included for matched-estimator comparison.','Blocked precedes interleaved and normal precedes skewed; time/order can affect cross-schedule/distribution comparisons, so no pure schedulecausality.','VariantSO is reused without rebuild; baseline retains ordinary inherited loader/cache and may lazilycompile.','No profiler/full14/holdout/fullmodel/official result is claimed.'],'evidence':[pin(C/'inventory.json'),pin(P/'result.json'),pin(P/'raw.jsonl'),pin(P/'two_tiles.so'),pin(R/'proj_rsi/scheduling/reviews/moe-original-marlin-sharedbank-01.md')],'new_GPU_API_queue_git_calls':0}
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'disposition':out['disposition'],'qualification':out['qualification'],'signs':{d:{'paired':v['interleaved']['anchor_minus_contrast']['all102_triplets_median_us'],'blocked_matched':{k:x['median_same_fixture_delta_us']for k,x in v['blocked_estimator_matched_same_seed_differences'].items()}}for d,v in outputs.items()}},indent=2))
if __name__=='__main__':main()

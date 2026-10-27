"""CPU-only exact-receipt analyzer for one M3012 T64 whole-serving experiment.
Never mutates a scheduler job, sources, collected artifacts, or promotion policy.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
JID='research.moe.ss.lut.t64.m3012.whole.01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize_valid(r):
 assert r['state']=='VALID_M3012_T64_WHOLE_SERVING_TWO_ARM_ROOT_REVIEW_REQUIRED'
 assert r['validated_direct_timings_completed']and not r['cleanup']and not r['full14_or_threefold_or_official_claim']
 assert len(r['jit'])==1 and len(r['raw'])==25
 checks=r['correctness'];assert len(checks)==39,len(checks)
 for c in checks:
  for key in ['pass_','bitwise_equal']:
   if key in c:assert c[key]is True,('invalid owncheck',c)
 for label in ['anchor','LUT_T64']:
  for kind in ['eager_original_task','captured_current_X_routing','captured_fresh_X_routing','valid_to_empty_captured_ids','own_paired_terminal_seed7907']:
   assert any(c.get('arm')==label and c.get('kind')==kind for c in checks),(label,kind)
  for kind,n in [('timed_current_output_F1_fixed_fixture',6 if label=='anchor'else 3),('timed_current_output_paired',8)]:
   assert sum(c.get('arm')==label and c.get('kind')==kind for c in checks)==n
 for block,label in enumerate(['anchor','LUT_T64','anchor']):
  assert any(c.get('arm')==label and c.get('kind')=='own_F1_block'+str(block)+'_seed7904'for c in checks)
 assert any(c.get('kind')=='canonical_original_SS_T64_codec_control'and c.get('bitwise_equal')for c in checks)
 resources=r['jit'][0]['resources'];functions=r['jit'][0]['functions'];assert{(x['T'],x['DIV'])for x in resources}=={(64,1),(64,8)}
 for x in resources:
  assert x['dynamic_shared']==42016 and x['local']==x['binary_resources']['STACK']==x['binary_resources']['LOCAL']==0 and x['occupancy']>=1
  assert x['registers']==x['binary_resources']['REG']and x['max_dynamic']>=42016 and'preferred_shared_carveout'in x
 for x in functions:
  assert x['T']==64 and x['four_HGMMA_per_group']and x['accumulator_lifetime_pass']and x['static_K64_groups']==x['math_reader_barriers']>0
 assert r['jit'][0]['gate_before_first_candidate_math']
 fixed=r['F1_fixed_fixture'];assert fixed['seed']==7904 and len(fixed['tensor_sha256'])==3
 hist=fixed['routing'];assert hist==r['routing']['7904']
 rows=[]
 for seed in range(7900,7908):
  raw=[x for x in r['raw']if x['kind']=='paired'and x['seed']==seed];assert len(raw)==2
  assert[x['arm']for x in raw]==(['anchor','LUT_T64']if seed%2==0 else['LUT_T64','anchor'])
  assert raw[0]['routing']==raw[1]['routing']==r['routing'][str(seed)]
  a=next(x['us']for x in raw if x['arm']=='anchor');b=next(x['us']for x in raw if x['arm']=='LUT_T64');assert 0<a<float('inf')and 0<b<float('inf')
  h=raw[0]['routing'];assert len(h['global_counts'])==256 and len(h['local_counts'])==32 and h['local_counts']==h['global_counts'][:32]
  assert sum(h['global_counts'])==3012*8 and sum(h['local_counts'])==h['local_routes']
  assert h['mapped_local_average']==h['local_routes']/32 and h['heuristic_unmapped_average']==753.0
  assert h['active_local_blocks']==sum((n+63)//64 for n in h['local_counts'])
  assert h['expected_post']==64*(h['active_local_blocks']+(h['nonlocal_routes']+63)//64)
  assert h['block_m']==h['T']==64 and h['source_alignment_capacity']==40288
  assert h['local_token_column_utilization']==h['local_routes']/(64*h['active_local_blocks'])
  rows.append({'seed':seed,'order':[x['arm']for x in raw],'anchor_us':a,'LUT_T64_us':b,'signed_anchor_minus_LUT_us':a-b,'anchor_over_LUT':a/b,'routing':h})
 f1={}
 for block,label in enumerate(['anchor','LUT_T64','anchor']):
  raw=[x for x in r['raw']if x['kind']=='F1_fixed_fixture'and x['block']==block]
  assert len(raw)==3 and all(x['arm']==label and x['seed']==7904 and x['routing']==hist for x in raw)
  f1[label+str(block)]=[x['us']for x in raw]
  assert f1[label+str(block)]==r['F1_raw_us'][label+str(block)]
 med={k:statistics.median(v)for k,v in f1.items()};assert med==r['F1_medians_us']
 a0=med['anchor0'];a2=med['anchor2'];b=med['LUT_T641'];save=statistics.median(x['signed_anchor_minus_LUT_us']for x in rows);ratio=statistics.median(x['anchor_over_LUT']for x in rows)
 assert save==r['paired']['median_anchor_minus_LUT_us']and ratio==r['paired']['median_anchor_over_LUT']
 assert sum(x['signed_anchor_minus_LUT_us']>0 for x in rows)==r['paired']['LUT_wins']
 memory=r['memory_after_capture'];assert r['runtime']['initial_free_bytes']>=32*(1<<30)and memory['free']>=8*(1<<30)and memory['allocated']<48*(1<<30)
 if save>0:disposition='VALID_GAIN'
 elif save<0:disposition='VALID_REGRESSION'
 else:disposition='VALID_NO_GAIN'
 return {'scope':'Only actualM3012normalEP8/busiest whole-serving versus exact05ec9a8anchor, weights5252. Nofull14/M8/16384/holdout/official/model/3xclaim. No inference that failedM8versions are fixed.','recommended_disposition':disposition,'correctness_records':len(checks),'all_recorded_checks_pass':True,'current_timed_outputs_checked':25,'own_F1_and_paired_terminals_checked':True,'raw_calls':25,'F1_calls':9,'paired_calls':16,'F1_fixed_fixture':fixed,'F1_raw_us':f1,'F1_medians_us':med,'F1_endpoint_average_anchor_us':(a0+a2)/2,'F1_endpoint_average_saving_us':(a0+a2)/2-b,'F1_endpoint_average_anchor_over_LUT':((a0+a2)/2)/b,'F1_before_saving_us':a0-b,'F1_after_saving_us':a2-b,'F1_conservative_saving_us':min(a0,a2)-b,'pairs':rows,'paired_median_saving_us':save,'paired_median_anchor_over_LUT':ratio,'paired_LUT_wins':sum(x['signed_anchor_minus_LUT_us']>0 for x in rows),'arm_median_us':{'anchor':statistics.median(x['anchor_us']for x in rows),'LUT_T64':statistics.median(x['LUT_T64_us']for x in rows)},'paired_order_4_AB_4_BA':True,'actual_resources':resources,'actual_function_gate':functions,'runtime':r['runtime'],'memory_after_capture':memory,'wall_seconds':r['elapsed_seconds'],'compile_seconds':r['jit'][0]['compile_seconds'],'routing_interpretation':'753 is only the existingunmappedblockMheuristic; actualmappedaverages/counts/utilization aboveareobservedseparately. Residentwave arithmetic isnotobservedactualscheduling. Preferredcarveoutisnotactivehardwareconfig.','stack_claim':'NONE; no arithmetic additionofearlierM8 codec/grid/stagemeasurements; onlythisactualT64wholepackagecomparison.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path);a=p.parse_args()
 job=json.loads((R/'results/proj-rsi/scheduler/checkpoint.json').read_text())['jobs'][JID]
 receipts=[x['transition']['evidence']for x in job.get('observations',[])if x.get('transition',{}).get('to')=='VALIDATING']
 if not receipts:raise SystemExit('WAITING_FOR_EXACT_COLLECTION: no VALIDATING receipt; no output written')
 receipt=receipts[-1];invpath=R/receipt['inventory']
 if a.inventory:assert a.inventory.resolve()==invpath.resolve(),'must use checkpoint-selected exact collection'
 assert sha(invpath)==receipt['sha256'];inv=json.loads(invpath.read_text());collection=invpath.parent
 assert not inv['missing']and inv['terminal_exit']==0
 for rel,h in inv['files'].items():assert sha(collection/rel)==h,rel
 for rel,h in inv['dispatch']['source_pins'].items():assert sha(R/rel)==h,rel
 clean=receipt['cleanup_observation'];assert receipt['cleanup_verified']and not clean['cleanup_failed']and clean['terminal_exit']==0 and not clean['compute_pids']and not clean['guard_pids']and not clean['container_exists']and clean['gpu_memory_mib']==0
 result=collection/'logs/scheduler/moe-ss-lut-t64-m3012-whole-01/result.json';r=json.loads(result.read_text())
 # Invalid attempts remain raw evidence; root diagnoses concrete phase rather than family rejection.
 if r['state']!='VALID_M3012_T64_WHOLE_SERVING_TWO_ARM_ROOT_REVIEW_REQUIRED':
  invalid={'state':'ACTUAL_INVALID_ATTEMPT_RETAINED_NO_PERFORMANCE_ADMISSION','result':str(result.relative_to(R)),'result_sha256':sha(result),'phase':r.get('phase'),'error':r.get('error'),'correctness':r.get('correctness'),'raw_signed_timings_unadmitted':r.get('raw'),'next':'Identifyandrepairconcretefailedgate innewversion; preservefailedattempt, nofamilyrejection.'};(D/'invalid-evidence.json').write_text(json.dumps(invalid,indent=2)+'\n');print(json.dumps(invalid,indent=2));return
 out=summarize_valid(r)
 source=R/'serving/experiments/moe_rsi_20261027/ss_lut_t64_m3012_v1/extension_source.cu';assert r['jit'][0]['production_sha256']==sha(source)
 assert(result.parent/'actual_T64.cu').read_text().startswith(source.read_text())and r['jit'][0]['SO_sha256']==sha(result.parent/'actual_T64.so')
 out.update(state='VALID_M3012_T64_WHOLE_SERVING_SCOPED_ANALYSIS',result=str(result.relative_to(R)),result_sha256=sha(result),inventory=str(invpath.relative_to(R)),inventory_sha256=sha(invpath),terminal_exit=0,cleanup_verified=True)
 if out['paired_median_saving_us']>0:
  next_decision='RetainmeasuredpositivewithinM3012scope, includinganyF1disagreement. Independentreviewbeforeintegration; anyruntimecaseguardselectionmustpreserveothercases. Noautomaticfull14/officialsubmission orfamilywideclaim; rootranksnextgeneralization/composedvalidation.'
 else:
  next_decision='Retainvalidsignedlossesandanyindividualpositivepairs/F1. M8andthisM3012versiondo notestablishallshapeimpossibility; donotlaunchbroadrewriteorunchangedfull14. Useactualrouting/resources/results todeterminewhetheroneconcretecausalfollowupwarrantsROI; otherincumbentworkremainseligible.'
 out['next_decision']=next_decision;(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 evidence={'job_id':JID,'recommended_state':'DONE','recommended_disposition':out['recommended_disposition'],'observed_state':r['state'],'scope':out['scope'],'paired':{k:out[k]for k in ['paired_median_saving_us','paired_median_anchor_over_LUT','paired_LUT_wins']},'F1':{k:out[k]for k in ['F1_endpoint_average_saving_us','F1_before_saving_us','F1_after_saving_us','F1_conservative_saving_us']},'positive_pairs_preserved':[x for x in out['pairs']if x['signed_anchor_minus_LUT_us']>0],'correctness_records':39,'all_current_timed_outputs_and_own_terminals_pass':True,'source_binary_resources_gates_pass':True,'raw_calls':25,'terminal_exit':0,'cleanup_verified':True,'cleanup_observation':clean,'missing_artifacts':[],'evidence_files':{str(result.relative_to(R)):sha(result),str(invpath.relative_to(R)):sha(invpath),str((D/'analysis.json').relative_to(R)):sha(D/'analysis.json')},'root_action':'Rootindependentlyvalidatesscope/cleanupthentransitionsDONEwithsigneddisposition; noincumbent/official/quota assumption.','next_decision':next_decision,'loop_contract':'profile->measure->hypothesis->optimize/reimplement->test->validate->stack->submit->learn->reprioritize->repeat; idle<=10s; proj_rsi/ITERATION.md'}
 (D/'done-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'disposition':out['recommended_disposition'],'paired':evidence['paired'],'F1':evidence['F1'],'done_evidence_sha256':sha(D/'done-evidence.json')},indent=2))
if __name__=='__main__':main()

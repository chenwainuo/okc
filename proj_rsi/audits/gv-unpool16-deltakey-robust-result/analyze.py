"""CPU-only collected exact-SO three-weight robustness recomputation."""
from pathlib import Path
import hashlib,json,math,statistics
D=Path(__file__).resolve().parent;ROOT=D.parents[2];JOB='research.gv.unpool16.deltakey.robust.01';WEIGHTS=(6203,6257,6287);SEEDS=tuple(range(92100,92108));LABELS=('control','deltakey');SO='d1c06fd80d1f5391986c5257765851e95eb749ded49129601a5183b0871f3c49';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def summary(d):return {'n':len(d),'median_us':statistics.median(d),'mean_us':statistics.mean(d),'win_tie_loss':[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)],'range_us':[min(d),max(d)]}
def main():
 B=ROOT/'results/proj-rsi/scheduler/artifacts'/JOB
 if not B.exists():print('NOT_COLLECTED');return
 C=sorted(B.glob('collection-*'))[-1];I=json.loads((C/'inventory.json').read_text());P=C/'logs/scheduler/gv-unpool16-robust-01/run';r=json.loads((P/'result.json').read_text())if(P/'result.json').exists()else{}
 for rel,h in I['files'].items():assert sha(C/rel)==h,rel
 out={'job_id':JOB,'source_state':r.get('state','NO_RESULT_JSON'),'terminal_exit':I['terminal_exit'],'evidence':[pin(C/'inventory.json')],'artifact_hashes_verified':len(I['files']),'new_GPU_API_queue_git_calls':0}
 for p in(P/'result.json',P/'raw.jsonl',C/'logs/scheduler'/f'{JOB}.host.log'):
  if p.exists():out['evidence'].append(pin(p))
 if r.get('state')!='THREE_FRESH_WEIGHTS_SAME_BANK_COMPLETE_ROOT_REVIEW_REQUIRED':
  out.update(state='SETUP_RUNTIME_OR_CLEANUP_FAILURE_PRESERVED_NO_PERFORMANCE_ADMISSION',phase=r.get('phase'),error=r.get('error'),completed_weights=r.get('fixed_cells'),resources=r.get('raw_attributes'),post_scope_cleanup=r.get('post_scope_cleanup'))
  host=C/'logs/scheduler'/f'{JOB}.host.log'
  if host.exists():out['host_tail']=host.read_text()[-12000:]
 else:
  assert I['terminal_exit']==0 and not I['missing']and not r['cleanup']and r['phase']=='complete'
  for rel,h in I['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
  assert sha(P/'actual.so')==r['build']['SO_sha256']==SO and r['build']['native_compile_calls']==0 and r['build']['exact_prior_SO_reused']and r['full_prepare_calls']==r['profiler_calls']==0
  journal=[json.loads(line)for line in(P/'raw.jsonl').read_text().splitlines()];assert len(journal)==270
  outputs={};flatten=[];all_pairs=[];strata={'normal':[],'outliers':[]};terminals=0
  assert set(r['fixed_cells'])==set(map(str,WEIGHTS))
  for wi,w in enumerate(WEIGHTS):
   s=r['fixed_cells'][str(w)];raw=s['raw'];assert len(raw)==90 and s['weight_seed']==w and s['pack']['W_seed']==w and s['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and s['cleanup']=='RESET_AFTER_DRAIN'
   assert s['fixture_hashes']==r['shared_X_hashes']and set(s['fixture_hashes'])==set(map(str,SEEDS))
   assert all(math.isfinite(q['us'])and q['us']>0 for q in raw);flatten.extend(dict(q,m=16,weight_seed=w)for q in raw)
   a,b=LABELS if wi%2==0 else LABELS[::-1];cursor=10;blocked={};matched=[]
   for tag,arms in(('F1_pre',(a,b,a)),('interleaved',()),('F1_post',(b,a,b))):
    if tag=='interleaved':
     for oi,order in enumerate((LABELS,LABELS[::-1])):
      for seed in SEEDS:
       group=raw[cursor:cursor+2];cursor+=2;assert[q['arm']for q in group]==list(order)and all(q['mode']==tag and q['seed']==seed and q['order']==oi for q in group)
    else:
     for bi,arm in enumerate(arms):
      group=raw[cursor:cursor+8];cursor+=8;assert[q['seed']for q in group]==list(SEEDS)and all(q['arm']==arm and q['mode']==tag and q['block']==bi for q in group)
      val={'arm':arm,'samples_us':[q['us']for q in group],'median_us':statistics.median(q['us']for q in group)};assert s['blocked'][tag+str(bi)]==val;blocked[tag+str(bi)]=val
   assert cursor==90
   pairs=[]
   for oi in range(2):
    for seed in SEEDS:
     times={q['arm']:q['us']for q in raw if q['mode']=='interleaved'and q['order']==oi and q['seed']==seed};pairs.append({'seed':seed,'dist':'normal'if seed<92104 else'outliers','order':oi,'times':times})
   assert pairs==s['paired'];deltas=[q['times']['control']-q['times']['deltakey']for q in pairs];assert statistics.median(deltas)==s['contrasts']['paired_control_minus_new_us'];assert statistics.median(q['times']['control']/q['times']['deltakey']for q in pairs)==s['contrasts']['paired_ratio']
   per_seed={}
   for seed in SEEDS:
    inter=[q['times']['control']-q['times']['deltakey']for q in pairs if q['seed']==seed];c=[q['us']for q in raw if q['mode'].startswith('F1')and q['arm']=='control'and q['seed']==seed];n=[q['us']for q in raw if q['mode'].startswith('F1')and q['arm']=='deltakey'and q['seed']==seed];assert len(c)==len(n)==3
    val={'dist':'normal'if seed<92104 else'outliers','interleaved_control_minus_new_us':statistics.median(inter),'blocked_control_minus_new_us':statistics.median(c)-statistics.median(n)};assert val==s['contrasts']['per_seed'][str(seed)];per_seed[str(seed)]=dict(val,interleaved_orders=inter,blocked_control_samples=c,blocked_new_samples=n)
   phases={}
   for tag,arms in(('F1_pre',(a,b,a)),('F1_post',(b,a,b))):
    x=[blocked[tag+str(i)]['median_us']for i in range(3)];orientation=1 if arms[0]=='control'else -1;endpoints={}
    for endpoint in(0,2):
     diffs=[orientation*(c-n)for c,n in zip(blocked[tag+str(endpoint)]['samples_us'],blocked[tag+'1']['samples_us'])]
     endpoints[str(endpoint)]={'all_same_fixture':summary(diffs),'normal':summary(diffs[:4]),'outliers':summary(diffs[4:])}
    phases[tag]={'arms':list(arms),'arm_medians_us':x,'control_minus_new_difference_of_medians_before_us':orientation*(x[0]-x[1]),'control_minus_new_difference_of_medians_after_us':orientation*(x[2]-x[1]),'matched_control_minus_new_endpoints':endpoints}
   conditions={}
   for dist in('normal','outliers'):
    ds=[q['times']['control']-q['times']['deltakey']for q in pairs if q['dist']==dist];bs=[v['blocked_control_minus_new_us']for v in per_seed.values()if v['dist']==dist];conditions[dist]={'interleaved':summary(ds),'blocked_fixture_matched':summary(bs)};strata[dist].extend(ds)
   order_signs={str(oi):summary([q['times']['control']-q['times']['deltakey']for q in pairs if q['order']==oi])for oi in range(2)}
   assert len(s['terminals'])==10
   for t in s['terminals']:assert t['before_any_replay_or_mutation']and t['seed']==SEEDS[-1]and t['oracle']['current_X']=='EXACT'
   assert len(s['mutations'])==10 and all(q['NaN_output_overwritten']for q in s['mutations']);assert all(s['validation'][label]['basis']=='EXACT'and len(s['validation'][label]['task'])==3 for label in LABELS)
   terminals+=10;all_pairs.extend(deltas);outputs[str(w)]={'pack':s['pack'],'interleaved':dict(summary(deltas),median_ratio=s['contrasts']['paired_ratio']),'conditions':conditions,'order_signs':order_signs,'blocked_brackets':phases,'blocked_fixture_matched':summary([v['blocked_control_minus_new_us']for v in per_seed.values()]),'per_seed':per_seed}
  assert journal==flatten and terminals==30
  cleanup=r['post_scope_cleanup'];assert cleanup['allocated']==0 and cleanup['reserved']==0 and cleanup['clear_error']is None and cleanup['blas_map_entries_workspace_bytes_after'][2]==0
  ext=I['cleanup_observation'];assert not ext['cleanup_failed']and not ext['container_exists']and not ext['compute_pids']and not ext['guard_pids']and not ext['containers']and ext['gpu_memory_mib']==0
  out.update(state='CPU_THREE_WEIGHT_ROBUSTNESS_VALIDATED',per_weight=outputs,aggregate_equal_weight_diagnostic=summary(all_pairs),aggregate_conditions_equal_weight_diagnostic={k:summary(v)for k,v in strata.items()},actual_resources=r['raw_attributes'],shared_X_hashes=r['shared_X_hashes'],validation={'launch_source_pins':len(I['dispatch']['source_pins']),'journal_rows':270,'clean_calls':240,'warmups':30,'own_timed_terminal_oracles':30,'retained_mutations':30,'basis_columns_per_arm_per_weight':16,'native_compile_calls':0,'profiler_calls':0,'post_scope_cleanup':cleanup,'external_cleanup':{k:v for k,v in ext.items()if k!='host_log_tail'}},cost=r['cost'],prior_preserved={'paired_us':-.07999967783689499,'F1_before_us':.03200024366378784,'F1_after_us':.1600012183189392},limits=['Weights/schedules/address/time are changed together; no pure causal hardware attribution.','SameXbank hashes match across3weights; twointerleavedorders/fixture are repeated same8fixtures, not16independentworkloads.','Blocked median differences and interleaved median ofdifferences are differentestimators; perfixturedata is retained.','Only ownterminaloutputs are numericallychecked, not all240timedsamples.','No newselector/fullpolicy/officialscore/additivityclaim.'])
 if out.get('state')=='CPU_THREE_WEIGHT_ROBUSTNESS_VALIDATED':
  primary=out['aggregate_equal_weight_diagnostic']['median_us'];out['disposition']='VALID_GAIN'if primary>0 else'VALID_REGRESSION'if primary<0 else'VALID_NO_GAIN'
  out['disposition_basis']='Equal-weight 48pair diagnostic point; individual+.016gain/tie/F1positives preserved. No productionfrequency or significanceclaim.'
  oldp=next((ROOT/'results/proj-rsi/scheduler/artifacts/research.gv.unpool16.deltakey.01').glob('*/logs/scheduler/*/run/result.json'));assert sha(oldp)==r['build']['prior_result_sha256'];old=json.loads(oldp.read_text());f=old['fixed_cells']['16']['raw'][10:43];oldmatched={}
  for endpoint in(0,2):
   c=f[endpoint*11:(endpoint+1)*11];n=f[11:22];assert[x['seed']for x in c]==[x['seed']for x in n];oldmatched[str(endpoint)]=summary([x['us']-y['us']for x,y in zip(c,n)])
  out['prior_estimator_supplement_newfile_only']={'input':pin(oldp),'matched_same_seed_F1_endpoints':oldmatched,'conclusion':'OldF1 remains positive after matching; nonlinear estimator difference alone does NOT explain oldpairednegative/F1positive. New6257pre/6287post show some nonlinear estimator signflips on same rawdata.'}
  out['cause_confidence']={'setup_source_SO_currentinput_cleanup':'HIGH within observed invariants: exactbinary/bytebanks/sourcepins/oracles/S1guards/drain/allocator andexternalcleanup PASS. No detected bug; not proof allpossiblebugs absent.','resource':'PROVEN actual110to108regs butbothoccupancy4/local0/shared5120; fewerregisters didnotraise residency.','performance_root_cause':'UNKNOWN. Schedule/time/order association visible, nonlinear estimators can flip some newendpoints; freshweights/address/time and cache/clock stateconfounded. No counters isolating GV16memory/issue/dependencies were collected.','no_causal_claim':'Static88instructionreduction isnotdynamicloop-time fraction; no bandwidth/stall cause is established.'}
  out['validation']['task_FP64_oracles']=78;out['validation']['exact_basis_launches']=6;out['validation']['basis_rows_across_arms_weights']=96
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('completed_weights','host_tail','per_weight','evidence','validation')},indent=2))
if __name__=='__main__':main()

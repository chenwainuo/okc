"""Read-only archive arithmetic/routes diagnosis; no causal cross-job claim."""
import csv,hashlib,json,math,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
OLD='9916d34357b745cf';NEW='ac49231d41d441f6'
def kernels(p):
 out=[]
 for line in p['report'].splitlines()[1:]:
  m=re.match(r'^(.*?)\s+([\d.]+)\s+(µs|us|ms|ns)\s+([\d.]+)%\s+(\d+)\s*$',line)
  if m:out.append({'name':re.sub(r'\s+',' ',m[1].strip()),'profile_us':float(m[2])*{'µs':1,'us':1,'ms':1000,'ns':.001}[m[3]],'share':float(m[4]),'calls':int(m[5])})
 assert out,p
 return out

def main():
 old=json.loads((R/'results/proj-rsi/jobs'/f'{OLD}.json').read_text());new=json.loads((R/'results/proj-rsi/jobs'/f'{NEW}.json').read_text());rows=[];T={};B={};base_score=old['summary']['score_us']
 for label,job in [('old',old),('new',new)]:
  assert job['summary']['passed']and job['summary']['tests_passed']==9
  T[label]={};B[label]={}
  for x in job['benchmarks']:
   assert math.isclose(x['median_ns'],x['baseline_ns']/x['paired_ratio'],abs_tol=1e-8)
   assert math.isclose(x['baseline_ns'],x['baseline_raw_ns']-x['replay_overhead']['overhead_ns'],abs_tol=1e-8)
   load=x['case']['load'];T[label][load]=T[label].get(load,0)+x['median_ns']/1000;B[label][load]=B[label].get(load,0)+x['baseline_ns']/1000
  score=4830.8*math.prod(T[label][l]/B[label][l]for l in T[label])**(1/6);assert math.isclose(score,job['summary']['score_us'],abs_tol=1e-8)
 for i,(o,n)in enumerate(zip(old['benchmarks'],new['benchmarks'])):
  assert all(o['case'][k]==n['case'][k]for k in ['load','layer','m']);load=o['case']['load'];kp,kq=kernels(old['profiles'][i]),kernels(new['profiles'][i]);on,nn=[p['name']for p in kp],[p['name']for p in kq]
  oc,nc=o['median_ns']/1000,n['median_ns']/1000;oraw,nraw=o['median_raw_ns']/1000,n['median_raw_ns']/1000;ob,nb=o['baseline_ns']/1000,n['baseline_ns']/1000;orr,nr=o['paired_ratio'],n['paired_ratio'];oh,nh=o['replay_overhead']['overhead_ns']/1000,n['replay_overhead']['overhead_ns']/1000
  baseline_term=(nb-ob)*(.5/orr+.5/nr);ratio_term=(1/nr-1/orr)*.5*(nb+ob);assert math.isclose(baseline_term+ratio_term,nc-oc,abs_tol=1e-12)
  row={'index':i,'load':load,'layer':o['case']['layer'],'m':o['case']['m'],'old_seed':o['case']['seed'],'new_seed':n['case']['seed'],'old_kernel_names':on,'new_kernel_names':nn,'same_displayed_route':on==nn,'intentional_keyed_target':i in[2,17],'old_calibrated_us':oc,'new_calibrated_us':nc,'calibrated_delta_new_minus_old_us':nc-oc,'old_raw_event_us':oraw,'new_raw_event_us':nraw,'raw_event_delta_new_minus_old_us':nraw-oraw,'old_replay_overhead_us':oh,'new_replay_overhead_us':nh,'old_raw_minus_overhead_us':oraw-oh,'new_raw_minus_overhead_us':nraw-nh,'raw_minus_overhead_delta_us':nraw-nh-(oraw-oh),'old_baseline_calibrated_us':ob,'new_baseline_calibrated_us':nb,'old_baseline_raw_us':o['baseline_raw_ns']/1000,'new_baseline_raw_us':n['baseline_raw_ns']/1000,'old_paired_baseline_over_candidate':orr,'new_paired_baseline_over_candidate':nr,'cal_delta_symmetric_algebra_baseline_term_us':baseline_term,'cal_delta_symmetric_algebra_ratio_term_us':ratio_term,'single_case_oldvector_calibrated_time_score_delta_us':base_score*((1+(nc-oc)/T['old'][load])**(1/6)-1),'old_profiler_rows':kp,'new_profiler_rows':kq}
  rows.append(row)
 loads={l:{'old_candidate_calibrated_sum_us':T['old'][l],'new_candidate_calibrated_sum_us':T['new'][l],'candidate_sum_delta_us':T['new'][l]-T['old'][l],'old_baseline_sum_us':B['old'][l],'new_baseline_sum_us':B['new'][l],'baseline_sum_delta_us':B['new'][l]-B['old'][l],'old_normalized_sum':T['old'][l]/B['old'][l],'new_normalized_sum':T['new'][l]/B['new'][l],'log_score_ratio_contribution':math.log((T['new'][l]/B['new'][l])/(T['old'][l]/B['old'][l]))/6}for l in T['old']}
 dT=math.prod(T['new'][l]/T['old'][l]for l in T['old'])**(1/6);dB=math.prod(B['old'][l]/B['new'][l]for l in T['old'])**(1/6)
 targetT=dict(T['old'])
 for r in rows:
  if r['intentional_keyed_target']:targetT[r['load']]+=r['calibrated_delta_new_minus_old_us']
 targetOnly=base_score*math.prod(targetT[l]/T['old'][l]for l in T['old'])**(1/6)
 a={'state':'OFFICIAL_CHECK_VALID_REGRESSION_EXPOSURE_CONFIRMED_CROSS_JOB_NONCAUSAL','old_job':OLD,'new_job':NEW,'old_kind':'scored','new_kind':'CHECK','old_summary':old['summary'],'new_summary':new['summary'],'score_delta_new_minus_old_us':new['summary']['score_us']-base_score,'both_keyed_targets_exposed':all('gv_keyed' in r['new_kernel_names'][0]for r in rows if r['intentional_keyed_target']),'new_keyed_route_indices':[r['index']for r in rows if any('gv_keyed' in k for k in r['new_kernel_names'])],'all30_input_seeds_differ':all(r['old_seed']!=r['new_seed']for r in rows),'displayed_routes_changed':sum(not r['same_displayed_route']for r in rows),'unaffected_source_displayed_routes_changed':[r['index']for r in rows if not r['intentional_keyed_target']and not r['same_displayed_route']],'load_sums':loads,'score_algebra_not_causal':{'candidate_sum_factor':dT,'baseline_normalizer_factor':dB,'actual_combined_factor':dT*dB,'new_candidate_times_with_old_baseline_score_us':base_score*dT,'target_two_calibrated_times_replaced_only_oldvector_score_us':targetOnly,'target_two_calibrated_time_only_oldvector_delta_us':targetOnly-base_score,'limits':'These are counterfactual accounting identities; neither term is a feature causal effect and contributions are not additive gains.'},'targets':[r for r in rows if r['intentional_keyed_target']],'largest_calibrated_losses':sorted(rows,key=lambda r:r['calibrated_delta_new_minus_old_us'],reverse=True)[:8],'all30_routes':rows,'limits':['Different jobs/fixtures/baselines/prepare histories: descriptive cross-job deltas, not old-vs-new matched pairs.','Within-job paired_ratio compares reference baseline against that submission, not new keyed against501d.','Official calibrated route_ns=baseline_ns/paired_ratio; baseline_ns=baseline_raw_ns-replay_overhead. Raw candidate minus overhead is a distinct diagnostic, not exactly calibrated median.','Rounded profiler durations, raw event medians and official calibrated values are separate.','Target exposure is proven by actual kernel names, not wholemodule/source inclusion.','No actual official SO/SASS/buildenvironment was archived, so remote machine-code equality cannot be proven from source or profiles.'],'evidence':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [R/'results/proj-rsi/jobs'/f'{OLD}.json',R/'results/proj-rsi/jobs'/f'{NEW}.json']}}
 (D/'analysis.json').write_text(json.dumps(a,indent=2)+'\n')
 with(D/'all30.csv').open('w')as f:
  keys=[k for k,v in rows[0].items()if not isinstance(v,(dict,list))];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(json.dumps({k:a[k]for k in ['state','score_delta_new_minus_old_us','displayed_routes_changed','unaffected_source_displayed_routes_changed','score_algebra_not_causal']},indent=2));print('largest',[ (r['load'],r['layer'],r['calibrated_delta_new_minus_old_us'],r['same_displayed_route'])for r in a['largest_calibrated_losses']])
if __name__=='__main__':main()

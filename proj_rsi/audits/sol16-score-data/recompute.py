#!/usr/bin/env python3
"""CPU-only exact same-job score sensitivities; no framework/device imports."""
import hashlib,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'audits/sol16-score-opportunity.json'
raw=ROOT/'candidates/rs32/profiles/submission.fb548b5dc6f04696.json'
log=raw.with_suffix('.log')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(raw.read_text()); rows=a['benchmarks']; S0=a['summary']['score_us']; pub=4830.8
loads=list(dict.fromkeys(r['case']['load'] for r in rows)); layers=['qkv_gate','o','pooler','unpooler','router']
tot={l:sum(r['median_ns']/1000 for r in rows if r['case']['load']==l) for l in loads}
base={l:sum(r['baseline_ns']/1000 for r in rows if r['case']['load']==l) for l in loads}
def score(ds):
    new={l:tot[l]-ds.get(l,0) for l in loads}
    if any(t<=0 for t in new.values()):return None
    return pub/math.exp(sum(math.log(base[l]/new[l]) for l in loads)/6)
def case_score(changes):
    ds={l:0 for l in loads}
    for (layer,m),d in changes.items():
        r=next(r for r in rows if r['case']['layer']==layer and r['case']['m']==m)
        if d>r['median_ns']/1000: raise ValueError('negative component latency')
        ds[r['case']['load']]+=d
    return {'delta_load_us':ds,'score_us':score(ds),'score_reduction_us':S0-score(ds)}
picks={}
for line in log.read_text().splitlines():
    m=re.match(r'PICK (\w+) m=(\d+) (\S+) ([0-9.]+)',line)
    if m:picks[(m[1],int(m[2]))]={'policy':m[3],'bake_us':float(m[4])}
profiles={r['index']:r['report'] for r in a['profiles']}
shape=[]
for r in rows:
    c=r['case']; l=c['load']; v=r['median_ns']/1000; key=(c['layer'],c['m'])
    shape.append({'layer':c['layer'],'m':c['m'],'load':l,'official_us':v,'same_job_sglang_us':r['baseline_ns']/1000,'load_fraction':v/tot[l],'local_score_derivative_per_us':S0/(6*tot[l]),'zero_shape_conditional_score_us':score({l:v}),'zero_shape_conditional_score_reduction_us':S0-score({l:v}),'actual_prepare_pick':picks.get(key,{'policy':'not printed; see actual timed profile'}),'actual_timed_profile':profiles[r['index']]})
scenarios={}
small=[('qkv_gate',16),('qkv_gate',32),('pooler',32),('unpooler',32),('o',32)]
six=[(l,m) for m in (128,256) for l in ('qkv_gate','pooler','unpooler')]
for name,group in [('smallm_actual_rq_five',small),('large_six',six),('native_x_only',[('unpooler',128)]),('future_m256_three',[(l,256) for l in ('qkv_gate','pooler','unpooler')])]:
    for d in (-1,0,0.5,1,2,3): scenarios[f'{name}_each_delta_{d}us']=case_score(dict.fromkeys(group,d))
# common per-component cut required; remains a conditional equality, not forecast
for name,group in [('smallm_actual_rq_five',small),('large_six',six)]:
    maxcut=min(next(r['median_ns']/1000 for r in rows if (r['case']['layer'],r['case']['m'])==k) for k in group)
    for label,target in [('3000',3000),('3x_incumbent',S0/3),('3x_published',pub/3)]:
        hi=maxcut
        if case_score(dict.fromkeys(group,hi))['score_us']>target:val=None
        else:
            lo=0
            for _ in range(100):
                mid=(lo+hi)/2
                if case_score(dict.fromkeys(group,mid))['score_us']>target:lo=mid
                else:hi=mid
            val=hi
        scenarios[f'{name}_equal_cut_to_{label}']={'required_per_shape_us':val,'null_means':'No feasible equal nonnegative-component-time cut in this restricted group, not a family bound'}
layerzero={layer:case_score({(r['case']['layer'],r['case']['m']):r['median_ns']/1000 for r in rows if r['case']['layer']==layer}) for layer in layers}
targets={}
for label,target in [('latest_user_3ms',3000),('3x_current_official_incumbent',S0/3),('3x_published_sglang_reference',pub/3)]:
    frac=(target/S0)**6
    targets[label]={'target_score_us':target,'score_reduction_us':S0-target,'relative_speedup_vs_incumbent':S0/target,'relative_speedup_vs_published':pub/target,'sum_log_load_gain_needed':6*math.log(S0/target),'all_load_equal_fractional_reduction':1-target/S0,'single_load_residual_fraction_required':frac,'single_load_required_reduction_us':{l:tot[l]*(1-frac) for l in loads}}
result={'authority':{'job':a['job_id'],'score_us':S0,'speedup':a['summary']['speedup'],'tests_passed':a['summary']['tests_passed'],'tests':a['summary']['tests'],'raw_path':str(raw.relative_to(ROOT)),'raw_sha256':sha(raw),'source_path':'candidates/rs32/submission.py','source_sha256':sha(ROOT/'candidates/rs32/submission.py'),'log_sha256':sha(log),'source_contracts':{p:sha(ROOT/p) for p in ['ITERATION.md','protocol.json','score.py','audits/group16-feasibility.md']},'exact_formula':'4830.8 / exp(sum(log(same_job_sglang_load_sum / proposed_candidate_load_sum))/6)','equivalent_delta_formula':'S0*exp(sum(log1p(-delta_load_us/official_load_sum_us))/6)','units':'component sums are single-call microseconds; multiply sums by 62 for weighted load times; ratio cancels 62','warning':'PICK bake_us and GPU profile times are not scored median_ns. No local minima substitution.'},'recomputed_score_us':score({}),'targets':targets,'loads':[{'load':l,'m':next(r['case']['m'] for r in rows if r['case']['load']==l),'ours_component_sum_us':tot[l],'sglang_component_sum_us':base[l],'ours_weighted_load_us':62*tot[l],'sglang_weighted_load_us':62*base[l],'score_derivative_per_genuine_us':S0/(6*tot[l])} for l in loads],'shapes':shape,'conditional_scenarios':scenarios,'zero_entire_layer_across_six_loads_conditional':layerzero,'status':'CPU arithmetic only. Attainable improvements UNKNOWN. No objective achieved claim.'}
result['evidence_snapshot']={p:sha(ROOT/p) for p in ['audits/sol04-smallm-increment.md','audits/sol05-qkv-coverage.md','audits/sol08-tma-async-refill.md','audits/sol08-tma-x4.md','audits/sol10-smallm-integration.md','audits/native-x-01/report.md']}
result['potential_roi_rank']=['smallm actual rq five-shape coverage; initial runtime scope qkv32 only','future qkv/pooler/unpooler128+256 broad coverage, conditional on useful seed','native-X U128 refinement, narrow ceiling and current incumbent deficit']
result['information_value_rank']=['repaired selected qkv32 cursor actual-policy lab','bounded native-X bottleneck profile then isolated async-refill/X4/store controls','QKV128 split4 full-pipeline coverage; then explicit M256 port']
result['local_native_x_evidence']={'hardware':'local H100, not official H200','paired_median_us':{'actual_pw36':23.968,'native_x':31.856,'DP':34.096,'DS':35.312},'paired_median_actual_minus_native_us':-7.952,'official_gain_admitted':False,'source':'audits/native-x-01/report.md','new_cpu_branch_gain':'UNKNOWN; GPU_UNRUN in reviewed branch snapshots','warning':'Do not substitute local times into official score or confuse gain versus failed parents with gain versus actual incumbent'}
assert abs(score({})-S0)<1e-9
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'score':score({}),'targets':targets,'layer_zero':layerzero,'equalcuts':{k:v for k,v in scenarios.items() if 'equal_cut' in k}},indent=2))

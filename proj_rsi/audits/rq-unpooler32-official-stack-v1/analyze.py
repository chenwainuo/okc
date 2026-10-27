"""CPU-only immutable CHECK/official comparisons; descriptive, not causal pairs."""
from pathlib import Path
import csv,hashlib,json,math,re
D=Path(__file__).resolve().parent;R=D.parents[2]
IDS=['71a76a2c3fcd431f','9916d34357b745cf','ac49231d41d441f6']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return (x['case']['load'],x['case']['layer'],x['case']['m'])
def labels(x):return [re.split(r'\s{2,}',l.strip())[0]for l in x['report'].splitlines()[1:-1]]
def timing(x):return {k:x[k] for k in ['median_raw_ns','median_ns','baseline_ns','baseline_raw_ns','paired_ratio','parent_ns','outside_ns','replay_overhead']}
jobs={};pins={}
for i in IDS:
 p=R/f'results/proj-rsi/jobs/{i}.json';j=json.loads(p.read_text());r=R/f'results/proj-rsi/jobs/{i}.submitted.json';receipt=json.loads(r.read_text());s=R/receipt['candidate'];assert sha(s)==receipt['source_sha256']
 assert j['summary']['passed'] and j['summary']['tests_passed']==9 and len(j['benchmarks'])==len(j['profiles'])==30
 for b in j['benchmarks']:assert b['status']=='pass' and math.isclose(b['median_ns'],b['baseline_ns']/b['paired_ratio'],rel_tol=1e-12)
 log=R/f'results/proj-rsi/jobs/{i}.log'
 jobs[i]={'summary':j['summary'],'kind':receipt['kind'],'source_sha256':receipt['source_sha256'],'b':{key(b):b for b in j['benchmarks']},'p':{key(b):b for b in j['profiles']},'target_bake':[l for l in log.read_text().splitlines() if 'unpooler m=32' in l or 'PREP done in' in l]}
 for f in [p,r,s,log]:pins[str(f.relative_to(R))]=sha(f)
comparisons={};target=('decode 32','unpooler',32)
for old in IDS[1:]:
 a,b=jobs[IDS[0]],jobs[old];rows=[];sums={}
 for k,n in a['b'].items():
  o=b['b'][k];nl,ol=labels(a['p'][k]),labels(b['p'][k]);row={'load':k[0],'layer':k[1],'m':k[2],'new_seed':n['case']['seed'],'old_seed':o['case']['seed'],'same_input_seed':n['case']['seed']==o['case']['seed'],'new_kernel_labels':nl,'old_kernel_labels':ol,'same_profile_visible_route':nl==ol,'new_timing':timing(n),'old_timing':timing(o),'raw_gain_us':(o['median_raw_ns']-n['median_raw_ns'])/1000,'calibrated_gain_us':(o['median_ns']-n['median_ns'])/1000}
  rows.append(row);ss=sums.setdefault(k[0],{'raw_gain_us':0,'calibrated_gain_us':0,'new_calibrated_sum_us':0,'old_calibrated_sum_us':0})
  ss['raw_gain_us']+=row['raw_gain_us'];ss['calibrated_gain_us']+=row['calibrated_gain_us'];ss['new_calibrated_sum_us']+=n['median_ns']/1000;ss['old_calibrated_sum_us']+=o['median_ns']/1000
 comparisons[old]={'score_gain_us':b['summary']['score_us']-a['summary']['score_us'],'changed_input_seeds':sum(not r['same_input_seed']for r in rows),'profile_visible_route_changes':sum(not r['same_profile_visible_route']for r in rows),'raw_wins_losses_ties':[sum(r['raw_gain_us']>0 for r in rows),sum(r['raw_gain_us']<0 for r in rows),sum(r['raw_gain_us']==0 for r in rows)],'calibrated_wins_losses_ties':[sum(r['calibrated_gain_us']>0 for r in rows),sum(r['calibrated_gain_us']<0 for r in rows),sum(r['calibrated_gain_us']==0 for r in rows)],'target':next(r for r in rows if (r['load'],r['layer'],r['m'])==target),'load_sums':sums,'all30':rows}
out={'state':'FROZEN_ACTUAL_CHECK_MIXED_TRANSFER_NOT_INCUMBENT_WIN','new_job':IDS[0],'jobs':{i:{k:v for k,v in j.items()if k not in ['b','p']}for i,j in jobs.items()},'comparisons':comparisons,'calibration':'Every official calibrated median equals calibrated baseline / paired_ratio. Raw event minus overhead is not the official calibrated formula. Profile snapshot durations and parent/outside timings are distinct.','cause_status':{'setup_correctness':{'status':'PROVEN','fact':'Official CHECK passes9/9; no setup/correctness failure observed.'},'actual_path':{'status':'PROVEN','fact':'New131107 unpooler32 selected in actual bake and timed profile. 9916 uses oldST6/35, ac492 oldST5/35. Budget did not hide new option.'},'implementation':{'status':'UNKNOWN','fact':'Focused native body equality and earlier own-output tests pass. No proof source/SO placement or prepare timing cannot change performance.'},'measurement_comparability':{'status':'PROVEN','fact':'Separate jobs change every input seed, baseline calibration and multiple routes. These30-row deltas are descriptive, not matched causal contrasts.'},'performance_transfer':{'status':'UNKNOWN','fact':'Newtarget slightly loses vs9916, wins vsac492; fullpackage loses vs9916 and wins vsac492. Cannot isolate RQ source contribution from GV source, route selection, seeds and calibration.'}},'next_smallest_falsifier':'If retained small gain merits further budget: one bounded unpoolerM32 same fixture/currentX, same run cold-L2 comparison of actualselected oldST5, actualincumbentST6, guardednew131107, with existing exact sources/codec/ownterminals and fixedF1+balancedpaired. Do not rerun unchanged full30 merely to explain this CHECK. Distinguishes geometry preference vs offset effect; officialtransfer still separate.','limits':['No minimum-size/significance rejection: preserve prior fixedunpooler+.048us paired/+.080usF1, newofficialraw/calibrated signs and poolernegative.','CHECK3182.348773us is25.310370us worse than actualscoredincumbent3157.038404us; not promoted as newbest.','CHECK improves43.544472us over239bCHECK3225.893246us; independentCHECKcontexts preventcausalstackclaim.','Shared-route labels may hide cuBLASLt configuration changes; no hiddenconfig identity claim.'],'files':pins,'CPU_only':True,'contract':'proj_rsi/ITERATION.md: profile->measure->hypothesis->optimize/test->validate->stack/submit->learn->reprioritize->repeat'}
def write(name,text):
 p=D/name;assert not p.exists()or p.read_text()==text;p.write_text(text)
write('analysis.json',json.dumps(out,indent=2)+'\n')
for i,c in comparisons.items():
 p=D/f'vs-{i}-all30.csv'
 import io
 f=io.StringIO();w=csv.writer(f);w.writerow(['load','layer','m','raw_gain_us','calibrated_gain_us','same_input_seed','same_profile_visible_route','old_kernel_labels','new_kernel_labels'])
 for r in c['all30']:w.writerow([r[k]for k in ['load','layer','m','raw_gain_us','calibrated_gain_us','same_input_seed','same_profile_visible_route']]+[json.dumps(r['old_kernel_labels']),json.dumps(r['new_kernel_labels'])])
 write(p.name,f.getvalue())
print(json.dumps({i:{k:v for k,v in c.items()if k not in ['all30','target']}for i,c in comparisons.items()},indent=2))

"""Tiny archived official comparison; offline only, never queries API or mutates release."""
from pathlib import Path
import hashlib,json,ast,re
D=Path(__file__).resolve().parent;R=D.parents[2];JOB='40d856e085274f51';OLD=['9916d34357b745cf','fe5c45197ca041bb'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):c=x['case'];return(c['load'],c['layer'],c['m'])
def profile_map(job):return{key(x):x['report']for x in job['profiles']}
def native_names(report):
 out=[]
 for line in(report or'').splitlines():
  m=re.match(r'^(.*?)\s+[0-9.,]+\s*(?:µs|us|ns|ms|s)\s+[0-9.]+%\s+\d+\s*$',line.strip())
  if m:out.append(m[1])
 return out
def receipts(value):
 found=[]
 def walk(x):
  if isinstance(x,str):
   for line in x.splitlines():
    for prefix in['GV_KEYED_CHOICE ','GV_EARLYLOAD_CHOICE ','GV_SELECTED ']:
     if line.startswith(prefix):
      try:record=ast.literal_eval(line[len(prefix):])
      except(ValueError,SyntaxError):record={'unparsed_line':line}
      found.append({'prefix':prefix.strip(),'record':record})
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(value);return found

def main():
 p=R/'results/proj-rsi/jobs'/f'{JOB}.json'
 if not p.exists():print('WAIT_ARCHIVED_COMPLETION');return
 cur=json.loads(p.read_text());assert cur['job_id']==JOB and cur['problem']=='muse_bf16_proj'
 source=R/'results/proj-rsi/releases/gv53-selected-earlyload-v1-17d44d1140ab/submission.py';assert sha(source)=='17d44d1140ab09cdbab1f2835841e2891fc1c7c01c1eb4d9a2833c242d99a3e6'
 cp=profile_map(cur);cb={key(x):x for x in cur['benchmarks']};targets=[k for k in cb if k[1]=='pooler'and k[2]in(1,8)]
 comparisons={}
 for old in OLD:
  op=R/'results/proj-rsi/jobs'/f'{old}.json';o=json.loads(op.read_text());ob={key(x):x for x in o['benchmarks']};pr=profile_map(o);rows=[]
  for k in cb:
   a=ob[k];b=cb[k]
   rows.append({'load':k[0],'layer':k[1],'m':k[2],'old_raw_us':a['median_raw_ns']/1000,'new_raw_us':b['median_raw_ns']/1000,'old_minus_new_raw_us':(a['median_raw_ns']-b['median_raw_ns'])/1000,'old_calibrated_us':a['median_ns']/1000,'new_calibrated_us':b['median_ns']/1000,'old_minus_new_calibrated_us':(a['median_ns']-b['median_ns'])/1000,'old_baseline_raw_us':a['baseline_raw_ns']/1000,'new_baseline_raw_us':b['baseline_raw_ns']/1000,'old_profile':pr.get(k),'new_profile':cp.get(k),'old_selected_native_names':native_names(pr.get(k)),'new_selected_native_names':native_names(cp.get(k)),'profile_route_changed':native_names(pr.get(k))!=native_names(cp.get(k)),'case_seed_equal':a['case'].get('seed')==b['case'].get('seed')})
  comparisons[old]={'old_score_us':o['summary']['score_us'],'new_score_us':cur['summary']['score_us'],'old_minus_new_score_us':o['summary']['score_us']-cur['summary']['score_us'],'targets':[x for x in rows if(x['load'],x['layer'],x['m'])in targets],'all_cases':rows,'causal_scope':'Cross-job descriptive only:baselinecalibration,choices,devicecontext/timing/weight/inputcaseidentity maydiffer; noisolatedearlyloadcausality'}
 status_path=source.parent/'official-status.json';status=json.loads(status_path.read_text());status_log=Path(status['output']['log']);assert status_log.is_file()and R in status_log.parents;status_receipts=receipts(status_log.read_text())
 rr=receipts(cur)
 # Root may separately archive full prepare stdout beside the official response.
 receipt_files=[]
 for q in sorted((R/'results/proj-rsi/jobs').glob(JOB+'*')):
  if q.suffix in('.txt','.log'):
   rr+=receipts(q.read_text(errors='replace'));receipt_files.append(str(q.relative_to(R)))
 # Actual logged receipt arithmetic and selected implementation must agree.
 assert len(rr)==2 and rr==status_receipts
 import statistics
 for x in rr:
  rec=x['record'];alt='keyed'if rec['m']==1 else'earlyload';effect=statistics.median(t['parent']-t[alt]for t in rec['pairs_us'])
  assert rec['status']=='COMPLETE_ZERO_THRESHOLD'and len(rec['pairs_us'])==4 and [t['order']for t in rec['pairs_us']]==['AB','BA','AB','BA']
  assert effect==rec['paired_median_parent_minus_'+alt+'_us']and rec['choice']==(alt if effect>0 else'parent')
 uk=next(k for k in cb if k[1]=='unpooler'and k[2]==16);ur=cp[uk];utuple=[int(x.strip())for x in re.search(r'gv_predxor<([^>]+)>',ur)[1].split(',')];assert utuple==[1,2,2,1,4,16,1,1,2,1]
 route=[]
 for k in targets:
  report=cp.get(k,'');implementation='earlyload'if'gv_earlyload<'in report else'keyed'if'gv_keyed<'in report else'predxor'if'gv_predxor<'in report else'other_or_missing'
  route.append({'load':k[0],'layer':k[1],'m':k[2],'actual_profile_implementation':implementation,'report':report,'receipt_match':[x for x in rr if isinstance(x.get('record'),dict)and x['record'].get('m')==k[2]]})
 out={'state':'ARCHIVED_OFFICIAL_RESULT_ANALYZED','job_id':JOB,'summary':cur['summary'],'source_sha256':sha(source),'benchmarks':len(cb),'profiles':len(cp),'actual_targets':route,'actual_unpooler_M16':{'tuple':utuple,'profile':ur,'eligibility':'matchesexactscopedunpoolerM16tuple;futurefreshprepareselectionstillunmeasured'},'prepare_choice_receipts':rr,'prepare_choice_receipt_status':'PRESENT'if rr else'ABSENT_FROM_CURRENT_ARCHIVE_NOT_PROOF_NO_SELECTION','searched_receipt_text_files':receipt_files,'comparisons':comparisons,'local_body_points_preserved':{'M1_keyed_minusparent_direction':'control_minuskeyed−.048000365us','M8_control_minusearlyload_us':.031999312341213226,'scope':'separatefixedH100sourcebodyvalidation,notofficialforecast'},'evidence':{str(x.relative_to(R)):sha(x)for x in[p,source,status_path,status_log,R/'results/proj-rsi/jobs'/f'{JOB}.log']+[R/'results/proj-rsi/jobs'/f'{j}.json'for j in OLD]},'promotion':'ROOT_RECORDED_REFINE_INCUMBENT9916_REMAINS_NO_ACTION_BY_ANALYZER','submission_receipt':json.loads((R/'results/proj-rsi/jobs'/f'{JOB}.submitted.json').read_text())}
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Official selected-earlyload result','',f"Job {JOB}; source17d44d11. Status{cur['summary']['status']}, passed{cur['summary']['tests_passed']}/{cur['summary']['tests']}; score{cur['summary']['score_us']:.9f}us. Root owns promotion; analyzer makes no API/queue/release mutation.",'', '## Actual selected targets','']
 for x in route:lines += [f"- {x['load']} M{x['m']}: {x['actual_profile_implementation']}; selected names: {native_names(x['report'])}."]
 lines += ['','Receipt arithmetic verified againstall4ABBApairs andofficial-status.output.log referenced archive: M1keyed selectedat+.016000122us; M8parent retainedatparent−early−.016000122us. BothCOMPLETE_ZERO_THRESHOLD. M8earlyloadwasactuallycomparedintuningbutnotselectedforofficialprofile; no officialearlyloadkernelperformanceeffect canbeassigned.','',f"Prepare-choice receipts: {out['prepare_choice_receipt_status']}. Actual native profile execution is separate evidence from tuning choice margins/status; absent stdout cannot establish why a choice won/skipped.",'', '## Cross-job descriptive comparisons','']
 for j,c in comparisons.items():
  lines += [f"Versus{j}: old{c['old_score_us']:.9f}→new{c['new_score_us']:.9f}us; old−new{c['old_minus_new_score_us']:+.9f}us.",'','|Target|Rawold−new us|Calibratedold−new us|Baseline raw old→new us|','|---|---:|---:|---:|']
  for x in c['targets']:lines += [f"|{x['load']} poolerM{x['m']}|{x['old_minus_new_raw_us']:+.9f}|{x['old_minus_new_calibrated_us']:+.9f}|{x['old_baseline_raw_us']:.6f}→{x['new_baseline_raw_us']:.6f}|"]
  lines += ['',f"Changed native routes across all30cases: {sum(x['profile_route_changed']for x in c['all_cases'])}. Native names ignore rounded profiler duration; full reports and every raw/calibrated case remain in analysis.json.",'']
 unpoolrow=next(x for x in comparisons[OLD[0]]['all_cases']if x['layer']=='unpooler'and x['m']==16)
 lines += ['## Actual unpooler M16 eligibility','',f"Currentjoband9916 selectsame gv_predxor tuple{utuple}, decode16M16,N8192/K4096. Currentraw{unpoolrow['new_raw_us']:.9f}us versusold{unpoolrow['old_raw_us']:.9f}, calibrated{unpoolrow['new_calibrated_us']:.9f} versusold{unpoolrow['old_calibrated_us']:.9f}. Rawold−new{unpoolrow['old_minus_new_raw_us']:+.9f}us whilecalibratedold−new{unpoolrow['old_minus_new_calibrated_us']:+.9f}us; baseline/contextdiffer. Exact-tupleoptionalXprefetcheligibilitymatches thisactualjob; futurepreparemayselectanothergeometry. Roundedprofile15.4versus14.2us remainsseparate.",'', 'RootrecordedREFINE:9916remainsincumbent; actualscore3175.899650724>3157.038403653andtarget<3000notmet. Preserve localbodygains andofficialchoicefail-to-expose distinction; no familydiscard.','', 'These are separate jobs, with calibration/baseline/prepare/device context differences. Rawtiming, calibratedscore, roundedprofileGPUduration and localH100fixedcells are distinct estimators. No isolatedearlyloadcausality, additivegain or three-way same-window comparison is claimed. Local mixedM1−.048/M8+.032us points remain preserved. ActualsourcecontainsoptionalM1parent/keyed andM8parent/early; no keyed+earlyload body.','', 'Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize. [Contract](../../ITERATION.md).']
 (D/'report.md').write_text('\n'.join(lines)+'\n')
 manifest={'state':'ARCHIVED_OFFICIAL_RESULT_EVIDENCE','files':{str(x.relative_to(R)):sha(x)for x in[D/'analyze.py',D/'analysis.json',D/'report.md']}}
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:out[k]for k in['summary','actual_targets','prepare_choice_receipt_status']},indent=2))
 for x in[D/'analysis.json',D/'report.md',D/'manifest.json']:print(x.name,sha(x))
if __name__=='__main__':main()

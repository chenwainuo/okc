"""Local archived official CHECK only. No API/GPU/queue/release mutation."""
import ast,collections,hashlib,json,math,pathlib,re,statistics
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];JOB='38821b954fb14978';PARENTS=['40d856e085274f51','9916d34357b745cf'];RELEASE=R/'results/proj-rsi/releases/gv-unpool16-xprefetch-choice-v1-48af8c797b40';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):c=x['case'];return c['load'],c['layer'],c['m']
def names(report):
 out=[]
 for line in (report or '').splitlines():
  m=re.match(r'^(.*?)\s+[0-9.,]+\s*(?:µs|us|ns|ms|s)\s+[0-9.]+%\s+\d+\s*$',line.strip())
  if m:out.append(m[1])
 return out
PREFIXES=['GV_KEYED_CHOICE ','GV_EARLYLOAD_CHOICE ','GV_XPREFETCH_CHOICE ']
def receipts(text):
 out=[]
 for line in text.splitlines():
  for prefix in PREFIXES:
   if line.startswith(prefix):out.append({'prefix':prefix.strip(),'record':ast.literal_eval(line[len(prefix):])})
 return out

def main():
 jp=R/f'results/proj-rsi/jobs/{JOB}.json'
 if not jp.exists():print('WAIT_ARCHIVED_TERMINAL');return
 cur=json.loads(jp.read_text());assert cur['job_id']==JOB and cur['problem']=='muse_bf16_proj'
 source=RELEASE/'submission.py';assert H(source)=='48af8c797b404f97eff2aefaf93b7a8354baf00869c0518fcc0e81298bc6ec66'
 evidence=[jp,source,R/f'results/proj-rsi/jobs/{JOB}.submitted.json'];rr=[];receipt_source=None
 status=RELEASE/'official-status.json'
 if status.exists():
  st=json.loads(status.read_text());log=pathlib.Path(st['output']['log']);assert log.is_file() and R in log.parents
  rr=receipts(log.read_text());receipt_source=str(log.relative_to(R));evidence.extend([status,log])
 else:
  logs=list((R/'results/proj-rsi/jobs').glob(JOB+'*.log'))
  if logs:rr=receipts(logs[0].read_text());receipt_source=str(logs[0].relative_to(R));evidence.append(logs[0])
 receipt_checks=[]
 for row in rr:
  rec=row['record'];alt={'GV_KEYED_CHOICE':'keyed','GV_EARLYLOAD_CHOICE':'earlyload','GV_XPREFETCH_CHOICE':'xprefetch'}[row['prefix']]
  if rec['status']=='COMPLETE_ZERO_THRESHOLD':
   assert len(rec['pairs_us'])==4 and [p['order']for p in rec['pairs_us']]==['AB','BA','AB','BA']
   delta=[p['parent']-p[alt]for p in rec['pairs_us']];effect=statistics.median(delta)
   assert effect==rec['paired_median_parent_minus_'+alt+'_us'] and rec['choice']==(alt if effect>0 else'parent')
   receipt_checks.append({'prefix':row['prefix'],'effect_parent_minus_new_us':effect,'all_pair_deltas_us':delta,'choice':rec['choice'],'arithmetic':'PASS'})
  else:receipt_checks.append({'prefix':row['prefix'],'status':rec['status'],'choice':rec['choice'],'performance_effect':'UNAVAILABLE_NO_COMPLETED_COMPARISON'})
 cb={key(x):x for x in cur['benchmarks']};cp={key(x):x['report']for x in cur['profiles']};comparisons={}
 for oldid in PARENTS:
  op=R/f'results/proj-rsi/jobs/{oldid}.json';old=json.loads(op.read_text());evidence.append(op);ob={key(x):x for x in old['benchmarks']};pp={key(x):x['report']for x in old['profiles']};rows=[]
  for k,b in cb.items():
   
   if k not in ob:continue
   a=ob[k];n1=names(pp.get(k));n2=names(cp.get(k));fields={}
   for f in ['median_raw_ns','median_ns','baseline_raw_ns','baseline_ns','baseline_parent_ns','parent_ns','event_median_ns','outside_ns']:
    if isinstance(a.get(f),(int,float)) and isinstance(b.get(f),(int,float)):fields[f.removesuffix('_ns')+'_us']={'old':a[f]/1000,'new':b[f]/1000,'old_minus_new':(a[f]-b[f])/1000}
   rows.append({'load':k[0],'layer':k[1],'m':k[2],'old_case':a['case'],'new_case':b['case'],'case_seed_equal':a['case'].get('seed')==b['case'].get('seed'),'timing_fields':fields,'old_native_names':n1,'new_native_names':n2,'profile_route_changed':n1!=n2,'old_profile_report':pp.get(k),'new_profile_report':cp.get(k),'case_weight':b['case']['weight'],'weighted_calibrated_cost_old_minus_new_us':(a['median_ns']-b['median_ns'])/1000*b['case']['weight'],'limits':'Cross-job descriptive; route/input/calibration/device and prepare context may differ. Weighted cost is a per-load component, not whole-score gain.'})
  ranked=sorted(rows,key=lambda x:abs(x['weighted_calibrated_cost_old_minus_new_us']),reverse=True)
  comparisons[oldid]={'old_summary':old['summary'],'new_summary':cur['summary'],'score_old_minus_new_us':old['summary']['score_us']-cur['summary']['score_us'] if isinstance(cur['summary'].get('score_us'),(int,float)) else None,'same_complete_case_coverage':cb.keys()==ob.keys(),'route_changes':sum(x['profile_route_changed']for x in rows),'changed_seed_count':sum(not x['case_seed_equal']for x in rows),'all_cases':rows,'dominant_weighted_calibrated_cost_changes':ranked[:10],'weighted_cost_caution':'Per-case weight*calibrated latency components are not direct additive score microseconds.'}
 targets=[]
 for k in cb:
  if (k[1]=='pooler'and k[2]in(1,8)) or (k[1]=='unpooler'and k[2]==16):
   report=cp.get(k);selected=names(report);rec=[r for r in rr if r['record']['m']==k[2]]
   impl='xprefetch'if any('gv_xprefetch<'in n for n in selected) else'earlyload'if any('gv_earlyload<'in n for n in selected) else'keyed'if any('gv_keyed<'in n for n in selected) else'predxor'if any('gv_predxor<'in n for n in selected) else'other_or_missing'
   expected=(rec[0]['record']['choice'] if rec[0]['record']['choice']!='parent' else 'predxor') if len(rec)==1 and rec[0]['record']['status']=='COMPLETE_ZERO_THRESHOLD' else None
   targets.append({'receipt_expected_implementation':expected,'receipt_profile_agreement':impl==expected if expected is not None else 'UNRESOLVED_SKIP_OR_MISSING_RECEIPT','load':k[0],'layer':k[1],'m':k[2],'selected_native_names':selected,'implementation':impl,'prepare_receipts':rec,'profile_report':report})
 out={'state':'ARCHIVED_OFFICIAL_CHECK_ANALYZED','job_id':JOB,'source_sha256':H(source),'summary':cur['summary'],'benchmarks':len(cb),'profiles':len(cp),'prepare_receipt_source':receipt_source,'prepare_receipts':rr,'receipt_arithmetic':receipt_checks,'actual_targets':targets,'comparisons':comparisons,'evidence':{str(p.relative_to(R)):H(p) for p in evidence},'prior_fixed_cell_H100':{'paired_parent_minus_xprefetch_us':.5760006606578827,'win_tie_loss':[21,0,3],'F1_endpoints_us':[-.12799911201000214,.5440004169940948],'scope':'Separate same-window fixed body result; not official forecast/causal crossjob estimate.'},'decision_owner':'ROOT; analyzer does not promote or mutate queue/release/protocol','limits':['CHECK result is not scored quota receipt or automatic incumbent promotion.','Actual selected profile is distinct from source inclusion and optional prepare comparison.','All raw/calibrated/reference and changed-route/seed facts retained; no isolated prefetch causality inferred from cross-job difference.']}
 (D/'analysis-v1.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'summary':cur['summary'],'targets':targets,'receipt_arithmetic':receipt_checks,'comparison_scores':{k:{'old_minus_new_score_us':v['score_old_minus_new_us'],'route_changes':v['route_changes'],'changed_seed_count':v['changed_seed_count']} for k,v in comparisons.items()}},indent=2))
if __name__=='__main__':main()

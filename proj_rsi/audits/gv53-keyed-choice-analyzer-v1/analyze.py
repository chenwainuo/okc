"""Local official archive reader; no service calls or frozen-output overwrite."""
import argparse,ast,hashlib,json,math,re,statistics
from pathlib import Path
EXPECTED='78f93581a011fc7c0088032542faccb7617c6a5402ef79aded44a1aa33988597'
TPL=(1,1,1,2,16,8,1,1,0,3)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kernels(p):
 out=[]
 for line in p.get('report','').splitlines():
  m=re.match(r'^(.*?)\s+([\d.]+)\s+(µs|us|ms|ns)\s+([\d.]+)%\s+(\d+)\s*$',line)
  if m:out.append({'name':re.sub(r'\s+',' ',m[1].strip()),'profile_us':float(m[2])*{'µs':1,'us':1,'ms':1000,'ns':.001}[m[3]],'share':float(m[4]),'calls':int(m[5])})
 return out
def literal(text):
 class Floats(ast.NodeTransformer):
  def visit_Name(self,node):
   if node.id in ("nan","inf"):return ast.copy_location(ast.Constant(float(node.id)),node)
   return node
 return ast.literal_eval(Floats().visit(ast.parse(text,mode="eval")))
def receipts(text):
 rows=[]
 for line_no,line in enumerate(text.splitlines(),1):
  marker='GV_KEYED_CHOICE '
  if marker not in line:continue
  r=literal(line.split(marker,1)[1]);assert isinstance(r,dict)
  assert r['m']in(1,8) and r['choice']in('parent','keyed')
  row={'log_line':line_no,'record':r,'same_prepare_effect_us':None,'evidence_class':'SKIP_NO_MATCHED_EFFECT'}
  assert all(math.isfinite(x['us'])and x['us']>0 for x in r.get('samples_us',[]) if r['status']!='INVALID_TIMING_PARENT_RETAINED')
  if r['status']=='COMPLETE_ZERO_THRESHOLD':
   pairs=r['pairs_us'];assert len(pairs)==4 and[p['order']for p in pairs]==['AB','BA','AB','BA']
   assert tuple(r['template'])==TPL
   assert all(math.isfinite(p[a])and p[a]>0 for p in pairs for a in ['parent','keyed'])
   ds=[p['parent']-p['keyed']for p in pairs];effect=statistics.median(ds)
   assert math.isclose(effect,r['paired_median_parent_minus_keyed_us'],abs_tol=1e-10)
   assert(r['choice']=='keyed')==(effect>0)
   samples=r['samples_us'];assert len(samples)==8
   for b,p in enumerate(pairs):
    ss=[x for x in samples if x['block']==b];assert [x['arm']for x in ss]==(['parent','keyed']if b%2==0 else['keyed','parent'])
    for x in ss:assert x['us']==p[x['arm']]
   row.update({'same_prepare_effect_us':effect,'evidence_class':'MATCHED_TARGET_GEOMETRY_TUNING_INPUT_NOT_OFFICIAL_CASE_PAIR','signed_block_effects_us':ds,'AB_median_effect_us':statistics.median(ds[::2]),'BA_median_effect_us':statistics.median(ds[1::2]),'parent_block_median_us':statistics.median(p['parent']for p in pairs),'keyed_block_median_us':statistics.median(p['keyed']for p in pairs)})
  else:assert r['choice']=='parent'
  rows.append(row)
 return rows
def official(job):
 rows=[];profiles={p['index']:p for p in job.get('profiles',[])}
 for b in job.get('benchmarks',[]):
  i=b['index'];c=b['case'];oh=b['replay_overhead']['overhead_ns']
  assert math.isclose(b['median_ns'],b['baseline_ns']/b['paired_ratio'],abs_tol=1e-7)
  assert math.isclose(b['baseline_ns'],b['baseline_raw_ns']-oh,abs_tol=1e-7)
  ks=kernels(profiles.get(i,{}));names=[x['name']for x in ks]
  target=c['layer']=='pooler'and c['m']in(1,8)
  selected='keyed'if any('gv_keyed<'in n for n in names)else'parent_predxor'if any('gv_predxor<'in n for n in names)else'OTHER_ROUTE_OR_UNAVAILABLE'
  rows.append({'index':i,'case':c,'target':target,'actual_profile_implementation':selected,'profile_rows':ks,'calibrated_us':b['median_ns']/1000,'raw_event_us':b['median_raw_ns']/1000,'overhead_us':oh/1000,'raw_minus_overhead_us':(b['median_raw_ns']-oh)/1000,'baseline_us':b['baseline_ns']/1000,'paired_reference_over_submission':b['paired_ratio']})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--log',type=Path,required=True);ap.add_argument('--control',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists(),'Use a new output path; frozen analyses are never overwritten'
 job=json.loads(a.job.read_text());rr=receipts(a.log.read_text());rows=official(job);matches=[]
 for r in rows:
  if not r['target']:continue
  rs=[x for x in rr if x['record']['m']==r['case']['m']]
  match={'m':r['case']['m'],'official_index':r['index'],'actual_profile_implementation':r['actual_profile_implementation'],'receipt_count':len(rs),'selection_receipt_agrees_with_profile':None}
  if len(rs)==1 and rs[0]['record']['status']=='COMPLETE_ZERO_THRESHOLD':
   expected='keyed'if rs[0]['record']['choice']=='keyed'else'parent_predxor';match['selection_receipt_agrees_with_profile']=r['actual_profile_implementation']==expected
  matches.append(match)
 evidence={str(a.job):sha(a.job),str(a.log):sha(a.log)}
 source=Path(job.get('submission',''));source_sha=sha(source)if source.is_file()else None
 report={'state':'LOCAL_ARCHIVE_ANALYSIS','job_id':job.get('job_id'),'summary':job.get('summary'),'source_sha256':source_sha,'source_inclusion_matches_expected':source_sha==EXPECTED,'choice_receipts':rr,'receipt_coverage':{'M1':sum(x['record']['m']==1 for x in rr),'M8':sum(x['record']['m']==8 for x in rr),'absent_receipt_means_unknown_not_budget_skip':True},'target_selection_profile_comparisons':matches,'official_targets':[x for x in rows if x['target']],'all30':rows,'evidence':evidence,'limits':['Same-prepare choice effects compare matched target geometry on a tuning input; they are not official scored-case paired old/new measurements.','Official paired ratios compare reference against this submission; cross-job calibrated deltas are descriptive, with different fixtures/selections/context.','Source inclusion, target eligibility, successful completed comparison, accepted marker and profiled execution are separate evidence.','Incomplete/skipped receipts give no keyed-vs-parent gain; preserve reason and any partial raw samples.','Profile durations, raw events, raw-minus-overhead and calibrated times remain separate.','Multiple preparation receipts are retained; no unverified last-record association to official profiles.']}
 if a.control:
  control=json.loads(a.control.read_text());old=official(control);assert len(rows)==len(old)
  cross=[]
  for o,n in zip(old,rows):
   assert all(o['case'][k]==n['case'][k]for k in ['load','layer','m'])
   cross.append({'index':n['index'],'case':n['case'],'target':n['target'],'same_seed':o['case']['seed']==n['case']['seed'],'old_kernels':[x['name']for x in o['profile_rows']],'new_kernels':[x['name']for x in n['profile_rows']],'calibrated_delta_new_minus_old_us':n['calibrated_us']-o['calibrated_us'],'raw_delta_new_minus_old_us':n['raw_event_us']-o['raw_event_us'],'overhead_delta_us':n['overhead_us']-o['overhead_us'],'old_paired_reference_over_submission':o['paired_reference_over_submission'],'new_paired_reference_over_submission':n['paired_reference_over_submission']})
  report['cross_job_NONCAUSAL']={'control_job':control.get('job_id'),'score_delta_new_minus_old_us':job['summary']['score_us']-control['summary']['score_us'],'all30':cross};evidence[str(a.control)]=sha(a.control)
 if source_sha:evidence[str(source)]=source_sha
 a.output.mkdir(parents=True);(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['job_id','summary','receipt_coverage','target_selection_profile_comparisons']},indent=2))
if __name__=='__main__':main()

"""Read immutable projection policy evidence; no Torch, GPU or queue imports."""
from pathlib import Path
import hashlib,json,collections
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
paths=['proj_rsi/candidates/gv_predxor_pw45_v2/submission.py','proj_rsi/audits/generic-policy-binding-proposal/proposal.md','proj_rsi/audits/generic-policy-binding-proposal/findings.json','proj_rsi/audits/selection-margin-source-audit.json','proj_rsi/audits/selection-margin-counterfactual.json','proj_rsi/audits/gv-direct-hotcell-selection-01.json','proj_rsi/audits/gv-keepchoice-policy-result/analysis.json','proj_rsi/audits/gv53-keyed-choice-result-v1/analysis.json','proj_rsi/audits/gv53-keyed-choice-result-v1/score-accounting.json','proj_rsi/audits/rq-unpooler32-official-stack-v1/analysis.json','results/proj-rsi/jobs/9916d34357b745cf.json','results/proj-rsi/jobs/9916d34357b745cf.log','results/proj-rsi/jobs/71a76a2c3fcd431f.json','results/proj-rsi/jobs/71a76a2c3fcd431f.log','results/proj-rsi/jobs/fe5c45197ca041bb.json','results/proj-rsi/jobs/fe5c45197ca041bb.log']
evidence={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in paths}
load=lambda p:json.loads((ROOT/p).read_text())
rq=load('proj_rsi/audits/rq-unpooler32-official-stack-v1/analysis.json');choice=load('proj_rsi/audits/gv53-keyed-choice-result-v1/analysis.json')
rows={}
for job,data in [('71a76a2c3fcd431f',rq['comparisons']['9916d34357b745cf']['all30']),('fe5c45197ca041bb',choice['cross_job_NONCAUSAL']['all30'])]:
 parsed=[]
 for x in data:
  if job.startswith('71'):
   case={k:x[k]for k in ['load','layer','m']};old=x['old_kernel_labels'];new=x['new_kernel_labels'];target=(x['layer']=='pooler'and x['m']in[1,8])or(x['layer']=='unpooler'and x['m']==32);changed=not x['same_profile_visible_route']
  else:
   case=x['case'];old=x['old_kernels'];new=x['new_kernels'];target=x['target'];changed=old!=new
  if changed:parsed.append({'case':case,'native_change_domain':target,'old':old,'new':new})
 rows[job]={'displayed_changed_routes':len(parsed),'outside_native_change_domains':sum(not x['native_change_domain']for x in parsed),'changed_rows':parsed}
logs={}
for job in ['9916d34357b745cf','71a76a2c3fcd431f','fe5c45197ca041bb']:
 text=(ROOT/f'results/proj-rsi/jobs/{job}.log').read_text();lines=text.splitlines()
 logs[job]={'prepare_receipts':[{'line':i+1,'text':s}for i,s in enumerate(lines)if s.startswith('PREP done')],'budget_skip_lines':[{'line':i+1,'text':s}for i,s in enumerate(lines)if 'skipped-budget'in s],'target_and_unrelated_bake_receipts':[{'line':i+1,'text':s}for i,s in enumerate(lines)if s.startswith(('gv qkv_gate m1 ','gv pooler m1 ','gv pooler m8 ','PICK unpooler m=32 ','BAKE unpooler m=32 rq:','PICK router m=128 ','PICK router m=256 '))]}
(D/'observations.json').write_text(json.dumps({'state':'ARCHIVED_DESCRIPTIVE_NOT_CAUSAL','route_comparisons':rows,'actual_log_receipts':logs,'scores':{j:load(f'results/proj-rsi/jobs/{j}.json')['summary']['score_us']for j in ['9916d34357b745cf','71a76a2c3fcd431f','fe5c45197ca041bb']},'all30_benchmark_seeds_changed':'Both comparisons30/30 per pinned prior analyses; none is a same-input experiment.','limits':['Profile label changes are lower-bound visible routes, not full binary/algorithm-plan equality.','Different route identity is observed; noise, fixture, thermals, clock, compilation and elapsed-time causal shares are UNKNOWN.','Prepare receipts within240s contradict blanket deadline-exhaustion; no new timing or performance inference.']},indent=2)+'\n')
(D/'evidence-manifest.json').write_text(json.dumps({'state':'PINNED_EXISTING_EVIDENCE','files':evidence},indent=2)+'\n')
print(json.dumps({j:{k:v[k]for k in ['displayed_changed_routes','outside_native_change_domains']}for j,v in rows.items()}))

"""Exact local upload contents and stack branches; no service/store calls."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}

def main():
 records=[]
 entries=[('Current GV best','gv-predxor-v2-501d901db8ee','9916d34357b745cf','SCORED_CURRENT_BEST'),('Keepchoice alternative','gv-keepchoice-v1-244e5b7d6bb8','c2b3b2f924e043a8','SCORED_PASSED_NOT_BEST'),('GV + RQ predxor','rq-gv-combined-v1-137e0dbce0f0','22d677fa4501409f','SCORED_PASSED_NOT_BEST'),('GV + private EH','gv-private-eh-v1-aaf40d39d38a','ed28c762f3274df1','SCORED_PASSED_NOT_BEST')]
 for label,release,jid,status in entries:
  base=ROOT/'results/proj-rsi/releases'/release;source=base/'submission.py';raw=base/'profiles'/('submission.'+jid+'.json');result=json.loads(raw.read_text());summary=result['summary'];assert summary['passed']is True and summary['tests_passed']==summary['tests']==9
  s=source.read_text();row={'immutable_id':'official-content.'+jid,'label':label,'status':status,'job_id':jid,'mode':'run','source':ref(source),'score_us':summary['score_us'],'tests_passed':9,'tests':9,'GV_predxor_present':'gv_fix_pred('in s,'RQ_predxor_present':'rq_fix_pred('in s,'private_EH_present':'ehpriv_hz_free'in s,'O32_positive_factor_present':'1.0 if (layer, m) == ("o", 32) else MARGIN'in s,'keepchoice_policy_present':'def _gvpred_confirm('in s,'raw':ref(raw)}
  receipt=ROOT/'results/proj-rsi/jobs'/(jid+'.submitted.json')
  if receipt.exists():
   r=json.loads(receipt.read_text());assert r['source_sha256']==row['source']['sha256'];row['receipt']=ref(receipt)
  records.append(row)
 for label,candidate,jobfile in [('GV + EH + O32','gv_private_eh_o32_v1','gv-private-eh-o32-policy-01.json'),('Complete compatible stack','complete_gv_rq_eh_o32_v1','complete-gv-rq-eh-o32-policy-01.json')]:
  p=ROOT/'proj_rsi/candidates'/candidate;source=p/'submission.py';s=source.read_text();descriptor=ROOT/'proj_rsi/scheduling/jobs'/jobfile;d=json.loads(descriptor.read_text());assert d['review']['decision']=='GO'
  records.append({'immutable_id':'queued-content.'+candidate,'label':label,'status':'READY_QUEUED_ROOT_REPORTED_NO_SCORE_CLAIM','job_id':None,'local_job_id':d['id'],'mode':'local_full30_planned','source':ref(source),'score_us':None,'GV_predxor_present':'gv_fix_pred('in s,'RQ_predxor_present':'rq_fix_pred('in s,'private_EH_present':'ehpriv_hz_free'in s,'O32_positive_factor_present':'1.0 if (layer, m) == ("o", 32) else MARGIN'in s,'keepchoice_policy_present':'def _gvpred_confirm('in s,'source_proof':ref(p/'source-proof.json'),'descriptor':ref(descriptor),'control':'localEH aaf40d'if candidate=='gv_private_eh_o32_v1'else'currentGV501d','status_evidence':'Root reports READY queued; descriptor independent GO and source pins verified. No scored result claimed.'})
 data={'schema_version':1,'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_best':records[0],'submissions_and_queued_sources':records,'common_base':'PW45 source835acb retains codec, original RQ, RTK, optionalEH and originalo32HALF52 contenders. CurrentGV501d adds standalone GVpredxor. O32factor column means new1.0selection factor, not HALF52 source presence.','branch_relationship':'Keepchoice244, RQ+GV137 and privateEH+GVaaf were parallel descendants of currentGV501d, not cumulative submissions. Queuedc425 combinesEHO32; queuedd1ca combinesGVRQEHO32.','prior_official_RQ':{'job_id':'7b2da0f017d54e3d','score_us':3193.818257232445,'scope':'RQ-only on PW45; distinct from GV+RQ job22d677'},'local_signed_evidence':['proj_rsi/audits/measured-candidate-reconciliation-20261027/analysis.json','proj_rsi/audits/gv-private-eh-policy-result/analysis.json','proj_rsi/audits/gv-o32-positive-policy-result/analysis.json'],'limits':['Source presence and submitted bytes do not prove actual per-shape selection or isolated benefit.','All three new scored outcomes passed9/9 and lost currentGV; local positive/mixed signs are preserved separately.','Complete package is READY queued and unmeasured; no sum or positive/scored claim.','Keepchoice is an alternative policy branch, excluded from compatible arithmetic/config stack.','No historical artifact was changed; no GPU/API/queue/git/ledger operation.']}
 (HERE/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
 text='''# Exact submission contents and queued stack

The three newer scored submissions were **parallel branches sharing current GV**, not a cumulative stack. All passed9/9; none beat GV's3157.038404us. Source presence below is distinct from actual per-shape selection and isolated benefit.

|Source / job|GV predxor|RQ predxor|Private EH|New O32 factor|Keepchoice policy|Official outcome / state|
|---|:---:|:---:|:---:|:---:|:---:|---|
'''
 for r in records:
  short=r['source']['sha256'][:12];jid=r['job_id']or r['local_job_id'];bits=['yes'if r[k]else'—'for k in ['GV_predxor_present','RQ_predxor_present','private_EH_present','O32_positive_factor_present','keepchoice_policy_present']]
  outcome=format(r['score_us'],'.6f')+'us;9/9'+(' **current best**'if r['status']=='SCORED_CURRENT_BEST'else'; not best')if r['score_us']is not None else'**READY queued; no score or win claim**'
  text+='|'+r['label']+' `'+short+'` / `'+jid+'`|'+'|'.join(bits)+'|'+outcome+'|\n'
 text+='''
```mermaid
flowchart TD
  GV[Current best GV 501d]
  GV --> K[Keepchoice 244e / scored3199.387]
  GV --> R[GV + RQ 137e / scored3234.205]
  GV --> E[GV + EH aaf4 / scored3207.568]
  E --> EO[GV + EH + O32 c425 / READY]
  EO --> ALL[GV + RQ + EH + O32 d1ca / READY]
  R -. same compatible composition .-> ALL
```

The queued complete source `d1ca973956a3a968796964679e8685741dba0d390e8f4c09f4b0794be108872b` includes all four compatible arithmetic/config changes. All six RQ/EH/O32 edit orders commute and invert; removingRQ recovers queuedc425 exactly. Keepchoice stays a separate alternative selector policy. Existing EH+O32 source/descriptor remains preserved.

Common PW45 ancestry already contains original RQ kernels, RTK, optionalEH and originalO32HALF52 contenders. The O32 column refers specifically to the new zero-minimum-gain comparison factor; it does not imply HALF52 was absent earlier. RQ-only scored3193.818257 is also preserved separately from the GV+RQ scored3234.204866 job.

Local EHfull30 points (+0.05417% paired/+0.22506%F1), O32-only mixed points (−0.01574%paired/+0.18669%F1), combined03mixed points, and every shape loss remain in linked reconciliation evidence. They are not added or converted into an official forecast. Source hashes, exact accepted receipts and raw completed results are in `analysis.json`. Complete-stack local control is currentGV; EH+O32 local control is measuredEH. Neither queued source has a scored claim.
'''
 (HERE/'report.md').write_text(text)
 print(json.dumps({'rows':len(records),'official_best':records[0]['job_id'],'new_official':[(r['job_id'],r['score_us'])for r in records[1:4]],'complete_source':records[-1]['source']['sha256']},indent=2))
if __name__=='__main__':main()

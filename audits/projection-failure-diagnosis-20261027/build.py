"""Read-only projection failure inventory; render only this audit's new artifacts."""
import ast,collections,csv,hashlib,json,re
from pathlib import Path
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
if (D/'report.json').exists():
 raise SystemExit('Published report exists; create a new version directory rather than overwrite consumed evidence.')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
h=json.loads((D/'inputs/history_catalog.json').read_text())
queue=json.loads((D/'inputs/projection-queue-snapshot-v2.json').read_text())['jobs']
local=list(csv.DictReader((D/'inputs/local_observations.csv').open()))
local=[x for x in local if 'moe'not in x['attempt_id'].lower() and 'moe'not in x.get('evidence','').lower()]
bylocal=collections.defaultdict(list)
for x in local:bylocal[x['attempt_id']].append(x)

def base(aid,channel,version,source,comparator):
 name=(aid+' '+str(version)).lower()
 family=next((label for token,label in [('splitcta','GV split-CTA/half selector'),('codec','GV lossless codec'),('privateeh','RQ private EH'),('eh.','RQ epilogue'),('rq32','RQ decode/repair'),('o32','O32 selection'),('balanced','Lt confirmation'),('cursor','Escape cursor'),('native','Native WGMMA'),('gv','GV repair/selection')]if token in name),'Projection source/selector lineage')
 return dict(attempt_id=aid,channel=channel,version=version,family=family,source=source,comparator=comparator,
  what_failed=[],proven_facts=[],cause='No specific failure mechanism established by the archived record.',
  cause_confidence='UNKNOWN',cause_scope='Failure cause, not confidence that the outcome occurred.',
  implementation_bug_not_excluded=True,bug_scope='Functional checks constrain tested numerics; selection, timing, packing/lifetime outside tested cases, integration and transfer bugs remain possible.',
  next_small_falsifier='Use the existing bounded harness to compare the exact selected parent/candidate tuple on the same current input and prepared state; inspect actual dispatch and own output before attributing a cause. No broad rerun or new proof gate.',
  reopen_ROI='LOW_PENDING_SPECIFIC_HYPOTHESIS',version_status='Retain this version and every sign.',family_status='OPEN: one losing version or invalid attempt does not reject the family.',evidence=[])

def apply(r,confidence,cause,falsifier,roi='MEDIUM',scope=None):
 r.update(cause_confidence=confidence,cause=cause,next_small_falsifier=falsifier,reopen_ROI=roi)
 if scope:r['cause_scope']=scope

setup={
 'pw38.hotpath.trace.01':('The trace parser conflated an event category/name with a selected-kernel category.','Use the repaired parser against the retained trace; trace02 already supplies the successful separate attempt.'),
 'pw38.selected.profile.v1.01':('CUDA context query returned201 before model import/prepare.','Use the separately reviewed context-initialization repair; do not rerun any kernel proof.'),
 'pw41.ticket.narrow.01':('A per-translation-unit four-EH-target resource gate was applied to a TU with different target scope, before candidate timing.','Check each TU against its actual function inventory and retain original kernels; reattempt only the bounded affected screen.'),
 'pw42.emptyescape.gpu.01':('Harness reconstruction compared canonical5D decoder output with original6D fragments, before new-kernel evaluation.','Apply the retained dimensional reconstruction repair; attempt02 already measured the specialization and lost.'),
 'pw45.privateeh.screen.01':('The load_inline wrapper read the module name as a keyword although it was supplied positionally; KeyError(name), zero target rows.','Use the separately repaired name extraction; screen02 already supplies a valid signed measurement.'),
 'research.combined.direct.policy.02':('Missing actual --review file caused an assertion before Torch/output directory and candidate timings.','Bind the existing reviewed file; attempt03 already contains the repaired direct current-GV comparison.'),
 'research.gv.codec.frontier.two-cell.01':('Internal cleanup asserted zero allocation while32MiB BLAS workspace remained; external GPU/container cleanup succeeded.','Inspect actual BLAS cache ownership and release after drain; preload successor explicitly accounts/clears it and is valid but still loses currentC12.'),
 'research.gv.optional.counters.01':('Fresh poolerM16 selected RQ, so the requested GV pair was absent; the family guard stopped before profiling.','Profile the actually selected family or use a separately labelled historical fixed control; v2 explicitly measured selected historicalM1.'),
 'research.gv.predxor.pw45.01':('An optional profiler returned no CUDA events and its assertion stopped the run before full30 timing; partial correctness/fixed-cell evidence survives.','Make optional trace failure local while keeping post-replay numerics mandatory; recover completed partial evidence, not an invented full-policy result.'),
}

def specialize(r):
 aid=r['attempt_id'];text=' '.join(r['proven_facts'])
 if aid in setup:
  cause,nxt=setup[aid];apply(r,'PROVEN',cause,nxt,'MEDIUM','Specific setup/selection-guard fault only; not a diagnosis of kernel performance.')
 if aid=='pw44.stack.fullscope.01':
  apply(r,'UNKNOWN','A wrapper resource assertion failed before correctness/timing; the exact rejected raw field is not retained. A resource gate mismatch is possible but its precise value cannot be invented.','Inspect retained actual attributes/SASS and the old wrapper assertion; compare with successful attempt02 before changing kernels.','MEDIUM')
 if aid=='pw38.selected.profile.v2.02':
  apply(r,'UNKNOWN','Nsight reported LaunchFailed at an actual clustered GV QKV node after prepare. Tool/driver/cluster interaction is a hypothesis; a candidate kernel failure is not established.','Use retained launch attributes and installed-version support; a noncluster diagnostic can isolate toolability without changing the candidate.','LOW')
 if aid=='pw38.selected.profile.v3.03':
  apply(r,'UNKNOWN','Capture failed and the required post-range own-output terminal is absent; metadata include malformed/NaN values. No counters or performance cause can be admitted.','Recover tool logs and validate the literal selected node/metadata on the minimal optional diagnostic; do not repeat a full prepare/proof campaign.','LOW')
 if aid in ('research.gv53.splitcta.fullpolicy.01','research.gv53.splitcta.m1.fullpolicy.01','research.complete.splitcta.m1.incremental.01'):
  apply(r,'PROVEN','New splitCTA tuples were not selected in the recorded actual policy. This proves missing exposure, not the cause of the whole-package loss or gain.','Key the intended half/split tuple in a separately versioned optional contender; compare it with the exact incumbent and record whether it wins/serves, within the existing bake budget.','HIGH','Selection exposure only; whole-package performance/official-transfer cause remains UNKNOWN.')
 if aid=='gv.lt.qkv256.balanced.fullpolicy.01':
  apply(r,'PROVEN','The balanced confirmation retained its legacy LT choice; downstream original RQ then served the local target. The new LT confirmer was not an observed changed serving mechanism.','Retain actual end-of-prepare family/config and chooser receipt. Use a single targeted state where confirmation actually changes the winner; do not label same-choice context variation a selector gain.','LOW','Why the intended chooser improvement was not exposed; official-transfer cause is not proved.')
 if aid=='gv.lt.unpooler256.balanced.focused.01':
  apply(r,'PROVEN','Confirmation retained the legacy choice and both policies served the same complete Lt configuration. The observed negative point is not evidence that a new Lt configuration is worse.','If revisiting, compare exact same selected Lt configuration/allocation with controlled prepare context; otherwise retain the optional helper and deprioritize.','LOW','Selection identity is proven; timing-loss cause remains UNKNOWN.')
 if aid=='gv.privateeh.screen.01':
  apply(r,'PROVEN','QKV128 did not select the positive fixed private-free arm. Printed old/private bake values tie at16.74us; exact ordering versus rounded tie is unknown. Pooler selected the private arm and gained.','One QKV128 same-packed-state comparison with unrounded bake/graph points; retain the incumbent policy and private contender separately. Pooler can be carried as a separately scoped recovery.','HIGH','QKV nonselection only; no causal claim about whole-policy or H200 loss.')
 if aid=='research.gv.o32.positive.policy.01':
  apply(r,'PROVEN','The old1% comparison rejects a same-prepare positive O32 contender while1.0 accepts it. The target wins both estimators; full30 paired loses and F1 gains.','Reuse the exact O32-only source/current-GV control if scheduling one authorized scored attempt; preserve both whole estimates and actual picks.','MEDIUM','Binding selector threshold only; mixed whole-package/transfer cause remains UNKNOWN.')
 if aid=='research.positive.selection.policy.01':
  apply(r,'PROVEN','O32 margin binds in this prepare, but the broad source also changes RQ implementation and independently selected plans; its whole mixed point is not an isolated margin gain.','Use the already measured O32-only branch rather than infer an additive gain from this broad package.','MEDIUM','Target gate behavior, not whole-package cause.')
 if aid=='research.gv.codec.frontier.preload.01':
  apply(r,'UNKNOWN','Preload improves versus slower v2 but both cells lose exact currentC12. Layout/lookup/repair/register scheduling explanations remain hypotheses without an isolated causal measurement.','Compare the same fixed actual tuple and packs with one source-local representation change; reuse successful cleanup and preserve currentC12 as control.','MEDIUM')
 if aid in ('complete.gv.rq.eh.o32.fullpolicy.01','gv.privateeh.fullpolicy.01','research.gv.keepchoice.policy.01','research.rq32.predxor.gpu.01'):
  r['proven_facts'].append('A later official outcome is a separate observation; local validity is not retroactively erased and a loss does not prove a kernel bug or hardware explanation.')
 return r

rows=[]
for aid,j in queue.items():
 v=j.get('validation',{});e=v.get('evidence',{});disp=e.get('disposition',j.get('disposition'))
 sources={p:s for p,s in j.get('files',{}).items()if '/candidates/'in p or p.endswith('submission.py')}
 control=j.get('annotations',{}).get('control','UNKNOWN: see pinned driver/manifest; not inferred from the job name')
 r=base(aid,'local_queue_'+j['resource'],aid,sources,control)
 r.update(state=j['state'],disposition=disp,queue_kind=j['kind'],source_review=j.get('review'),validation_reason=v.get('reason',''),validation_evidence=e)
 if j['state']!='DONE':r['what_failed']=['NOT_RUN_OR_HELD'];r['proven_facts']=['No completed attempt claimed by this snapshot.'];r['reopen_ROI']='FOLLOW_EXISTING_ROOT_QUEUE'
 elif str(disp).startswith('INVALID'):
  r['what_failed']=[str(disp)];r['proven_facts']=[v.get('reason','Invalid disposition; inspect retained evidence before classifying cause.')]
 elif j['resource']=='cpu':r['what_failed']=['NO_RUNTIME_FAILURE_CLAIM'];r['proven_facts']=['CPU readiness/review is not hardware correctness or a performance measurement.',v.get('reason','')];r['reopen_ROI']='SUPPORTING_ONLY'
 else:
  r['what_failed']=['LOCAL_PERFORMANCE_REGRESSION'if disp=='VALID_REGRESSION'else'VALID_SCOPED_OR_MIXED_RESULT'];r['proven_facts']=[v.get('reason','Scoped valid result; no failure cause asserted.')]
  if disp in ('VALID_NO_GAIN','CPU_PASS'):r['what_failed']=['DIAGNOSTIC_NO_GAIN_NOT_EXPERIMENT_FAILURE'];r['reopen_ROI']='SUPPORTING_ONLY'
 obs=bylocal.get(aid,[]);g=[float(x['gain_us'])for x in obs if x['gain_us']]
 if obs:r['comparator']=list(dict.fromkeys(x['control']for x in obs))
 r['signed_local_observations']={'rows':len(obs),'positive':sum(x>0 for x in g),'zero':sum(x==0 for x in g),'negative':sum(x<0 for x in g),'caution':'Rows can include fixed controls/seed-level observations; not independent shipping gains.'}
 if g and min(g)<0:r['what_failed'].append('NEGATIVE_CELLS_RETAINED')
 r['evidence']=[{'path':'inputs/projection-queue-snapshot.json','record':aid}]
 for x in obs:
  if x.get('evidence'):
   try:ref=json.loads(x['evidence'])
   except ValueError:continue
   if ref not in r['evidence']:r['evidence'].append(ref)
 rows.append(specialize(r))

# Local evidence with no queue record stays separate; never silently omit it.
for aid,obs in bylocal.items():
 if aid in queue:continue
 r=base(aid,'local_catalog_only',obs[0]['version'],'See hashed observation evidence',list(dict.fromkeys(x['control']for x in obs)))
 r['what_failed']=['LOCAL_SIGNED_OR_MIXED_RECORD_NO_QUEUE_BINDING'];r['proven_facts']=['All canonical signed rows retained; absence of a queue binding is an evidence gap, not a failed execution.']
 r['signed_local_observations']=obs;r['evidence']=[json.loads(x['evidence'])for x in obs if x.get('evidence')];rows.append(specialize(r))

official=list(h['all_official_jobs'])
new_receipt=ROOT/'results/proj-rsi/jobs/dc78c2473d204a92.submitted.json'
new_result=ROOT/'results/proj-rsi/releases/gv-o32-positive-selection-v1-e0f1f712868f/profiles/submission.dc78c2473d204a92.json'
if new_receipt.exists()and new_result.exists()and not any(x['job_id']=='dc78c2473d204a92'for x in official):
 receipt=json.loads(new_receipt.read_text());result=json.loads(new_result.read_text())
 official.append({'job_id':receipt['job'],'submitted_utc':receipt['submitted_at'],'mode':'check','status':'done','candidate':'gv_o32_positive_selection_v1','passed':result['summary']['passed'],'score_us':result['summary']['score_us'],'archive_status':'ACCEPTED_RECEIPT_AND_OWN_RESULT_SUPPLEMENT','parent':'gv_predxor_pw45_v2 /501d901d','source':{'path':receipt['candidate'],'sha256':receipt['source_sha256']},'source_identity_status':'Receipt-bound source; own completed result, not leaderboard authority','raw':{'path':str(new_result.relative_to(ROOT)),'sha256':sha(new_result)},'status_evidence_source':{'path':str(new_receipt.relative_to(ROOT)),'sha256':sha(new_receipt)}})
prior_best=None
for x in sorted(official,key=lambda x:(x.get('submitted_utc')or'',x['job_id'])):
 score=x.get('score_us');version=x.get('candidate')or'UNKNOWN_SOURCE'
 r=base(x['job_id'],'official_'+x['mode'],version,x.get('source'),x.get('parent')or'UNKNOWN: no paired comparator')
 r.update(state=x.get('status'),passed=x.get('passed'),score_us=score,archive_status=x.get('archive_status'),source_identity_status=x.get('source_identity_status'))
 r['proven_facts']=[f"Mode={x['mode']}; passed={x.get('passed')}; score={score}; archive={x.get('archive_status')}.",'Official jobs are independent; a score difference is not an isolated feature delta.']
 r['evidence']=[ref for ref in(x.get('raw'),x.get('log'),x.get('status_evidence_source'))if ref]
 if not x.get('passed'):
  r['what_failed']=['OFFICIAL_FAILED_CHECK_OR_BUILD_UNCLASSIFIED']
  apply(r,'UNKNOWN','Passed=false is recorded, but complete source/error/test metadata are absent. Do not infer a numerical or kernel cause.','Recover original submission bytes/build logs and first failing test if still available; otherwise retain UNKNOWN rather than fabricate a diagnosis.','LOW_ARCHIVE_RECOVERY')
  if x['job_id']=='5c994bf40c144571':
   apply(r,'PROVEN','Retained ptxas log rejects WGMMA instructions for .target sm_90; build target lacks the required architecture-specific support. Exact candidate source association remains unknown.','Recover/upload identity if possible, then check generated target sm_90a on CPU. Current correctly targeted kernels do not need a rerun to rediscover this old build failure.','LOW','Compiler target mismatch only; no runtime/numerical failure occurred.')
   r['what_failed']=['BUILD_TARGET_SETUP'];r['proven_facts'].append("Archived log lines59–65 reject WGMMA on sm_90 and terminate assembly.")
 elif score is not None:
  if prior_best is not None and score>prior_best:r['what_failed']=['OFFICIAL_SCORE_ABOVE_PREVIOUS_BEST'];r['proven_facts'].append(f'Previous recorded best at this chronology={prior_best}us; score loss is an outcome, not a cause.')
  else:r['what_failed']=['NO_FAILURE_HISTORICAL_BEST_OR_BASELINE'];r['reopen_ROI']='ALREADY_INCLUDED_OR_SUPERSEDED'
  prior_best=score if prior_best is None else min(prior_best,score)
 if version in ('rq32_gv_predxor_pw45_v1','gv_private_eh_screen_v1','complete_gv_rq_eh_o32_v1','complete_gv_rq_eh_o32_splitcta_m1_v1','gv_predxor_keepchoice_v1','gv_small_original_choice_v2','gv_lt_qkv256_balanced_v1','gv53_splitcta_selection_v1','gv53_splitcta_m1_selection_v1'):
  r['what_failed'].append('LOCAL_TO_OFFICIAL_TRANSFER_OR_SELECTION_MISMATCH')
  apply(r,'UNKNOWN','Valid local package/fixed positive evidence did not yield a better official result. Different selected paths, hardware/context and same-job normalization are observed or possible confounders, not a proven explanation.','Inspect the exact official selected target and compare with its local control/actual tuple. Use the existing smallest affected route/selector experiment; do not repeat a broad proof or declare the family invalid.','MEDIUM')
  if version.startswith('gv53_splitcta')or version.endswith('splitcta_m1_v1'):
   r['proven_facts'].append('Local new tuple was unselected; a fixed-cell gain is not an actual-policy gain. Keyed tuple exposure is the relevant next falsifier.');r['reopen_ROI']='HIGH'
 if x['job_id']=='dc78c2473d204a92':
  r['what_failed'].append('LOCAL_TO_OFFICIAL_WHOLE_TRANSFER')
  r['proven_facts']+=['New CHECK9/9 score3191.721918275971us loses3157.0384036526784us incumbent.','Actual official O32 kernel changes rq<32,5,1,0,0,0,34> to rq<32,5,1,0,0,0,35>; H100 local target was GV45 to RQST6.','Across these official jobs O32 raw10.496→10.400us and calibrated6.765882→6.638172us improve despite whole score loss. CURRENT_STACK generic incumbentFL35 differs from the actual rawprofileFL34; bookkeeping discrepancy, not a kernel diagnosis.']
  r['cause']='Why the whole normalized score loses despite a positive O32 target is UNKNOWN. Different selected flags and target-vs-whole signs are observed; do not call them a proven causal explanation.'
  r['next_small_falsifier']='Reconcile all30 actual kernels and same-job calibrated/reference load vectors against incumbent, then isolate the changed selected route(s) on the existing bounded runner. No immediate repeat or source rewrite required.'
  r['reopen_ROI']='MEDIUM'
 rows.append(r)

# All canonical source versions, including unlaunched versions, are inventoried.
versions=[]
for c in h['candidates']:
 if 'moe'in c['candidate'].lower():continue
 decisions=[]
 for p in c.get('historical_summary_paths',[]):
  if p.endswith(('decision.json','diagnosis.json'))and(ROOT/p).exists():
   try:decisions.append({'path':p,'sha256':sha(ROOT/p),'record':json.loads((ROOT/p).read_text())})
   except (ValueError,OSError):pass
 versions.append({'version':c['candidate'],'parent':c.get('parent'),'source':c.get('source'),'official_jobs':c.get('official_jobs',[]),'decision_records':decisions,'evidence_paths':c.get('historical_summary_paths',[]),'review_refs':c.get('review_and_manifest_refs',[]),'family_status':'OPEN unless an explicit source-specific contraindication is documented; no retirement inferred from loss.','coverage':'Source version is not itself proof of an executed attempt; unindexed/absent logs remain unknown.'})
known={v['version']for v in versions}
for source in sorted((ROOT/'proj_rsi/candidates').glob('*/submission.py')):
 if source.parent.name in known or'moe'in source.parent.name.lower():continue
 versions.append({'version':source.parent.name,'parent':'UNKNOWN_UNINDEXED_DRAFT','source':{'path':str(source.relative_to(ROOT)),'sha256':sha(source)},'official_jobs':[],'decision_records':[],'evidence_paths':[],'review_refs':[],'family_status':'OPEN; active source not a failed experiment.','coverage':'Filesystem source version beyond canonical catalog snapshot; no runtime outcome invented.'})

findings=[x for x in h['improvements']if not re.search(r'\bmoe\b|ss[_ -]lut|serving expert',json.dumps(x),re.I)]
positives=list(csv.DictReader((D/'inputs/positive_observations.csv').open()))
match=sum(ast.literal_eval(x['actual_final_pick'])[0]==x['mode']for x in positives)
report={'schema_version':1,'scope':'ONLY muse_bf16_proj, sub3000us target. Historical serving/expert research excluded.','answer':'No: every failed reproduction has NOT been explained. Some setup faults are proven; most performance/official-transfer causes remain UNKNOWN, and implementation bugs are not excluded by a losing point or passing finite tests.','confidence_semantics':{'PROVEN':'Direct retained error/selector/source evidence establishes the stated narrow cause only.','LIKELY':'Supported hypothesis with counterevidence; not proof. No automatic assignment from a loss.','UNKNOWN':'No discriminating causal evidence; do not infer a cause from outcome.'},'coverage':{'canonical_official_jobs':len(h['all_official_jobs']),'own_result_official_supplements':len(official)-len(h['all_official_jobs']),'official_jobs':len(official),'projection_queue_records':len(queue),'projection_queue_GPU_records':sum(j['resource']=='gpu'for j in queue.values()),'local_catalog_only_attempts':sum(r['channel']=='local_catalog_only'for r in rows),'source_versions':len(versions),'projection_local_observation_rows':len(local),'projection_findings':len(findings),'attempt_records':len(rows),'unknown_source_failed_official_jobs':5,'failed_official_without_archive':4,'caveat':'Complete over these frozen canonical/queue inputs, not an assertion that every private/unlogged experiment is known. CPU readiness/HELD/version records are not counted as runtime failures. Collection replicas are not invented as separate executions.'},'positive_index_correction':{'rows':len(positives),'matched_logged_bake_winner_label':match,'did_not_match_logged_bake_winner_label':len(positives)-match,'not_actual_dispatch_counts':True,'feature_level_true':sum(x['feature_level']=='True'for x in positives)},'attempts':rows,'source_version_inventory':versions,'finding_inventory':findings,'inputs':json.loads((D/'inputs/provenance.json').read_text()),'policy':'Preserve every negative/mixed/invalid result. Reopen by a narrow falsifiable hypothesis and source version; this audit adds no numerical gate, significance bar, repeated proof or mandatory rerun. Root owns ROI/admission and all hardware/service actions.','actions':'CPU/read-only evidence plus new audit files only; no GPU/SSH/API/queue/git mutation.','loop':'profile->measure->hypothesis->optimize/reimplement->test->validate->stack->submit measuredgains->learn->reprioritize->repeat; idle<=10s','contract':'proj_rsi/ITERATION.md'}
(D/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')

def short(v,n=120):
 s=str(v).replace('|','/').replace('\n',' ')
 return s if len(s)<=n else s[:n-1]+'…'
md=['# Projection failure and transfer diagnosis',report['answer'],'','This is an evidence inventory, not a gate on useful measurements. **Family ≠ version ≠ attempt.** A valid loss retires neither its family nor its possible future source repair. Passing correctness constrains observed numerical errors; it does not exclude selection, integration, lifetime outside tested cases, or measurement bugs.','','## Coverage',json.dumps(report['coverage'],indent=2),'','The1,066 positive bake rows include321 mode labels matching their logged bake winner and745 that did not. These are **not actual dispatch counts** or isolated feature gains. All per-attempt detail, source/hash/evidence, cause scope and bug caveats are in [report.json](report.json).','','## Highest-ROI causal questions','','|Question|Proven facts|Cause confidence and bug status|Small next falsifier|','|---|---|---|---|','|GV53 keyed half/split exposure|FixedM1 gain; new tuple unselected in policy; official packages lost.|Nonselection PROVEN; whole loss UNKNOWN; selector/integration bug open.|Key only the intended tuple as an optional contender, measure against exact incumbent and record actual selection.|','|RQ32 selected-loop rewrite transfer|Local full30 positive; GV+RQ official3234.205us and complete3187.142us lose3157.038 incumbent.|Transfer cause UNKNOWN; finite correctness does not exclude performance or integration bugs.|Use actual selected current tuple and same prepared packs; change one loop/repair mechanism, retain own terminals.|','|GV16 repair/decode|Current incumbent source has selected predxor routes; other codecs can improve slower controls while losing currentC12.|Codec-frontier current losses proven; reason UNKNOWN; decode/packing/placement hypotheses open.|Exact current tuple/control, one source-local decode/repair change; no full source sweep.|','|Private EH QKV128/pooler128|Pooler selected positive; QKV positive fixed arm unselected or mixed. Fullaaf4 official3207.568us loses.|Selection exposure known; exact tie and transfer cause UNKNOWN; bugs open.|One-cell same-packed-state selection/context comparison; preserve original policy versus fixed arms.|','|O32-only threshold|Binding1% gate proven in local prepare; target+.128/.080us; whole paired loss/F1 gain.|Gate cause PROVEN, whole/transfer UNKNOWN.|Existing exact single-edit source and actual selected target; no broad margin/RQ bundle.|','|Old harness failures|Wrong shape/name/review/TU/context/cleanup guards stopped specific attempts.|Several setup causes PROVEN; no kernel-performance claim from them.|Reuse already repaired successor evidence; do not rediscover full suites.|','','## Every recorded attempt','','|Attempt / channel|Version / source|Comparator|Outcome / failure class|Cause confidence / facts|Bug excluded?|Small falsifier / reopen ROI|','|---|---|---|---|---|---|---|']
for r in sorted(rows,key=lambda x:(0 if x['reopen_ROI']=='HIGH'else 1,x['channel'],x['attempt_id'])):
 source=r['source'];source=source.get('path','UNKNOWN')if isinstance(source,dict)and'path'in source else r['version']
 md.append('|'+ '|'.join([short(r['attempt_id'],70)+' / '+r['channel'],short(source,80),short(r['comparator'],80),short(', '.join(r['what_failed']),100),r['cause_confidence']+': '+short(r['cause'],220),'No; scoped checks only',short(r['next_small_falsifier'],200)+' / '+r['reopen_ROI']])+'|')
md+=['','## Every source version (including unrun versions)','','A source directory is not an execution. The JSON keeps parent, source hash, decision excerpts and review paths; no missing attempt or numerical result is invented.','','|Version|Parent|Official job count|Recorded decisions / coverage|','|---|---|---:|---|']
for v in versions:md.append('|'+ '|'.join([v['version'],short(v['parent'],70),str(len(v['official_jobs'])),short(v['evidence_paths'],130)])+'|')
md+=['','## Remaining gaps','','Five older failed official jobs have no source association; four have no archive. One retained partial log proves a WGMMA/sm_90 compiler-target error, but not which candidate bytes were uploaded. Most valid performance losses have no causal experiment that excludes implementation or measurement bugs. No H100→H200 explanation, calibration explanation, occupancy explanation or family retirement is inferred just from losing.','',report['policy'],'','Loop: profile → measure → hypothesis → optimize/reimplement → test → validate → stack → submit measured gains → learn → reprioritize → repeat; idle≤10s. [Contract](../../proj_rsi/ITERATION.md).']
(D/'report.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'coverage':report['coverage'],'cause_confidence':dict(collections.Counter(r['cause_confidence']for r in rows)),'report_json_sha256':sha(D/'report.json'),'report_md_sha256':sha(D/'report.md')},indent=2))

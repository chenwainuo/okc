"""Read-only admission checks with inert local store and no remote transport."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from proj_rsi.scheduling.model import validate_job,load_policy
from proj_rsi.scheduling.dispatch import GPUDispatcher
HERE=Path(__file__).resolve().parent

def main():
    job=json.loads((HERE/'job.json').read_text());errors=validate_job(job,for_admission=True);assert not errors,errors
    # No persistent Store is opened, read or written; no transport method exists.
    store=SimpleNamespace(policy=load_policy(),snapshot=lambda:{'jobs':[]})
    dispatcher=GPUDispatcher(store,ROOT,transport=object());launch,pins=dispatcher._validate(job)
    assert job['state']=='BACKLOG' and job['id']=='pw42.emptyescape.gpu.01'
    assert job['cost']=={'gpu_seconds':300,'followup_gpu_seconds':3420,'cpu_minutes':90}
    assert launch['inner_seconds']==240 and launch['outer_seconds']==300
    negative={}
    for name,mutate in [
        ('wrong_source_hash',lambda x:x['files'].__setitem__('proj_rsi/candidates/pw42/submission.py','0'*64)),
        ('same_author_review',lambda x:x['review'].__setitem__('actor',x['author'])),
        ('missing_guard',lambda x:x['files'].pop('proj_rsi/tools/h100_trial.sh')),
        ('deadline',lambda x:x['launch'].__setitem__('outer_seconds',240)),
        ('shell_launch',lambda x:x['launch'].__setitem__('argv',['bash','-lc','true'])),
        ('wrong_review_hash',lambda x:x['review'].__setitem__('sha256','0'*64)),
    ]:
        mutant=copy.deepcopy(job);mutate(mutant)
        try:dispatcher._validate(mutant)
        except ValueError as error:negative[name]=str(error)
        else:raise AssertionError('admission negative accepted '+name)
    mutant=copy.deepcopy(job);mutant['estimate']['direct']['scenario_score_us']=1
    assert validate_job(mutant,for_admission=True),'diagnostic direct credit admitted'
    negative['diagnostic_direct_gain']='REJECT_MODEL'
    from proj_rsi.scheduling._reviewed_inbox_proposal import inspect_descriptor
    inbox='proj_rsi/scheduling/jobs/pw42-emptyescape-01.json'
    assert (ROOT/inbox).read_bytes()==(HERE/'job.json').read_bytes()
    policy=json.loads((ROOT/'proj_rsi/scheduling/reviewed_inbox_policy.proposal.json').read_text())
    simulated={'jobs':{'research.pw42.readiness.01':{'id':'research.pw42.readiness.01','state':'DONE','disposition':'CPU_PASS'}},'sequence':0}
    proposal=inspect_descriptor(ROOT,inbox,simulated,load_policy(),policy)
    assert proposal['code']=='APPROVED',proposal
    simulated['jobs']['research.pw42.readiness.01']['state']='RUNNING'
    assert inspect_descriptor(ROOT,inbox,simulated,load_policy(),policy)['code']=='DEPENDENCY_NOT_DONE_VALID'
    proof=subprocess.run([sys.executable,'-m','proj_rsi.candidates.pw42.verify_frozen'],cwd=ROOT,text=True,capture_output=True,timeout=60);assert proof.returncode==0,proof.stderr
    evidence=dict(state='PASS_AUTHOR_CPU_LOCAL_ADMISSION_VALIDATION_PENDING_ROOT_STATE_TRANSITION',job_id=job['id'],descriptor_sha256=hashlib.sha256((HERE/'job.json').read_bytes()).hexdigest(),candidate_manifest_sha256=hashlib.sha256((ROOT/'proj_rsi/candidates/pw42/manifest.json').read_bytes()).hexdigest(),review=job['review'],source_pins=len(pins),model='PASS_FOR_ADMISSION',dispatch='PASS_PURE_LOCAL_INERT_STORE',monitor='APPROVED_WITH_SIMULATED_CPU_PASS_AND_BLOCKED_WITH_SIMULATED_RUNNING; NOT_ACTUAL_JOB_STATE',negative_controls=negative,readonly_candidate_proof=json.loads(proof.stdout),GPU='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE',scheduler='NO_PERSISTENT_STORE_OR_QUEUE_MUTATION',CPU_disposition='ROOT_MUST_MARK_research.pw42.readiness.01_CPU_PASS_AFTER_INDEPENDENT_GO')
    (HERE/'cpu-validation.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()

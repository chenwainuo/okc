"""Author CPU-only descriptor assembly. Never touches scheduler state or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',required=True);args=ap.parse_args()
    review=ROOT/args.review
    assert review.resolve().is_relative_to(ROOT) and review.is_file()
    files={}
    def pin(rel):
        path=ROOT/rel;assert path.resolve()==path and path.is_file();files[rel]=sha(path)
    for candidate in ('pw38','pw42'):
        base='proj_rsi/candidates/'+candidate;manifest=json.loads((ROOT/base/'manifest.json').read_text());pin(base+'/manifest.json')
        for name,digest in manifest['files'].items():
            pin(base+'/'+name);assert files[base+'/'+name]==digest
    for rel in ('proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh','muse_bf16_proj/reference.py','muse_bf16_proj/checks.py','muse_bf16_proj/baseline.py','proj_rsi/correctness.py','proj_rsi/candidates/pw42_admission/README.md','proj_rsi/candidates/pw42_admission/build_descriptor.py','proj_rsi/candidates/pw42_admission/validate.py','proj_rsi/tools/pw42_emptyescape_lab.py',args.review):pin(rel)
    for path in sorted((ROOT/'proj_rsi/audits').glob('pw42*independent*')):
        if path.is_file():pin(str(path.relative_to(ROOT)))
    job=dict(id='pw42.emptyescape.gpu.01',resource='gpu',kind='diagnostic',state='BACKLOG',author='sol12',created_at=time.time(),depends_on=[dict(id='research.pw42.readiness.01',accepted_dispositions=['CPU_PASS'])],write_paths=[],
        estimate=dict(direct=dict(scenario_score_us=0,p_gain=0,p_valid=1),information=dict(scenario_score_us=14.631009688068389,p_resolve=.7,p_followup_gain=.1,unblocks=['pw42.stack.fullscope.01'],decision='Resolve actual CUDA empty-group prevalence and guarded-vs-exact-generic behavior before deciding whether a separate actual PW42/PW38 whole-policy comparison is justified.'),basis='Subjective readiness prior inherited from research.pw42.readiness.01: conditional14.631score-us scenario,70% resolution,10% followup gain. No direct diagnostic speed credit, measured PW42 gain, or H100-to-H200 transfer. Actual CUDA empty prevalence remains unknown. Full planning cap300narrow+3420followup GPUseconds and90CPUminutes; expected180+2400 GPUseconds and65CPUminutes. Costs include transfer/JIT/prepare/validation/timing/cleanup; runtime may exceed narrow cap and time out.',confidence='low',evidence=['proj_rsi/candidates/pw42/cost-quote.json','proj_rsi/candidates/pw42/PROVENANCE.md','proj_rsi/candidates/pw42/learning-query.json','proj_rsi/candidates/pw42_admission/README.md',args.review]),
        cost=dict(gpu_seconds=300,followup_gpu_seconds=3420,cpu_minutes=90),files=files,review=dict(actor='sol07',path=args.review,sha256=sha(review),decision='GO'),
        launch=dict(host='root@107.170.6.60',remote_repo='/root/okc',container='okc-proj-pw42-emptyescape-01',inner_seconds=240,outer_seconds=300,profile=False,env=dict(MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_CUDA_ARCH_LIST='9.0a'),argv=['python3','/okc/proj_rsi/tools/pw42_emptyescape_lab.py','--json','/okc/logs/scheduler/pw42-emptyescape-01/result.json'],artifacts=['/root/okc/logs/scheduler/pw42-emptyescape-01']),
        annotations=dict(state='INDEPENDENT_GO_PROPOSED_BACKLOG_ONLY_NOT_REGISTERED_OR_LAUNCHED',scope='Actual CUDA prevalence plus fixed two-geometry QKV M1/8 mechanism discriminator over untouched PW38 prepare. Not actual PW42 whole-policy prepare or cumulative score admission.',zero_empty_outcome='VALID_METADATA_DIAGNOSTIC_EMPTY_PATH_UNUSED_NO_TIMING; valid metadata-only diagnostic, no timing or gain claim.',nonzero_scope='Four contexts (M1/8,output-role swaps); exact actual JIT/control/resource gates;8192basis FP64/task oddM checks;1132warm/F1/pairedcalls with40strictterminaloutputs before replay/mutation/profile;80currentXmutations;single-node profiles;immutable fullstorage dormant counters;drain/reset.',unsupported='Unsupported selected policy invalidates setup; never force geometry. Invalid elapsed retains raw and invalidates.',cost='Planning cap chain3720GPUseconds+90CPUminutes; expected2580GPUseconds+65CPUminutes, not measured cost. CPU common90cap includes implementation/review/analysis; no second CPU charge intended.',reservation='Root accepted provisional240inner/300outer; fullwall feasibility unmeasured, timeouts retained.',promotion='No official/H200/full-score claim or automatic promotion. Tiny signed effects retained without fixed cutoff; followup planningID is not registration.'))
    job['annotations']['source_and_lab_review']=dict(job['review'])
    job['annotations']['binary_evidence_limit']='Actual loaded SASS/source/attributes/occupancy are archived and gated; actual loaded SO hash/binary is not archived. Offline cubins retained. No binary attestation claimed.'
    (HERE/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    print(json.dumps(dict(path=str((HERE/'job.json').relative_to(ROOT)),sha256=sha(HERE/'job.json'),pins=len(files)),indent=2))
if __name__=='__main__':main()

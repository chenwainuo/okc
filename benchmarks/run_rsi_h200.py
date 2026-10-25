#!/usr/bin/env python3
"""Host launcher for reviewed RSI runs. Requires supervisor-issued live lease."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import uuid
from rsi_bundle import materialize, verify_tree

ROOT=Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--lease',type=Path,required=True);p.add_argument('--experiment',required=True)
    p.add_argument('--origin-root',type=Path,required=True,help='Original artifact namespace in the plan')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--timeout-seconds',type=int,default=2400);p.add_argument('--control',action='store_true')
    args=p.parse_args()
    def cancelled(signum,frame):
        raise InterruptedError(f'Launcher cancelled by signal {signum}')
    signal.signal(signal.SIGTERM,cancelled)
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Refusing to overwrite measurement evidence')
    plan=json.loads(args.plan.read_text());review=json.loads(args.review.read_text());lease=json.loads(args.lease.read_text())
    plan_hash=canonical(plan)
    if review.get('plan_sha256')!=plan_hash or review.get('preflight_reviewer') in (None,review.get('author')):
        raise ValueError('Independent reviewed plan required')
    if review.get('state') not in ('reviewed','running') or not all(review.get('preflight_checklist',{}).get(k) is True for k in
            ('baseline_bundle_complete','candidate_bundle_complete','baseline_isolated','hypothesis_falsifiable','suite_and_holdouts_frozen','precision_preserved')):
        raise ValueError('Incomplete preflight review')
    if lease.get('experiment')!=args.experiment or lease.get('deadline_unix',0)<=time.time():
        raise ValueError('Live matching supervisor GPU lease required')
    pairs = plan.get('control_policy',{}).get('pairs',plan['min_pairs']) if args.control else plan['min_pairs']
    if type(pairs) is not int or pairs < plan['min_pairs'] or pairs % 2:
        raise ValueError('Invalid preregistered pair count')
    if args.control and args.seed not in plan['screen_seeds']:
        raise ValueError('Controls must not consume holdout seeds')
    if args.seed not in plan['screen_seeds']+plan['holdout_seeds']:
        raise ValueError('Unregistered seed')
    def local(path):
        return ROOT/Path(path).relative_to(args.origin_root)
    for a in review['artifacts']:
        if digest(local(a['path']))!=a['sha256']:raise ValueError(f'Reviewed artifact changed: {a["path"]}')
    # The runner itself must be part of the preregistered transitive dependency set.
    required={str(Path(__file__).resolve()),str(ROOT/'benchmarks/rsi_bundle.py'),str(ROOT/'benchmarks/rsi_pair.py'),str(ROOT/'benchmarks/rsi_worker.py'),str(ROOT/'benchmarks/measurement.py'),str(ROOT/'serving/benchmark-contract.json')}
    if not required.issubset({str(local(d).resolve()) for d in plan['dependencies']}):
        raise ValueError('Runner dependency manifest incomplete')
    baseline=local(plan['baseline']);candidate=baseline if args.control else local(plan['candidate'])
    trees=ROOT/'results/rsi/immutable-trees';trees.mkdir(parents=True,exist_ok=True)
    roots=[];manifests=[]
    for role,archive in [('baseline',baseline),('candidate',candidate)]:
        root=trees/(role+'-'+digest(archive))
        manifest_path=Path(str(archive)+'.manifest.json')
        if str(manifest_path.resolve()) not in {str(local(d).resolve()) for d in plan['dependencies']}:
            raise ValueError('Bundle manifest not frozen by proposal')
        manifests.append(materialize(archive,manifest_path,root));roots.append(root)
    contract=json.loads((ROOT/'serving/benchmark-contract.json').read_text());image=contract['base']['image']
    lock=(ROOT/'.gpu-cache/rsi-device0.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def compute_pids():return subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if compute_pids():raise RuntimeError('Existing GPU compute process; refusing overlap')
        gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip().split(',')
        timeout=min(args.timeout_seconds,int(lease['deadline_unix']-time.time())-10)
        if timeout<=0:raise TimeoutError('Lease deadline exhausted')
        name='okc-rsi-'+uuid.uuid4().hex[:12]
        cache=ROOT/'.gpu-cache/rsi';cache.mkdir(parents=True,exist_ok=True)
        command=['docker','run','--rm','--init','--name',name,'--gpus','"device=0"','--shm-size=16g','--network','none',
            '--mount',f'type=bind,src={roots[0]},dst=/baseline,readonly',
            '--mount',f'type=bind,src={roots[1]},dst=/candidate,readonly',
            '--mount',f'type=bind,src={ROOT / "benchmarks"},dst=/runner,readonly',
            '--mount',f'type=bind,src={output.parent},dst=/output',
            '--mount',f'type=bind,src={cache},dst=/cache',
            '--env','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python3',image,
            '/runner/rsi_pair.py','--baseline-root','/baseline','--candidate-root','/candidate',
            '--output','/output/'+output.name,'--seed',str(args.seed),'--pairs',str(pairs),
            '--timeout-seconds',str(timeout-5),'--cache-root','/cache','--batches',*map(str,plan['suite'])]
        process_file=output.with_suffix('.process.json')
        save(process_file,{'container':name,'experiment':args.experiment,'status':'starting','deadline_unix':lease['deadline_unix']})
        proc=None
        try:
            with output.with_suffix('.log').open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(process_file,{'container':name,'client_pid':proc.pid,'experiment':args.experiment,'status':'running'})
                code=proc.wait(timeout=timeout)
                if code:raise RuntimeError(f'Container failed: {code}')
        finally:
            # Finish bounded teardown even if cancellation is repeated.
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            # Docker daemon children are not in the CLI process group. Remove the
            # named container explicitly, then reap the CLI before releasing lock.
            cleanup=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
            remaining=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if remaining.returncode==0:
                save(process_file,{'container':name,'status':'termination_unknown','reason':'container still exists'})
                raise RuntimeError('Container still exists; keep supervisor lease held')
            if 'No such object:' not in remaining.stderr and 'No such container:' not in remaining.stderr:
                save(process_file,{'container':name,'status':'termination_unknown','reason':remaining.stderr})
                raise RuntimeError('Docker absence unverified; keep supervisor lease held')
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            save(process_file,{'container':name,'experiment':args.experiment,'status':'terminal','client_returncode':None if proc is None else proc.returncode})
            signal.signal(signal.SIGTERM,cancelled)
            signal.signal(signal.SIGINT,signal.default_int_handler)
        for artifact in review['artifacts']:
            if digest(local(artifact['path']))!=artifact['sha256']:
                raise ValueError(f'Reviewed artifact changed during run: {artifact["path"]}')
        for root,manifest in zip(roots,manifests):verify_tree(root,manifest)
        report=json.loads(output.read_text())
        if report.get('status')!='complete' or report.get('process_groups_terminated') is not True:
            raise ValueError('Worker completion evidence missing')
        runtimes=[report['workers'][r]['init']['runtime'] for r in ('baseline','candidate')]
        if runtimes[0]!=runtimes[1]:raise ValueError('Worker runtime mismatch')
        for role,manifest in zip(('baseline','candidate'),manifests):
            if report['workers'][role]['archive_sha256']!=manifest['archive_sha256']:raise ValueError('Worker bundle mismatch')
        report.update(candidate_sha256=digest(candidate),baseline_sha256=digest(baseline),rsi_plan_sha256=plan_hash,
                      dependency_hashes={path:digest(local(path)) for path in plan['dependencies']},
                      session_id=uuid.uuid4().hex,profiling=False,run_kind='control' if args.control else 'candidate',
                      environment={'gpu_uuid':gpu[0].strip(),'driver':gpu[1].strip(),'torch':runtimes[0]['torch'],
                                   'cuda':runtimes[0]['cuda'],'container_digest':image.split('@')[1],'exclusive_gpu':True})
        save(output,report)
        correctness=dict(report['correctness'],report_sha256=digest(output),precision_preserved=True,
                         precision_review_source=str(args.review),precision_reviewer=review['preflight_reviewer'])
        save(output.with_suffix('.correctness.json'),correctness)
        print(json.dumps({'status':'complete','output':str(output),'run_kind':report['run_kind']}),flush=True)


if __name__=='__main__':main()

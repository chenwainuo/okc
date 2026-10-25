#!/usr/bin/env python3
"""Serialized AB/BA measurements using isolated persistent worker processes."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from measurement import summarize


class WorkerFailure(RuntimeError):
    pass


class WorkerCancelled(RuntimeError):
    pass


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.pair-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def bundle_identity(root):
    metadata=json.loads((Path(root)/'.rsi-bundle.json').read_text())
    identity=metadata.get('archive_sha256')
    if not isinstance(identity,str) or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
        raise WorkerFailure('Verified .rsi-bundle.json archive SHA256 required')
    return identity


class WorkerClient:
    def __init__(self, command, cwd, environment, log, deadline):
        self.log=open(log,'wb')
        try:
            self.process=subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,stderr=self.log,start_new_session=True,bufsize=0)
        except BaseException:
            self.log.close();raise
        self.group=self.process.pid
        self.deadline=deadline
        self.sequence=0
        self.buffer=b''
        self.selector=selectors.DefaultSelector()
        os.set_blocking(self.process.stdout.fileno(),False)
        self.selector.register(self.process.stdout,selectors.EVENT_READ)

    def rpc(self, op, **parameters):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Paired worker session deadline exhausted')
        self.sequence+=1
        request={'id':self.sequence,'op':op,**parameters}
        payload=(json.dumps(request,allow_nan=False)+'\n').encode()
        if len(payload)>4096:raise WorkerFailure('RPC request exceeds bounded pipe message size')
        try:
            self.process.stdin.write(payload);self.process.stdin.flush()
        except (BrokenPipeError,OSError) as error:
            raise WorkerFailure(f'Worker stdin closed: {error}') from error
        while b'\n' not in self.buffer:
            remaining=self.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError(f'Worker timed out during {op}')
            if not self.selector.select(remaining):raise TimeoutError(f'Worker timed out during {op}')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise WorkerFailure(f'Worker exited/closed stdout during {op}; inspect stderr log')
            self.buffer+=chunk
            if len(self.buffer)>16*1024*1024:raise WorkerFailure('Worker RPC response exceeded 16 MiB')
        line,self.buffer=self.buffer.split(b'\n',1)
        try:response=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError) as error:raise WorkerFailure('Invalid worker RPC JSON') from error
        if response.get('id')!=self.sequence or response.get('ok') is not True:
            raise WorkerFailure(f'Worker RPC rejected {op}: {response}')
        return response['result']

    def wait_clean(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Worker shutdown deadline exhausted')
        result=self.process.wait(timeout=remaining)
        if result!=0:raise WorkerFailure(f'Worker returned nonzero exit {result}')

    def signal_group(self, sig):
        try:os.killpg(self.group,sig)
        except ProcessLookupError:pass

    def close_handles(self):
        self.selector.close()
        for pipe in (self.process.stdin,self.process.stdout):
            if pipe:pipe.close()
        self.log.close()


def reap_group(group):
    while True:
        try:
            child,_=os.waitpid(-group,os.WNOHANG)
            if child==0:return
        except ChildProcessError:return


def group_exists(group):
    try:os.killpg(group,0);return True
    except ProcessLookupError:return False


def cleanup(workers):
    # Parent is a Linux child subreaper, so orphaned compiler grandchildren are
    # reparented here and can be waited, not merely signaled and left as zombies.
    for worker in workers:worker.signal_group(signal.SIGTERM)
    deadline=time.monotonic()+1
    while time.monotonic()<deadline:
        alive=False
        for worker in workers:
            worker.process.poll();reap_group(worker.group)
            alive=group_exists(worker.group) or alive
        if not alive:break
        time.sleep(.02)
    for worker in workers:worker.signal_group(signal.SIGKILL)
    for worker in workers:
        try:worker.process.wait(timeout=3)
        except subprocess.TimeoutExpired:raise WorkerFailure('Failed to reap worker after SIGKILL')
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        for worker in workers:reap_group(worker.group)
        if not any(group_exists(worker.group) for worker in workers):break
        time.sleep(.02)
    survivors=[w.group for w in workers if group_exists(w.group)]
    for worker in workers:worker.close_handles()
    if survivors:raise WorkerFailure(f'Worker process groups survived cleanup: {survivors}')


def subreaper(enabled):
    libc=ctypes.CDLL(None,use_errno=True)
    previous=ctypes.c_int()
    if libc.prctl(37,ctypes.byref(previous),0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PR_GET_CHILD_SUBREAPER failed')
    if libc.prctl(36,int(enabled),0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
    return bool(previous.value)


def input_match(left,right,context):
    digest=left.get('input_sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or digest!=right.get('input_sha256'):
        raise WorkerFailure(f'Actual input fingerprints differ: {context}')
    return digest


def run_pair(baseline_root,candidate_root,output,seed,pairs,batches,timeout_seconds,
             cache_root=None,worker_script=None):
    if type(pairs) is not int or pairs<2 or pairs%2:raise ValueError('Pairs must be even and >=2')
    if type(seed) is not int:raise ValueError('Integer seed required')
    if not batches or len(set(batches))!=len(batches) or any(type(bs) is not int or bs<1 for bs in batches):
        raise ValueError('Distinct positive integer batches required')
    if not isinstance(timeout_seconds,(int,float)) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:
        raise ValueError('Positive finite timeout required')
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    roots=[Path(baseline_root).resolve(strict=True),Path(candidate_root).resolve(strict=True)]
    identities=[bundle_identity(root) for root in roots]
    script=Path(worker_script or Path(__file__).with_name('rsi_worker.py')).resolve(strict=True)
    cache_root=Path(cache_root or output.parent/'.rsi-worker-cache').resolve()
    deadline=time.monotonic()+timeout_seconds
    report={'status':'initializing','seed':seed,'pairs':pairs,'batches':batches,'results':[],
            'measurement':'serialized isolated-process CUDA graph AB/BA replay; cold128MiB; official reference outside timing',
            'baseline_root':str(roots[0]),'candidate_root':str(roots[1]),
            'bundle_sha256':dict(zip(('baseline','candidate'),identities)),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'worker_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
            'profiling':False,'workers':{},'input_fingerprint_checks':0,
            'process_groups_terminated':False}
    workers=[];failed=None
    previous_subreaper=subreaper(True)
    previous_sigterm=signal.getsignal(signal.SIGTERM)
    previous_sigint=signal.getsignal(signal.SIGINT)
    def cancelled(signum, frame):
        raise WorkerCancelled(f'Coordinator received signal {signum}')
    signal.signal(signal.SIGTERM,cancelled)
    try:
        for index,role in enumerate(('baseline','candidate')):
            cache=cache_root/role/identities[index];cache.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy()
            env.update(PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                       XDG_CACHE_HOME=str(cache),SGLANG_JIT_CACHE_DIR=str(cache/'sglang'),
                       TRITON_CACHE_DIR=str(cache/'triton'),TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'))
            worker=WorkerClient([sys.executable,'-u',str(script),'--root',str(roots[index]),'--role',role],
                                roots[index],env,output.parent/f'{output.stem}-{role}.stderr.log',deadline)
            workers.append(worker)
            # Initialization does GPU work, so the second worker is not initialized
            # until the first explicitly synchronizes and replies.
            reply=worker.rpc('init',seed=seed)
            report['workers'][role]={'pid':worker.process.pid,'init':reply,'cache_root':str(cache),
                                     'archive_sha256':identities[index], 'source_root':str(roots[index]),
                                     'runtime':reply.get('runtime',{}),'imports':reply.get('imports',{}),
                                     'weight_seed':reply.get('weight_seed')}
            if reply.get('weight_seed')!=seed:
                raise WorkerFailure('Worker weight seed differs from requested seed')
            imports=reply.get('imports',{})
            if imports.get('archive_sha256')!=identities[index] or imports.get('source_root')!=str(roots[index]):
                raise WorkerFailure('Worker imported bundle identity differs from verified root')
            runtime=reply.get('runtime',{})
            if not all(isinstance(runtime.get(key),str) and runtime[key] for key in ('torch','cuda')):
                raise WorkerFailure('Worker runtime identity incomplete')
            if index:
                first=report['workers']['baseline']['init']
                if any(runtime[key]!=first['runtime'][key] for key in ('torch','cuda')):
                    raise WorkerFailure('Worker Torch/CUDA runtime differs between paired processes')
                for name in ('reference','baseline','nvfp4'):
                    first_hash=first['imports'].get('modules',{}).get(name,{}).get('sha256')
                    second_hash=imports.get('modules',{}).get(name,{}).get('sha256')
                    if not first_hash or first_hash!=second_hash:
                        raise WorkerFailure(f'Pinned fixture source differs between workers: {name}')
            atomic(output,report)
        for bs in batches:
            setup=[worker.rpc('configure',batch=bs) for worker in workers]
            input_match(*setup,f'graph setup M={bs}')
            report['input_fingerprint_checks']+=1
            times=[[],[]];fingerprints=[];orders=[]
            report['status']=f'measuring_bs{bs}';atomic(output,report)
            for pair in range(pairs):
                input_seed=seed+2000+bs*100+pair
                order=(0,1) if pair%2==0 else (1,0)
                replies={}
                for index in order:
                    reply=workers[index].rpc('measure',input_seed=input_seed)
                    elapsed=reply.get('elapsed_us')
                    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:
                        raise WorkerFailure('Nonfinite/nonpositive CUDA event duration')
                    if reply.get('batch')!=bs or reply.get('input_seed')!=input_seed or reply.get('reference_pass') is not True or reply.get('input_immutable_pass') is not True:
                        raise WorkerFailure('Worker measurement identity/correctness mismatch')
                    replies[index]=reply;times[index].append(elapsed)
                fingerprints.append(input_match(replies[0],replies[1],f'M={bs} pair={pair}'))
                report['input_fingerprint_checks']+=1
                orders.append('AB' if order==(0,1) else 'BA')
            row={'bs':bs,'baseline_samples_us':times[0],'candidate_samples_us':times[1],
                 'baseline_us':statistics.median(times[0]),'candidate_us':statistics.median(times[1]),
                 'input_sha256':fingerprints,'orders':orders,**summarize(*times)}
            report['results'].append(row);atomic(output,report)
        for role,worker in zip(('baseline','candidate'),workers):
            terminal=worker.rpc('close')
            counts=terminal.get('counts',{})
            required_count=pairs*len(batches)
            if (terminal.get('reference_pass') is not True or terminal.get('empty_local_eager_pass') is not True or
                terminal.get('empty_local_graph_pass') is not True or terminal.get('input_immutable_pass') is not True or
                counts.get('eager_reference_cases')!=5 or counts.get('timed_reference_cases')!=required_count or
                counts.get('captured_valid_cases')!=len(batches) or counts.get('empty_local_graph_cases')!=len(batches)):
                raise WorkerFailure('Worker terminal correctness accounting incomplete')
            report['workers'][role]['terminal']=terminal
            worker.wait_clean()
            report['workers'][role].update(clean_exit=True,exit_code=0,
                imports=terminal.get('imports',{}),counts=counts,
                reference_pass=terminal['reference_pass'],
                empty_local_eager_pass=terminal['empty_local_eager_pass'],
                empty_local_graph_pass=terminal['empty_local_graph_pass'],
                input_immutable_pass=terminal['input_immutable_pass'])
        report['status']='workers_exited_pending_cleanup'
    except BaseException as error:
        failed=error;report.update(status='failed',error=repr(error))
    finally:
        # A second termination signal must not interrupt process-tree cleanup.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:
            cleanup(workers)
            report['process_groups_terminated']=True
        except BaseException as error:
            report.update(status='cleanup_failed',cleanup_error=repr(error))
            if failed is None:failed=error
        finally:
            subreaper(previous_subreaper)
            signal.signal(signal.SIGTERM,previous_sigterm)
            signal.signal(signal.SIGINT,previous_sigint)
        if failed is None:
            report['status']='complete'
            report['correctness']={'reference_pass':True,'empty_local_eager_pass':True,
                                   'empty_local_graph_pass':True,'input_immutable_pass':True,
                                   'reference_cases':sum(w['terminal']['counts']['reference_cases'] for w in report['workers'].values())}
        atomic(output,report)
    if failed is not None:raise failed
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--pairs',type=int,required=True);p.add_argument('--batches',type=int,nargs='+',required=True)
    p.add_argument('--timeout-seconds',type=int,required=True);p.add_argument('--cache-root',type=Path)
    args=p.parse_args()
    run_pair(args.baseline_root,args.candidate_root,args.output,args.seed,args.pairs,args.batches,args.timeout_seconds,args.cache_root)
    print(f'Completed isolated paired run: {args.output}',flush=True)


if __name__=='__main__':main()

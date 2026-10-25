#!/usr/bin/env python3
"""Persistent GPU worker rooted in one immutable bundle; JSON-line RPC only on stdout."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback


def fingerprint(data):
    digest = hashlib.sha256()
    digest.update(json.dumps(data['config'], sort_keys=True, separators=(',', ':')).encode())
    for name in ('hidden_states', 'topk_ids', 'topk_weights'):
        tensor = data[name].detach().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.cpu().view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def setup_paths(root):
    own = [root / 'muse_moe_w4a16', root / 'benchmarks', root / 'serving/sglang/python']
    if not all(path.is_dir() for path in own):
        raise RuntimeError('Bundle must contain fixtures, benchmarks and serving/sglang/python')
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    # Remove ambient PYTHONPATH and central script directory; only the bundle may
    # supply project imports. System stdlib/site-packages remain for pinned runtime.
    ambient = {Path(p).resolve() for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p}
    remainder = []
    for value in sys.path:
        path = Path(value or '.').resolve()
        if path in ambient or path in (script_dir, cwd):
            continue
        remainder.append(value)
    sys.path[:] = [str(p) for p in own] + remainder
    os.chdir(root)


class GPUWorker:
    def __init__(self, root, role):
        self.root, self.role = root, role
        self.bundle = json.loads((root/'.rsi-bundle.json').read_text())
        self.torch = None
        self.seed = None
        self.graph = self.output = self.data = None
        self.counts = dict(reference_cases=0, eager_reference_cases=0,
                           captured_valid_cases=0, timed_reference_cases=0,
                           empty_local_eager_cases=0, empty_local_graph_cases=0, input_immutability_checks=0)

    def identities(self):
        identities = {}
        for name in ('reference','baseline','nvfp4','serving_moe','sglang'):
            module = sys.modules[name]
            source = Path(module.__file__).resolve()
            if not source.is_relative_to(self.root):
                raise RuntimeError(f'Import escaped bundle: {name} -> {source}')
            relative = str(source.relative_to(self.root))
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if self.bundle['files'].get(relative) != source_hash:
                raise RuntimeError(f'Imported source differs from extracted bundle manifest: {relative}')
            identities[name] = dict(path=str(source),relative_path=relative,sha256=source_hash)
        sglang_modules = {}
        for name,module in list(sys.modules.items()):
            if name.startswith('sglang.') and getattr(module,'__file__',None):
                source = Path(module.__file__).resolve()
                if not source.is_relative_to(self.root / 'serving/sglang/python'):
                    raise RuntimeError(f'Transitive import escaped bundle: {name} -> {source}')
                sglang_modules[name] = str(source)
        jit_libraries = {}
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(None,5)
            if len(fields) != 6 or 'sgl_kernel_jit_' not in fields[5] or '.so' not in fields[5]:
                continue
            path = Path(fields[5])
            if not path.is_file():
                raise RuntimeError(f'Loaded JIT object no longer inspectable: {path}')
            if str(path) in jit_libraries:
                continue
            deps = path.parent/'sgl_deps.json'
            if not deps.is_file():
                raise RuntimeError(f'Loaded JIT dependency manifest missing: {deps}')
            jit_libraries[str(path)] = dict(so_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dependency_manifest=str(deps),dependency_manifest_sha256=hashlib.sha256(deps.read_bytes()).hexdigest())
        if self.seed is not None and not any('moe_wna16_marlin' in path for path in jit_libraries):
            raise RuntimeError('Actual Marlin JIT shared object not found in process maps')
        return dict(modules=identities,sglang_modules=sglang_modules,jit_libraries=jit_libraries,
                    torch_path=str(Path(self.torch.__file__).resolve()), source_root=str(self.root),
                    archive_sha256=self.bundle['archive_sha256'],
                    cache_environment={key:os.environ.get(key) for key in
                                       ('XDG_CACHE_HOME','SGLANG_JIT_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR')})

    def unchanged_input(self, data, expected):
        if fingerprint(data) != expected:
            raise RuntimeError('Candidate modified read-only input tensors or configuration')
        self.counts['input_immutability_checks'] += 1

    def check(self, data, output, kind, input_hash):
        self.unchanged_input(data,input_hash)
        passed, detail = self.reference.check_implementation(data, output, self.weights)
        if not passed:
            raise RuntimeError(f'Official reference failed: {detail}')
        self.counts['reference_cases'] += 1
        self.counts[kind] += 1

    def init(self, seed):
        if self.seed is not None:
            raise RuntimeError('Worker already initialized')
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.reference = importlib.import_module('reference')
        self.adapter = importlib.import_module('serving_moe')
        self.seed = seed
        self.weights = self.reference.generate_weights(seed)
        self.state = self.adapter.prepare({k:v.clone() for k,v in self.weights.items()})
        torch.cuda.synchronize()
        no_local = self.reference.generate_input(4,'ep8',seed+99)
        no_local['topk_ids'][:] = torch.arange(32,40,device='cuda',dtype=torch.int32)
        no_local_hash = fingerprint(no_local)
        output = self.adapter.kernel_fn(no_local,self.state)
        self.unchanged_input(no_local,no_local_hash)
        torch.testing.assert_close(output,torch.zeros_like(no_local['hidden_states']),rtol=0,atol=0)
        self.counts['empty_local_eager_cases'] += 1
        for index,(bs,dist) in enumerate(((4,'normal'),(33,'normal'),(257,'normal'),(128,'skewed'),(1,'normal'))):
            data = self.reference.generate_input(bs,'ep8',seed+100+index,dist=dist)
            input_hash = fingerprint(data)
            self.check(data,self.adapter.kernel_fn(data,self.state),'eager_reference_cases',input_hash)
        self.flush = torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record();self.end_event.record();self.end_event.synchronize()
        torch.cuda.synchronize()
        return dict(imports=self.identities(),counts=dict(self.counts),weight_seed=seed,
                    weight_fingerprint='not_collected; pinned fixture/runtime, seed, cloned preparation and reference validation',
                    runtime=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))

    def configure(self, batch):
        if self.seed is None or type(batch) is not int or batch < 1:
            raise RuntimeError('Initialize first; positive integer batch required')
        torch = self.torch
        torch.cuda.synchronize()
        self.graph = self.output = self.data = None
        self.data = self.reference.generate_input(batch,'ep8',self.seed+1000+batch)
        original_input_hash = fingerprint(self.data)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.adapter.kernel_fn(self.data,self.state)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.unchanged_input(self.data,original_input_hash)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.adapter.kernel_fn(self.data,self.state)
        self.graph.replay()
        self.check(self.data,self.output,'captured_valid_cases',original_input_hash)
        original = self.data['topk_ids'].clone()
        self.data['topk_ids'][:] = torch.arange(32,40,device='cuda',dtype=torch.int32)
        empty_input_hash = fingerprint(self.data)
        self.graph.replay()
        self.unchanged_input(self.data,empty_input_hash)
        torch.testing.assert_close(self.output,torch.zeros_like(self.data['hidden_states']),rtol=0,atol=0)
        self.counts['empty_local_graph_cases'] += 1
        self.data['topk_ids'].copy_(original)
        input_hash = fingerprint(self.data)
        torch.cuda.synchronize()
        return dict(batch=batch,input_sha256=input_hash,counts=dict(self.counts))

    def measure(self, input_seed):
        if self.graph is None:
            raise RuntimeError('Configure graph before measuring')
        torch = self.torch
        batch = self.data['hidden_states'].shape[0]
        fresh = self.reference.generate_input(batch,'ep8',input_seed)
        for name in ('hidden_states','topk_ids','topk_weights'):
            self.data[name].copy_(fresh[name])
        # Hash actual captured-input buffers after copy, not merely RNG seed.
        input_hash = fingerprint(self.data)
        self.flush.zero_()
        self.start_event.record()
        self.graph.replay()
        self.end_event.record()
        self.end_event.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)*1000
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f'Invalid CUDA event duration: {elapsed}')
        self.check(self.data,self.output,'timed_reference_cases',input_hash)
        torch.cuda.synchronize()
        return dict(batch=batch,input_seed=input_seed,input_sha256=input_hash,
                    elapsed_us=elapsed,reference_pass=True,input_immutable_pass=True,counts=dict(self.counts))

    def close(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()
        return dict(imports=self.identities() if self.torch is not None else {},counts=dict(self.counts),
                    reference_pass=self.counts['eager_reference_cases']==5 and self.counts['timed_reference_cases']>0,
                    empty_local_eager_pass=self.counts['empty_local_eager_cases']==1,
                    empty_local_graph_pass=self.counts['empty_local_graph_cases']>0,
                    input_immutable_pass=self.counts['input_immutability_checks'] ==
                        6+3*self.counts['captured_valid_cases']+self.counts['timed_reference_cases'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',choices=('baseline','candidate'),required=True)
    args=p.parse_args()
    # Preserve original fd1 solely for RPC. Native/CUDA/compiler stdout also goes
    # to stderr, unlike redirect_stdout which only redirects Python prints.
    rpc=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    sys.stdout=sys.stderr
    root=args.root.resolve(strict=True)
    setup_paths(root)
    worker=GPUWorker(root,args.role)
    for line in sys.stdin:
        request_id=None
        try:
            request=json.loads(line)
            request_id=request['id']
            op=request['op']
            if op=='init': result=worker.init(request['seed'])
            elif op=='configure': result=worker.configure(request['batch'])
            elif op=='measure': result=worker.measure(request['input_seed'])
            elif op=='close': result=worker.close()
            else: raise ValueError(f'Unknown operation: {op}')
            rpc.write(json.dumps(dict(id=request_id,ok=True,result=result),allow_nan=False)+'\n')
            rpc.flush()
            if op=='close':return
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            rpc.write(json.dumps(dict(id=request_id,ok=False,error=repr(error)))+'\n')
            rpc.flush()
            raise SystemExit(1)


if __name__=='__main__':main()

"""Development-only official-reference checks; no timing, run with root GPU lease.

--bundle-root must be an immutable verified bundle. No heldout seed accepted:
seed910231 is intentionally development-only. All inputs fingerprinted.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

DEVELOPMENT_SEED=910231

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle-root',required=True,type=Path)
    parser.add_argument('--runner-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.bundle_root.resolve()
    candidate_dir=Path(__file__).resolve().parent
    runner_path=args.runner_root.resolve()/'rsi_worker.py'
    worker=load('_rsi_frozen_paths',runner_path)
    worker.setup_paths(root)
    import torch
    import reference
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    candidate=load('_rsi_full_wgmma_candidate',candidate_dir/'serving_wgmma_down.py')
    weights=reference.generate_weights(DEVELOPMENT_SEED)
    state=candidate.prepare({k:v.clone() for k,v in weights.items()})
    results=[]
    def unchanged(data,before):
        if worker.fingerprint(data)!=before:
            raise AssertionError('candidate mutated input before reference check')
    def verify(data,out,before,kind,empty=False):
        unchanged(data,before)
        if empty:
            torch.testing.assert_close(out,torch.zeros_like(data['hidden_states']),rtol=0,atol=0)
            detail='exact zero'
        else:
            passed,detail=reference.check_implementation(data,out,weights)
            if not passed:raise AssertionError(detail)
        torch.cuda.synchronize()
        results.append({'kind':kind,'pass':True,'detail':str(detail)})
    for index,(bs,dist) in enumerate(((4,'normal'),(33,'normal'),(257,'normal'),(128,'skewed'),(1,'normal'))):
        data=reference.generate_input(bs,'ep8',DEVELOPMENT_SEED+100+index,dist=dist)
        before=worker.fingerprint(data)
        verify(data,candidate.kernel_fn(data,state),before,f'eager_{bs}_{dist}')
        if worker.fingerprint(data)!=before:raise AssertionError('candidate mutated input')
    data=reference.generate_input(4,'ep8',DEVELOPMENT_SEED+99)
    data['topk_ids'][:]=torch.arange(32,40,device='cuda',dtype=torch.int32)
    before=worker.fingerprint(data)
    verify(data,candidate.kernel_fn(data,state),before,'eager_empty',empty=True)
    if worker.fingerprint(data)!=before:raise AssertionError('candidate mutated input')
    data=reference.generate_input(8,'ep8',DEVELOPMENT_SEED+108)
    before=worker.fingerprint(data)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate.kernel_fn(data,state)
            unchanged(data,before)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    unchanged(data,before)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=candidate.kernel_fn(data,state)
    unchanged(data,before)
    graph.replay();verify(data,out,before,'graph_valid')
    if worker.fingerprint(data)!=before:raise AssertionError('candidate mutated input')
    data['topk_ids'][:]=torch.arange(32,40,device='cuda',dtype=torch.int32)
    before=worker.fingerprint(data);graph.replay();verify(data,out,before,'graph_empty_after_valid',empty=True)
    if worker.fingerprint(data)!=before:raise AssertionError('candidate mutated input')
    # Captured shapes fixed, fresh independently generated input bytes.
    fresh=reference.generate_input(8,'ep8',DEVELOPMENT_SEED+109,dist='skewed')
    for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
    before=worker.fingerprint(data);graph.replay();verify(data,out,before,'graph_fresh_skewed')
    if worker.fingerprint(data)!=before:raise AssertionError('candidate mutated input')
    torch.cuda.synchronize()
    report={'status':'complete','scope':'full expert synthetic correctness only; not performance or official serving score',
            'development_seed':DEVELOPMENT_SEED,'results':results,
            'runner_sha256':hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
            'candidate_sha256':{f:hashlib.sha256((candidate_dir/f).read_bytes()).hexdigest() for f in ('serving_wgmma_down.py','routed_adapter.py','routed_down.cuh','check_full_expert.py')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
    print(json.dumps(report))
if __name__=='__main__':main()

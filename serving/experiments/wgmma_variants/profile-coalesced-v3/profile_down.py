"""Root-leased observational single-down-launch counters; never a performance test."""
import hashlib,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=Path('/candidate')
PIN=json.loads((P/'source-pins.json').read_text())
for relative,expected in PIN['files'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
worker_path=Path('/runner/rsi_worker.py')
assert hashlib.sha256(worker_path.read_bytes()).hexdigest()==PIN['worker_sha256']
spec=importlib.util.spec_from_file_location('_profile_worker_paths',worker_path)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
worker.setup_paths(ROOT)
import torch
import torch.cuda.profiler as profiler
import reference
import serving_wgmma_down as candidate
from serving_wgmma_down import _down

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False
weight_seed=940033
input_seed=weight_seed+1000+128
weights=reference.generate_weights(weight_seed)
data=reference.generate_input(128,'ep8',input_seed)
before=worker.fingerprint(data)
state=candidate.prepare({k:v.clone() for k,v in weights.items()})
# Diagnostic-only wrapper captures actual operands from the unchanged adapter.
# Restore before validation/profile; no frozen file or kernel code modifications.
original=_down.project
captured={}
def capture(*args,**kwargs):
    captured['args']=args;captured['kwargs']=kwargs
    return original(*args,**kwargs)
_down.project=capture
try:
    out=candidate.kernel_fn(data,state)
finally:
    _down.project=original
torch.cuda.synchronize()
assert worker.fingerprint(data)==before,'input mutated'
ok,detail=reference.check_implementation(data,out,weights)
assert ok,detail
args=captured['args'];kwargs=captured['kwargs']
a,bank,sorted_ids,expert_ids,post,block_m=args
def intermediate_fingerprint():
    digest=hashlib.sha256()
    for tensor in (a,sorted_ids,expert_ids,post):
        digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
intermediate_before=intermediate_fingerprint()
expected=kwargs['out'].clone()
original(*args,**kwargs)
torch.cuda.synchronize()
torch.testing.assert_close(kwargs['out'],expected,rtol=0,atol=0)
# Metadata transfer is outside profiler scope.
sorted_cpu=sorted_ids.cpu();experts_cpu=expert_ids.cpu();post_count=int(post.item())
grid_y=sorted_ids.numel()//8
# Coalesced layout is already tiled: [E,N/64,K/16,128] int32.
assert bank.packed.dtype==torch.int32 and bank.packed.ndim==4
assert bank.packed.shape[2:]==(256,128),tuple(bank.packed.shape)
assert tuple(bank.scales.shape)==(*bank.packed.shape[:3],64)
grid_x=bank.packed.shape[1]
assert kwargs['out'].shape[1]==grid_x*64 and a.shape[1]==4096
valid_groups=0;local_rows=0;all_padding_groups=0
for y in range(grid_y):
    pos=y*8
    if pos>=post_count or pos//block_m>=expert_ids.numel():continue
    e=int(experts_cpu[pos//block_m])
    if 0<=e<bank.packed.shape[0]:
        valid_groups+=1
        rows_here=sum(0<=int(v)<a.shape[0] for v in sorted_cpu[pos:pos+8])
        local_rows+=rows_here
        all_padding_groups+=int(rows_here==0)
assert intermediate_fingerprint()==intermediate_before,'isolated down mutated captured inputs'
jit_libraries={}
for line in Path('/proc/self/maps').read_text().splitlines():
    fields=line.split(None,5)
    if len(fields)!=6 or 'rsi_wgmma_coalesced_down' not in fields[5] or '.so' not in fields[5]:continue
    path=Path(fields[5])
    jit_libraries[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
assert jit_libraries,'actual routed-down JIT shared object not observed'
report={'scope' :'observational Nsight counters; no benchmark timing',
        'weight_seed':weight_seed,'input_seed':input_seed,'batch':128,'input_sha256':before,
        'full_expert_reference_pass':True,'full_expert_detail':str(detail),
        'isolated_down_repeat_exact_pass':True,'bundle_sha256':PIN['bundle_sha256'],
        'bank_packed_shape':list(bank.packed.shape),'bank_scales_shape':list(bank.scales.shape),
        'bank_packed_storage_bytes':bank.packed.untyped_storage().nbytes(),
        'bank_scales_storage_bytes':bank.scales.untyped_storage().nbytes(),
        'grid':[grid_x,grid_y,1],'block':[128,1,1], 'total_ctas':grid_x*grid_y,
        'local_expert_groups_of8':valid_groups,'all_padding_groups_of8':all_padding_groups,
        'useful_ctas':grid_x*(valid_groups-all_padding_groups),'all_padding_ctas':grid_x*all_padding_groups,
        'padded_rows_in_local_groups':valid_groups*8-local_rows,'captured_input_sha256':intermediate_before,'loaded_down_jit_libraries':jit_libraries,'nonlocal_or_beyond_post_ctas':grid_x*(grid_y-valid_groups),
        'local_route_rows':local_rows,'sorted_capacity':sorted_ids.numel(),'num_post':post_count,'block_m':block_m,
        'cache_mode':'NCU cache-control all; kernel replay; not clean latency',
        'source_pins':PIN}
Path('/output/setup.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
torch.cuda.synchronize()
# One launch of target kernel. Output-zeroing remains real work but excluded by
# exact NCU kernel filter. NCU replay may execute this launch multiple times.
profiler.start()
torch.cuda.nvtx.range_push('okc_wgmma_coalesced_v3_down_m128')
original(*args,**kwargs)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
profiler.stop()
torch.testing.assert_close(kwargs['out'],expected,rtol=0,atol=0)
assert intermediate_fingerprint()==intermediate_before,'profile mutated captured inputs'
Path('/output/terminal.json').write_text(json.dumps({'status':'complete','post_profile_down_repeat_exact_pass':True,'performance_claim':False})+'\n')

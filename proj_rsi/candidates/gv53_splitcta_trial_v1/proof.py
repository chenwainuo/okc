"""Focused host proof for two direct template bindings and geometry-specific CSR."""
import ast,hashlib,importlib.util,json,re,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]

def cpu_proof():
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('splitcta_driver',D/'driver.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    pilot=(D/'pilot.cu').read_text();original=(ROOT/'proj_rsi/candidates/gv53_maskgate_v1/pilot.cu').read_text()
    marker='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__ void __launch_bounds__'
    original=original[:original.index(marker,original.index('gv_predxor(G p)')+20)]
    pilot=pilot[:pilot.index('template __global__')]
    def canonical(text):
        a=text.index('  static_assert(');b=text.index('  // pw14 probes',a)
        return text[:a]+text[b:]
    assert canonical(pilot).rstrip()==canonical(original).rstrip(),'predxor body/helper/prefix drift'
    env={'torch':torch};exec(compile(ast.Module(body=driver.codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<original-codec>','exec'),env)
    # Every output row tile and K chunk exactly once, with the same per-output K-warp summation order.
    geometry={0:(2,16,8,1,1,0),1:(1,16,8,1,1,0)};grids={0:128,1:256}
    for arm,(wr,wk,ku,s,rt,xs)in geometry.items():
        units=[(bx*wr+w%wr,(w//wr)*ku+u)for bx in range(grids[arm])for w in range(wr*wk)for u in range(ku)]
        assert len(units)==len(set(units))==256*128
        assert set(units)=={(t,c)for t in range(256)for c in range(128)}
    # CSR order and correction target must follow each actual geometry, including distant tiles/last chunk.
    entries=[]
    for t in range(256):
        for wk in range(16):
            for offset in (0,7):
                entries.append([t,wk*8+offset,(t+wk)%4,(t+wk+offset)%32,(t+offset)%8,1+(t+wk)%127,0])
    esc=torch.tensor(entries,dtype=torch.int64)
    for arm,geo in geometry.items():
        packed=env['_gvesc'](esc,256,geo);assert packed.numel()==131072
        wr,wk,ku,s,rt,xs=geo;decoded=[]
        for pos in torch.nonzero(packed!=-1).flatten().tolist():
            gw=pos//32;value=int(packed[pos])&0xffffffff
            tile=(gw//(wr*wk))*wr+gw%wr;chunk=((gw//wr)%wk)*ku+(value>>20)
            reg=(value>>11)&15
            decoded.append([tile,chunk,reg//4,(value>>15)&31,2*(reg%4)+((value>>10)&1),value&127,(value>>9)&1])
        assert sorted(decoded)==sorted(entries),'geometry CSR correction mapping'
    exact32=torch.tensor([[0,0,i%4,i,2*(i%4),1,0]for i in range(32)],dtype=torch.int64)
    for geo in geometry.values():
        assert env['_gvesc'](exact32,256,geo)is not None
        assert env['_gvesc'](torch.cat((exact32,exact32[:1])),256,geo)is None
    assert not torch.cuda.is_initialized()
    return {'state':'PASS_FOCUSED_SPLIT_CTA_CPU_GPU_UNRUN','same_predxor_template_body_except_scope_assertion':True,'scope':'Direct pilot explicitly enables predxor for new tuple. Current serving whitelist does not enable this tuple; no policy comparison claim.','geometry_units_per_arm':32768,'CSR_adversarial_entries_per_arm':len(entries),'CSR_exact32_and_overflow33':'PASS','separate_geometry_lists':True,'original_codec_AST':'EXACT','GPU_calls':0,'all95_compile':False}


def compile_binding():
    import torch
    from torch.utils.cpp_extension import include_paths
    source=(D/'pilot.cu').read_text()+'\n'+(D/'binding.cu').read_text()
    source=source.replace('#include <torch/extension.h>','#include <ATen/core/Tensor.h>\nnamespace torch {using Tensor=at::Tensor;}').replace('#include <ATen/cuda/CUDAContext.h>','namespace at {namespace cuda {cudaStream_t getCurrentCUDAStream();}}')
    (D/'focused-binding.cu').write_text(source)
    argv=['/usr/local/cuda-12.8/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    for p in include_paths():argv+=['-I',p]
    argv+=['-c',str(D/'focused-binding.cu'),'-o',str(D/'focused-binding.o')]
    begin=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=90);(D/'focused-binding.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    assert len(re.findall(r'Used 64 registers',p.stderr))==2
    assert len(re.findall(r'0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads',p.stderr))==2
    result={'state':'PASS_TWO_KERNEL_HOST_BINDING_CPU_COMPILE','argv':argv,'returncode':p.returncode,'seconds_observed':time.monotonic()-begin,'resources':'Two exactsymbols,64regs/zero stack/spills each','local_Torch':torch.__version__,'limits':'Real localTensor API with declaration-only CUDA current-stream seam/C++20. RemoteABI/JIT/load/occupancy/numeric unmeasured.','GPU_calls':0}
    (D/'compile-result.json').write_text(json.dumps(result,indent=2)+'\n');assert not torch.cuda.is_initialized()


if __name__=='__main__':
    result=cpu_proof();compile_binding();(D/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

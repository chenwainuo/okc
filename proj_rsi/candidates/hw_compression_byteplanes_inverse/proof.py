"""CPU all-bits/lane/WGMMA-input mapping + inverse-only SM90a SASS; no CUDA API."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'pw36/submission.py'
PARENT_SHA = '2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
CUDA = Path('/usr/local/cuda-12.8/bin')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prmt(a,b,s):
    out=np.zeros_like(a,dtype=np.uint32)
    for j in range(4):
        n=(s>>(4*j))&7
        out |= (((a if n<4 else b)>>(8*(n%4)))&255) << (8*j)
    return out


def main():
    assert sha(PARENT)==PARENT_SHA
    n,k=128,512
    W=np.arange(n*k,dtype=np.uint16).reshape(n,k)
    # [KC,warp,q,lane,value], v=4*kh+2*half+e; lane=4*g+t.
    c,warp,q,lane,v=np.ogrid[:8,:8,:4,:32,:8]
    rows=16*warp+8*((v//2)%2)+lane//4
    cols=64*c+16*q+8*(v//4)+2*(lane%4)+v%2
    rows,cols=np.broadcast_arrays(rows,cols)
    ids=rows*k+cols
    assert np.array_equal(np.sort(ids.reshape(-1)),np.arange(n*k))
    f=W[rows,cols]
    # Independently compare symbolic mapping to the actual shipped packer's reshape/permute.
    tree=ast.parse(PARENT.read_text()); node=next(t for t in tree.body if isinstance(t,ast.FunctionDef) and t.name=='_pack_nat')
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(PARENT),'exec'),env)
    actual=env['_pack_nat'](torch.from_numpy(W.view(np.int16))).numpy().view(np.uint16)
    assert np.array_equal(actual.transpose(1,0,2,3,4),f)
    lo=(f&255).astype(np.uint8);hi=(f>>8).astype(np.uint8)
    pairs=[]
    for b in [lo,hi]:
        for start in [0,2]:
            pairs.append(b[:,:,start:start+2].transpose(0,1,3,2,4).copy().reshape(8,8,32,16))
    # [KC,warp,4planes,32lanes,16bytes]; one warp=2048B, one128row stage=16384B.
    packed=np.stack(pairs,axis=2)
    assert packed.nbytes==W.nbytes and packed.shape==(8,8,4,32,16)
    words=packed.view('<u4')
    a=np.empty((8,8,4,32,4),dtype=np.uint32)
    for j in range(4):
        l=words[:,:,j//2];h=words[:,:,2+j//2];o=2*(j%2)
        a[:,:,j,:,0]=prmt(l[...,o],h[...,o],0x5140)
        a[:,:,j,:,1]=prmt(l[...,o],h[...,o],0x7362)
        a[:,:,j,:,2]=prmt(l[...,o+1],h[...,o+1],0x5140)
        a[:,:,j,:,3]=prmt(l[...,o+1],h[...,o+1],0x7362)
    decoded=np.empty_like(f)
    decoded[...,0::2]=(a&65535).astype(np.uint16);decoded[...,1::2]=(a>>16).astype(np.uint16)
    assert np.array_equal(decoded,f)
    out=np.empty_like(W);out[rows,cols]=decoded
    assert np.array_equal(out,W) and np.unique(out).size==65536
    assert np.array_equal(decoded[:,:4],f[:,:4]) and np.array_equal(decoded[:,4:],f[:,4:])
    # Four uint4 loads/lane, no overlap within each warp/stage, full raw-plane span.
    ptr=(np.arange(8)[:,None,None]*2048+np.arange(4)[None,:,None]*512+np.arange(32)[None,None,:]*16)
    assert np.array_equal(np.sort(ptr.reshape(-1)),np.arange(0,16384,16))
    cmd=[str(CUDA/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_90a','--ptxas-options=-v','--cubin',
         str(HERE/'inverse_only.cu'),'-o',str(HERE/'inverse.sm90a.cubin')]
    build=subprocess.run(cmd,check=True,text=True,capture_output=True)
    (HERE/'build.txt').write_text(build.stdout+build.stderr)
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in build.stderr
    sass=subprocess.run([str(CUDA/'cuobjdump'),'--dump-sass',str(HERE/'inverse.sm90a.cubin')],check=True,text=True,capture_output=True).stdout
    (HERE/'inverse.sass').write_text(sass)
    assert len(re.findall(r'\bPRMT\b',sass))==16
    assert len(re.findall(r'\bLDG\.E\.128\b',sass))==4 and len(re.findall(r'\bSTG\.E\.128\b',sass))==4
    assert not re.search(r'\b(?:LDL|STL)\b',sass)
    result={'state':'CPU_ALL_BITS_AND_LANE_MAPPING_PLUS_INVERSE_ONLY_SASS_PASS_GPU_UNRUN','CUDA_API_calls':0,
            'parent_sha256':PARENT_SHA,'source_sha256':sha(HERE/'inverse_only.cu'),
            'all65536bit_patterns':True,'all128rows_x512K_positions_bijection':True,
            'actual_pack_nat_mapping_equal':True,'both64row_HALFs_exact':True,
            'all16B_load_offsets_bijection_per_stage':True,'plane_stage_bytes':16384,'stock_stage_bytes':12288,
            'perlane_weights':32,'SASS_PRMT':16,'SASS_LDG128':4,'SASS_STG128':4,
            'spill_stack_bytes':0,'cubin_sha256':sha(HERE/'inverse.sm90a.cubin'),
            'limits':['Inverse-only compile is not a WGMMA resource/launch proof',
                      'Fragment planes differ from raw-Wt byteplanes64 diagnostic payload',
                      'No GEMM/device launch/API execution or hardware-compression ratio'],
            'compile_command':cmd}
    (HERE/'proof.json').write_text(json.dumps(result,indent=2)+'\n')
    m={'state':'CPU_ONLY_CONDITIONAL_CONSUMER_PROOF','sha256':{p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='manifest.json'}}
    (HERE/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(result,indent=2));print('MANIFEST',sha(HERE/'manifest.json'))


if __name__=='__main__':main()

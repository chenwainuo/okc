"""Local static-cost diagnostic, NEVER launched: actual parent GV bodies + two isolated decode helpers.

No rebased GV candidate, epilogue modification, Torch wrapper or GPU lab is implemented.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"pw36"/"submission.py"
SHA="2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"

PROBES=r'''
__device__ __forceinline__ void dec_or(uint32_t L0,uint32_t L1,uint32_t C,uint32_t Tb,uint32_t* a) {
 const uint32_t hl=(C&0x87878787u)|Tb, hh=(__funnelshift_r(C,C,4)&0x87878787u)|Tb;
 a[0]=prmt(L0,hl,0x5140);a[1]=prmt(L0,hl,0x7362);a[2]=prmt(L1,hh,0x5140);a[3]=prmt(L1,hh,0x7362);
}
extern "C" __global__ void diagnostic_decode_base(const uint4* p,uint32_t Tb,uint4* o) {
 uint4 v=p[threadIdx.x];uint32_t a[4];dec1(v.x,v.y,v.z,Tb,0,a);o[threadIdx.x]=make_uint4(a[0],a[1],a[2],a[3]);
}
extern "C" __global__ void diagnostic_decode_or(const uint4* p,uint32_t Tb,uint4* o) {
 uint4 v=p[threadIdx.x];uint32_t a[4];dec_or(v.x,v.y,v.z,Tb,a);o[threadIdx.x]=make_uint4(a[0],a[1],a[2],a[3]);
}
template __global__ void gv<1,2,2,2,8,16,1,1,0,1>(G);
template __global__ void gv<1,2,2,1,4,16,1,1,2,1>(G);
'''


def main():
 src=PARENT.read_text();assert hashlib.sha256(src.encode()).hexdigest()==SHA
 tree=ast.parse(src)
 cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="GVCU" for t in n.targets))
 cuda=cuda[:cuda.index("__global__ void p1zero")]
 cuda=cuda.replace("#include <torch/extension.h>","#include <cuda_runtime.h>\n#include <type_traits>")
 cuda=cuda.replace("#include <ATen/cuda/CUDAContext.h>","")
 cuda+=PROBES
 (HERE/"diagnostic.cu").write_text(cuda)
 nvcc=["/usr/local/cuda/bin/nvcc","-arch=sm_90a","-cubin","-O3","-lineinfo","-Xptxas=-v",str(HERE/"diagnostic.cu"),"-o",str(HERE/"diagnostic.sm90a.cubin")]
 r=subprocess.run(nvcc,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (HERE/"local-compile.log").write_text(r.stdout)
 assert r.returncode==0,r.stdout
 sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",str(HERE/"diagnostic.sm90a.cubin")],text=True)
 (HERE/"diagnostic.sm90a.sass").write_text(sass)
 functions={}
 for section in sass.split("Function : ")[1:]:
  name=section.splitlines()[0].strip()
  ops=CounterLike(section)
  functions[name] = ops
 report={"state":"LOCAL_STATIC_DIAGNOSTIC_ONLY_NEVER_LAUNCHED","parent_sha256":SHA,
         "scope":"Two actual selected m16 GV bodies UNCHANGED; isolated7opORhelper only; no aligned GV candidate",
         "diagnostic_source_sha256":hashlib.sha256(cuda.encode()).hexdigest(),"functions":functions,"compile_log":r.stdout}
 (HERE/"source-sass-cost.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))


def CounterLike(section):
 lines=[s for s in section.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s)]
 names=[]
 for line in lines:
  tail=line.split('*/',1)[1].strip()
  if tail.startswith('@'):tail=tail.split(None,1)[1]
  names.append(tail.split()[0])
 return {"instructions_including_padding":len(names),"instructions_excluding_NOP":sum(n!='NOP;' for n in names),
         "PRMT":sum(n.startswith('PRMT') for n in names),
         "LOP3":sum(n.startswith('LOP3') for n in names),"SHF":sum(n.startswith('SHF') for n in names),
         "IMAD_IADD":sum(n.startswith('IMAD.IADD') for n in names),
         "VIADD":sum(n=='VIADD' for n in names),
         "IADD3":sum(n.startswith('IADD3') for n in names),"HMMA":sum(n.startswith('HMMA') for n in names),
         "LDL_STL":sum(n.startswith(('LDL','STL')) for n in names)}


if __name__=="__main__":main()

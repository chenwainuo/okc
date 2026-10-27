"""Focused compile of changed Tensor exports and two actual GV routes; CPU only.
Local CPU Torch lacks generated CUDA context config. We use real at::Tensor,
CUDA12.8 headers and a declaration-only current-stream seam; no context/link/load.
"""
import ast,hashlib,json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
SOURCE=ROOT/'proj_rsi/candidates/gv_predxor_keepchoice_v1/submission.py'
SHA='244e5b7d6bb8a68b503eebf09dd5c2d6cfba5a579680c24381018c918a802b4a'
TARGETS={51:(1,1,2,2,8,16,1,1,0,1),0:(1,1,2,1,8,16,1,1,0,1)}
def function(text,name):
 start=text.index('int64_t '+name+'(');start=text.rfind('\n',0,start)+1
 brace=text.index('{',start);depth=1;end=brace+1
 while depth:
  if text[end]=='{':depth+=1
  elif text[end]=='}':depth-=1
  end+=1
 return text[start:end]
def main():
 import torch
 from torch.utils.cpp_extension import include_paths
 text=SOURCE.read_text();assert hashlib.sha256(text.encode()).hexdigest()==SHA
 tree=ast.parse(text);cu=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
 prefix=cu[:cu.index('// FP32 router, small m:')]
 prefix=prefix.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>\n#include <ATen/core/Tensor.h>\nnamespace torch {using Tensor=at::Tensor;}\nnamespace at {namespace cuda {cudaStream_t getCurrentCUDAStream();}}').replace('#include <ATen/cuda/CUDAContext.h>','')
 go=function(cu,'gvgo');pred=function(cu,'gvpredrun')
 tail=cu[cu.index('// Keep-choice host control:'):]
 cases='\n'.join('case %d: return gvl<%s>(p,st,occ);'%(i,','.join(map(str,t)))for i,t in TARGETS.items())
 source=(prefix+'\n'+go+'\n'+pred+'\n'+tail).replace('@@CASES@@',cases).replace('@@OLDCASES@@',cases.replace('return gvl<','return gv_keepoldlaunch<'))
 assert '@@'not in source
 path=D/'focused-wrapper.cu';path.write_text(source)
 argv=['/usr/local/cuda-12.8/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
 for include in include_paths():argv+=['-I',include]
 argv+=['-c',str(path),'-o',str(D/'focused-wrapper.o')]
 begin=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=180)
 (D/'compile.log').write_text(p.stdout+p.stderr)
 result={'source_sha256':SHA,'targets':TARGETS,'compiler_argv':argv,'returncode':p.returncode,'elapsed_seconds_observed':time.monotonic()-begin,'local_torch':torch.__version__,'CPU_only':True,'limits':'Compile real local at::Tensor exports with CUDA12.8 and declaration-only at::cuda current stream seam; does not link/load extension, attest remote Torch ABI, initialize CUDA or validate runtime.'}
 (D/'compile-result.json').write_text(json.dumps(result,indent=2)+'\n')
 assert p.returncode==0,p.stderr[-4000:]
 for suffix,flag in [('sass','--dump-sass'),('resources','--dump-resource-usage')]:
  (D/('focused-wrapper.'+suffix)).write_text(subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump',flag,str(D/'focused-wrapper.o')],text=True,timeout=30))
 assert not torch.cuda.is_initialized()
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()

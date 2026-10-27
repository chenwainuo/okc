"""CPU NVCC only; no extension/context/prepare. --full includes all archived GVI."""
import argparse,importlib.util,json,re,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
spec=importlib.util.spec_from_file_location('gv_pred_build',D/'build.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
ARCHIVE=ROOT/'results/proj-rsi/scheduler/artifacts/research.pw45.gpu.01/collection-1793111000996134576/logs/scheduler/pw45-stack-01/run'
FLAGS=['-std=c++17','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']
def prefix(text):
 cu=B.value(text,'GVCU');end=cu.index('// FP32 router, small m:')
 return cu[:end].replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args();parent=B.PARENT.read_text();candidate=(D/'submission.py').read_text();assert B.sha(parent)==B.SHA
 tuples=B.TARGETS
 if args.full:
  archived=(ARCHIVE/'pw38.gv20_7bb50cec.original.cu').read_text();tuples=list(dict.fromkeys(tuple(map(int,s.split(',')))for s in re.findall(r'return gvl<([0-9,]+)>',archived)))
  assert all(t in tuples for t in B.TARGETS)
 result={'state':'CPU_COMPILE_STARTED','full_GVI':args.full,'original_instantiations':len(tuples),'new_scoped_instantiations':len(B.TARGETS),'compiler_flags':FLAGS,'CUDA_contexts':0}
 for label,text in [('control',parent),('candidate',candidate)]:
  source=prefix(text)+'\n'+''.join('template int64_t gvl<'+','.join(map(str,t))+'>(const G&,cudaStream_t,int);\n'for t in tuples)
  stem=label+('-full'if args.full else'-pilot');(D/(stem+'.cu')).write_text(source)
  begin=time.monotonic();p=subprocess.run(['/usr/local/cuda-12.8/bin/nvcc',*FLAGS,'-Xptxas=-v','-c',str(D/(stem+'.cu')),'-o',str(D/(stem+'.o'))],capture_output=True,text=True,timeout=240);(D/(stem+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
  (D/(stem+'.sass')).write_text(subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(D/(stem+'.o'))],text=True,timeout=30))
  (D/(stem+'.resources')).write_text(subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(D/(stem+'.o'))],text=True,timeout=30))
  result[label]={'seconds':time.monotonic()-begin,'stem':stem}
 result['state']='CPU_NVCC_PASS_NO_GPU';(D/('compile-full.json'if args.full else'compile-pilot.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

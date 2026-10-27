"""Offline CUDA compilation only: no context, linking, extension load, prepare or GPU."""
import ast,subprocess,sys,sysconfig
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else BASE
assert not torch.cuda._initialized
flags=['-std=c++17','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a'];nvcc='/usr/local/cuda-12.8/bin/nvcc';dump='/usr/local/cuda-12.8/bin/cuobjdump'
f=next(n for n in ast.parse((D/'adapter.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='module');host=next(ast.literal_eval(n.value)for n in ast.walk(f)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='host'for t in n.targets));(D/'extension_source.cu').write_text((D/'kernel.cuh').read_text()+host)
torchroot=Path(torch.__file__).parent
jobs=[([nvcc,*flags,'-Xptxas=-v','-cubin',str(D/'compile.cu'),'-o',str(D/'device.cubin')],'resource.log'),([nvcc,*flags,'-I'+str(torchroot/'include'),'-I'+str(torchroot/'include/torch/csrc/api/include'),'-I'+sysconfig.get_paths()['include'],'-c',str(D/'extension_source.cu'),'-o',str(D/'extension_compile.o')],'extension_compile.log')]
for cmd,log in jobs:
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(D/log).write_text(p.stdout+p.stderr);assert p.returncode==0,(log,p.stderr[-3000:])
for obj,out in [('device.cubin','sass.txt'),('extension_compile.o','extension_sass.txt')]:
 (D/out).write_text(subprocess.check_output([dump,'--dump-sass',str(D/obj)],text=True,timeout=30))
subprocess.run([sys.executable,str(BASE/'cpu_proof.py'),str(D)],check=True);assert not torch.cuda._initialized

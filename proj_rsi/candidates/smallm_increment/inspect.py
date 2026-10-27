import ast,hashlib,pathlib,subprocess
D=pathlib.Path(__file__).resolve().parent
p=D.parent/'rs32/submission.py'
s=p.read_text(); assert hashlib.sha256(s.encode()).hexdigest()=='a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'
t=ast.parse(s); env={}
for n in t.body:
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RQCU' for x in n.targets): env['RQCU']=ast.literal_eval(n.value)
 if isinstance(n,ast.FunctionDef) and n.name in ('_rqwg','_rqss'): exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),env)
c=env['RQCU'].split('typedef CUresult')[0].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
c=c.replace('@@WGMMA@@',''.join(env['_rqwg'](n) for n in (16,32,128,256))+''.join(env['_rqss'](n) for n in (128,256)))
c+='\ntemplate __global__ void rq<32,5,1,0,0,0,35>(const RqP,__grid_constant__ const RqTmap,__grid_constant__ const RqTmap);\ntemplate __global__ void rq<32,6,1,0,0,0,35>(const RqP,__grid_constant__ const RqTmap,__grid_constant__ const RqTmap);\ntemplate __global__ void rq<16,5,1,0,0,0,35>(const RqP,__grid_constant__ const RqTmap,__grid_constant__ const RqTmap);\n'
(D/'before.cu').write_text(c)
cmd=['/usr/local/cuda/bin/nvcc','-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr','-cubin','-Xptxas=-v',str(D/'before.cu'),'-o',str(D/'before.cubin')]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(D/'before.log').write_text(r.stdout);assert r.returncode==0,r.stdout
sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(D/'before.cubin')],text=True);(D/'before.sass').write_text(sass)
print(r.stdout)

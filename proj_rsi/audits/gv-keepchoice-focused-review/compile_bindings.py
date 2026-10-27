"""Syntax-only real Torch/pybind export bindings; no extension load or CUDA."""
import ast,json,subprocess,sysconfig,time
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
text=(ROOT/'proj_rsi/candidates/gv_predxor_keepchoice_v1/submission.py').read_text();tree=ast.parse(text)
cu=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_gvext')
captured={};env={'GVCU':cu,'GVI':[(1,1,2,2,8,16,1,1,0,1),(1,1,2,1,8,16,1,1,0,1)],'_gvmod':None,'load_inline':lambda *a,**kw:captured.update(kw)or object()}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-factory>','exec'),env);env['_gvext']()
source='#include <torch/extension.h>\n'+captured['cpp_sources']+'\nPYBIND11_MODULE(review_keepchoice,m){\n'+''.join('m.def("'+name+'",&'+name+');\n'for name in captured['functions'])+'}\n'
path=D/'focused-bindings.cpp';path.write_text(source)
argv=['g++','-std=c++20','-fsyntax-only','-O0','-fPIC']
for include in include_paths()+[sysconfig.get_path('include')]:argv+=['-I',include]
argv += [str(path)]
begin=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=120);(D/'bindings.log').write_text(p.stdout+p.stderr)
result={'returncode':p.returncode,'functions':captured['functions'],'compiler_argv':argv,'elapsed_seconds_observed':time.monotonic()-begin,'torch':torch.__version__,'GPU_calls':0,'limits':'Syntax-only pybind/real Tensor declarations; not a linked/loaded GPU extension or remote Torch ABI attestation.'}
(D/'bindings-result.json').write_text(json.dumps(result,indent=2)+'\n');assert p.returncode==0,p.stderr[-3000:];assert not torch.cuda.is_initialized();print(json.dumps(result,indent=2))

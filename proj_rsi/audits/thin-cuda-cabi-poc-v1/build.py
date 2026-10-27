"""Bounded CPU-only compile/link; never loads the CUDA DLL or invokes CUDA."""
import hashlib,json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
p=ROOT/'proj_rsi/candidates/gv53_keyed_half_v1/pilot.cu';raw=p.read_text();assert hashlib.sha256(raw.encode()).hexdigest()=='f79c659ff30e548c51482a6b132c3f48ff5ee81bf7ad6802edfe679bd43be65d'
assert all(x not in raw for x in ['torch/','ATen/','pybind'])
(D/'target.cu').write_text(raw+'\n'+(D/'bridge.cuh').read_text());nvcc='/usr/local/cuda-12.8/bin/nvcc';stages=[]
for name,cmd in [('native_compile',[nvcc,'-c','-std=c++17','-arch=sm_90a','-O3','-lineinfo','-Xptxas=-v','-Xcompiler','-fPIC',str(D/'target.cu'),'-o',str(D/'target.o')]),('host_link',[nvcc,'-shared','-cudart','shared','-Xcompiler','-fPIC',str(D/'target.o'),'-o',str(D/'libgv53_thin.so')])]:
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);stages.append({'name':name,'argv':cmd,'seconds':time.monotonic()-t,'exit':r.returncode});(D/(name+'.stdout')).write_text(r.stdout);(D/(name+'.stderr')).write_text(r.stderr);assert r.returncode==0,r.stderr
r=subprocess.run(['nm','-D','--defined-only',str(D/'libgv53_thin.so')],capture_output=True,text=True,check=True);(D/'exports.txt').write_text(r.stdout)
assert all(' '+name+'\n'in r.stdout for name in ['gv53_run','gv53_attrs','gv53_error'])
record={'state':'CPU_COMPILE_LINK_ONLY_GPU_UNRUN','source_pilot':str(p.relative_to(ROOT)),'source_pilot_sha256':hashlib.sha256(raw.encode()).hexdigest(),'stages':stages,'total_compile_link_seconds':sum(x['seconds']for x in stages),'CUDA_version':subprocess.run([nvcc,'--version'],capture_output=True,text=True,check=True).stdout,'shared_library_sha256':hashlib.sha256((D/'libgv53_thin.so').read_bytes()).hexdigest(),'Torch_headers_or_libraries_used':False,'CUDA_DSO_loaded':False,'GPU_calls':0,'limits':['LocalCUDA12.8 CPU compile/link; targetimage launch/ABI/graph capture/numerics/occupancy untested.','Not a paired build comparison with remote Torch load_inline; savings unknown.']};(D/'build-proof.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

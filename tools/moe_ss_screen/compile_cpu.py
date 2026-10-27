"""Compile-only diagnostic host wrapper proof: no torch import/context/load/link."""
import ast,hashlib,importlib.util,json,re,subprocess,sysconfig
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2'
def main():
 tree=ast.parse((ROOT/'tools/moe_ss_pipeline_screen.py').read_text())
 diagnostic=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='DIAGNOSTIC'for t in n.targets))
 cu=(SOURCE/'extension_source.cu').read_text()+diagnostic
 out=HERE/'offline-v2';out.mkdir(exist_ok=False);(out/'screen.cu').write_text(cu)
 torchroot=Path(importlib.util.find_spec('torch').origin).parent
 cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-I'+str(torchroot/'include'),'-I'+str(torchroot/'include/torch/csrc/api/include'),'-I'+sysconfig.get_paths()['include'],'-c',str(out/'screen.cu'),'-o',str(out/'screen.o')]
 process=subprocess.run(cmd,text=True,capture_output=True,timeout=240);(out/'compile.log').write_text(process.stdout+process.stderr)
 assert process.returncode==0,process.stderr[-6000:]
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(out/'screen.o')],text=True,timeout=60);(out/'screen.sass').write_text(sass)
 def streams(s):return {t.splitlines()[0].strip():[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)]for t in s.split('Function : ')[1:]if '_ZN9moe_ss_v27project' in t.splitlines()[0]}
 assert streams(sass)==streams((SOURCE/'sass.txt').read_text()),'appendeddiagnostic changed production streams'
 resources=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(SOURCE/'device.cubin')],text=True,timeout=60);(out/'canonical.resources').write_text(resources)
 actual_resources=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(out/'screen.o')],text=True,timeout=60);(out/'screen.resources').write_text(actual_resources)
 proof={'state':'COMPILE_ONLY_PASS_GPU_UNRUN','source_sha256':hashlib.sha256(cu.encode()).hexdigest(),'same_production_streams':len(streams(sass)),'command':cmd,'torch_imported':False,'CUDA_calls':0,'link_or_extension_load':False}
 (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()

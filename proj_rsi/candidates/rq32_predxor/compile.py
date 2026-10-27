"""CPU-only offline CUDA cubin discriminator; never loads a CUDA extension/context."""
import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
TARGET={(32,5,1,0,0,0,35),(32,6,1,0,0,0,35)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value(text,name):
 for node in ast.parse(text).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
 raise KeyError(name)
def device(text):
 env={}
 for node in ast.parse(text).body:
  if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss'):exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure WGMMA macro factory>','exec'),env)
 cu=value(text,'RQCU').split('typedef CUresult (*RqEnc)')[0]
 cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
 cu=cu.replace('@@WGMMA@@',''.join(env['_rqwg'](n) for n in (16,32,128,256))+env['_rqss'](128)+env['_rqss'](256))
 variants={v[0]:v for v in value(text,'RQV')};names=value(text,'RQPROD')+list(dict.fromkeys(name for choices in value(text,'RQSBAKE').values() for name,ks in choices))
 cu+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
 for name in names:
  spec=','.join(map(str,variants[name][1:8]));cu+=f'rq<{spec}><<<dim3(18,7),384,RqC<{spec}>::SMEM>>>(p,x,o);\n'
 return cu+'}\n'
def streams(sass):
 out={}
 for table in re.split(r'\n\s*Function : ',sass)[1:]:
  key=tuple(map(int,re.findall(r'Li(\d+)E',table.splitlines()[0])));assert key not in out
  out[key]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
 return out
def ops(rows):return collections.Counter(re.sub(r'^@!?\w+\s+','',ins).split()[0] for pc,ins in rows)
def registers(log):
 out={};current=None
 for line in log.splitlines():
  m=re.search(r"Compiling entry function '([^']+)'",line)
  if m:current=tuple(map(int,re.findall(r'Li(\d+)E',m[1])))
  m=re.search(r'Used (\d+) registers',line)
  if m:out[current]=int(m[1])
 return out

def run():
 parent=HERE.parent/'pw45';old=(parent/'submission.py').read_text();new=(HERE/'submission.py').read_text()
 assert device(old)==(parent/'device_source.cu').read_text(),'offline parent factory must reconstruct exactly'
 cu=device(new);(HERE/'device_source.cu').write_text(cu)
 result=subprocess.run(['/usr/local/cuda-12.8/bin/nvcc',*FLAGS,'-Xptxas=-v','-cubin',str(HERE/'device_source.cu'),'-o',str(HERE/'device.sm90a.cubin')],capture_output=True,text=True,timeout=240)
 log=result.stdout+result.stderr;(HERE/'device-resource.log').write_text(log);assert result.returncode==0,log
 assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log),log
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/'device.sm90a.cubin')],text=True,timeout=60);(HERE/'sass.txt').write_text(sass)
 resources=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(HERE/'device.sm90a.cubin')],text=True,timeout=60);(HERE/'resources.txt').write_text(resources)
 a,b=streams((parent/'sass.txt').read_text()),streams(sass);assert set(a)==set(b) and len(a)==21
 regs=registers(log);assert set(regs)==set(a)
 for args in set(a)-TARGET:assert a[args]==b[args],('unrelated main changed',args)
 records={};changed=[]
 for args in sorted(TARGET):
  before,after=ops(a[args]),ops(b[args]);assert regs[args]<=168 and ((regs[args]+7)//8)*8*384<=65536
  for opcode in set(before)|set(after):
   if opcode.startswith(('HGMMA','WARPGROUP','SYNCS','LDS','UTMA','UBLKCP','DEPBAR','STSM','STG','ATOM','RED','TRAP','BAR','FENCE','USETMAXREG')):assert before[opcode]==after[opcode],(args,opcode,before[opcode],after[opcode])
  if a[args]!=b[args]:changed.append(args)
  records[','.join(map(str,args))]={'before_static_instructions':len(a[args]),'after_static_instructions':len(b[args]),'registers':regs[args],'local_or_spills':0,'shared_bytes':1024+args[1]*(4096+12288)+24*args[1]+16,'all_sync_math_LDS_TMA_store_opcode_counts_unchanged':True,'opcode_deltas':{op:[before[op],after[op]] for op in sorted(set(before)|set(after)) if before[op]!=after[op]},'predicated_XOR_sites':[{'pc':hex(pc),'instruction':ins} for pc,ins in b[args] if ins.startswith('@') and 'LOP3.LUT' in ins and ', 0x3c,' in ins]}
 output={'state':'CPU_COMPILED_DIFFERENT_SASS_GPU_UNRUN' if changed else 'STOP_NVCC_REPRODUCES_PARENT','changed_types':changed,'unchanged_exact_other_mains':19,'parent_factory_exact':True,'targets':records,'flags':FLAGS,'source_sha256':sha(HERE/'submission.py'),'cubin_sha256':sha(HERE/'device.sm90a.cubin'),'GPU_contexts':0,'extensions_loaded':0}
 (HERE/'compile-proof.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':run()

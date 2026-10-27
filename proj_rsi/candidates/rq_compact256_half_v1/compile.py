"""Exactly one offline SM90a kernel compile. Never loads CUDA/Torch."""
from pathlib import Path
import importlib.util,subprocess,json,re,hashlib,time
D=Path(__file__).resolve().parent;R=D.parents[2]
s=importlib.util.spec_from_file_location('rq_cpu_helpers',R/'proj_rsi/candidates/rq32_predxor/compile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
T=(32,6,1,0,0,0,262195);t=','.join(map(str,T))
cu=m.device((D/'submission.py').read_text()).split('void instantiate(')[0]+f'\nvoid instantiate(RqP p,RqTmap x,RqTmap o){{rq<{t}><<<dim3(128,2),256,RqC<{t}>::SMEM>>>(p,x,o);}}\n'
(D/'device.cu').write_text(cu);start=time.monotonic()
p=subprocess.run(['/usr/local/cuda-12.8/bin/nvcc',*m.FLAGS,'-Xptxas=-v','-cubin',str(D/'device.cu'),'-o',str(D/'device.cubin')],capture_output=True,text=True,timeout=120)
log=p.stdout+p.stderr;(D/'resources.log').write_text(log)
o={'state':'CPU_COMPILE_FAILED','compile_returncode':p.returncode,'elapsed_seconds':time.monotonic()-start,'one_offline_kernel_compiled':list(T),'GPU_context':False}
if p.returncode==0:
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(D/'device.cubin')],text=True)
 (D/'device.sass').write_text(sass)
 usage=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(D/'device.cubin')],text=True);(D/'resources.txt').write_text(usage)
 rows=m.streams(sass)[T];regs=m.registers(log)[T];spill=bool(re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log))
 counts=m.ops(rows);rounded=((regs+7)//8)*8
 o.update(state='CPU_COMPILED_COMPACT_RESOURCE_POINT_GPU_UNRUN' if regs<=128 and not spill else 'CPU_COMPILED_RESOURCE_COUNTEREVIDENCE',registers=regs,rounded_registers=rounded,threads=256,entry_CTA_register_estimate=rounded*256,register_CTA_limit_estimate=65536//(rounded*256),dynamic_register_budget=128*(216+40),shared_request_bytes=1024+6*(4096+6144)+24*6+16,spill_or_stack_nonzero=spill,resource_log=log,static_instructions=len(rows),opcode_counts=dict(counts),USETMAXREG=[{'pc':hex(pc),'instruction':ins} for pc,ins in rows if 'SETMAXREG' in ins],local_memory_instructions=[{'pc':hex(pc),'instruction':ins}for pc,ins in rows if re.search(r'\b(?:LDL|STL)\b',ins)],limits='Register arithmetic is an estimate, not cudaOccupancyMaxActiveBlocksPerMultiprocessor or actual scheduling. No runtime outputs or lifetime validation performed.',source_sha256=hashlib.sha256((D/'submission.py').read_bytes()).hexdigest(),cubin_sha256=hashlib.sha256((D/'device.cubin').read_bytes()).hexdigest())
(D/'compile-proof.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({k:v for k,v in o.items()if k not in ['opcode_counts','resource_log']},indent=2))

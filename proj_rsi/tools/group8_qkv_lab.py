#!/usr/bin/env python3
"""Frozen CPU research admission lab. Runtime entrypoints exposed by runtime.Pipeline.
No CUDA call, extension load, prepare or benchmark occurs under --cpu-check.
"""
import argparse,hashlib,importlib.util,json,sys,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'candidates/group8_qkv_trial'
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
R=module(H/'runtime.py','qkv_trial_runtime')
def cpu_check():
 import torch
 torch.set_num_threads(2)
 # Trap accidental CUDA entrypoints, even on a CUDA-enabled installation.
 def forbidden(*a,**k):raise AssertionError('CPU check attempted CUDA')
 for name in ('_lazy_init','init','is_available','device_count','synchronize','current_device'):
  setattr(torch.cuda,name,forbidden)
 source=R.source();(H/'extension_source.cu').write_text(source)
 P=module(R.PARENT/'proofs.py','qkv_frozen_proofs');proof=P.proof()
 c=R.codec()
 # Every BF16 bit pattern incl signs, infinities and NaNs, in real fragment pack.
 bits=torch.arange(65536,dtype=torch.int32).reshape(128,512)
 w=bits.to(torch.int16).view(torch.bfloat16)
 packed=c['_rqpack'](w);assert packed is not None
 lay,tab,esc=packed
 assert torch.equal(c['_rqunpack'](lay,tab,esc,128,512),bits)
 # All global sequence bits 0..127 and all entry fields; local seq is ch%32.
 for seq in range(128):
  for lane in range(32):
   for reg in range(16):
    for half in range(2):
     for xr in (0,1,63,64,127):
      ent=(seq<<17)|(lane<<12)|(reg<<8)|(half<<7)|xr
      assert (ent>>17,(ent>>12)&31,(ent>>8)&15,(ent>>7)&1,ent&127)==(seq,lane,reg,half,xr)
 # Actual pinned _rqesc; unsorted varying-density escapes span all slices.
 rows=[]
 for ch in range(128):
  for lane in range(32):
   rows.append((ch%144,ch,ch%4,lane,(ch+lane)%8,1+(ch+lane)%127))
 e=torch.tensor(rows,dtype=torch.int64);csr,_=c['_rqesc'](e,18,128,4,'cpu');cap=(csr.numel()-32)//576
 assert cap%32==0 and cap<=32800
 expected={}
 for tile,ch,q,ln,v,xr in rows:
  slot=((ch//32)*18+tile//8)*8+tile%8
  expected.setdefault(slot,[]).append(((ch%32)<<17)|(ln<<12)|((q*4+v//2)<<8)|((v%2)<<7)|xr)
 for slot in range(576):
  data=[int(x)&0xffffffff for x in csr[slot*cap:(slot+1)*cap]];stop=data.index(0xffffffff)
  assert sorted(data[:stop])==sorted(expected.get(slot,[]))
  assert all(x==0xffffffff for x in data[stop:])
 assert torch.all(csr[-32:]==-1)
 # Empty metadata guard and fixed bounds.
 empty,_=c['_rqesc'](torch.empty((0,6),dtype=torch.int64),18,128,4,'cpu')
 assert empty.numel()==576*32+32 and torch.all(empty==-1)
 result={'status':'CPU_PASS_GPU_UNRUN','gpu_calls':0,'official_submissions':0,'extension_identity':R.identity(),'device_source_sha256':R.PIN,'helper_sha256':R.HELPER_PIN,'proofs':proof,'codec_all_65536_BF16_patterns':'bit exact','CSR_global_seq_0_127_all_fields':'pass','CSR_pinned_helper_split4_and_empty':'pass','synthetic_CSR_capacity':cap,'actual_task_W_codec':'MANDATORY_RUNTIME_GATE_UNRUN','wrapper_compile':'not yet checked','runtime_admission':'HOLD independent source/resource critic and actual JIT SASS; CPU pass is not numerical/timing admission'}
 return result

def compile_host():
 import sysconfig,torch
 inc=Path(torch.__file__).parent/'include'
 cmd=['/usr/local/cuda-12.8/bin/nvcc','-c','-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','-O3','--expt-relaxed-constexpr','-Xptxas=-v','-I'+str(inc),'-I'+str(inc/'torch/csrc/api/include'),'-I'+sysconfig.get_paths()['include'],str(H/'extension_source.cu'),'-o',str(H/'extension.cpu.o')]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(H/'host_compile.log').write_text(r.stdout+r.stderr)
 assert r.returncode==0,'offline host compile failed: retained host_compile.log'
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(H/'extension.cpu.o')],text=True);(H/'host.sass').write_text(sass)
 import re
 L=module(ROOT/'candidates/operand_reverse_group8/liveness.py','qkv_trial_liveness')
 table=next(t for t in re.split(r'\n\s*Function : ',sass)[1:] if 'rq_qkv_g8_sk4' in t.splitlines()[0]);return L.check_one(table)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true',required=True);p.add_argument('--compile-host',action='store_true');a=p.parse_args()
 result=cpu_check()
 if a.compile_host:
  try:result['host_physical']=compile_host();result['wrapper_compile']='PASS offline only'
  except Exception as e:result['wrapper_compile']='FAIL retained';result['compile_error']=str(e)
 (H/'cpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
 files={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.name!='manifest.json' and '__pycache__' not in f.parts and f.name!='check.stdout'}
 manifest={'status':result['status'],'extension_identity':R.identity(),'extension_source_sha256':hashlib.sha256(R.source().encode()).hexdigest(),'lab_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':files,'gpu_calls':0,'official_submissions':0}
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(result,indent=2))
 if result['wrapper_compile']=='FAIL retained':sys.exit(1)
if __name__=='__main__':main()

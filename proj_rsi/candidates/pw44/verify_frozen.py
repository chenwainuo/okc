"""Read-only PW44 closure and pure proof reproduction; no compilation/import torch."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
 m=json.loads((HERE/'manifest.json').read_text())
 for f,h in m['files'].items():assert sha(HERE/f)==h,f
 assert sha(HERE/'submission.py')==m['source_sha256']
 for f,h in m['external_pins'].items():assert sha(ROOT/f)==h,f
 from proj_rsi.candidates.pw44.cpu_proof import source_proof,physical
 from proj_rsi.candidates.pw44.protocol_proof import proof as protocol
 from proj_rsi.candidates.pw44.codegen_proof import proof as codegen
 from proj_rsi.candidates.pw44.learning_proof import proof as learning
 p=json.loads((HERE/'cpu-proof.json').read_text());assert source_proof()==p['source_proof'];assert json.dumps(physical((HERE/'sass.txt').read_text()),sort_keys=True)==json.dumps(p['physical'],sort_keys=True)
 for f,fn in [('protocol-proof.json',protocol),('codegen-proof.json',codegen),('learning-proof.json',learning)]:assert json.dumps(fn(),sort_keys=True)==json.dumps(json.loads((HERE/f).read_text()),sort_keys=True),f
 assert not p.get('failure') and p['guard_host_compiled'] and p['CUDA_calls']==0
 return {'state':'PW44_FROZEN_CPU_CLOSURE_PASS_GPU_UNRUN','source_sha256':m['source_sha256'],'files':len(m['files']),'external_pins':len(m['external_pins']),'GPU_calls':0,'production_imports':0,'NVCC_calls':0}
if __name__=='__main__':print(json.dumps(check(),indent=2))

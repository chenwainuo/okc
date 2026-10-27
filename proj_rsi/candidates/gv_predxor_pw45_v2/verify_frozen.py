"""Read-only closure verification; no torch/CUDA/runtime import."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
def main():
 m=json.loads((D/'manifest.json').read_text());count=0
 for key in ('files','external_inputs'):
  for row in m[key]:
   p=(D if key=='files' else ROOT)/row['path'];assert p.is_file(),str(p);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p);count+=1
 print(json.dumps({'state':'PASS','pins':count,'manifest_sha256':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'gpu_actions':0}))
if __name__=='__main__':main()

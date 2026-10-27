"""Read-only exact lab closure; no production/CUDA imports."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
m=json.loads((D/'manifest.json').read_text())
for rel,pin in m['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==pin,rel
print(json.dumps({'state':'PASS','pins':len(m['files']),'manifest_sha256':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'GPU_calls':0}))

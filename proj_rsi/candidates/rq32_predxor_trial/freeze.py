"""CPU-owned trial freeze only. No queue, torch, GPU, production import or API."""
from pathlib import Path
import difflib,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
files={}
def pin(p):
 p=Path(p);files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def kernel(name):
 base=ROOT/'proj_rsi/candidates'/name;m=json.loads((base/'manifest.json').read_text());pin(base/'manifest.json')
 for rel in m['files']:pin(base/rel)
 externalbase=ROOT if m.get('external_files_base')=='repository_root' else ROOT/'proj_rsi'
 for rel in m.get('external_files',{}):pin(externalbase/rel)
for name in ('rq32_predxor','pw45','pw38','rs32_eh_backport'):kernel(name)
old=ROOT/'proj_rsi/candidates/pw45_trial/manifest.json';pin(old)
for f in json.loads(old.read_text())['files']:pin(ROOT/f)
for rel in ('proj_rsi/scheduling/reviews/rq32-predxor-source-01.md','proj_rsi/audits/rq32-predxor-source-independent/proof.json','proj_rsi/correctness.py','muse_bf16_proj/reference.py','muse_bf16_proj/checks.py','proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh','proj_rsi/tools/rq32_predxor_lab.py'):
 p=ROOT/rel
 if p.exists():pin(p)
for before,after,target in ((ROOT/'proj_rsi/tools/pw45_stack_lab.py',ROOT/'proj_rsi/tools/rq32_predxor_lab.py',HERE/'driver-diff.patch'),(ROOT/'proj_rsi/candidates/pw45_trial/support.py',HERE/'support.py',HERE/'support-diff.patch')):
 target.write_text(''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile=str(before.relative_to(ROOT)),tofile=str(after.relative_to(ROOT)))))
for p in HERE.rglob('*'):
 if p.is_file() and '__pycache__' not in str(p) and not(p.parent==HERE and p.name in ('manifest.json','job-proposed.json','cpu-proof.json')):pin(p)
m={'version':'rq32-predxor-runtime-v1','state':'CPU_FROZEN_GPU_UNRUN_INDEPENDENT_RUNTIME_REVIEW_REQUIRED','files_base':'repository_root','files':dict(sorted(files.items())),'bounds':{'inner_seconds':900,'outer_seconds':1020,'profile':False},'scope':{'actual_policies':2,'timed_shapes':30,'actual_external_cases':148,'actual_raw_calls':3810,'actual_terminal_oracles':120,'actual_mutations':240,'actual_affected_M32_basis_routes':4,'fixed_shapes':2,'fixed_arms_per_shape':4,'fixed_raw_calls':590,'fixed_terminal_oracles':16,'fixed_mutations':32,'fixed_fresh_escape_W_entries_per_shape':1026},'GPU_calls':0}
p=HERE/'manifest.json';p.write_text(json.dumps(m,indent=2)+'\n')
r=subprocess.run([sys.executable,str(ROOT/'proj_rsi/tools/rq32_predxor_lab.py'),'--cpu-check'],capture_output=True,text=True,check=True,cwd=ROOT);(HERE/'cpu-proof.json').write_text(r.stdout);pin(HERE/'cpu-proof.json');m['files']=dict(sorted(files.items()));p.write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'manifest':hashlib.sha256(p.read_bytes()).hexdigest(),'driver':hashlib.sha256((ROOT/'proj_rsi/tools/rq32_predxor_lab.py').read_bytes()).hexdigest(),'files':len(files)}))

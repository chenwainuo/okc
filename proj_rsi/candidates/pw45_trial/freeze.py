"""Rebuild only this CPU-owned trial manifest; does not import production or use a queue."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];TRIAL=Path(__file__).resolve().parent
files={}
def pin(p):
 p=Path(p);files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('pw45','pw38','rs32_eh_backport'):
 base=ROOT/'proj_rsi/candidates'/name;m=json.loads((base/'manifest.json').read_text());pin(base/'manifest.json')
 for rel in m['files']:pin(base/rel)
 for rel in m.get('external_files',{}):pin(ROOT/'proj_rsi'/rel)
for rel in ['proj_rsi/candidates/rs32/submission.py','proj_rsi/scheduling/reviews/pw45-source-01.md','proj_rsi/audits/pw45-independent.json','proj_rsi/audits/pw45_independent.py','proj_rsi/correctness.py','muse_bf16_proj/reference.py','muse_bf16_proj/checks.py','proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh','proj_rsi/learning_archive/registry-v3-43ede5d0.json','proj_rsi/tools/pw45_stack_lab.py']:pin(ROOT/rel)
for p in TRIAL.rglob('*'):
 if p.is_file() and '__pycache__' not in str(p) and not (p.parent==TRIAL and p.name in ('manifest.json','job-proposed.json','cpu-proof.json')):pin(p)
manifest={'version':'pw45-stack-lab-v5-profile-clone-drain','state':'CPU_FROZEN_GPU_UNRUN_INDEPENDENT_REVIEW_REQUIRED','files_base':'repository_root','files':dict(sorted(files.items())),'GPU_calls':0,'bounds':{'inner_seconds':3600,'outer_seconds':3720,'profile':False},'scope':{'actual_timed_shapes':30,'actual_raw_calls':4350,'actual_terminal_oracles':180,'actual_external_cases':222,'actual_mutations':360,'fixed_raw_calls':210,'fixed_terminal_oracles':6,'fixed_mutations':12,'basis_columns_per_o32_policy':1024,'o32_policies_with_full_basis':6}}
path=TRIAL/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
result=subprocess.run([sys.executable,str(ROOT/'proj_rsi/tools/pw45_stack_lab.py'),'--cpu-check'],capture_output=True,text=True,check=True,cwd=ROOT)
(TRIAL/'cpu-proof.json').write_text(result.stdout);pin(TRIAL/'cpu-proof.json');manifest['files']=dict(sorted(files.items()));path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(files),'manifest':hashlib.sha256(path.read_bytes()).hexdigest(),'driver':hashlib.sha256((ROOT/'proj_rsi/tools/pw45_stack_lab.py').read_bytes()).hexdigest()}))

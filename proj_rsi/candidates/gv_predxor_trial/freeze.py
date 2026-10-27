"""CPU freeze new lab only. No queue/API/GPU imports or launch."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;CAND=ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2';files={}
def pin(p):p=Path(p);files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
# Reuse the known fullpolicy protocol closure, pinning rather than editing it.
base=ROOT/'proj_rsi/candidates/pw45_trial/manifest.json';pin(base)
for rel in json.loads(base.read_text())['files']:pin(ROOT/rel)
pin(CAND/'manifest.json');m=json.loads((CAND/'manifest.json').read_text())
for row in m['files']:pin(CAND/row['path'])
for row in m['external_inputs']:pin(ROOT/row['path'])
for p in D.rglob('*'):
 if p.is_file()and '__pycache__'not in str(p)and not(p.parent==D and p.name in ('manifest.json','cpu-proof.json','job-proposed.json')):pin(p)
pin(ROOT/'proj_rsi/tools/gv_predxor_lab.py')
pin(ROOT/'proj_rsi/scheduling/reviews/gv-predxor-pw45-v2-source.md')
for rel in ('proj_rsi/audits/gv-predxor-independent/review.py','proj_rsi/audits/gv-predxor-independent/v2.json','proj_rsi/audits/gv-predxor-independent/source-review.md'):pin(ROOT/rel)
base=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.02/collection-1793115060902266190/logs/scheduler/pw38-hotpath-trace-02/run'
for name in ('capture.json','JIT01.gv20_7bb50cec.sass','JIT01.gv20_7bb50cec.resources','JIT01.gv20_7bb50cec.cu','JIT01.gv20_7bb50cec.so'):pin(base/name)
manifest={'version':'gv-predxor-pw45-v2-lab-v3-postprepare-fixedpacks','state':'CPU_FROZEN_GPU_UNRUN_REVIEW_REQUIRED','files_base':'repository_root','files':dict(sorted(files.items())),'GPU_calls':0,'bounds':{'inner_seconds':900,'outer_seconds':1020,'profile':False},'scope':{'fullpolicy_shapes':30,'external_cases':148,'fullpolicy_raw_calls':2910,'fullpolicy_terminal_oracles':120,'fullpolicy_mutations':240,'fixed_geometry_pairs':11,'fixed_arms':3,'fixed_raw_calls':1716,'fixed_terminal_oracles':66,'fixed_mutations':165,'fixed_basis_columns':'allK4096or8192 perarm/geometry','actual_o32_fullbasis_policies':2},'promotion':'All30freshpolicy controls required; fixed-arm gain alone cannot promote; all tiny signed outcomes retained'}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
p=subprocess.run([sys.executable,str(ROOT/'proj_rsi/tools/gv_predxor_lab.py'),'--cpu-check','--cpu-artifacts',str(D)],text=True,capture_output=True,cwd=ROOT);(D/'cpu-check.stdout').write_text(p.stdout);(D/'cpu-check.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
(D/'cpu-proof.json').write_text(p.stdout)
for p in D.rglob('*'):
 if p.is_file()and '__pycache__'not in str(p)and not(p.parent==D and p.name in ('manifest.json','job-proposed.json')):pin(p)
manifest['files']=dict(sorted(files.items()));(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'manifest':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'driver':hashlib.sha256((ROOT/'proj_rsi/tools/gv_predxor_lab.py').read_bytes()).hexdigest(),'pins':len(files)}))

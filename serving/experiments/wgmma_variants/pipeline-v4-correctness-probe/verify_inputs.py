"""Host CPU preflight: source/runner/tree pins, root lease; no GPU work."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('baseline','manifest','source','runner','lease','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--experiment',required=True)
a=p.parse_args();pins=json.loads(Path(__file__).with_name('source-pins.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(a.manifest)==pins['baseline_manifest_sha256'],'baseline manifest changed'
m=json.loads(a.manifest.read_text());assert m['archive_sha256']==pins['baseline_archive_sha256']
for f,h in pins['source_files'].items():assert sha(a.source/f)==h,f
for f,h in pins['runner_files'].items():assert sha(a.runner/f)==h,f
sys.path.insert(0,str(a.runner))
from rsi_bundle import verify_tree
verify_tree(a.baseline,m)
lease=json.loads(a.lease.read_text())
assert isinstance(lease,dict) and lease.get('experiment')==a.experiment,'matching root lease required'
remaining=lease['deadline_unix']-time.time()
assert remaining>=285,'need240s run+15s forcedkill+25s cleanup+5s margin inside live lease'
a.output.write_text(json.dumps({'status':'verified','pins':pins,'lease':{k:v for k,v in lease.items() if k!='token'},'remaining_seconds':remaining},indent=2)+'\n')

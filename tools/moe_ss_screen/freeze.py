"""CPU-only deterministic screen closure. No torch/candidate imports."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 original=json.loads((SOURCE/'manifest.json').read_text())
 for rel,pin in original['files'].items():assert sha(SOURCE/rel)==pin,('source freeze drift',rel)
 for rel,pin in original['external_pins'].items():assert sha(ROOT/rel)==pin,('source external drift',rel)
 paths={str((SOURCE/p).relative_to(ROOT))for p in original['files']}
 paths.update(original['external_pins']);paths.add(str((SOURCE/'manifest.json').relative_to(ROOT)))
 paths.update(['tools/moe_ss_pipeline_screen.py','proj_rsi/tools/moe_ss_pipeline_screen.py','proj_rsi/tools/h100_container.sh','proj_rsi/tools/h100_trial.sh','muse_moe_w4a16/nvfp4.py','muse_moe_w4a16/reference.py','proj_rsi/scheduling/reviews/moe-ss-pipeline-source-v2-01.md'])
 paths.update(str(p.relative_to(ROOT))for p in HERE.rglob('*')if p.is_file()and'__pycache__'not in str(p)and p.name not in ('manifest.json','cpu-proof.json'))
 result={'schema_version':1,'state':'CPU_FROZEN_EXECUTABLE_SCREEN_REVIEW_PENDING_GPU_UNRUN','source_manifest_sha256':sha(SOURCE/'manifest.json'),'files':{p:sha(ROOT/p)for p in sorted(paths)},'transfer':{'sum_pinned_file_bytes':sum((ROOT/p).stat().st_size for p in paths),'time_seconds':None,'note':'Existing image, no new wheel/image expected; actual transfer/cache/bandwidth unknown, independently recorded by root.'},'scope':'isolated stage correctness/measurement screen; no current/frozen full-expert comparison and no3x/full-MoE/official claim','image':'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel','cost':{'CPU_minutes_additional':90,'GPU_inner_seconds':600,'GPU_outer_seconds':720,'actual_compile_prepare_transfer_seconds':None,'full14_holdout_and_official_followup_seconds':None},'review_binding':'Separate review SHA bound in descriptor+argv after independent CPU review; no self-hash cycle','CUDA_calls':0}
 (HERE/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(sha(HERE/'manifest.json'))
if __name__=='__main__':main()

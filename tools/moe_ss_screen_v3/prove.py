"""Stdlib-only verification of preserved actual failure and immutable repair."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('screen_v3_proof',ROOT/'tools/moe_ss_pipeline_screen_v3.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 cpu=s.cpu_check();e=json.loads((HERE/'failure-evidence.json').read_text())
 for p,h in e['evidence_sha256'].items():assert sha(ROOT/p)==h,('actual evidence drift',p)
 actual=ROOT/e['result'];r=json.loads(actual.read_text());assert r['state']=='INVALID_SCREEN_NO_PERFORMANCE_ADMISSION' and r['checks']==r['timings']==[] and r['graph_replays']==0
 assert s.resource_tables((actual.parent/'actual.resources').read_text())==s.resource_tables((s.OFFLINE/'canonical.resources').read_text())
 assert (actual.parent/'actual.cu').read_text()==(s.SOURCE/'extension_source.cu').read_text()+s.DIAGNOSTIC
 assert sha(actual.parent/'actual.sass')==sha(s.OFFLINE/'screen.sass')
 row=e['actual_outcome']['actual_resource_failure'];canonical=s.resource_tables((s.OFFLINE/'canonical.resources').read_text())[8,8]
 assert s.resource_gate(row,canonical,2*(4096+8*64)*2+32+1024)
 old=ast.parse((ROOT/'tools/moe_ss_pipeline_screen.py').read_text());new=ast.parse((ROOT/'tools/moe_ss_pipeline_screen_v3.py').read_text())
 # Only resource admission in run() differs; CUDA diagnostic byte identity is
 # additionally proved against the actual failed run above.
 oldf={n.name:ast.dump(n)for n in old.body if isinstance(n,ast.FunctionDef)};newf={n.name:ast.dump(n)for n in new.body if isinstance(n,ast.FunctionDef)}
 assert {k for k in oldf if oldf[k]!=newf[k]}=={'run'} and set(newf)-set(oldf)=={'resource_gate'}
 assert 'torch' not in sys.modules
 proof={'state':'CPU_PASS_V3_GPU_UNRUN_REVIEW_PENDING','cpu_check':cpu,'actual_failed_evidence_sha256':e['evidence_sha256'],'actual_augmented_CUDA_source_unchanged':True,'actual_all8_canonical_resources_exact':True,'actual_augmented_SASS_matches_offline_exact':True,'old_functions_changed':['run'],'new_functions':['resource_gate'],'torch_imported':False,'CUDA_calls':0,'scope':'stage-only; original fullMoE3x goal unvalidated'}
 (HERE/'cpu-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()

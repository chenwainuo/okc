"""Independent focused read-only source/compile receipt audit; CPU only."""
import ast,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
SOURCE=ROOT/'proj_rsi/candidates/gv_predxor_keepchoice_v1/submission.py'
SHA='244e5b7d6bb8a68b503eebf09dd5c2d6cfba5a579680c24381018c918a802b4a'
MANIFEST='5868701f983fe7488d09906be64afbbeaed06f5d468d21e37a5e24d53a655ef6'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value(text,name):return next(ast.literal_eval(n.value)for n in ast.parse(text).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
def main():
 assert digest(SOURCE)==SHA
 m=SOURCE.parent/'manifest.json';assert digest(m)==MANIFEST
 spec=importlib.util.spec_from_file_location('review_compile_source',D/'compile_wrapper.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
 new=SOURCE.read_text();newcu=value(new,'GVCU');oldcu=value((ROOT/'proj_rsi/candidates/pw45/submission.py').read_text(),'GVCU');parentcu=value((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text(),'GVCU')
 assert newcu.replace('int64_t gvpredrun(', 'int64_t gvrun(',1).startswith(parentcu)
 assert C.function(newcu,'gv_keepoldlaunch').replace('gv_keepoldlaunch','gvl')==C.function(oldcu,'gvl')
 assert C.function(newcu,'gv_keepoldgo').replace('gv_keepoldgo','gvgo').replace('gv_keepoldlaunch','gvl').replace('@@OLDCASES@@','@@CASES@@')==C.function(oldcu,'gvgo')
 assert C.function(newcu,'gvrun').replace('gv_keepoldgo','gvgo')==C.function(oldcu,'gvrun')
 compile_receipt=json.loads((D/'compile-result.json').read_text());binding_receipt=json.loads((D/'bindings-result.json').read_text())
 assert compile_receipt['returncode']==binding_receipt['returncode']==0 and compile_receipt['source_sha256']==SHA
 assert binding_receipt['functions']==['gvpredrun','gvrun','gvocc','gvregs','rrrun','rsrun','prrun']
 helper=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)and n.name=='_gvpred_confirm')
 helper_source=ast.get_source_segment(new,helper)
 assert 'deadline = _T0 + PREPARE_BUDGET_S'in helper_source and 'if effect > 0:'in helper_source
 assert 'tuple(p) + ("gvpredxor",)'in helper_source
 assert 'if rp is not None and rp["n"] == n:'in helper_source
 assert helper_source.count('if time.time() >= deadline:')==4
 assert 'sample[index] = _tm(lambda: invoke(index),flush,9)'in helper_source
 assert '((0,1) if block % 2 == 0 else (1,0))'in helper_source
 assert 'torch.equal(x.view(torch.uint8),xb.view(torch.uint8))'in helper_source
 assert new.index('_gvpred_confirm(st, torch.empty(')<new.index('st.pop("gvc12", None)')
 out={'state':'GO_SOURCE_FOCUSED_WRAPPER_CPU_COMPILE_PASS_GPU_UNRUN','source_sha256':SHA,'manifest_sha256':MANIFEST,'all_device_bodies_exact_parent':True,'original_gvrun_and_oldlaunch_argument_pack_and_grid_exact_PW45':True,'separate_gvpredrun_export_compiled':True,'two_compiled_target_ids':[51,0],'pybind_exports_syntax_checked':binding_receipt['functions'],'route_semantics':'Unmarked direct/native calls original; only actual same-template positive confirmed plan gains trailing marker and launches predxor. Same seven original plan fields/packs held by references after temporary cache removal.','timing':'Four AB/BA blocks,9observations perarm; current commonX/separateoutputs; own immediate terminal checked. Positive paired median only; ties/loss/invalidtime/validation decline/budget skips retain old.','budget':'Original import clock and240s soft deadline; in-flight validation/timing can overrun. Additional cost unmeasured and charged to prepare.','CPU_boundary':'Actual helper transport win/loss/tie/tinypositive + ten focused skip/retention/currentX/output fixtures; no CUDA numeric/runtime claim.','compile_boundary':'CUDA12.8+real local CPU Torch2.13 at::Tensor compile and pybind syntax pass. CUDAcurrentstream declaration seam only; C++20 localheaders; no link/load or remoteTorch ABI attestation.','runtime_followup':'Existing root smoke/wholepolicy run must retain chosen marker,gvpred_choices,actualsymbol/status,currentoutputs and original deadline/cleanup. No additional all95 proof or mandatory profiler gate.','GPU_calls':0}
 (D/'review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

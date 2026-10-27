"""Focused inherited-driver diff receipt; no torch/production imports or compilation."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
def check():
 old=(ROOT/'proj_rsi/tools/gv_predxor_lab.py').read_text()
 assert hashlib.sha256(old.encode()).hexdigest()=='0353af7299f41a05fd6954db021e2f5da5943c8114fb482c2b84b5ff1e5bb5bf'
 expected=old.replace('ROOT=Path(__file__).resolve().parents[2]','ROOT=Path(__file__).resolve().parents[3]').replace("TRIAL=ROOT/'proj_rsi/candidates/gv_predxor_trial'","TRIAL=ROOT/'proj_rsi/candidates/policy_integration_trial_v1'")
 a=expected.index('        import mechanism\n');b=expected.index('        immutable_check()\n',a);removed_fixed=expected[a:b];expected=expected[:a]+expected[b:]
 a=expected.index('                from torch.profiler import profile,ProfilerActivity\n');b=expected.index('                for x,out,g in work.values():',a);removed_profiler=expected[a:b];expected=expected[:a]+expected[b:]
 expected=expected.replace("        result['all30_package_controls_required_for_promotion']=True","        result['profiling']='NOT_REQUESTED_NO_PROFILER_DEPENDENCY';result['fixed_mechanism']='OMITTED_ACTUAL_POLICY_ONLY'\n        result['all30_package_controls_required_for_promotion']=True")
 expected=expected.replace('    return G.cpu_check(sys.modules[__name__],output)',"    import focused_check\n    return focused_check.check()")
 actual=(D/'driver.py').read_text();assert actual==expected,'unexpected inherited flow change'
 ast.parse(actual)
 for name in ('support.py','gv_support.py'):assert (D/name).read_bytes()==(ROOT/'proj_rsi/candidates/gv_predxor_trial'/name).read_bytes()
 assert 'from torch.profiler' not in actual and 'mechanism.run' not in actual
 # Existing terminal ordering, AB/BA schedule, mutation checks, clock/status and finally cleanup are byte-inherited.
 assert actual.index("terminal('F1')")<actual.index("for i in range(args.pairs)")<actual.index("terminal('paired')")<actual.index('mutations=[')
 for marker in ('budget_clock_reset','for thread in build_threads:thread.join','counter_check(baseline[label]','out.fill_(float(\'nan\'))','g.reset();graphs_live.remove(g)','INVALID_NO_PERFORMANCE_ADMISSION'):
  assert marker in actual,marker
 return {'state':'PASS_FOCUSED_SOURCE_DIFF_GPU_UNRUN','driver_sha256':hashlib.sha256(actual.encode()).hexdigest(),'inherited_driver_sha256':hashlib.sha256(old.encode()).hexdigest(),'removed_fixed_stage_sha256':hashlib.sha256(removed_fixed.encode()).hexdigest(),'removed_profiler_stage_sha256':hashlib.sha256(removed_profiler.encode()).hexdigest(),'support_snapshots_byte_exact':True,'all30_actual_policy':True,'fixed_mechanism_and_profiler_dependencies':False,'new_compilation_or_exhaustive_proof':False}
if __name__=='__main__':print(json.dumps(check(),indent=2))

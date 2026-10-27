"""Bounded CPU driver readiness, not another codec or matrix proof."""
import ast,hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
assert 'torch'not in sys.modules
pilot=load(D/'pilot.py','decode_lut_v2_readiness_driver');assert 'torch'not in sys.modules
helptext=subprocess.check_output([sys.executable,str(D/'pilot.py'),'--help'],text=True);assert all(k in helptext for k in ['--prior','--lut-prior','--rolled-prior','--output','--review','--profile'])
manifest=json.loads((D/'manifest.json').read_text())
for rel,h in manifest['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
# Recording must use the identical code object and independent globals. This
# executable small example validates the mechanism without importing serving.
ns={'_engine':types.SimpleNamespace(project=lambda x:('original',x))}
exec('def method(self,x):return _engine.project(x)',ns);method=ns['method'];own=dict(method.__globals__);own['_engine']=types.SimpleNamespace(project=lambda x:('recorded',x));recorded=types.FunctionType(method.__code__,own,method.__name__,method.__defaults__,method.__closure__)
assert recorded.__code__ is method.__code__ and method(None,7)==('original',7)and recorded(None,7)==('recorded',7)
parent=load(ROOT/'serving/experiments/moe_rsi_20261027/ss_pipeline_v2/cpu_proof.py','decode_lut_v2_parent_readiness')
sass=(ROOT/'serving/experiments/moe_rsi_20261027/ss_decode_two_value_scatter_v1/sass.txt').read_text();positive=pilot.physical(sass,parent);assert len(positive)==1
negatives=[]
for label,bad in [('missing_wait0',sass.replace('WARPGROUP.DEPBAR.LE','NO_DRAIN',1)),('missing_reader_barrier',sass.replace('BAR.SYNC.DEFER_BLOCKING 0x3, 0x80','NO_READER_BARRIER',1))]:
 try:pilot.physical(bad,parent)
 except AssertionError:negatives.append(label)
 else:raise AssertionError('unsafe actual-SASS gate admitted '+label)
tree=ast.parse((D/'pilot.py').read_text());orders=next(ast.literal_eval(n.value)for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='orders'for t in n.targets))
assert len(orders)==8 and all(set(o)=={0,1,2}for o in orders)
pair_order={}
for a,b in[(0,1),(0,2),(1,2)]:
 count=sum(o.index(a)<o.index(b)for o in orders);assert count==4;pair_order[str((a,b))]=[count,len(orders)-count]
import torch
import torch.utils.cpp_extension as cpp
builder=cpp.load_inline
factories=[]
try:
 for label,path,source in [('v1_LUT_control',manifest['LUT_adapter'],manifest['LUT_extension_source']),('failed_v2',manifest['rolled_adapter'],manifest['rolled_extension_source']),('two_value_scatter',manifest['candidate_adapter'],manifest['candidate_extension_source'])]:
  calls=[]
  def mocked(*a,**kw):calls.append((a,kw));return object()
  cpp.load_inline=mocked;engine=load(ROOT/path,'readiness_'+label);engine.module();assert len(calls)==1
  assert calls[0][1]['cuda_sources']==(ROOT/source).read_text()and calls[0][1]['functions']==['run']
  factories.append({'arm':label,'exact_factory_source':True})
finally:cpp.load_inline=builder
assert not torch.cuda._initialized
out={'state':'CPU_RUNNABLE_DRIVER_GPU_UNRUN','driver_sha256':hashlib.sha256((D/'pilot.py').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'import_and_help_without_Torch':True,'manifest_files_match':True,'same_code_private_globals_mechanism':True,'reused_onefunction_gate':positive,'gate_negative_rejections':negatives,'exact_factories':factories,'pairwise_three_arm_order':pair_order,'no_new_codec_or_matrix_campaign':True,'CUDA_context_initialized':torch.cuda._initialized}
(D/'cpu-readiness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

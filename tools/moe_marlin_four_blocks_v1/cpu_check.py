"""Narrow stdlib control/source/route tests. No torch import/native compile."""
import ast,contextlib,hashlib,importlib.util,json,types
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
def main():
 env={'__file__':str(D/'pilot.py'),'__name__':'cpu_import'};exec(compile((D/'pilot.py').read_text(),'<actual-driver>','exec'),env)
 sp=importlib.util.spec_from_file_location('anchor_tiles_builder',D/'build.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b);proof=b.check()
 for blocks in range(1,5):
  for mode,(k,n,t)in enumerate(((128,128,256),(64,128,128)),1):
   encoded=blocks*1000000000+t*1000000+n*1000+k;assert env['decode'](encoded)==dict(blocks_per_sm=blocks,threads=t,thread_n=n,thread_k=k);assert env['opposite'](encoded)==3-mode
 for bad in(0,1000000000+128*1000000+64*1000+64):
  try:env['opposite'](bad)
  except AssertionError:pass
  else:raise AssertionError('invalid config accepted')
 orders=env['ORDERS'];assert len(orders)==12
 for arm in range(3):assert [sum(order[pos]==arm for order in orders)for pos in range(3)]==[4]*3
 for a in range(3):
  for c in range(a+1,3):assert sum(order.index(a)<order.index(c)for order in orders)==6
 # Execute exact nested route wrapper with fake original transport.
 tree=ast.parse((D/'pilot.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');node=next(n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)and n.name=='routed_gemm')
 calls=[];native=object();factory=object();api=types.SimpleNamespace(_jit_moe_wna16_marlin_module=factory);active=['anchor'];counts={l:0 for l in env['LABELS']}
 def original(*a,**kw):calls.append(api._jit_moe_wna16_marlin_module);return 0
 e={'active':active,'host_up_calls':counts,'LABELS':env['LABELS'],'api':api,'original_factory':factory,'original_gemm':original,'variant':native,'torch':types.SimpleNamespace(bfloat16='bf16')};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-route-wrapper>','exec'),e)
 args=[types.SimpleNamespace(dtype='bf16'),None,None,None,types.SimpleNamespace(size=lambda dim:256),object()]
 kw=dict(size_m=8,size_n=8192,size_k=4096,top_k=8,is_ep=True,use_atomic_add=False,use_fp32_reduce=True,is_k_full=True,is_zp_float=False,moe_block_size=8,mul_topk_weights=False)
 for label in env['LABELS']:
  active[0]=label;assert e['routed_gemm'](*args,**kw)==0;observed=calls[-1]
  if label=='anchor':assert observed is factory
  else:assert observed('bf16',True,False)is native
  assert api._jit_moe_wna16_marlin_module is factory
  down=dict(kw,size_m=64,size_n=4096,top_k=1);e['routed_gemm'](*args,**down);assert calls[-1]is factory
 active[0]='contrast'
 def failure(*a,**kw):raise RuntimeError('declined')
 e['original_gemm']=failure
 try:e['routed_gemm'](*args,**kw)
 except RuntimeError:pass
 else:raise AssertionError('backend failure hidden')
 assert api._jit_moe_wna16_marlin_module is factory
 # Exact changed driver block: no forced launch or GPU timer exists before feasibility.
 delta=json.loads((D/'driver-delta.json').read_text());parent=(R/delta['parent']).read_text();new=(D/'pilot.py').read_text();expect=parent
 for edit in delta['edits']:
  assert expect.count(edit['before'])==1;expect=expect.replace(edit['before'],edit['after'],1)
 assert expect==new
 for edit in reversed(delta['edits']):
  assert new.count(edit['after'])==1;new=new.replace(edit['after'],edit['before'],1)
 assert new==parent
 body=run.body[run.body.index(next(n for n in run.body if isinstance(n,ast.Try)))] .body
 begin=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='fields'for t in n.targets))
 end=next(i for i,n in enumerate(body[begin:],begin)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='original_gemm'for t in n.targets))
 block=body[begin:end]
 fun=ast.FunctionDef(name='actual_feasibility_guard',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=block+[ast.Assign(targets=[ast.Subscript(value=ast.Name(id='r',ctx=ast.Load()),slice=ast.Constant(value='admitted'),ctx=ast.Store())],value=ast.Constant(value=True))],decorator_list=[]);ast.fix_missing_locations(fun)
 # Numeric field fixtures are independent raw query results, not timing predictions.
 legal=[1,128,0,0,128,65536,233472,232448,2048,32,76458,57088,28736,3,4,65536,228352,1024]
 for kind,vals in [('legal',legal),('unsupported',[0]+legal[1:]),('query_error',[-101]*18),('lying_occupancy',legal[:14]+[3]+legal[15:]),('lying_registers',legal[:15]+[65537]+legal[16:])]:
  queries=[];record={};v=types.SimpleNamespace(anchor_block_info=lambda i:(queries.append(i),vals[i])[1]);g={'variant':v,'r':record,'save':lambda:None};exec(compile(ast.Module(body=[fun],type_ignores=[]),'<actual-pre-timing-feasibility>','exec'),g)
  if kind.startswith('lying_'):
   try:g['actual_feasibility_guard']()
   except AssertionError:pass
   else:raise AssertionError('inconsistent hardware receipt admitted')
  else:
   g['actual_feasibility_guard']();assert bool(record.get('admitted'))==(kind=='legal')
   if kind!='legal':assert record['GPU_timing_calls']==0 and 'UNSUPPORTED_NO_TIMING'in record['state']
  assert queries==list(range(18))
 return {'state':'CPU_FOCUSED_SOURCE_SCHEDULE_ROUTE_PASS_GPU_UNRUN','native_source_proof':proof,'test_cases':{'config_decode_opposite':8,'invalid_config':2,'same_position_balanced_orders':12,'actual_wrapper_anchor_default_contrast':3,'down_original_fallthrough':3,'failure_restores_factory':1,'actual_feasibility_guard_supported_unsupported_queryerror':3,'inconsistent_hardware_receipt_rejected':2,'strict_driver_inverse':1},'timed_calls':5*5+12*3,'own_timed_terminal_oracles':5+3,'optional_profiles_after_validated_direct_results':True,'native_compile_or_GPU_calls':0}
if __name__=='__main__':
 p=main();(D/'cpu-proof.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))

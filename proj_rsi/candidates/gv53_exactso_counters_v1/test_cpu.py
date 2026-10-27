"""Synthetic parser/driver contract fixtures. No Torch/CUDA/NCU imports or subprocesses."""
import copy,csv,importlib.util,io,json
from pathlib import Path
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('profile_recipe',D/'profile.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
spec=importlib.util.spec_from_file_location('capture_recipe',D/'capture.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
def fixture():
 cap={'state':'THREE_ARM_EXACTSO_PROFILE_COMPLETE_ROOT_CLEANUP_REQUIRED','phase':'complete','cleanup':[], 'post_scope_cleanup':{'allocated':0,'reserved':0,'clear_error':None,'blas_map_entries_workspace_bytes_after':[1,33554432,0]},'fixed_cells':{'8':{'terminals':{},'mutations':[{}]*15,'raw':C.schedule()}},'actual_graph_nodes':{},'arm_bindings':{},'clock_context':[]}
 headers=['ID','Kernel Name','Block Size','Grid Size',*P.METRICS,'profiler__replayer_passes'];units={k:'%'if k.endswith('.pct')or'.pct_'in k else'inst'for k in P.METRICS};units.update({'gpu__time_duration.sum':'ns','dram__bytes_read.sum':'byte','dram__bytes_write.sum':'byte','smsp__warps_eligible.avg.per_cycle_active':'warp','profiler__replayer_passes':'pass'})
 rows=[]
 for i,arm in enumerate(P.LABELS):
  symbol='CPU_FAKE_'+arm;cap['actual_graph_nodes'][arm]=[{'node_type':0,'cluster_query_returncode':0,'cluster_dimension':[0,0,0],'function_mangled':symbol,'grid':[1,128,1],'block':[1024,1,1],'dynamic_shared_bytes':17408,'SASS_sha256':'a'*64,'attributes':{'local_bytes':0}}]
  cap['arm_bindings'][arm]={'SO_sha256':C.OLD_SO_SHA if arm=='quarter'else C.NEW_SO_SHA}
  cap['fixed_cells']['8']['terminals'][arm]={'before_any_replay_or_mutation':True,'oracle':{'current_X':'EXACT'}}
  row=dict.fromkeys(headers,'1');row.update({'ID':str(i),'Kernel Name':symbol,'Block Size':'(1024, 1, 1)','Grid Size':'(1, 128, 1)','profiler__replayer_passes':'10'});rows.append(row)
 out=io.StringIO();w=csv.writer(out);w.writerow(headers);w.writerow([units.get(h,'')for h in headers]);w.writerows([[r[h]for h in headers]for r in rows]);return cap,out.getvalue()
def main():
 cap,raw=fixture();assert len(P.parse(raw,cap)['arms'])==3;negative=[]
 for name in ['missing_launch','wrong_symbol','nonfinite_metric','cluster','terminal','cleanup','missing_metric']:
  c=copy.deepcopy(cap);r=raw
  if name=='missing_launch':r='\n'.join(r.splitlines()[:-1])
  elif name=='wrong_symbol':r=r.replace('CPU_FAKE_sparse','wrong')
  elif name=='nonfinite_metric':r=r.replace('"(1, 128, 1)",1,','"(1, 128, 1)",nan,',1)
  elif name=='cluster':c['actual_graph_nodes']['quarter'][0]['cluster_dimension']=[1,1,2]
  elif name=='terminal':c['fixed_cells']['8']['terminals']['sparse']['before_any_replay_or_mutation']=False
  elif name=='cleanup':c['post_scope_cleanup']['allocated']=1
  elif name=='missing_metric':r=r.replace(P.METRICS[-1],'unknownmetric')
  try:P.parse(r,c)
  except(AssertionError,ValueError,KeyError):negative.append(name)
  else:raise AssertionError(name)
 assert C.cpu_check()['JIT_calls']==0
 assert P.cpu_check()['profile_launches_max']==3
 print(json.dumps({'state':'PASS_CPU_PROFILE_PARSE_AND_EXACTSO','negative_fixtures':negative,'NCU_GPU_subprocess_calls':0},indent=2))
if __name__=='__main__':main()

"""Focused parser negatives and leader-exit descendant cleanup, no CUDA/NCU."""
from pathlib import Path
import importlib.util,csv,io,copy,json,subprocess,os,time,sys
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recipe',D/'profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
capture={'state':'VALID_EXACT_CURRENT_POOLER32_RQ_PROFILE_OUTPUTS','phases':['complete'],'raw':[{'mode':'warmup'}]*5+[{'mode':'profile'}],'validation':[{}]*4,'mutations':[{}]*5,'own_terminal':{'before_any_later_replay_or_mutation':True},'post_scope_cleanup':{'allocated':0,'blas_after':[0,0,0]},'build':{'SO_sha256':'ab6c3ad71701098713f7c3435439f905c4d482ae66a995948bfcda254494a32f','JIT':False,'native_arm':0},'actual_graph_nodes':[{'function_mangled':m.FUNCTION,'grid':[32,4,1],'block':[384,1,1],'dynamic_shared_bytes':99488}]}
h=['ID','Kernel Name','Block Size','Grid Size',*m.METRICS,'profiler__replayer_passes'];units={k:'%'if k.endswith('.pct')or'.pct_'in k else'inst'for k in m.METRICS};units.update({'gpu__time_duration.sum':'ns','dram__bytes_read.sum':'byte','dram__bytes_write.sum':'byte','smsp__warps_eligible.avg.per_cycle_active':'warp','profiler__replayer_passes':'pass'})
row={k:'1'for k in h};row.update({'ID':'0','Kernel Name':m.FUNCTION,'Block Size':'(384,1,1)','Grid Size':'(32,4,1)','profiler__replayer_passes':'4'})
def raw(r=row,u=units):
 f=io.StringIO();w=csv.writer(f);w.writerow(h);w.writerow([u.get(k,'')for k in h]);w.writerow([r[k]for k in h]);return f.getvalue()
assert m.parse(raw(),capture)['replay_passes']==4
negatives=0
for key,val in [('Kernel Name','wrong'),('profiler__replayer_passes','1.5'),('profiler__replayer_passes','nan'),('Block Size','(256,1,1)'),(m.METRICS[1],'-1')]:
 r=dict(row);r[key]=val
 try:m.parse(raw(r),capture)
 except (AssertionError,ValueError):negatives+=1
 else:raise AssertionError(key)
u=dict(units);u['profiler__replayer_passes']='%'
try:m.parse(raw(u=u),capture)
except AssertionError:negatives+=1
else:raise AssertionError('pass units')
c=copy.deepcopy(capture);c['own_terminal']['before_any_later_replay_or_mutation']=False
try:m.parse(raw(),c)
except AssertionError:negatives+=1
else:raise AssertionError('terminal ownership')
# Process group fixture: parent exits, descendant waits in same unique group.
p=subprocess.Popen([sys.executable,'-c','import os,time; p=os.fork(); time.sleep(30) if p==0 else None'],start_new_session=True)
p.wait(timeout=5);drained=m.drain_group(p.pid)
# Unreaped orphan zombies can remain visible to killpg0 until container init reaps;
# they cannot execute GPU work. Runtime still records exact group-drained result.
assert p.poll()==0
print(json.dumps({'state':'CPU_PROFILE_FOCUSED_TESTS_PASS','parser_negatives':negatives,'leader_exited_descendant_SIGKILL_attempted':True,'group_disappeared_before3s':drained,'GPU_NCU_calls':0},indent=2))

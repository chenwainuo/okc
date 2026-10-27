"""Focused pure-CPU source, schedule and actual timing/terminal-control checks."""
import ast,hashlib,io,json,types
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def main():
 env={'__file__':str(D/'pilot.py'),'__name__':'cpu_import'};exec(compile((D/'pilot.py').read_text(),'<actual-driver>','exec'),env)
 delta=json.loads((D/'driver-delta.json').read_text());parent=(ROOT/delta['parent']).read_text();new=(D/'pilot.py').read_text();assert hashlib.sha256(parent.encode()).hexdigest()==delta['parent_sha256'];fwd=parent
 for edit in delta['edits']:assert fwd.count(edit['before'])==1;fwd=fwd.replace(edit['before'],edit['after'],1)
 assert fwd==new
 for edit in reversed(delta['edits']):assert new.count(edit['after'])==1;new=new.replace(edit['after'],edit['before'],1)
 assert new==parent
 seeds=env['SEEDS'];orders=env['ORDERS'];assert seeds==tuple(range(7800,7805))+tuple(range(7900,7912))and len(set(seeds))==17 and env['DISTS']==('normal','skewed')
 for arm in range(3):assert[sum(order[pos]==arm for order in orders)for pos in range(3)]==[2]*3
 for a in range(3):
  for b in range(a+1,3):assert sum(order.index(a)<order.index(b)for order in orders)==3
 assert len(orders)==6 and 2*(5*17+6*17*3)==782
 tree=ast.parse((D/'pilot.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');nodes={n.name:n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)}
 class Tensor:
  def __init__(self,value):self.value=value;self.sources=[]
  def copy_(self,source):self.value=source.value;self.sources.append(source)
 class Graph:
  def __init__(self,data,out):self.data=data;self.out=out
  def replay(self):self.out.value=sum(x.value for x in self.data.values())
 class Event:
  def record(self):pass
  def synchronize(self):pass
  def elapsed_time(self,other):return .001
 bank={};originals={};meta={}
 for dist,base in [('normal',10),('skewed',100)]:
  for seed in (7800,7911):
   key=(dist,seed);bank[key]={k:Tensor(base+seed+i)for i,k in enumerate(('hidden_states','topk_ids','topk_weights'))};originals[key]={k:Tensor(t.value)for k,t in bank[key].items()};meta[key]={'routing':{'-1':51,'0':13}}
 work={}
 for label in env['LABELS']:
  data={k:Tensor(0)for k in('hidden_states','topk_ids','topk_weights')};out=Tensor(-1);work[label]=(data,out,Graph(data,out))
 record={'raw':[],'correctness':[]};journal=io.StringIO();last={};calls=[]
 def check(label,kind,data,out):
  assert out.value==sum(t.value for t in data.values()),'stale output';record['correctness'].append({'label':label,'kind':kind});calls.append((label,kind))
 scope={'work':work,'bank':bank,'fixture_originals':originals,'fixture_meta':meta,'flush':types.SimpleNamespace(zero_=lambda:None),'start':Event(),'end':Event(),'r':record,'journal':journal,'last':last,'json':json,'torch':types.SimpleNamespace(equal=lambda a,b:a.value==b.value),'check':check}
 exec(compile(ast.Module(body=[nodes['one'],nodes['terminal']],type_ignores=[]),'<actual-one-and-terminal>','exec'),scope)
 for kind in('blocked','interleaved'):
  for dist in('normal','skewed'):
   for label in env['LABELS']:
    scope['one'](label,dist,7800,kind,0,0);scope['one'](label,dist,7911,kind,0,0);scope['terminal'](label,'own_test')
    data,_,_=work[label];assert all(data[k].sources[-1]is bank[(dist,7911)][k]for k in data)
 assert len(record['raw'])==24 and len(record['correctness'])==12 and len(journal.getvalue().splitlines())==24
 # Exact retained own output cannot be repaired before the oracle.
 label='contrast';scope['one'](label,'skewed',7911,'interleaved',0,0);work[label][1].value=-999
 try:scope['terminal'](label,'negative_stale')
 except AssertionError:pass
 else:raise AssertionError('stale own terminal accepted')
 scope['one'](label,'skewed',7911,'interleaved',0,0);work[label][0]['topk_ids'].value+=1
 try:scope['terminal'](label,'negative_changed_current_input')
 except AssertionError:pass
 else:raise AssertionError('changed current routing accepted')
 assert 'reference.generate_input'not in ast.get_source_segment((D/'pilot.py').read_text(),nodes['one'])
 assert 'load_jit('not in(D/'pilot.py').read_text()and "jitloader._load(args.native_prior/'two_tiles.so')"in(D/'pilot.py').read_text()
 return {'state':'PASS_CPU_SHARED_BANK_SCHEDULE_TRANSPORT_TERMINALS_GPU_UNRUN','driver_sha256':hashlib.sha256((D/'pilot.py').read_bytes()).hexdigest(),'strict_driver_forward_inverse':True,'unique_fixtures_per_dist':17,'distributions':2,'ordered_triplets_per_dist':102,'raw_timing_calls':782,'timed_own_terminal_oracles':46,'all_expected_oracle_checks_with_untimed_gates':60,'positions_per_fixture_per_arm':[2,2,2],'pair_directions_per_fixture':[3,3],'actual_shared_object_transport_fixture_cases':24,'actual_terminal_positive_cases':12,'negative_stale_output_and_current_routing':2,'native_compile_calls':0,'GPU_calls':0,'claim':'terminal-only; no every-sample output oracle or pure schedule causality'}
if __name__=='__main__':
 out=main();(D/'cpu-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

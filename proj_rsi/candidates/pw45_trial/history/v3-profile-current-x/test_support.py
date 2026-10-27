"""CPU-only adversarial wrapper contracts; no torch import or GPU execution."""
import importlib.util,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import support as S
import mechanism as M
spec=importlib.util.spec_from_file_location('pw45_cpu_driver',S.ROOT/'proj_rsi/tools/pw45_stack_lab.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
class Contracts(unittest.TestCase):
 def test_serving_status_forwards_single_call_and_rejects_failure(self):
  class Ext:
   def __init__(self):self.calls=0;self.status=0
   def lt_mm(self,*args):self.calls+=1;return self.status
   def run(self,*args):self.calls+=1;return None
  ext=Ext();proxy=S.ServingStatus(ext);self.assertEqual(proxy.lt_mm(123),0);self.assertEqual(ext.calls,1)
  ext.status=-1
  with self.assertRaises(AssertionError):proxy.lt_mm(123)
  self.assertEqual(ext.calls,2);self.assertIsNone(proxy.run(123));self.assertEqual(ext.calls,3)
 def test_plain_keeps_nonzero_full_physical_arena(self):
  before={'a':{'name':'cnt','values':[7,3,19,23]}}
  A.counter_check(before,before,{},77)
  for changed in ([0,3,19,23],[7,3,19,0]):
   with self.assertRaises(AssertionError):A.counter_check(before,{'a':{'name':'cnt','values':changed}}, {},77)
 def test_zi_only_active_prefix_and_offset(self):
  before={'a':{'name':'cnt','values':[11,3,4,17]}};plan={'a':{'mode':'reset0','offset':1,'reset_indices':[0,1]}}
  A.counter_check(before,{'a':{'name':'cnt','values':[11,0,0,17]}},plan,3)
  for bad in ([0,0,0,17],[11,0,0,0],[11,0,4,17]):
   with self.assertRaises(AssertionError):A.counter_check(before,{'a':{'name':'cnt','values':bad}},plan,3)
 def test_physical_storage_set_cannot_change(self):
  with self.assertRaises(AssertionError):A.counter_check({}, {'x':{}},{},1)
 def test_progress_preserves_dormant_and_counts_calls(self):
  self.assertEqual(S.counter_expected([10,21,17],'progress',0,changes={0:2},calls=3),[16,21,17])
  with self.assertRaises(AssertionError):S.counter_expected([2**31-1],'progress',changes={0:1},calls=1)
 def test_exact_resource_gate(self):
  attrs=dict(regs=168,local=0,shared=42096,occupancy=1,max_dynamic=42096,max_threads=384)
  S.check_attrs((32,4,1,0,0,0,52),attrs)
  for key,val in [('regs',167),('local',4),('shared',52360),('occupancy',0),('max_dynamic',42095),('max_threads',383)]:
   with self.assertRaises(AssertionError):S.check_attrs((32,4,1,0,0,0,52),dict(attrs,**{key:val}))
 def test_helper_only_appends_and_enumerates_all_mains(self):
  cu=S.cpu_factory();new,args=S.append_attrs(cu)
  self.assertEqual(new[:len(cu)],cu);self.assertEqual(set(args),set(S.streams((S.CAND/'sass.txt').read_text())))
  self.assertEqual(len(args),21)
  for a in S.FIXED.values():self.assertIn(a[0],args)
 def test_all_canonical_mains_and_new_physical(self):
  cu=S.cpu_factory();sass=(S.CAND/'sass.txt').read_text();resources=S.synthetic_resources(sass)
  result=S.canonical_gate(cu,sass,resources);self.assertEqual(len(result['args']),21);self.assertIsNotNone(result['new_plain_fullK_physical'])
 def test_pc_and_instruction_tamper_rejected(self):
  cu=S.cpu_factory();sass=(S.CAND/'sass.txt').read_text();resources=S.synthetic_resources(sass)
  entries=S.streams(sass);entry=entries[(32,4,1,0,0,0,52)];table=entry['table'];badtable=table.replace('/*0000*/','/*0001*/',1);self.assertNotEqual(table,badtable)
  with self.assertRaises(AssertionError):S.canonical_gate(cu,sass.replace(table,badtable),resources)
  with self.assertRaises(AssertionError):S.canonical_gate(cu,sass.replace(table,''),resources)
 def test_original_table_scalar_and_void_source_signatures(self):
  text=(S.CAND/'submission.py').read_text()
  self.assertIn('return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc',text)
  for name in ('run','run2','run3'):self.assertIn('void '+name+'(torch::Tensor',text)
  source=(HERE/'mechanism.py').read_text();self.assertIn('assert type(tab) is int',source);self.assertNotIn('tab.clone()',source)
 def test_resource_dump_negative(self):
  symbol='foo';S.resource('Function foo\n REG:168 LOCAL:0\n',symbol)
  for res in ('Function foo\n REG:169 LOCAL:0\n','Function foo\n REG:168 LOCAL:8\n'):
   with self.assertRaises(AssertionError):S.resource(res,symbol)
 def test_balanced_fixtures_and_counts(self):
  rows=M.plan();self.assertEqual(len(rows),210)
  pairs=rows[120:]
  for i in range(30):self.assertEqual(len({r['seed'] for r in pairs[3*i:3*i+3]}),1);self.assertEqual({r['label'] for r in pairs[3*i:3*i+3]},set(M.LABELS))
  for pos in range(3):self.assertEqual([sum(pairs[3*i+pos]['label']==label for i in range(30)) for label in M.LABELS],[10,10,10])
 def test_hardware_and_build_admission_order(self):
  source=Path(A.__file__).read_text();self.assertLess(source.index('HOLD incompatible hardware'),source.index('weights=reference.generate_weights'))
  self.assertLess(source.index('S.canonical_gate(original_cu'),source.index('return ext'))
  self.assertLess(source.index('S.check_attrs(geometry'),source.index('return ext'))
 def test_terminal_rejects_wrong_actual_last_fixture(self):
  class T:
   def __init__(self,v,ptr):self.v=v;self.ptr=ptr
   def data_ptr(self):return self.ptr
  class Torch:
   @staticmethod
   def equal(a,b):return a.v==b.v
  out={l:T(1,i) for i,l in enumerate(M.LABELS)};rows=[dict(label=l,seed=i,mode='F1') for i,l in enumerate(M.LABELS)];fixtures={i:T(i,9) for i in range(3)}
  with self.assertRaises(AssertionError):M.terminal(out,T(0,7),fixtures,{l:i for i,l in enumerate(M.LABELS)},rows,None,None,Torch)
if __name__=='__main__':unittest.main()

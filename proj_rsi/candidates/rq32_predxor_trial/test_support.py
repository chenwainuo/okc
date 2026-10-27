"""CPU-only adversarial wrapper contracts; no torch import or GPU execution."""
import importlib.util,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import support as S
import mechanism as M
spec=importlib.util.spec_from_file_location('rq32_predxor_cpu_driver',S.ROOT/'proj_rsi/tools/rq32_predxor_lab.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
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
 def test_profile_expected_clone_rejects_joint_x_output_mutation(self):
  class Tensor:
   def __init__(self,bits):self.bits=list(bits)
   def view(self,dtype):return self
  class Torch:
   uint8='uint8'
   @staticmethod
   def equal(a,b):return a.bits==b.bits
  expected=Tensor([0,0,128,63]);x=Tensor(expected.bits)
  M.current_x_bytes(Torch,x,expected)
  # Even a correspondingly changed output oracle cannot authorize altered input.
  x.bits[1]=128
  consistent_output=Tensor(x.bits)
  self.assertTrue(Torch.equal(x,consistent_output))
  with self.assertRaises(AssertionError):M.current_x_bytes(Torch,x,expected)
 def test_profile_expected_clone_drains_before_profiler(self):
  source=(HERE/'mechanism.py').read_text();start=source.index('expected_fixture=x.clone()');sync=source.index('torch.cuda.synchronize()',start);context=source.index('with profile(',start)
  self.assertLess(start,sync);self.assertLess(sync,context)
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
  for a in S.FIXED.values():self.assertIn(a[1],args)
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
 def test_actual_parent_new_canonical_and_wrong_source_label_rejects(self):
  for label in ('pw45','rq32_predxor'):
   cu=S.cpu_factory(label);sass=(S.CAND.parent/label/'sass.txt').read_text();r=S.canonical_gate(cu,sass,S.synthetic_resources(sass));self.assertEqual(r['source_kind'],'predxor' if label=='rq32_predxor' else 'PW45_parent')
  sass=(S.CAND/'sass.txt').read_text()
  with self.assertRaises(AssertionError):S.canonical_gate(S.cpu_factory('pw45'),sass,S.synthetic_resources(sass))
 def test_escape_fixture_exact_owner_refill_tail_and_negatives(self):
  rows=S.expected_edge_rows();r=S.verify_edge_rows(rows,56*0x01010101);self.assertEqual(r['stage0_unique_lane_word_half_owners'],1024)
  for changed in (rows[:-1],rows[:-1]+[[0,30,3,31,7,24]],rows[:-1]+[[0,31,3,31,6,24]]):
   with self.assertRaises(AssertionError):S.verify_edge_rows(changed,56*0x01010101)
  with self.assertRaises(AssertionError):S.verify_edge_rows(rows,57*0x01010101)
 def test_invalid_elapsed_raw_is_retained_before_rejection(self):
  import json
  for us in (0.,-1.,float('nan'),float('inf'),-float('inf')):
   rows=[]
   with self.assertRaises(AssertionError):S.elapsed_gate(us,{'label':'pw45','seed':9001,'mode':'paired'},rows.append)
   self.assertEqual(len(rows),1);self.assertEqual(rows[0]['seed'],9001);self.assertEqual(rows[0]['elapsed_us_repr'],repr(us));json.dumps(rows,allow_nan=False)
  rows=[];self.assertEqual(S.elapsed_gate(.001,{'label':'pw45'},rows.append),.001);self.assertEqual(rows,[])
 def test_balanced_fixtures_and_counts(self):
  rows=M.plan();self.assertEqual(len(rows),295);pairs=rows[167:]
  for i in range(32):self.assertEqual(len({r['seed'] for r in pairs[4*i:4*i+4]}),1);self.assertEqual({r['label'] for r in pairs[4*i:4*i+4]},set(M.LABELS))
  for pos in range(4):self.assertEqual([sum(pairs[4*i+pos]['label']==label for i in range(32)) for label in M.LABELS],[8,8,8,8])
 def test_real_policy_schedule_all32rounds_and127rows(self):
  rows=S.policy_plan();self.assertEqual(len(rows),127)
  for i in range(32):self.assertEqual(S.policy_order(i),((0,1),(1,0))[i%2]);self.assertEqual({r['label'] for r in rows[63+2*i:65+2*i]},{'pw45','rq32_predxor'})
  for i in (-1,32):
   with self.assertRaises(AssertionError):S.policy_order(i)
  source=Path(A.__file__).read_text();self.assertIn('for index in S.policy_order(i)',source);self.assertNotIn('orders[i%6]',source)
  with self.assertRaises(IndexError):[(0,1),(1,0)][2%6]
 def test_binary_storage_receipt_raw_bytes_and_digest(self):
  import tempfile,hashlib
  previous=S.ARTIFACT_ROOT
  with tempfile.TemporaryDirectory() as directory:
   S.ARTIFACT_ROOT=Path(directory)/'physical-storage';a=S.arena_receipt({('cuda:0',123,16):{'name':'cnt','values':[17,0,-1,23]}})[0]
   payload=(S.ARTIFACT_ROOT/Path(a['artifact']).name).read_bytes();self.assertEqual(len(payload),16);self.assertEqual(a['sha256'],hashlib.sha256(payload).hexdigest())
   (S.ARTIFACT_ROOT/Path(a['artifact']).name).write_bytes(b'corrupt')
   with self.assertRaises(AssertionError):S.arena_receipt({('cuda:0',123,16):{'name':'cnt','values':[17,0,-1,23]}})
  S.ARTIFACT_ROOT=previous
 def test_hardware_and_build_admission_order(self):
  source=Path(A.__file__).read_text();self.assertLess(source.index('HOLD incompatible hardware'),source.index('weights=reference.generate_weights'))
  self.assertLess(source.index('S.canonical_gate(original_cu'),source.index('return ext'))
  self.assertLess(source.index('S.check_attrs(geometry'),source.index('return ext'))
 def test_terminal_rejects_wrong_actual_last_fixture(self):
  class T:
   def view(self,dtype):return self
   def __init__(self,v,ptr):self.v=v;self.ptr=ptr
   def data_ptr(self):return self.ptr
  class Torch:
   uint8='uint8'
   @staticmethod
   def equal(a,b):return a.v==b.v
  out={l:T(1,i) for i,l in enumerate(M.LABELS)};rows=[dict(label=l,seed=i,mode='F1') for i,l in enumerate(M.LABELS)];fixtures={i:T(i,9) for i in range(4)}
  with self.assertRaises(AssertionError):M.terminal(None,Torch,None,out,T(0,7),fixtures,{l:i for i,l in enumerate(M.LABELS)},rows,None,None,{},'F1')
if __name__=='__main__':unittest.main()

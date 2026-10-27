"""CPU-only numeric/route/terminal negative controls; no torch import."""
import ast,importlib.util,math,struct,sys,unittest
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[1]/'moe_ss_pipeline_screen.py'
spec=importlib.util.spec_from_file_location('ss_screen_cpu',P);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def bf16(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def e4m3(bits):
 exp=(bits>>3)&15;mant=bits&7
 return (mant*2.**-9 if exp==0 else (1+mant/8)*2.**(exp-7))
class ScreenTests(unittest.TestCase):
 def test_independent_e2m1_and_e4m3_scaling(self):
  count=0
  for q in range(16):
   kernel=bf16(((q&8)<<12)|((q&7)<<6))
   self.assertEqual(kernel,S.decode_nibble(q)*2.**-126)
   for raw in range(127):
    numeric=e4m3(raw);half=struct.unpack('<H',struct.pack('<e',numeric*128))[0];processed=((half<<1)>>8)&255
    decoded=bf16(((processed&128)<<7)|((processed&127)<<4))
    self.assertEqual(decoded,numeric*128)
    self.assertEqual(kernel*decoded,S.decode_nibble(q)*numeric*2.**-119);count+=1
  self.assertEqual(count,2032)
 def test_all_tiles_and_route_scopes(self):
  for div in S.DIVISORS:
   for bm in (8,16,32,48,64):
    for kind in ('balanced','skew','duplicates','mixed','one','empty','tail'):
     ids,experts,post,assign=S.route_fixture(17*div,bm,kind)
     self.assertTrue(all(0<=r//div<17 for r in assign));self.assertEqual(len(ids)%4,0)
  self.assertEqual(S.tile(48),16)
 def test_route_duplicate_write_post_and_expert_negatives(self):
  for args in [([0,0]+[8]*6,[0],8,8,8),([0]+[8]*7,[32],8,8,8),([0]+[8]*7,[0],9,8,8),([0]+[8]*7,[0],1,8,8)]:
   with self.assertRaises(AssertionError):S.validate_routes(*args)
 def test_terminal_own_fixture_and_metadata(self):
  class Tensor:
   device='cuda:0';dtype='bf16';shape=(8,)
   def __init__(self,p,value):self.p=p;self.value=value;self.offset=0
   def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.p,nbytes=lambda:16)
   def storage_offset(self):return self.offset
   def stride(self):return (1,)
  x,out=Tensor(1,3),Tensor(2,3);fresh=Tensor(3,3);meta={'x':S.metadata(x),'out':S.metadata(out)}
  equal=lambda a,b:a.value==b.value
  def oracle(a,b):assert a.value==b.value;return 'pass'
  self.assertEqual(S.own_terminal(x,out,fresh,meta,equal,oracle),'pass')
  with self.assertRaises(AssertionError):S.own_terminal(x,out,Tensor(4,7),meta,equal,oracle)
  out.offset=1
  with self.assertRaises(AssertionError):S.own_terminal(x,out,fresh,meta,equal,oracle)
  out.offset=0;out.value=99
  with self.assertRaises(AssertionError):S.own_terminal(x,out,fresh,meta,equal,oracle)
 def test_import_cpu_only_and_production_extension_source(self):
  self.assertNotIn('torch',sys.modules)
  source=S.SOURCE/'adapter.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='module')
  host=next(ast.literal_eval(n.value)for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='host'for t in n.targets))
  self.assertEqual((S.SOURCE/'kernel.cuh').read_text()+host,(S.SOURCE/'extension_source.cu').read_text())
  self.assertEqual(S.DIAGNOSTIC.count('CASE(8,1)'),2)
if __name__=='__main__':unittest.main()

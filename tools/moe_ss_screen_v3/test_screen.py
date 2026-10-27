"""CPU-only numeric/route/terminal negative controls; no torch import."""
import ast,importlib.util,math,re,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'moe_ss_pipeline_screen_v3.py'
spec=importlib.util.spec_from_file_location('ss_screen_cpu',P);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def bf16(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def e4m3(bits):
 exp=(bits>>3)&15;mant=bits&7
 return (mant*2.**-9 if exp==0 else (1+mant/8)*2.**(exp-7))
class ScreenTests(unittest.TestCase):
 def test_independent_e2m1_and_all_finite_signed_e4m3(self):
  count=0
  for q in range(16):
   for sf in range(256):
    if not S.valid_scale_code(sf):continue
    value=S.decode_nibble(q)*S.numeric_scale(sf)
    bits=S.numeric_weight_bits(q,sf)
    self.assertEqual(bf16(bits),value);count+=1
  self.assertEqual(count,4064)
  self.assertFalse(S.valid_scale_code(127));self.assertFalse(S.valid_scale_code(255))
  self.assertEqual(S.numeric_weight_bits(8,0),32768) # negative zero is preserved
  self.assertEqual(S.numeric_weight_bits(8,128),0) # -zero × -zero
 def test_deterministic_tile_covers_all_q_scale_pairs(self):
  scales=list(range(127))+list(range(128,255))+[0,0]
  pairs={(q,scales[row*4+group]) for row in range(64)for group in range(4)for q in range(16)}
  expected={(q,sf)for q in range(16)for sf in range(256)if S.valid_scale_code(sf)}
  self.assertEqual(pairs,expected)
 def test_all_tiles_and_route_scopes(self):
  for div in S.DIVISORS:
   for bm in (8,16,32,48,64):
    for kind in ('balanced','skew','duplicates','mixed','one','empty','tail'):
     ids,experts,post,assign=S.route_fixture(17*div,bm,kind)
     self.assertTrue(all(0<=r//div<17 for r in assign));self.assertEqual(len(ids)%4,0)
  self.assertEqual(S.tile(48),16)
 def test_actual_allocator_both_branches(self):
  for bm in (8,16,32,48,64):
   self.assertEqual(S.route_capacity(256,bm),256*bm)
   self.assertEqual(S.route_capacity(257,bm),(257+257*(bm-1)+3)//4*4)
  self.assertNotEqual(S.route_capacity(8,64),((8+256*63+3)//4)*4)
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
 def test_augmented_offline_wrapper_identity_and_resources(self):
  folder=S.OFFLINE
  self.assertEqual((folder/'screen.cu').read_text(),(S.SOURCE/'extension_source.cu').read_text()+S.DIAGNOSTIC)
  expected=S.resource_tables((folder/'canonical.resources').read_text());actual=S.resource_tables((folder/'screen.resources').read_text())
  self.assertEqual(len(expected),8);self.assertEqual(actual,expected)
  changed=re.sub(r'(Function _ZN9moe_ss_v27project[^\n]+\n\s+REG:)',r'\g<1>999',(folder/'screen.resources').read_text(),count=1)
  self.assertNotEqual(S.resource_tables(changed),expected)
 def test_actual_augmented_sass_gate_filters_only_production(self):
  # Exact frozen gate.check body and exact frozen physical body, with only its
  # module-loader replaced to avoid the irrelevant top-level Torch CPU import.
  gate=S.module_from(S.SOURCE/'actual_sass_gate.py','ss_cpu_gate')
  tree=ast.parse((S.SOURCE/'cpu_proof.py').read_text());physical=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='physical')
  env={'HERE':S.SOURCE,'re':re};exec(compile(ast.Module(body=[physical],type_ignores=[]),'<frozen-physical-only>','exec'),env)
  spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None));module=SimpleNamespace(physical=env['physical'])
  full=(S.OFFLINE/'screen.sass').read_text();canonical=(S.SOURCE/'sass.txt').read_text()
  main=S.production_sass(full,canonical)
  with patch.object(gate.importlib.util,'spec_from_file_location',return_value=spec),patch.object(gate.importlib.util,'module_from_spec',return_value=module):
   self.assertEqual(len(gate.check(main)['actual_cubin_functions']),8)
   with self.assertRaises(AssertionError):gate.check(full)
  with self.assertRaises(AssertionError):S.production_sass(main.split('Function : ',1)[1],canonical)
 def test_container_name_matches_frozen_guard(self):
  script=(S.ROOT/'proj_rsi/tools/h100_trial.sh').read_text();self.assertIn('^okc-proj-[a-z0-9-]+$',script)
  self.assertRegex('okc-proj-moe-ss-stage-screen-v2-01',r'^okc-proj-[a-z0-9-]+$')
 def test_launcher_preserves_existing_attempt_and_new_startup_error(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);old=root/'existing';old.mkdir();(old/'sentinel').write_bytes(b'prior immutable attempt')
   launcher=S.ROOT/'proj_rsi/tools/moe_ss_pipeline_screen_v3.py'
   def call(output):return subprocess.run([sys.executable,str(launcher),'--output',str(output),'--review',str(root/'missing-review'),'--review-sha256','invalid'],text=True,capture_output=True)
   before={p.name:p.read_bytes()for p in old.iterdir()};result=call(old)
   self.assertNotEqual(result.returncode,0);self.assertIn('refuse existing attempt output',result.stderr)
   self.assertEqual({p.name:p.read_bytes()for p in old.iterdir()},before)
   new=root/'new';result=call(new);self.assertNotEqual(result.returncode,0);self.assertTrue((new/'launcher-error.json').is_file())
 def test_import_cpu_only_and_production_extension_source(self):
  self.assertNotIn('torch',sys.modules)
  source=S.SOURCE/'adapter.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='module')
  host=next(ast.literal_eval(n.value)for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='host'for t in n.targets))
  self.assertEqual((S.SOURCE/'kernel.cuh').read_text()+host,(S.SOURCE/'extension_source.cu').read_text())
  self.assertEqual(S.DIAGNOSTIC.count('CASE(8,1)'),2)
 def test_actual_v2_failure_resource_regression_and_negative_controls(self):
  canonical=S.resource_tables((S.OFFLINE/'canonical.resources').read_text())[8,8]
  row={'registers':32,'local':0,'static_shared':0,'max_threads':256,'max_dynamic':49152,'blocks_per_SM':8}
  required=2*(4096+8*64)*2+32+1024
  self.assertTrue(S.resource_gate(row,canonical,required))
  self.assertTrue(S.resource_gate(dict(row,static_shared=1024),canonical,required))
  for name,value in [('registers',33),('local',4),('static_shared',1),('static_shared',2048),('max_threads',128),('max_dynamic',required-1),('blocks_per_SM',0)]:
   with self.assertRaises(AssertionError):S.resource_gate(dict(row,**{name:value}),canonical,required)
  with self.assertRaises(AssertionError):S.resource_gate(row,dict(canonical,SHARED=0),required)
if __name__=='__main__':unittest.main()

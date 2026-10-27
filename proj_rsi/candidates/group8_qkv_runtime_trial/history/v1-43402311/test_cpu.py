"""No GPU imports or parent mutations. Invoke directly with Python."""
import importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'tools/group8_qkv_runtime_lab.py'
spec=importlib.util.spec_from_file_location('qkv_runtime_cpu_tested',P);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
class Checks(unittest.TestCase):
 def test_complete_balanced_plan(self):
  plan=D.measurement_plan();self.assertEqual(len(plan),210)
  paired=plan[120:];self.assertEqual(len(paired),90)
  for start in range(0,90,3):
   self.assertEqual({r['label'] for r in paired[start:start+3]},set(D.LABELS))
   self.assertEqual(len({r['input_seed'] for r in paired[start:start+3]}),1)
 def test_missing_or_misbound_terminal_fixture_rejected(self):
  plan=D.measurement_plan();last={k:next(r['input_seed'] for r in reversed(plan) if r['label']==k) for k in D.LABELS}
  self.assertTrue(D.terminal_record_contract(last,plan))
  for bad in ({'new':last['new']},dict(last,pw36=last['pw36']-1)):
   with self.assertRaises(AssertionError):D.terminal_record_contract(bad,plan)
 def test_actual_kernel_missing_ambiguous_or_changed_instructions(self):
  text='\n Function : kernel\n /*0000*/ LDG.E R1, [R2]; /*code*/\n'
  original=D.instructions(D.sass_body(text,'kernel'))
  self.assertEqual(original,['LDG.E R1, [R2];'])
  self.assertNotEqual(original,D.instructions(D.sass_body(text.replace('R1','R9'),'kernel')))
  with self.assertRaises(AssertionError):D.sass_body(text,'missing')
  with self.assertRaises(AssertionError):D.sass_body(text+text,'kernel')
 def test_source_lifecycle_gate_avoids_unsafe_terminal_replay(self):
  import ast
  tree=ast.parse(P.read_text());gate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='terminal_gate')
  names={n.func.attr for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
  self.assertFalse(names & {'copy_','fill_','replay','reset','zero_'})
if __name__=='__main__':unittest.main()

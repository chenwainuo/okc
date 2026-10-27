"""CPU-only adversarial binding and null/unit checks on actual collected exports."""
import copy
import importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actual_counter_analysis',HERE/'counter_analysis.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

class ActualCounterChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  collection=sorted(A.EXPORTS.glob('collection-*'))[-1]
  cls.tables=A.source_csv(collection/'logs/scheduler/native-profile-export-01/run/source.csv')
  sass=(A.CAPTURE/'logs/scheduler/native-profile-01/driver.DP.jit.sass').read_text()
  cls.instructions=A.instruction_rows(sass,'rq_px_dp')
  cls.table=next(t for t in cls.tables if 'rq_px_dp'in t['kernel'])
 def test_actual_complete_PC_predicate_instruction_positive(self):
  rows,base=A.bind_source(self.table,self.instructions);self.assertEqual(len(rows),848)
 def test_PC_only_tamper_rejected(self):
  bad=copy.deepcopy(self.table);bad['rows'][0]['Address']=hex(int(bad['rows'][0]['Address'],16)+16)
  with self.assertRaises(AssertionError):A.bind_source(bad,self.instructions)
 def test_predicate_only_tamper_rejected(self):
  bad=copy.deepcopy(self.table);row=next(r for r in bad['rows']if '@P1'in r['Source']);row['Source']=row['Source'].replace('@P1','@!P1',1)
  with self.assertRaises(AssertionError):A.bind_source(bad,self.instructions)
 def test_instruction_only_tamper_rejected(self):
  bad=copy.deepcopy(self.table);bad['rows'][0]['Source']='NOP'
  with self.assertRaises(AssertionError):A.bind_source(bad,self.instructions)
 def test_unknown_is_not_zero(self):
  self.assertIsNone(A.strict_sum([0,None]));self.assertIsNone(A.metric({}, {},'absent')['base_value'])
  self.assertIsNone(A.number('N/A'));self.assertIsNone(A.number('-'));self.assertEqual(A.number('0'),0)
 def test_scaled_export_bytes_are_converted_explicitly(self):
  m=A.metric({'read':'51.466240'},{'read':'Mbyte'},'read')
  self.assertEqual(m['value'],51.466240);self.assertEqual(m['base_value'],51466240);self.assertEqual(m['base_unit'],'byte')

if __name__=='__main__':unittest.main()

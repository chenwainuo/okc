"""Meaningful CPU format/accounting tests; no torch/candidate imports or GPU."""
import ast,hashlib,importlib.util,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load(ROOT/'proj_rsi/tools/rq11_actualescape.py','actualescape')
C=load(ROOT/'proj_rsi/candidates/rq11_codec/codec.py','frozen_codec')
OS=(0,8,16,24,4,12,20,28);SS=(7,15,23,31,11,19,27,3)
class Proof(unittest.TestCase):
 def test_exhaustive_storage_patterns_all_legal_windows(self):
  raw=np.arange(65536,dtype=np.uint16).reshape(64,1,4,32,8)
  for width in (4,8):
   for base in range(129-width):
    _,bad,coords,xor=A.encode_decode(raw,base,width,OS,SS)
    self.assertEqual(len(xor),int(bad.sum()))
 def test_exact_frozen11_byte_layout_all_patterns_three_windows(self):
  raw=np.arange(65536,dtype=np.uint16).reshape(64,1,4,32,8)
  for base in (0,57,124):
   ours,*_=A.encode_decode(raw,base,4,OS,SS)
   for i in range(64):
    frozen,entries=C.pack(raw[i,0],base)
    self.assertTrue(np.array_equal(ours[i,0],frozen))
    self.assertTrue(np.array_equal(C.unpack(ours[i,0],base,entries),raw[i,0]))
 def test_real_artifact_and_all_cost_identities(self):
  d=json.loads(Path(__file__).with_name('actual-cpu-reference.json').read_text());self.assertEqual(d['status'],'PASS_CPU_REFERENCE_FIXTURE_ACCOUNTING_GPU_DISTRIBUTION_UNKNOWN');self.assertFalse(d['CUDA_initialized_after'])
  for p,h in d['sources'].items():self.assertEqual(hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),h)
  self.assertEqual(sum(l['values'] for l in d['layers'].values()),94371840)
  for layer in d['layers'].values():
   self.assertEqual(sum(layer['h7_histogram']),layer['values'])
   for bits,f in layer['formats'].items():
    self.assertEqual(sum(i*n for i,n in enumerate(f['escapes_per_warp_unit_histogram'])),f['escape_count'])
    for g in f['geometry']:
     self.assertEqual(sum(g['group_lengths']),f['escape_count']);self.assertGreater(g['CAP_entries'],max(g['group_lengths']))
     self.assertEqual(g['device_codec_total_bytes'],g['low_bytes']+g['selector_header_bytes']+g['actual_escape_bytes']+g['fixed_stride_unused_bytes']+g['trailing_guard_bytes']+g['escape_sink_bytes'])
     self.assertEqual(g['selector_header_padding_bits'],0)
 def test_independent_group_mapping_crosses_unequal_split_boundaries(self):
  coords=(np.array([0,7,2,6]),np.array([0,8,9,15]),np.array([0,3,2,1]),np.array([31,0,15,23]),np.array([7,0,3,4]));xors=np.array([1,127,64,3],np.uint16)
  count,groups=A.entry_proof(coords,xors,16,7);self.assertEqual(count,4);self.assertEqual(groups.shape,(7,8));self.assertEqual(groups[0,0],1);self.assertEqual(groups[3,7],1);self.assertEqual(groups[4,2],1);self.assertEqual(groups[6,6],1)
 def test_capacity_sentinel_rounding_cannot_truncate(self):
  self.assertEqual(A.unit_tests()['CAP_boundary_controls'],8)
  groups=[[C.entry(0,0,0,0,1)]*32];self.assertIsNone(C.csr(groups,32));a,cap=C.csr(groups,64);self.assertEqual(cap,64);self.assertEqual(a[32],C.SENTINEL)
 def test_nonfinite_serving_and_mutation_rejected(self):
  self.assertFalse(C.serving_admissible([0x7f80],[0x7f80]));self.assertFalse(C.serving_admissible([0x7fc1],[0x7fc1]));self.assertFalse(C.serving_admissible([1],[0]));self.assertTrue(C.serving_admissible([0,0x8000,0x7f7f],[0,0x8000,0x7f7f]))
if __name__=='__main__':unittest.main(verbosity=2)

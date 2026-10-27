import copy
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import TARGETS,LATE,PARENT,apply,edits,catalog
from protocol_proof import admit,expected_storage,check_storage,model

class PW41Tests(unittest.TestCase):
    def test_exact_parent_inverse_and_old_indices(self):
        old=PARENT.read_text();new=(Path(__file__).with_name('submission.py')).read_text()
        self.assertEqual(apply(new,edits(),True),old)
        self.assertEqual(catalog(new)['GVI'][:95],catalog(old)['GVI'])
        self.assertEqual(catalog(new)['GVI'][95:],[a[:-1]+(a[-1]|LATE,) for a in TARGETS])
    def test_epoch_placement_equivalence_and_repeated_calls(self):
        for s in (4,8,16):
            for epoch in (0,1,73):
                admit(epoch*s,epoch,s)
                self.assertEqual({(epoch*s+k)//s+1 for k in reversed(range(s))},{epoch+1})
    def test_epoch_invalid_overflow_and_concurrent_streams_hold(self):
        for args,kw in [((3,0,4),{}),((8,1,4),{}),((0,0,4),{'stream_owners':2}),((0,0,4),{'co_resident':False}),(((1<<31)-16,(1<<27)-1,16),{})]:
            with self.assertRaises(ValueError):admit(*args,**kw)
    def test_actual_global_write_requires_zero_release_acquire(self):
        for kw in ({'omit_acquire':True},{'early_publish':True}):
            with self.assertRaises(AssertionError):model(2,True,True,**kw)
    def test_dormant_alias_fullstorage_and_offset_identity(self):
        before=[11]*2064;offset=8
        for t in range(16):before[offset+2*t]=32;before[offset+2*t+1]=2
        after=expected_storage(before,offset,16,16,3);identity=('cpu',1234,8256,offset,2048,1,'int32')
        check_storage(before,after,offset,16,16,3,identity,identity)
        for pos in (0,offset,offset+1,offset+100,2063):
            wrong=after.copy();wrong[pos]+=1
            with self.assertRaises(AssertionError):check_storage(before,wrong,offset,16,16,3,identity,identity)
        with self.assertRaises(AssertionError):check_storage(before,after,offset,16,16,3,identity,identity[:3]+(offset+1,2048,1,'int32'))
    def test_static_proof_and_no_gpu_import(self):
        from cpu_proof import source_proof,machine_proof
        self.assertEqual(source_proof()['old_GVI_prefix'],95)
        self.assertEqual(machine_proof()['current_actual_JIT_GV_exact'],95)
if __name__=='__main__':unittest.main()

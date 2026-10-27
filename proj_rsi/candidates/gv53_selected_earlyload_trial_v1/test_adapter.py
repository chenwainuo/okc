"""CPU C-stub tests for ctypes widths, stream, ownership and error propagation."""
import contextlib,ctypes as C,hashlib,importlib.util,subprocess,tempfile,types,unittest,weakref,gc
from pathlib import Path
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('thin_adapter',D/'adapter.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
STUB='''#include <stdint.h>
static uintptr_t stream_seen=0;static uint64_t table_seen=0;static int status=0;static int arm_seen=-1;
int gv53_run(int arm,const void*w,const void*e,const void*x,void*o,void*s,void*c,uint64_t table,int m,void*stream){stream_seen=(uintptr_t)stream;table_seen=table;arm_seen=arm;return status;}
int gv53_attrs(int arm,int field,int64_t*out){*out=field==6?1:64;return status;}
const char*gv53_error(int code){return "stub failure";}
int stub_arm(){return arm_seen;} void stub_status(int x){status=x;} uintptr_t stub_stream(){return stream_seen;} uint64_t stub_table(){return table_seen;}
'''
class Tensor:
 def __init__(self,shape,dtype,count=None,ptr=0x10000000100):self.shape=shape;self.dtype=dtype;self.device='cuda:1';self.is_cuda=True;self.count=count;self.ptr=ptr
 def numel(self):
  if self.count is not None:return self.count
  import math
  return math.prod(self.shape)
 def data_ptr(self):return self.ptr
 def is_contiguous(self):return True
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name);(p/'stub.c').write_text(STUB);cls.dll=p/'stub.so'
  subprocess.run(['cc','-shared','-fPIC',str(p/'stub.c'),'-o',str(cls.dll)],check=True);cls.sha=hashlib.sha256(cls.dll.read_bytes()).hexdigest()
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def setUp(self):
  self.synced=[];self.stream=types.SimpleNamespace(cuda_stream=0x123456789abc);self.cleaned=[]
  cuda=types.SimpleNamespace(device=lambda d:contextlib.nullcontext(),current_stream=lambda device:self.stream,synchronize=lambda d:self.synced.append(d))
  self.t=types.SimpleNamespace(cuda=cuda,uint8='u8',int32='i32',bfloat16='bf16',float32='f32')
  self.owner=A.ThinGV53(self.dll,self.sha,self.t,lambda:self.cleaned.append('clear')or[1,33554432,0]);self.owner.lib.stub_status.argtypes=[C.c_int];self.owner.lib.stub_status(0)
  self.owner.lib.stub_stream.restype=C.c_uint64;self.owner.lib.stub_table.restype=C.c_uint64
  self.buf=[Tensor((50331648,),'u8'),Tensor((131072,),'i32'),Tensor((1,8192),'bf16'),Tensor((1,4096),'bf16'),Tensor((65536,),'f32'),Tensor((65536,),'i32')]
 def test_explicit_current_stream_and_uint64(self):
  bound=self.owner.bind(*self.buf,0xfedcba9876543210);bound.run(1)
  self.assertEqual(self.owner.lib.stub_stream(),0x123456789abc);self.assertEqual(self.owner.lib.stub_table(),0xfedcba9876543210);self.assertFalse(self.synced)
  self.stream.cuda_stream=0x555555555;bound.run(0);self.assertEqual(self.owner.lib.stub_stream(),0x555555555)
 def test_strong_owners_until_explicit_drain(self):
  refs=[weakref.ref(x)for x in self.buf];bound=self.owner.bind(*self.buf,0);self.buf=None;gc.collect();self.assertTrue(all(x()is not None for x in refs));bound.run(0)
  bound.close_after_graph_reset();gc.collect();self.assertTrue(all(x()is None for x in refs));self.assertEqual(self.synced,['cuda:1'])
  with self.assertRaises(A.ThinError):bound.run(0)
 def test_immediate_native_error_propagates(self):
  b=self.owner.bind(*self.buf,0);self.owner.lib.stub_status(700)
  with self.assertRaisesRegex(A.ThinError,'700'):b.run(0)
 def test_negative_argument_status_and_attrs_error_propagate(self):
  self.assertEqual(self.owner.attrs(0,6,'cuda:1'),1);self.owner.lib.stub_status(-90)
  with self.assertRaisesRegex(A.ThinError,'-90'):self.owner.attrs(0,1,'cuda:1')
 def test_metadata_rejected_before_native_launch(self):
  self.buf[2].shape=(2,8192)
  with self.assertRaises(A.ThinError):self.owner.bind(*self.buf,0)
 def test_alignment_rejected(self):
  self.buf[0].ptr+=1
  with self.assertRaises(A.ThinError):self.owner.bind(*self.buf,0)
 def test_wrong_dtype_and_device_rejected(self):
  self.buf[0].dtype='wrong'
  with self.assertRaises(A.ThinError):self.owner.bind(*self.buf,0)
  self.buf[0].dtype='u8';self.buf[0].device='cuda:0'
  with self.assertRaises(A.ThinError):self.owner.bind(*self.buf,0)
 def test_dll_pin_and_cleanup_binding(self):
  with self.assertRaises(A.ThinError):A.ThinGV53(self.dll,'0'*64,self.t,lambda:None)
  with self.assertRaises(A.ThinError):A.ThinGV53(self.dll,self.sha,self.t,None)
  self.assertEqual(self.owner.clear_blas_cache(),[1,33554432,0]);self.assertEqual(self.cleaned,['clear'])
 def test_three_native_arm_M_whitelist(self):
  b=self.owner.bind(*self.buf,0)
  with self.assertRaises(A.ThinError):b.run(2)
  self.buf[2].shape=(8,8192);self.buf[3].shape=(8,4096);b=self.owner.bind(*self.buf,0)
  with self.assertRaises(A.ThinError):b.run(1)
  b.run(2);self.assertEqual(self.owner.lib.stub_arm(),2);b.run(0);self.assertEqual(self.owner.lib.stub_arm(),0)
if __name__=='__main__':unittest.main(verbosity=2)

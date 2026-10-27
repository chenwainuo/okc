"""Small fixed-GV53 ctypes seam, opt-in only; no Torch import or CUDA at import.

The harness owns BoundBuffers alongside every graph that can use its pointers.
It must reset those graphs before close_after_graph_reset(). No automatic DLL
unload, asynchronous lifetime guess, stream allocation or cleanup substitution.
"""
import ctypes as C
import hashlib
from pathlib import Path

class ThinError(RuntimeError):pass
class ThinGV53:
 def __init__(self,path,sha256,torch_module,cleanup_bridge):
  p=Path(path).resolve()
  if hashlib.sha256(p.read_bytes()).hexdigest()!=sha256:raise ThinError('DLL source/artifact pin mismatch')
  if not callable(cleanup_bridge):raise ThinError('existing verified Torch BLAS cleanup bridge required')
  self.torch=torch_module;self.cleanup_bridge=cleanup_bridge;self.lib=C.CDLL(str(p))
  self.lib.gv53_run.argtypes=[C.c_int]+[C.c_void_p]*6+[C.c_uint64,C.c_int,C.c_void_p];self.lib.gv53_run.restype=C.c_int
  self.lib.gv53_attrs.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_int64)];self.lib.gv53_attrs.restype=C.c_int
  self.lib.gv53_error.argtypes=[C.c_int];self.lib.gv53_error.restype=C.c_char_p
 def check(self,code):
  if code:raise ThinError('CUDA/thin ABI status %d: %s'%(code,self.lib.gv53_error(code).decode()))
 def attrs(self,arm,field,device):
  with self.torch.cuda.device(device):
   value=C.c_int64();self.check(self.lib.gv53_attrs(arm,field,C.byref(value)));return value.value
 def bind(self,W,esc,x,out,ws,cnt,table):return BoundBuffers(self,(W,esc,x,out,ws,cnt),table)
 def clear_blas_cache(self):
  # Called by the original harness after ALL graph consumers drain/reset;
  # this per-kernel DLL neither links Torch nor fabricates a clear operation.
  return self.cleanup_bridge()

class BoundBuffers:
 def __init__(self,owner,buffers,table):
  t=owner.torch;W,esc,x,out,ws,cnt=buffers
  if not all(z.is_cuda and z.is_contiguous()and z.device==x.device for z in buffers):raise ThinError('same-device contiguous CUDA tensors required')
  if tuple(z.dtype for z in buffers)!=(t.uint8,t.int32,t.bfloat16,t.bfloat16,t.float32,t.int32):raise ThinError('dtype mismatch')
  if len(x.shape)!=2 or x.shape[0]not in(1,8)or x.shape[1]!=8192 or tuple(out.shape)!=(x.shape[0],4096):raise ThinError('fixed GV53 shape mismatch')
  if W.numel()!=50331648 or esc.numel()!=131072 or ws.numel()<65536 or cnt.numel()<65536:raise ThinError('backing size mismatch')
  if any(z.data_ptr()%align for z,align in zip(buffers,(16,4,16,2,16,4))):raise ThinError('pointer alignment mismatch')
  if not isinstance(table,int)or isinstance(table,bool)or not -(1<<63)<=table<(1<<64):raise ThinError('table must fit original64bit ABI')
  # These strong references cover asynchronous execution and graph replay.
  self.owner=owner;self.buffers=buffers;self.table=table&((1<<64)-1);self.device=x.device;self.m=x.shape[0];self.closed=False
 def run(self,arm):
  if self.closed:raise ThinError('buffers retired after graph reset')
  if arm not in(0,1)or isinstance(arm,bool):raise ThinError('invalid arm')
  # Query CURRENT stream on every invocation, including Torch capture stream;
  # never assume stream0 and never create/synchronize a stream per launch.
  with self.owner.torch.cuda.device(self.device):
   stream=self.owner.torch.cuda.current_stream(device=self.device)
   code=self.owner.lib.gv53_run(arm,*[C.c_void_p(z.data_ptr())for z in self.buffers],self.table,self.m,C.c_void_p(stream.cuda_stream))
   self.owner.check(code)
  return 0
 def close_after_graph_reset(self):
  # Caller must first reset every graph retaining these pointers. Synchronizing
  # only a capture stream is insufficient if graphs replayed on other streams.
  if not self.closed:
   self.owner.torch.cuda.synchronize(self.device);self.buffers=None;self.closed=True

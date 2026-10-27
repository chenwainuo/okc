"""Fixed-device QKV128 research wrapper. Import never imports torch or creates CUDA state."""
from pathlib import Path
import hashlib, ast, json
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'group8_qkv'
PIN='d568e2612056b5bb37a1ebaadeb5c50d5c63345a736d660b144c25f4dac0d864'
HELPER=HERE.parent/'rs32/submission.py'
HELPER_PIN='a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'
FLAGS=['-O3','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas=-v']
CPP='int64_t qkv_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); int64_t qkv_info(int64_t,int64_t);'
def source():
 raw=(PARENT/'device_source.cu').read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
 # Drop only the offline instantiate function; exact kernel declarations/body restored.
 src=raw.decode();cut=src.index('void instantiate(RqP p,RqTmap x,RqTmap o){')
 return src[:cut]+(HERE/'host.cuh').read_text()
def identity():
 return hashlib.sha256((source()+CPP+repr(FLAGS)).encode()+(HERE/'runtime.py').read_bytes()).hexdigest()
def codec():
 raw=HELPER.read_bytes();assert hashlib.sha256(raw).hexdigest()==HELPER_PIN
 import torch
 tree=ast.parse(raw);names={'_c12','_unc12','_pack_nat','_rqpack','_rqunpack','_rqesc'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 scope={'torch':torch,'_OS':((0,4,8,12,16,20,24,28),(0,8,16,24,4,12,20,28)),'_SS':((3,11,19,27,7,15,23,31),(7,15,23,31,11,19,27,3))}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HELPER),'exec'),scope)
 return scope
_ext=None
def extension():
 global _ext
 if _ext is None:
  manifest=json.loads((HERE/'manifest.json').read_text())
  assert manifest['extension_identity']==identity(),'frozen wrapper identity drift'
  for name,sha in manifest['files'].items():
   assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,('trial artifact drift',name)
  from torch.utils.cpp_extension import load_inline
  _ext=load_inline('group8_qkv_trial_'+identity()[:20],cpp_sources=CPP,cuda_sources=source(),functions=['qkv_run','qkv_info'],extra_cuda_cflags=FLAGS,verbose=True)
 return _ext
class Pipeline:
 """Own every workspace through completion; caller retains this object for graph lifetime."""
 def __init__(self,w):
  import torch
  assert w.is_cuda and w.shape==(2304,8192) and w.dtype==torch.bfloat16 and w.is_contiguous()
  c=codec();packed=c['_rqpack'](w);assert packed is not None
  self.W,self.tab,esc6=packed
  assert torch.equal(c['_rqunpack'](self.W,self.tab,esc6,2304,8192),(w.view(torch.int16).int()&65535)),'actual W codec mismatch'
  # Admission guard BEFORE allocating CSR: cap derived from actual per-slot counts.
  if len(esc6):
   tile,ch=esc6[:,0].long(),esc6[:,1].long();gw=((ch//32)*18+tile//8)*8+tile%8
   longest=int(torch.bincount(gw,minlength=576).max());cap=((longest+32)//32)*32
  else:cap=32
  assert cap<=32800 and (576*cap+32)*4<=76_000_000,'CSR allocation guard'
  self.esc,_=c['_rqesc'](esc6,18,128,4,w.device)
  assert self.esc.numel()==576*cap+32
  self.xp=torch.empty(524288,dtype=torch.int32,device=w.device)
  self.partial=torch.empty((4,128,2304),dtype=torch.float32,device=w.device)
  self.original_weight=w
  with torch.cuda.device(w.device):
   self.mod=extension();self.occupancy=self.mod.qkv_info(1,5)
  self.retained=[];self.cap=cap
 def bind(self,x,out):
  """Retain capture buffers explicitly for the lifetime of this Pipeline."""
  self.retained.append((x,out))
  return lambda:self(x,out)
 def __call__(self,x,out):
  # No input-derived cache; every call launches pack, main, reduce on current stream.
  return self.mod.qkv_run(self.W,x,out,self.esc,self.xp,self.partial,self.tab)

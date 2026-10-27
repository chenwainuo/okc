"""Read-only CUDA Driver graph metadata, loaded only inside authorized capture.
ABI follows CUDA12.8 cuda.h CUDA_KERNEL_NODE_PARAMS_v2; no launch/init/set API.
"""
import ctypes as C

class KernelParamsV2(C.Structure):
 _fields_=[('func',C.c_void_p),('gridX',C.c_uint),('gridY',C.c_uint),('gridZ',C.c_uint),('blockX',C.c_uint),('blockY',C.c_uint),('blockZ',C.c_uint),('shared',C.c_uint),('params',C.c_void_p),('extra',C.c_void_p),('kern',C.c_void_p),('ctx',C.c_void_p)]
class UUID(C.Structure):_fields_=[('bytes',C.c_ubyte*16)]

class Inspector:
 def __init__(self,api=None):
  self.api=api if api is not None else C.CDLL('libcuda.so.1')
  self.functions={}
  specs={'cuGraphGetNodes':[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],'cuGraphNodeGetType':[C.c_void_p,C.POINTER(C.c_int)],
   'cuGraphKernelNodeGetParams_v2':[C.c_void_p,C.POINTER(KernelParamsV2)],'cuKernelGetFunction':[C.POINTER(C.c_void_p),C.c_void_p],
   'cuFuncGetName':[C.POINTER(C.c_char_p),C.c_void_p],'cuFuncGetAttribute':[C.POINTER(C.c_int),C.c_int,C.c_void_p],
   'cuOccupancyMaxActiveBlocksPerMultiprocessor':[C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t],
   'cuCtxGetDevice':[C.POINTER(C.c_int)],'cuDeviceGetName':[C.POINTER(C.c_char),C.c_int,C.c_int],
   'cuDeviceGetUuid':[C.POINTER(UUID),C.c_int],'cuDeviceGetAttribute':[C.POINTER(C.c_int),C.c_int,C.c_int],
   'cuDriverGetVersion':[C.POINTER(C.c_int)]}
  for name,args in specs.items():
   fn=getattr(self.api,name)
   if isinstance(fn,C._CFuncPtr):fn.argtypes=args;fn.restype=C.c_int
   self.functions[name]=fn
 def call(self,name,*args):
  rc=int(self.functions[name](*args));assert rc==0,(name,'CUDA metadata query',rc)
 def hardware(self):
  device=C.c_int();self.call('cuCtxGetDevice',C.byref(device))
  name=C.create_string_buffer(256);self.call('cuDeviceGetName',name,256,device.value)
  uuid=UUID();self.call('cuDeviceGetUuid',C.byref(uuid),device.value)
  sm=C.c_int();self.call('cuDeviceGetAttribute',C.byref(sm),16,device.value)
  major=C.c_int();minor=C.c_int()
  self.call('cuDeviceGetAttribute',C.byref(major),75,device.value)
  self.call('cuDeviceGetAttribute',C.byref(minor),76,device.value)
  version=C.c_int();self.call('cuDriverGetVersion',C.byref(version))
  b=bytes(uuid.bytes).hex();assert sm.value>0
  return {'name':name.value.decode(),'UUID_raw_hex':b,'UUID_formatted':'GPU-'+b[:8]+'-'+b[8:12]+'-'+b[12:16]+'-'+b[16:20]+'-'+b[20:],
   'SM_count':sm.value,'compute_capability':[major.value,minor.value],'current_device_ordinal':device.value,'CUDA_driver_version_integer':version.value}
 def inspect(self,graph):
  n=C.c_size_t();self.call('cuGraphGetNodes',C.c_void_p(graph),None,C.byref(n));assert 0<n.value<=64
  nodes=(C.c_void_p*n.value)();self.call('cuGraphGetNodes',C.c_void_p(graph),nodes,C.byref(n));records=[]
  for node in nodes:
   kind=C.c_int();self.call('cuGraphNodeGetType',C.c_void_p(node),C.byref(kind))
   record={'node_handle':hex(node),'node_type':kind.value}
   if kind.value==0:
    p=KernelParamsV2();self.call('cuGraphKernelNodeGetParams_v2',C.c_void_p(node),C.byref(p))
    func=C.c_void_p(p.func)
    if not func.value:
     assert p.kern,'kernel node lacks CUfunction/CUkernel';self.call('cuKernelGetFunction',C.byref(func),C.c_void_p(p.kern))
    name=C.c_char_p();self.call('cuFuncGetName',C.byref(name),func);assert name.value
    attrs={}
    for key,attribute in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers',4),('PTX_version',5),('binary_version',6),('max_dynamic_shared_bytes',8)]:
     value=C.c_int();self.call('cuFuncGetAttribute',C.byref(value),attribute,func);attrs[key]=value.value
    block=[p.blockX,p.blockY,p.blockZ];threads=block[0]*block[1]*block[2];assert threads>0
    occ=C.c_int();self.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',C.byref(occ),func,threads,p.shared)
    assert occ.value>0 and attrs['max_threads']>=threads
    record.update(function_mangled=name.value.decode(),function_handle=hex(func.value),grid=[p.gridX,p.gridY,p.gridZ],block=block,threads=threads,dynamic_shared_bytes=p.shared,attributes=attrs,
     theoretical_CTA_occupancy_per_SM=occ.value,occupancy_scope='Read-only driver calculation for actual block/dynamic shared; not measured achieved occupancy or cluster occupancy')
   records.append(record)
  return records

def one_kernel(records):
 assert len(records)==1 and records[0]['node_type']==0,('HOLD actual serving graph is not exactly one kernel node',records)
 return records[0]


def require_prepare_hardware(hardware):
 assert hardware['compute_capability']==[9,0],('HOLD prepare requires SM90',hardware)
 assert hardware['SM_count']>=128,('HOLD full prepare P1 grid requires at least 128 SMs',hardware)
 return hardware

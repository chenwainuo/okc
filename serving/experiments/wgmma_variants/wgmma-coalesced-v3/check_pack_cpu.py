"""CPU-only lossless repack+fragment+scale invariants, no CUDA API calls."""
import json
import sys
import torch
from routed_adapter import pack_weight,pack_scale

OFFSETS=(0,8,4,12,16,24,20,28)

def main():
 assert sys.byteorder=='little'
 checked=0
 for e,n,k in ((1,64,16),(2,128,64),(1,64,4096)):
  size=e*n*k//2
  raw=(torch.arange(size,dtype=torch.int64)%256).to(torch.uint8).reshape(e,n,k//2)
  packed=pack_weight(raw)
  assert packed.shape==(e,n//64,k//16,128) and packed.dtype==torch.int32
  assert packed.numel()*4==raw.numel()
  rebuilt=bytearray(size);seen=set()
  words=packed.reshape(-1).tolist();original=raw.reshape(-1).tolist()
  for expert in range(e):
   for tile in range(n//64):
    for group in range(k//16):
     for tid in range(128):
      word=words[((expert*(n//64)+tile)*(k//16)+group)*128+tid]&0xffffffff
      codes=[]
      for v in range(8):
       # AMAP coordinate equation is checked against CUTLASS by C++ static_assert.
       row=(tid//4)%8+16*(tid//32)+8*((v//2)%2)
       kk=2*(tid%4)+(v%2)+8*(v//4)
       index=(expert*n+tile*64+row)*k+group*16+kk
       assert index not in seen;seen.add(index)
       code=(word>>OFFSETS[v])&15;codes.append(code)
       assert code==((original[index//2]>>((index%2)*4))&15)
       rebuilt[index//2]|=code<<((index%2)*4)
      for h in (0,1):
       q=codes[h*4:h*4+4]
       old=q[0]<<8|q[1]<<24|q[2]<<12|q[3]<<28
       half=word>>(h*16);new=((half&255)<<8)|((half&65280)<<16)
       assert old==new
      checked+=8
  assert len(seen)==e*n*k and bytes(rebuilt)==bytes(original)
  scale=(torch.arange(e*n*(k//16),dtype=torch.int64)%256).to(torch.uint8).reshape(e,n,k//16)
  sw=pack_scale(scale)
  assert torch.equal(sw.permute(0,1,3,2).contiguous().reshape_as(scale),scale)
  assert sw.numel()==scale.numel()
  # Everylane's eight operands use exactly two rows; scale repeats overh.
  for tid in range(128):
   rows=[(tid//4)%8+16*(tid//32)+8*((v//2)%2) for v in range(8)]
   assert rows==[rows[0],rows[0],rows[2],rows[2]]*2
 print(json.dumps({'status':'pass','CPU_only':True,'CUDA_calls':0,'nibbles_checked':checked,
  'cases':3,'all_raw_byte_values_covered':True,'lossless_roundtrip':True,'old_new_dequant_input_words_equal':True,
  'scale_roundtrip':True,'payload_size_preserved':True}))
if __name__=='__main__':main()

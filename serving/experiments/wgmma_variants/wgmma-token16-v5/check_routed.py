"""Run only with root's GPU lease. Isolated actual alignment/JIT correctness.
Uses canonical packed weights and independent CPU raw-format reference.
"""
import json
import torch
from routed_adapter import prepare,project,module
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

def main():
    gen=torch.Generator().manual_seed(761)
    e,n,k,rows=2,64,4096,24
    codes=torch.randint(0,16,(e,n,k),generator=gen,dtype=torch.uint8)
    packed=(codes[...,::2]|(codes[...,1::2]<<4)).contiguous()
    rawscale=torch.tensor([.5,1,2,4],dtype=torch.float32)[torch.randint(0,4,(e,n,k//16),generator=gen)]
    rawglobal=torch.tensor([1,.5],dtype=torch.float32)
    x=(torch.randint(-7,8,(rows,k),generator=gen).float()/8).to(torch.bfloat16)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float32)
    w=lut[(codes&7).long()]*torch.where((codes&8)!=0,-1.,1.)
    w=(w*rawscale.repeat_interleave(16,-1)).to(torch.bfloat16).float()
    # CPU exactdyadic fixture reference, independent of Marlin decode/GPU GEMM.
    refs=[((x.float()@w[i].T).to(torch.bfloat16).float()*rawglobal[i]).to(torch.bfloat16) for i in range(e)]
    bank=prepare(packed.cuda(),rawscale.to(torch.float8_e4m3fn).cuda(),rawglobal.cuda())
    a=x.cuda();out=torch.empty((rows,n),dtype=torch.bfloat16,device='cuda')
    module() # compile outside graph
    tested=[]
    for name,ids in [('mixed',torch.tensor([0,1,-1]*8)),('skew',torch.tensor([1]*23+[-1])),('empty',torch.full((rows,),-1))]:
        ids=ids.to(torch.int32)
        expected=torch.zeros((rows,n),dtype=torch.bfloat16)
        for i in range(e):expected[ids==i]=refs[i][ids==i]
        meta=moe_align_block_size(ids.reshape(3,8).cuda(),16,e)
        project(a,bank,*meta,16,out=out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)
        tested.append(name)
    # Capture with valid metadata, then mutate metadata in-place to emptylocal.
    ids=torch.zeros((3,8),dtype=torch.int32,device='cuda')
    meta=moe_align_block_size(ids,16,e)
    project(a,bank,*meta,16,out=out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):project(a,bank,*meta,16,out=out)
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(),refs[0],rtol=0,atol=0)
    meta[1].fill_(-1)
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(),torch.zeros_like(refs[0]),rtol=0,atol=0)
    print(json.dumps({'pass':True,'cases':tested+['graph_valid','graph_empty_after_valid'],
                      'scope':'canonical layout actual route alignment, down projection N64 K4096; no timing'}))
if __name__=='__main__':main()

"""Root-leased T16/T8 mapping tests; independent CPU oracle, no timing."""
import hashlib,json,sys
from pathlib import Path
from rsi_worker import setup_paths
setup_paths(Path('/baseline'))
sys.path.insert(0,'/probe')
import torch
from routed_adapter import prepare,project,module
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

def fingerprint(tensors):
    h=hashlib.sha256()
    for t in tensors:
        h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    gen=torch.Generator().manual_seed(963017)
    e,n,k,rows=2,64,4096,64
    codes=torch.randint(0,16,(e,n,k),generator=gen,dtype=torch.uint8)
    packed=(codes[...,::2]|(codes[...,1::2]<<4)).contiguous()
    scales=torch.tensor([.5,1,2,4])[torch.randint(0,4,(e,n,k//16),generator=gen)]
    globals=torch.tensor([1,.5])
    x=(torch.randint(-7,8,(rows,k),generator=gen).float()/8).to(torch.bfloat16)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6])
    w=(lut[(codes&7).long()]*torch.where((codes&8)!=0,-1.,1.)*scales.repeat_interleave(16,-1)).to(torch.bfloat16).float()
    refs=[((x.float()@w[i].T).to(torch.bfloat16).float()*globals[i]).to(torch.bfloat16) for i in range(e)]
    bank=prepare(packed.cuda(),scales.to(torch.float8_e4m3fn).cuda(),globals.cuda())
    a=x.cuda();out=torch.empty((rows,n),dtype=torch.bfloat16,device='cuda');module()
    records=[]
    def verify(meta,expected,before,block,kind):
        torch.cuda.synchronize()
        assert fingerprint((a,*meta))==before,'candidate mutated activation or routing'
        torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)
        records.append({'block_m':block,'token_tile':16 if block%16==0 else 8,'kind':kind,'pass':True})
    for block in (8,16,24,32,48,64):
        for kind,ids in [('partial',torch.tensor([0]*17+[1]*9+[-1]*38)),
                         ('skew',torch.tensor([1]*63+[-1])),
                         ('empty',torch.full((rows,),-1)),('allpad',torch.full((rows,),-1))]:
            expected=torch.zeros((rows,n),dtype=torch.bfloat16)
            for i in range(e):expected[ids==i]=refs[i][ids==i]
            if kind=='allpad':
                # Valid expert and nonzero post, every route sentinel padded:
                # exercises GMMA load/output predicates, not expert earlyreturn.
                meta=(torch.full((block,),rows,dtype=torch.int32,device='cuda'),
                      torch.zeros((1,),dtype=torch.int32,device='cuda'),
                      torch.tensor([block],dtype=torch.int32,device='cuda'))
            else:meta=moe_align_block_size(ids.to(torch.int32).reshape(8,8).cuda(),block,e)
            assert 0<=int(meta[2].item())<=meta[0].numel() and int(meta[2].item())%block==0
            before=fingerprint((a,*meta));out.fill_(float('nan'))
            project(a,bank,*meta,block,out=out);verify(meta,expected,before,block,kind)
        meta=moe_align_block_size(torch.zeros((8,8),dtype=torch.int32,device='cuda'),block,e)
        before=fingerprint((a,*meta));saved=meta[1].clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):project(a,bank,*meta,block,out=out)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        assert fingerprint((a,*meta))==before
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):project(a,bank,*meta,block,out=out)
        assert fingerprint((a,*meta))==before
        graph.replay();verify(meta,refs[0],before,block,'graph_valid')
        meta[1].fill_(-1);before=fingerprint((a,*meta))
        graph.replay();verify(meta,torch.zeros_like(refs[0]),before,block,'graph_empty_after_valid')
        meta[1].copy_(saved);before=fingerprint((a,*meta))
        graph.replay();verify(meta,refs[0],before,block,'graph_valid_restored')
    report={'status':'complete','scope':'routed mapping correctness only, no performance claim','development_seed':963017,'case_count':len(records),'cases':records}
    Path('/output/routed-extended.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

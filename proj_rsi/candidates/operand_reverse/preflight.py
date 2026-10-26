"""CPU bit/mapping/lifetime proof plus one SM90a compile. Never launches CUDA."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

import torch
import prototype as P

HERE=P.HERE


def prmt(a,b,s):
    raw=(a | b<<32).to_bytes(8,"little")
    return sum(raw[(s>>(4*i))&7]<<(8*i) for i in range(4))


def dec(a,b,c,tb):
    lo=((c&0x87878787)+tb)&0xffffffff
    hi=((((c>>4)|(c<<28))&0x87878787)+tb)&0xffffffff
    return [prmt(a,lo,0x5140),prmt(a,lo,0x7362),prmt(b,hi,0x5140),prmt(b,hi,0x7362)]


def shuffle_transpose(quad):
    v=[r.copy() for r in quad]
    for bit in (1,2):
        old=[r.copy() for r in v]
        for lane in range(4):
            for reg in range(4):
                if (lane&bit)!=(reg&bit):v[lane][reg]=old[lane^bit][reg^bit]
    assert all(v[l][r]==quad[r][l] for l in range(4) for r in range(4))
    return v


def val(row,col):return ((row*19937+col*73+0x83d1)^(row*col*31))&65535
def pair(row,col):return val(row,col)|(val(row,col+1)<<16)


def xproof():
    n=0
    for stage in range(64):
        for wg in range(2):
            for warp in range(4):
                for g in range(8):
                    trans=[]
                    for half in range(2):
                        for band in range(2):
                            row=wg*64+warp*16+g+8*half
                            quad=[[pair(row,stage*64+band*32+t*8+2*r) for r in range(4)] for t in range(4)]
                            trans.append(shuffle_transpose(quad))
                    for t in range(4):
                        for q in range(4):
                            for i in range(4):
                                got=trans[(i%2)*2+q//2][t][2*(q%2)+i//2]
                                row=wg*64+warp*16+g+8*(i%2)
                                col=stage*64+q*16+(i//2)*8+2*t
                                assert got==pair(row,col)
                                n+=2
    assert n==128*4096
    return {"exact_BF16_values":n,"coalesced_LDG128_per_thread_stage":4,
            "SHFL_per_thread_stage":32,"SELP_per_thread_stage":32}


def wproof():
    gen=torch.Generator().manual_seed(5729)
    raw=torch.randint(65536,(128,4096),generator=gen,dtype=torch.int32)
    raw[0,:8]=torch.tensor([0,0x8000,0x0001,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1])
    lay,tb,es=P.S._rqpack(raw.to(torch.int16).view(torch.bfloat16))
    assert torch.equal(P.S._rqunpack(lay,tb,es,128,4096),raw)
    esc,_=P.S._rqesc(es,1,64,1,"cpu");cap=(esc.numel()-32)//8
    entries=[x&0xffffffff for x in esc.tolist()]
    patches={}
    for tile in range(8):
        for e in entries[tile*cap:(tile+1)*cap]:
            if e!=0xffffffff:patches.setdefault((tile,e>>17),[]).append(e)
    packed=bytes(lay.flatten().tolist());count=0
    for half in range(2):
        for s in range(64):
            bmem=bytearray(8192);seen=set()
            for w in range(4):
                tile=4*half+w;a=[]
                for lane in range(32):
                    p=(s*8+tile)*1536+lane*16
                    q=[struct.unpack_from("<4I",packed,p+o) for o in (0,512,1024)]
                    a.append(dec(q[0][0],q[0][1],q[2][0],tb)+dec(q[0][2],q[0][3],q[2][1],tb)
                             +dec(q[1][0],q[1][1],q[2][2],tb)+dec(q[1][2],q[1][3],q[2][3],tb))
                for e in patches.get((tile,s),[]):a[(e>>12)&31][(e>>8)&15]^=(e&127)<<(8+((e>>7)&1)*16)
                for lane in range(32):
                    g,t=lane//4,lane%4
                    for q in range(4):
                        for i in range(4):
                            row=16*w+8*(i%2)+g
                            dst=row*128+(((2*q+i//2)^(row&7))*16)+4*t
                            assert dst not in seen;seen.add(dst)
                            struct.pack_into("<I",bmem,dst,a[lane][q*4+i])
            assert len(seen)==2048
            for row in range(64):
                for col in range(64):
                    byte=col*2;addr=row*128+(((byte//16)^(row&7))*16)+byte%16
                    assert struct.unpack_from("<H",bmem,addr)[0]==int(raw[half*64+row,s*64+col])
                    count+=1
    assert count==128*4096 and cap>32
    return {"exact_BF16_values":count,"escapes":len(es),"cap":cap,"escape_lists_span32entry_windows":True,
            "escape_control_flow":"Bit proof reconstructs CSR entries; unchanged inherited rq_fix supplies reload semantics, GPU runtime pending",
            "shared_B_SW128_bijection":True}


def outputproof():
    slab=bytearray(128*64*2);seen=set();used=set()
    for wg in range(2):
        for w in range(4):
            for lane in range(32):
                g,t=lane//4,lane%4
                for j in range(8):
                    row=wg*64+w*16+g;col=j*8+2*t
                    dst=row*128+((j^(row&7))*16)+4*t
                    for h in (0,1):
                        addr=dst+h*8*128
                        assert addr not in seen;seen.add(addr)
                        struct.pack_into("<I",slab,addr,pair(row+8*h,col))
                        used.add((wg,w,lane,4*j+2*h));used.add((wg,w,lane,4*j+2*h+1))
    assert len(seen)==4096 and len(used)==2*4*32*32
    for row in range(128):
        for col in range(64):
            byte=2*col;addr=row*128+(((byte//16)^(row&7))*16)+byte%16
            assert struct.unpack_from("<H",slab,addr)[0]==val(row,col)
    return {"all8192_D_scalars_observed":True,"single128x64_slab_bijection":True}


def lifetime():
    q=list(range(4));b=list(range(8));w=list(range(6));flight=[]
    for s in range(64):
        assert q[s%4]==s and w[s%6]==s and b[s%8]==s
        assert s%2 not in flight;flight.append(s%2)
        if s+4<64:q[s%4]=s+4
        if s+6<64:w[s%6]=s+6
        while len(flight)>1:flight.pop(0)
        if s>0 and s-1+8<64:b[(s-1)%8]=s-1+8
    return "PASS sequential XQ4/A2/W6/B8 tag consistency; not an asynchronous barrier interleaving simulation"


def main():
    cu=P.source();dev=P.device_source()
    (HERE/"extension_source.cu").write_text(cu);(HERE/"device_source.cu").write_text(dev)
    result={"status":"CPU_DEVICE_PROOF_ONLY_GPU_UNRUN","X":xproof(),"W":wproof(),"output":outputproof(),"lifetime":lifetime(),
            "source_sha256":hashlib.sha256(cu.encode()).hexdigest(),"device_sha256":hashlib.sha256(dev.encode()).hexdigest(),
            "CTA_register_pool":{"initial_static":168*384,"producer64_math216":128*64+256*216,"spare":1024},
            "shared_bytes":103600,"geometry":{"m":128,"n":8192,"k":4096,"channel_tile":64,"CTAs":128,"threads":384,"KS":1}}
    nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory(prefix="reverse-") as tmp:
        out=Path(tmp)/"reverse.cubin"
        r=subprocess.run([nvcc,*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),"-o",str(out)],text=True,capture_output=True)
        log=r.stdout+r.stderr;assert r.returncode==0 and not re.search(r"C75\d\d|warning|error",log,re.I),log
        assert "Used 168 registers" in log and "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in log,log
        sass=subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")),"--dump-sass",str(out)],text=True)
        assert "USETMAXREG.DEALLOC.CTAPOOL 0x40" in sass and "USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xd8" in sass,sass[:150]
        assert not re.search(r"\b(?:LDL|STL)\b",sass)
        counts={"HGMMA":sass.count("HGMMA."),"WARPGROUP_ARRIVE":sass.count("WARPGROUP.ARRIVE"),"UTMASTG":sass.count("UTMASTG"),
                "UBLKCP_W":sass.count("UBLKCP"),"UTMALDG_X":sass.count("UTMALDG"),"SHFL":sass.count("SHFL.")}
        assert counts["HGMMA"]==16 and counts["WARPGROUP_ARRIVE"]==4 and counts["UTMASTG"]==1 and counts["UTMALDG_X"]==0,counts
        result["ptxas"]=log;result["SASS"]=counts
        result["SASS_sha256"]=hashlib.sha256(sass.encode()).hexdigest()
    result["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"prototype.py",HERE/"kernel_body.cuh",Path(__file__)]}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

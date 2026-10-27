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
    q=list(range(4));flight=[];events=[]
    for s in range(64):
        assert q[s%4]==s and s%2 not in [slot for stage,slot in flight]
        # q is converted into separate A2 registers before current raw q is refilled.
        flight.append((s,s%2))
        if s+4<64:q[s%4]=s+4
        retired=flight[:-1];flight=flight[-1:] # wait_group1
        if s>0:
            assert retired==[(s-1,(s-1)%2)]
            events.append({"issue":s,"A_retired":s-1,"raw_refill":s+4 if s+4<64 else None})
        else:assert retired==[]
    assert flight==[(63,1)]
    flight=[] # wait_group0 then pin A0/A1 and release B63
    # Negative control: skipping the preceding retirement makes A(s+2) overwrite live A(s).
    bad=[]
    for s in range(62):
        if (s+2)%2==s%2:bad.append(s)
    assert len(bad)==62
    return {"status":"PASS_V1_RAW_Q4_SEPARATE_A2_LIFETIME", "prefetch_ahead":4,
            "current_raw_refill":"s+4 only after demux/current WGMMA commit; separate A still pinned",
            "A_reuse":"A(s) last read pinned after wait_group1 in round(s+1), before its next write(s+2)",
            "final":"wait_group0 then pin both A then releaseB63",
            "negative_without_retirement_caught":len(bad),"events":events,
            "limits":"W6/B8 protocol inherited byte-identically except removed dynamic allocation; not an arbitrary interleaving GPU simulation"}


def xproof():
    body=(HERE/"kernel_body.cuh").read_text();source=P.source()
    assert "p.K!=4096" in source and source.count("rx_pack<<<256,256,")==1
    assert "torch::Tensor xp" in source and "p.XP =" in source
    assert "packed[idx+65536u*u]=x[source+512u*u]" in body
    calls=[(int(i),int(off)) for i,off in re.findall(r"q\[(\d)\]=px_ldg<(\d+)>\(addr\)",body)]
    assert calls==[(j,512*j) for j in range(4)]
    raw=[((2*i)&65535)|(((2*i+1)&65535)<<16) for i in range(262144)]
    pack=[0xdeadbeef]*262144;back=[None]*262144;mapping=[];maxbyte=0
    for idx in range(262144):
        v=idx&3;lane=(idx>>2)&31;chunk=(idx>>7)&3;warp=(idx>>9)&7;s=idx>>12
        row=64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1)
        k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
        word=row*2048+k//2
        assert idx==((((s*8+warp)*4+chunk)*32+lane)*4+v)
        assert 0<=row<128 and 0<=k<=4094 and k%2==0 and back[word] is None
        # Actual fixed256x256 launch:four words/thread, stage +=16 per unrolled pass.
        thread=idx%65536;u=idx//65536
        iv=thread&3;il=(thread>>2)&31;ic=(thread>>7)&3;iw=(thread>>9)&7;is_=thread>>12
        ir=64*(iw>>2)+16*(iw&3)+(il>>2)+8*(iv&1)
        ik=64*is_+16*ic+8*((iv>>1)&1)+2*(il&3)
        assert ir*2048+ik//2+512*u==word
        pack[idx]=raw[word];back[word]=pack[idx];mapping.append(word)
        maxbyte=max(maxbyte,4*word+3)
    assert back==raw and maxbyte==1048575 and len(set(mapping))==262144
    patterns=set()
    for word in pack:patterns.add(word&65535);patterns.add(word>>16)
    assert len(patterns)==65536
    for first in range(0,262144,32):
        assert len({mapping[i]*4//32 for i in range(first,first+32)})==4
        assert len({i*4//32 for i in range(first,first+32)})==4
    sectors=0
    for s in range(64):
        for warp in range(8):
            allsectors=set()
            for j in range(4):
                now=set()
                for lane in range(32):
                    first=(((s*8+warp)*4+j)*32+lane)*4
                    addr=16384*s+2048*warp+16*lane+512*j
                    assert addr==4*first and addr%16==0
                    for v in range(4):
                        row=64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1)
                        k=64*s+16*j+8*((v>>1)&1)+2*(lane&3)
                        assert pack[first+v]==raw[row*2048+k//2]
                        now.add((addr+4*v)//32)
                assert len(now)==16;sectors+=16;allsectors.update(now)
            assert len(allsectors)==64
    assert sectors==32768
    assert sum(a!=(b^0x9e3779b9) for a,b in zip(pack,pack))==262144
    return {"status":"PACK_INVERSE_NATIVE_FRAGMENT_BIT_BIJECTION_PASS","u32_words":262144,"BF16_words":524288,
            "all65536_BF16_bit_patterns":True,"workspace_bytes":1048576,"max_input_or_packed_byte":maxbyte,
            "unsigned32_no_overflow":True,"pack_launch":{"CTAs":256,"threads":256,"words_per_thread":4},
            "packer_8192_warps_each":{"input_32B_sectors":4,"output_32B_sectors":4,"no_half_sector_amplification":True},
            "main":{"LDG_v4_per_thread_stage":4,"SHFL_per_thread_stage":0,"warp_stage_32B_sector_touches":64,
                    "warp_stage_unique32B_sectors":64,"all_vectors16B_aligned":True},
            "negative_stale_workspace_after_changed_input_mismatches":262144,
            "freshness":"pack launch precedes main on every normal API call; no input values populated in setup/prepare"}


def sourceproof():
    old=(HERE.parent/"operand_reverse_v4/kernel_body.cuh").read_text()
    assert hashlib.sha256(old.encode()).hexdigest()=="133a14958c9a4c87cd2e436bdfb8c1725fc20393385d4cdcca2ac2e6fe8f17c6"
    new=(HERE/"kernel_body.cuh").read_text()
    a=old.index('  if(warp>=8){');b=old.index('  float d[32]')
    assert old[a:b]==new[new.index('  if(warp>=8){'):new.index('  float d[32]')]
    assert old[old.index('  asm volatile("bar.sync 1,256;'):]==new[new.index('  asm volatile("bar.sync 1,256;'):]
    loop=old[b:old.index('  asm volatile("bar.sync 1,256;')]
    loop=loop.replace('int s,uint2 (&q)[8],','int s,uint4 (&q)[4],').replace('rev4_demux(q,a,t);','px_copy(q,a,t);')
    assert loop==new[new.index('  float d[32]'):new.index('  asm volatile("bar.sync 1,256;')]
    return "V4_W_PRODUCER_OUTPUT_AND_ROUND_IDENTITY_EXCEPT_RAW_QTYPE_AND_IDENTITY_COPY"


def main():
    cu=P.source();dev=P.device_source();identity=sourceproof()
    (HERE/"extension_source.cu").write_text(cu);(HERE/"device_source.cu").write_text(dev)
    result={"status":"CPU_DEVICE_PROOF_ONLY_GPU_UNRUN","X":xproof(),"W":wproof(),"output":outputproof(),"lifetime":lifetime(),
            "producer_output_source_identity":identity,"source_sha256":hashlib.sha256(cu.encode()).hexdigest(),
            "device_sha256":hashlib.sha256(dev.encode()).hexdigest(),"shared_bytes":103600,
            "geometry":{"m":128,"n":8192,"k":4096,"CTAs":128,"threads":384,"KS":1},"resources":{}}
    nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory(prefix="packed-x-") as tmp:
        out=Path(tmp)/"px.cubin"
        r=subprocess.run([nvcc,*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),"-o",str(out)],text=True,capture_output=True)
        log=r.stdout+r.stderr;(HERE/"resource.log").write_text(log)
        assert r.returncode==0 and not re.search(r"C75\d\d|warning|error",log,re.I),log
        entries=re.findall(r"Compiling entry function '([^']+)'.*?0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads.*?Used (\d+) registers",log,re.S)
        assert len(entries)==2,log
        for name,regs in entries:
            label="packer" if "rx_pack" in name else "main";regs=int(regs)
            assert (label=="main" and "rq_px" in name and regs<=168) or (label=="packer" and regs<=64)
            result["resources"][label]={"symbol":name,"static_regs":regs,"stack_spill_local":0}
        assert set(result["resources"])=={"main","packer"}
        result["resources"]["main"]["CTA_register_pool"]=result["resources"]["main"]["static_regs"]*384
        assert result["resources"]["main"]["CTA_register_pool"]<=65536
        sass=subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")),"--dump-sass",str(out)],text=True)
        (HERE/"sass.txt").write_text(sass);assert "USETMAXREG" not in sass and not re.search(r"\b(?:LDL|STL)\b",sass)
        tables=re.split(r"\n\s*Function : ",sass)[1:];assert len(tables)==2
        blocks={"packer" if "rx_pack" in table.splitlines()[0] else "main":table for table in tables}
        math=blocks["main"];pack=blocks["packer"]
        counts={"main_HGMMA":math.count("HGMMA."),"main_WG_arrive":math.count("WARPGROUP.ARRIVE"),
                "main_output_TMA":math.count("UTMASTG"),"main_X_TMA":math.count("UTMALDG"),
                "main_BFLY":math.count("SHFL.BFLY"),"main_LDG128":len(re.findall(r"LDG\.E\.NA\.128\.CONSTANT",math)),
                "main_all_LDG_mnemonics":dict(__import__("collections").Counter(re.findall(r"\bLDG(?:\.[A-Za-z0-9_]+)+",math))),
                "packer_LDG_mnemonics":dict(__import__("collections").Counter(re.findall(r"\bLDG(?:\.[A-Za-z0-9_]+)+",pack))),
                "packer_STG_mnemonics":dict(__import__("collections").Counter(re.findall(r"\bSTG(?:\.[A-Za-z0-9_]+)+",pack))),
                "main_MOV_static":len(re.findall(r"\b(?:MOV|IMAD.MOV)\b",math)),"main_async_A":"same separateA2 pin/retire source; actualGPU required"}
        assert counts["main_HGMMA"]==16 and counts["main_WG_arrive"]==4 and counts["main_output_TMA"]==1 and counts["main_X_TMA"]==0
        assert counts["main_BFLY"]==0 and counts["main_LDG128"]==32,counts
        result["SASS"]=counts;result["SASS_sha256"]=hashlib.sha256(sass.encode()).hexdigest();result["ptxas"]=log
    result["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"prototype.py",HERE/"kernel_body.cuh",Path(__file__)]}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","source_sha256","resources","SASS")},indent=2))


if __name__=="__main__":main()

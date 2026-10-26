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
    # Direct native WGMMA-A map is checked against v1's independent four-lane transpose.
    n=0;touches=0;unique=0;indices=set();patterns=set()
    for stage in range(64):
        for wg in range(2):
            for warp in range(4):
                trans=[]
                for g in range(8):
                    groups=[]
                    for half in range(2):
                        for band in range(2):
                            row=wg*64+warp*16+g+8*half
                            quad=[[((row,stage*64+band*32+t*8+2*r),
                                    (row,stage*64+band*32+t*8+2*r+1)) for r in range(4)] for t in range(4)]
                            groups.append(shuffle_transpose(quad))
                    trans.append(groups)
                sectors=set()
                for kstep in range(4):
                    for i in range(4):
                        this=set()
                        for lane in range(32):
                            g,t=lane//4,lane%4
                            row=wg*64+warp*16+g+8*(i%2)
                            col=stage*64+16*kstep+8*(i//2)+2*t
                            assert 0<=row<128 and 0<=col<=4094 and col%2==0
                            got=((row,col),(row,col+1))
                            want=trans[g][(i%2)*2+kstep//2][t][2*(kstep%2)+i//2]
                            assert got==want
                            indices.update(got)
                            direct_bits=((row*4096+col)&65535)|(((row*4096+col+1)&65535)<<16)
                            transpose_bits=((want[0][0]*4096+want[0][1])&65535)|(((want[1][0]*4096+want[1][1])&65535)<<16)
                            assert direct_bits==transpose_bits
                            patterns.update((direct_bits&65535,direct_bits>>16))
                            this.add((2*(row*4096+col))//32);n+=2
                        assert len(this)==8
                        touches+=len(this);sectors.update(this)
                assert len(sectors)==64;unique+=len(sectors)
    assert n==128*4096
    # Across the input, every possible BF16 encoding appears eight times; no numeric conversion.
    assert len(patterns)==65536 and len(indices)==524288
    return {"exact_BF16_values":n,"scalar_u32_LDG_per_thread_stage":16,
            "SHFL_SELP_for_X":0,"cache_policy":"ld.global.ca.u32",
            "warp_stage_32B_sector_touches":128,"warp_stage_unique32B_sectors":64,
            "total_sector_touches":touches,"total_unique_sectors":unique,
            "all65536_BF16_patterns_covered":True,
            "cache_reuse":"Two half-sector accesses; actual hit/transport ratio requires GPU measurement"}


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
    q=list(range(4));inflight=[];events=[]
    for s in range(64):
        current=s%4;old=(s-1)%4
        assert q[current]==s
        assert all(slot!=current for stage,slot in inflight)
        inflight.append((s,current))
        # WGMMA wait_group1 retires every group except the newest committed group.
        retired=inflight[:-1];inflight=inflight[-1:]
        if s>0:
            assert retired==[(s-1,old)]
            assert all(slot!=old for stage,slot in inflight)
            if s+3<64:
                assert (s+3)%4==old
                q[old]=s+3
            events.append({"issue":s,"retire":s-1,"refill":s+3 if s+3<64 else None,"slot":old})
        else:assert retired==[]
    assert inflight==[(63,3)]
    inflight=[] # final wait_group0, all four arrays pinned before epilogue
    # Negative control: old v1 refill s+4 into current q, if used directly as A, is illegal.
    bad=[]
    for s in range(60):
        live={s%4}
        if (s+4)%4 in live:bad.append(s)
    assert len(bad)==60
    return {"status":"PASS_A4_ASYNC_LIFETIME_BY_COMMIT_ORDER", "initial_stages":[0,1,2,3],
            "refill":"after wait_group1+pin retired(s-1), prefetch(s+3) into its four-slot buffer",
            "final_group":"wait_group0 then pin all4 then release B63",
            "negative_current_slot_refill_caught":len(bad),"retirement_events":events,
            "limits":"W6/B8 producer protocol unchanged; proof is register async lifetime, not arbitrary GPU barrier interleaving"}


def main():
    cu=P.source();dev=P.device_source()
    old=(HERE.parent/"operand_reverse/kernel_body.cuh").read_text()
    new=(HERE/"kernel_body.cuh").read_text()
    producer=old[old.index('  if(warp>=8){'):old.index('  asm volatile("setmaxnreg.inc')]
    producer=producer.replace('    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\\n");\n','')
    assert producer==new[new.index('  if(warp>=8){'):new.index('  // Uniform static')]
    assert old[old.index('  asm volatile("bar.sync 1,256;'):]==new[new.index('  asm volatile("bar.sync 1,256;'):]
    assert 'q[4*kstep+i]=rev_direct_ldg(p.X+(size_t)row*p.K+k)' in new
    (HERE/"extension_source.cu").write_text(cu);(HERE/"device_source.cu").write_text(dev)
    result={"status":"CPU_DEVICE_PROOF_ONLY_GPU_UNRUN","X":xproof(),"W":wproof(),"output":outputproof(),"lifetime":lifetime(),
            "producer_output_source_identity":"UNCHANGED_EXCEPT_REMOVED_DYNAMIC_REGISTER_DECLARATION",
            "source_sha256":hashlib.sha256(cu.encode()).hexdigest(),"device_sha256":hashlib.sha256(dev.encode()).hexdigest(),
            "CTA_register_pool":{"mode":"UNIFORM_STATIC_NO_SETMAXNREG","dynamic_role_requests":0},
            "shared_bytes":103600,"geometry":{"m":128,"n":8192,"k":4096,"channel_tile":64,"CTAs":128,"threads":384,"KS":1}}
    nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory(prefix="reverse-") as tmp:
        out=Path(tmp)/"reverse.cubin"
        r=subprocess.run([nvcc,*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),"-o",str(out)],text=True,capture_output=True)
        log=r.stdout+r.stderr;assert r.returncode==0 and not re.search(r"C75\d\d|warning|error",log,re.I),log
        regs=int(re.search(r"Used (\d+) registers",log).group(1))
        assert regs<=168 and "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in log,log
        result["CTA_register_pool"].update({"actual_static_regs":regs,"pool":regs*384,"pool_limit":65536})
        sass=subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")),"--dump-sass",str(out)],text=True)
        assert "USETMAXREG" not in sass, "unexpected dynamic register allocation"
        assert "SHFL.BFLY" not in sass, "X transpose survived"
        assert not re.search(r"\b(?:LDL|STL)\b",sass)
        counts={"HGMMA":sass.count("HGMMA."),"WARPGROUP_ARRIVE":sass.count("WARPGROUP.ARRIVE"),"UTMASTG":sass.count("UTMASTG"),
                "UBLKCP_W":sass.count("UBLKCP"),"UTMALDG_X":sass.count("UTMALDG"),"SHFL":sass.count("SHFL."),"LDG_mnemonics":dict(__import__("collections").Counter(re.findall(r"\bLDG(?:\.[A-Za-z0-9_]+)+",sass)))}
        assert counts["HGMMA"]==16 and counts["WARPGROUP_ARRIVE"]==4 and counts["UTMASTG"]==1 and counts["UTMALDG_X"]==0,counts
        (HERE/"resource.log").write_text(log);(HERE/"sass.txt").write_text(sass)
        result["ptxas"]=log;result["SASS"]=counts
        result["SASS_sha256"]=hashlib.sha256(sass.encode()).hexdigest()
    result["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"prototype.py",HERE/"kernel_body.cuh",Path(__file__)]}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","source_sha256","device_sha256","CTA_register_pool","SASS")},indent=2))


if __name__=="__main__":main()

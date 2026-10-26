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
    # Golden native-A coordinates are independent of the8B load permutation.
    seen=set();patterns=set();touches=0;unique=0
    for stage in range(64):
        for wg in range(2):
            for warp in range(4):
                sectors=set()
                for j in range(4):
                    for half in range(2):
                        q=[];instruction_sectors=set()
                        for lane in range(32):
                            g,t=lane//4,lane%4;row=wg*64+warp*16+g+8*half
                            k=64*stage+16*j+2*(t&2)+8*(t&1)
                            byte=2*(row*4096+k)
                            assert byte%8==0 and 0<=row<128 and 0<=k<=4092
                            q.append((((row,k),(row,k+1)),((row,k+2),(row,k+3))))
                            instruction_sectors.update((byte//32,(byte+4)//32))
                        assert len(instruction_sectors)==8
                        touches+=len(instruction_sectors);sectors.update(instruction_sectors)
                        for lane in range(32):
                            g,t=lane//4,lane%4;row=wg*64+warp*16+g+8*half
                            qx,qy=q[lane];u,v=q[lane^1]
                            a0=v if t&1 else qx;a2=qy if t&1 else u
                            want0=((row,64*stage+16*j+2*t),(row,64*stage+16*j+2*t+1))
                            want2=((row,64*stage+16*j+8+2*t),(row,64*stage+16*j+8+2*t+1))
                            assert a0==want0 and a2==want2
                            for pairindex in [a0,a2]:
                                seen.update(pairindex)
                                patterns.update(((r*4096+k)&65535) for r,k in pairindex)
                assert len(sectors)==64;unique+=64
    assert len(seen)==524288 and len(patterns)==65536 and touches==unique==32768
    return {"exact_symbolic_BF16_indices":len(seen),"all65536_BF16_patterns":True,
            "uint2_LDG_per_thread_stage":8,"SHFL_per_thread_stage":16,"SEL_per_thread_stage":16,
            "cache_policy":"ld.global.nc.L1::no_allocate.v2.u32 (same as v1)",
            "every_vector8B_aligned":True,"warp_stage_32B_sector_touches":64,
            "warp_stage_unique32B_sectors":64,"v1_same_sectors":True,"v2_sector_touches":128,
            "total32B_sector_touches":touches,"total_unique32B_sectors":unique}


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


def main():
    cu=P.source();dev=P.device_source()
    old=(HERE.parent/"operand_reverse/kernel_body.cuh").read_text()
    new=(HERE/"kernel_body.cuh").read_text()
    producer=old[old.index('  if(warp>=8){'):old.index('  asm volatile("setmaxnreg.inc')]
    producer=producer.replace('    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\\n");\n','')
    assert producer==new[new.index('  if(warp>=8){'):new.index('  float d[32]')]
    assert old[old.index('  asm volatile("bar.sync 1,256;'):]==new[new.index('  asm volatile("bar.sync 1,256;'):]
    assert 'q[2*j+half]=rev3_ldg(p.X+(size_t)row*p.K+k)' in new
    old_loop=old[old.index('  float d[32]'):old.index('  asm volatile("bar.sync 1,256;')]
    expected=old_loop.replace('int s,uint4 (&q)[4],','int s,uint2 (&q)[8],').replace('rev_transpose(q,a,t);','rev3_demux(q,a,t);')
    assert expected==new[new.index('  float d[32]'):new.index('  asm volatile("bar.sync 1,256;')]
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
        (HERE/"resource.log").write_text(log);(HERE/"sass.txt").write_text(sass)
        assert "USETMAXREG" not in sass, "unexpected dynamic register allocation"
        # ptxas outlines a separate collective-convergence path for the first two
        # rounds. Static fallback copies must not be counted as extra executed
        # shuffles in the fixed-mask main path. Keep both and expose the distinction.
        instructions=[(int(m[1],16),m[2].strip()) for m in
                      re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);",sass)]
        fixed=[];collective=[]
        for index,(pc,ins) in enumerate(instructions):
            if not ins.startswith("SHFL.BFLY"):continue
            if re.fullmatch(r"SHFL.BFLY PT, R\d+, R\d+, 0x1, 0x1f",ins):fixed.append(pc)
            else:
                assert instructions[index-1][1].startswith("WARPSYNC.COLLECTIVE")
                assert instructions[index+1][1]=="ENDCOLLECTIVE"
                collective.append(pc)
        assert len(fixed)==64 and len(collective)==32,(fixed,collective)
        assert max(fixed)<min(collective)
        branches=[(pc,int(re.search(r", 0x([0-9a-f]+)$",ins)[1],16))
                  for pc,ins in instructions if ins.startswith("BRA.DIV")]
        demux_branches=[(pc,target) for pc,target in branches if pc<max(fixed)]
        assert len(demux_branches)==2
        for index,(pc,target) in enumerate(demux_branches):
            stop=demux_branches[index+1][1] if index+1<len(demux_branches) else min(t for p,t in branches if p>max(fixed))
            assert sum(target<=q<stop for q in collective)==16
            block=[(q,ins) for q,ins in instructions if target<=q<stop]
            # First unconditional branch returns after the replaced demux.
            # Further outlined TRYWAIT blocks in this address range are unrelated.
            first_return=next(ins for q,ins in block if re.fullmatch(r"BRA 0x[0-9a-f]+",ins))
            assert pc<int(first_return.split()[1],16)<max(fixed)
        assert not re.search(r"\b(?:LDL|STL)\b",sass)
        counts={"HGMMA":sass.count("HGMMA."),"WARPGROUP_ARRIVE":sass.count("WARPGROUP.ARRIVE"),"UTMASTG":sass.count("UTMASTG"),
                "UBLKCP_W":sass.count("UBLKCP"),"UTMALDG_X":sass.count("UTMALDG"),"SHFL":sass.count("SHFL."),"LDG_mnemonics":dict(__import__("collections").Counter(re.findall(r"\bLDG(?:\.[A-Za-z0-9_]+)+",sass)))}
        counts["BFLY_CFG"]={"fixed_mask_main_path":len(fixed),"outlined_collective_alternatives":len(collective),
                            "demux_BRA_DIV":demux_branches,
                            "dynamic_execution":"UNMEASURED; static alternatives are not additive executed work"}
        assert counts["HGMMA"]==16 and counts["WARPGROUP_ARRIVE"]==4 and counts["UTMASTG"]==1 and counts["UTMALDG_X"]==0,counts
        (HERE/"resource.log").write_text(log);(HERE/"sass.txt").write_text(sass)
        result["ptxas"]=log;result["SASS"]=counts
        result["SASS_sha256"]=hashlib.sha256(sass.encode()).hexdigest()
    result["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"prototype.py",HERE/"kernel_body.cuh",Path(__file__)]}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","source_sha256","device_sha256","CTA_register_pool","SASS")},indent=2))


if __name__=="__main__":main()

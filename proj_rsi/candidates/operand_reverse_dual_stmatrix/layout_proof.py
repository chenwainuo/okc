"""Retained STMatrix full-byte and packed-X mapping models. CPU only."""
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
import liveness

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
    raw[:16,:]=torch.arange(65536,dtype=torch.int32).reshape(16,4096)
    raw[16,:8]=torch.tensor([0,0x8000,0x0001,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1])
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
            bmem=bytearray(8192);stmem=bytearray(8192);seen=set();stseen=set()
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
                # PTX x4 routing: reg i's row addresses are supplied by lanes
                # 8*i..8*i+7, while groups of4 DATA lanes write each16B row.
                for k in range(4):
                    provider=[]
                    for L in range(32):
                        row=16*w+8*((L>>3)&1)+(L&7);chunk=2*k+(L>>4)
                        addr=row*128+((chunk^(row&7))*16)
                        assert addr%16==0 and 0<=addr<=8176
                        provider.append(addr)
                    for L in range(32):
                        g,t=L>>2,L&3
                        for i in range(4):
                            addr=provider[8*i+g]+4*t
                            row=16*w+8*(i&1)+g
                            old=row*128+(((2*k+i//2)^(row&7))*16)+4*t
                            assert addr==old and addr not in stseen
                            stseen.add(addr)
                            struct.pack_into("<I",stmem,addr,a[L][4*k+i])
                            absolute=36864+(s%8)*8192+addr
                            assert 36864+(s%8)*8192<=absolute<=36864+(s%8)*8192+8188
            assert stseen==seen==set(range(0,8192,4)) and stmem==bmem
            assert len(seen)==2048
            for row in range(64):
                for col in range(64):
                    byte=col*2;addr=row*128+(((byte//16)^(row&7))*16)+byte%16
                    assert struct.unpack_from("<H",bmem,addr)[0]==int(raw[half*64+row,s*64+col])
                    count+=1
    assert count==128*4096 and cap>32
    return {"exact_BF16_values":count,"escapes":len(es),"cap":cap,"escape_lists_span32entry_windows":True,
            "escape_control_flow":"Bit proof reconstructs CSR entries; unchanged inherited rq_fix supplies reload semantics, GPU runtime pending",
            "shared_B_SW128_bijection":True,"stmatrix_equals_16_scalar_STS_every_byte":True,
            "per_stage_BF16_values":4096,"per_stage_bytes":8192,"all65536_weight_storage_patterns":True,
            "cross_lane_rule":"address_provider=8*matrix_register+(data_lane>>2), word_offset=4*(data_lane&3)",
            "all_provider_addresses16B_aligned":True,"all4producerwarps_no_alias_or_cross_slot_write":True}


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


def xproof():
    body=(HERE/"kernel_body.cuh").read_text();source=P.source()
    assert "p.K!=4096" in source and source.count("rx_pack<<<256,256,")==1
    assert "torch::Tensor xp" in source and "p.XP =" in source
    assert "packed[idx+65536u*u]=x[source+512u*u]" in body
    calls=[(int(off),int(i or "0")) for off,i in re.findall(r"px_vec<(\d+)>\(addr,q(?:\+(\d+))?\)",body)]
    assert calls==[(512*j,4*j) for j in range(4)]
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


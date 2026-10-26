"""CPU layout/bit/schedule proof and SM90a device compile. Never calls CUDA."""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent


def prmt(a,b,s):
    v = (a | b << 32).to_bytes(8,"little")
    return sum(v[(s >> (4*i)) & 7] << (8*i) for i in range(4))


def dec(a,b,c,tb):
    hl = ((c & 0x87878787)+tb) & 0xffffffff
    hh = ((((c >> 4) | (c << 28)) & 0x87878787)+tb) & 0xffffffff
    return [prmt(a,hl,0x5140),prmt(a,hl,0x7362),prmt(b,hh,0x5140),prmt(b,hh,0x7362)]


def bits():
    # Dense arbitrary BF16 bit patterns force >32 escapes/list, exercising window reload.
    gen = torch.Generator().manual_seed(7731)
    raw = torch.randint(65536,(128,1024),generator=gen,dtype=torch.int32)
    raw[0,:8] = torch.tensor([0,0x8000,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0x0001])
    w = raw.to(torch.int16).view(torch.bfloat16)
    lay,tb,esc6 = P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tb,esc6,128,1024),raw)
    esc,_ = P.S._rqesc(esc6,1,16,1,"cpu")
    cap = (esc.numel()-32)//8
    el = [x & 0xffffffff for x in esc.tolist()]
    packed = bytes(lay.flatten().tolist())
    checked = 0
    for half in range(2):
        for warp in range(4):
            tile = 4*half+warp
            entries = el[tile*cap:(tile+1)*cap]
            assert entries[-1] == 0xffffffff and cap > 32
            for s in range(16):
                a = []
                for lane in range(32):
                    p = (s*8+tile)*1536 + lane*16
                    q = [struct.unpack_from("<4I",packed,p+o) for o in (0,512,1024)]
                    a.append(dec(q[0][0],q[0][1],q[2][0],tb)
                             +dec(q[0][2],q[0][3],q[2][1],tb)
                             +dec(q[1][0],q[1][1],q[2][2],tb)
                             +dec(q[1][2],q[1][3],q[2][3],tb))
                for e in entries:
                    if e >> 17 == s:
                        lane,reg = (e >> 12)&31,(e >> 8)&15
                        a[lane][reg] ^= (e & 127) << (8+((e >> 7)&1)*16)
                for lane in range(32):
                    g,t = lane//4,lane%4
                    for q in range(4):
                        for i in range(4):
                            kh,hh = i//2,i%2
                            row = tile*16+8*hh+g
                            col = s*64+q*16+kh*8+2*t
                            got = a[lane][q*4+i]
                            assert got == int(raw[row,col]) | int(raw[row,col+1]) << 16
                            checked += 2
    assert checked == 128*1024
    return {"BF16_values_exact":checked,"escapes":len(esc6),"escape_cap":cap,
            "includes_zero_sign_subnormal_inf_nan_bits":True,"escape_window_reload_required":True}


def schedule():
    q = list(range(4))
    x = list(range(8))
    inflight = []
    for s in range(16):
        assert q[s%4] == s and x[s%8] == s
        # A is overwritten only after the previous group using its parity has retired.
        assert s%2 not in inflight
        inflight.append(s%2)
        if s+4<16: q[s%4]=s+4
        while len(inflight)>1: inflight.pop(0)
        if s>0 and s-1+8<16: x[(s-1)%8]=s-1+8
    assert len(inflight)==1
    return "PASS Q4/static slots, A2 wait_group1 lifetimes, X8 release-before-refill"


def main():
    source = P.source()
    device = P.device_source()
    (HERE/"extension_source.cu").write_text(source)
    (HERE/"device_source.cu").write_text(device)
    result = {"status":"CPU_PROOF_AND_DEVICE_COMPILE_ONLY_GPU_UNRUN",
              "bits":bits(),"schedule":schedule(),"geometry":{"NT":[32,128],"channel_tile":64,"KS":1,"Q_depth":4,"X_slots":8,"threads":256,"CTAs":128},
              "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
              "device_sha256":hashlib.sha256(device.encode()).hexdigest(),
              "parent_rs32_sha256":hashlib.sha256((HERE.parent/"rs32/submission.py").read_bytes()).hexdigest()}
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory(prefix="od-w-") as tmp:
        cubin = Path(tmp)/"od.cubin"
        cmd = [nvcc,*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),"-o",str(cubin)]
        c = subprocess.run(cmd,text=True,capture_output=True)
        log = c.stdout+c.stderr
        assert c.returncode==0 and not re.search(r"C75\d\d|warning|error",log,re.I),log
        assert log.count("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")==2,log
        sass = subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")),"--dump-sass",str(cubin)],text=True)
        counts = {}
        for sec in re.split(r"Function : ",sass)[1:]:
            nt = int(re.search(r"rqILi(32|128)E",sec).group(1))
            counts[nt] = {"HGMMA":sec.count("HGMMA."),"WARPGROUP_ARRIVE":sec.count("WARPGROUP.ARRIVE"),
                          "LDG128":len(re.findall(r"\bLDG[^\n]*\.128\b",sec)),"LDS128":sec.count("LDS.128"),
                          "UBLKCP_W":sec.count("UBLKCP"),"UTMALDG_X":sec.count("UTMALDG"),"UTMASTG_OUT":sec.count("UTMASTG")}
            assert counts[nt]["HGMMA"]==16 and counts[nt]["WARPGROUP_ARRIVE"]==4,counts
            assert counts[nt]["LDG128"]>=12 and counts[nt]["LDS128"]==0 and counts[nt]["UBLKCP_W"]==0,counts
            assert not re.search(r"\b(?:LDL|STL)\b",sec),sec[:100]
        result["ptxas"] = log
        result["SASS"] = counts
    result["files"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"prototype.py",HERE/"kernel_body.cuh",Path(__file__)]}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

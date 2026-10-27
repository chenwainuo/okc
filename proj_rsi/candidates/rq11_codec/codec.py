"""CPU-only RQ11 format/proofs; does not import submission or initialize CUDA.

One warp-unit: 1024 BF16 values, indexed [step=4,lane=32,value=8].
Low planes preserve stock RQ's 1024 bytes. Three header wordplanes add 384
bytes. CSR escape lists belong to (launch split, block, math warp), as in RQ.
"""
import json
from pathlib import Path
import numpy as np

SENTINEL = 0xffffffff
MAX_SEQ = 32766  # reserve seq 32767 so no valid entry aliases sentinel


def pack(bits, base):
    b = np.asarray(bits, dtype=np.uint16)
    if b.shape != (4, 32, 8) or not 0 <= base <= 124:
        raise ValueError("expected [4,32,8] BF16 bits and base in [0,124]")
    h = (b >> 8) & 127
    good = (h >= base) & (h < base + 4)
    code = np.where(good, h - base, 0).astype(np.uint32) | ((b.astype(np.uint32) >> 15) << 2)
    lo = (b & 255).astype(np.uint8)
    out = np.zeros(1408, dtype=np.uint8)
    out[:512] = lo[:2].transpose(1, 0, 2).reshape(-1)
    out[512:1024] = lo[2:].transpose(1, 0, 2).reshape(-1)
    # Per lane concatenate 4*24 bits, then transpose to 3 wordplanes.
    heads = np.zeros((3, 32), dtype=np.uint32)
    for lane in range(32):
        z = sum(int(code[q, lane, v]) << (24*q + 3*v) for q in range(4) for v in range(8))
        for word in range(3):
            heads[word, lane] = (z >> (32*word)) & 0xffffffff
    out[1024:] = heads.astype('<u4').view(np.uint8).reshape(-1)
    escapes = [(int(q), int(l), int(v), int(h[q,l,v]) ^ base) for q,l,v in np.argwhere(~good)]
    return out, escapes


def unpack(payload, base, escapes):
    p = np.asarray(payload, dtype=np.uint8)
    if p.shape != (1408,):
        raise ValueError("invalid payload length")
    out = np.empty((4,32,8), dtype=np.uint16)
    out[:2] = p[:512].reshape(32,2,8).transpose(1,0,2)
    out[2:] = p[512:1024].reshape(32,2,8).transpose(1,0,2)
    h = p[1024:].copy().view('<u4').reshape(3,32)
    for lane in range(32):
        z = sum(int(h[i,lane]) << (32*i) for i in range(3))
        for q in range(4):
            for v in range(8):
                c = (z >> (24*q+3*v)) & 7
                out[q,lane,v] |= ((base+(c&3)) | ((c>>2)<<7)) << 8
    for q,l,v,x in escapes:
        out[q,l,v] ^= x << 8
    return out


def entry(seq, q, lane, value, xor7):
    if not (0 <= seq <= MAX_SEQ and 0 <= q < 4 and 0 <= lane < 32 and 0 <= value < 8 and 0 <= xor7 < 128):
        raise ValueError("escape coordinate/seq/xor overflow")
    return (seq << 17) | (lane << 12) | ((q*4+value//2) << 8) | ((value%2) << 7) | xor7


def csr(groups, cap_limit=1<<24):
    """Exact stock-compatible fixed-stride CSR: strict > maxcount, 32 guard.

    Caller must reject before allocation on seq/entry/cap overflow and fall
    back to stock RQ12. No finite hard per-list capacity silently drops hits.
    cap_limit is an explicit allocation bound, not a format assumption.
    """
    if not groups or cap_limit < 32:
        raise ValueError("invalid group count/cap limit")
    if any(any(e < 0 or e >= SENTINEL or e >> 17 > MAX_SEQ for e in g) for g in groups):
        raise ValueError("invalid escape entry")
    cap = ((max(map(len, groups)) + 1 + 31)//32)*32
    if cap > cap_limit:
        return None
    a = np.full(len(groups)*cap+32, SENTINEL, dtype=np.uint32)
    for i,g in enumerate(groups):
        a[i*cap:i*cap+len(g)] = sorted(g, key=lambda e:e>>17)
    return a, cap


def bf16_rne(f):
    u = np.asarray(f, dtype=np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u>>16)&1)) >> 16).astype(np.uint16)


def serving_admissible(original, decoded):
    """Weight-only prepare gate; output numerical/probe gates remain separate."""
    a=np.asarray(original,dtype=np.uint16)
    b=np.asarray(decoded,dtype=np.uint16)
    return (a.shape==b.shape and np.array_equal(a,b)
            and bool(np.all((a&0x7f80)!=0x7f80))
            and bool(np.all((b&0x7f80)!=0x7f80)))


def prove():
    cases = 0
    # Every 16-bit pattern: signed zero/subnormals/normals/Inf/NaN payloads.
    # All bases cover exponent/carry boundaries, including base 124.
    raw = np.arange(65536, dtype=np.uint16)
    for c in range(4096):
        x=(c&0x3f)|((c&0xfc0)<<2)
        expanded=(x&0x0707)|((x&0x3838)<<1)
        assert expanded==sum(((c>>(3*i))&7)<<(4*i) for i in range(4))
    for base in range(125):
        for block in raw.reshape(-1,4,32,8):
            p,e = pack(block,base)
            assert np.array_equal(unpack(p,base,e),block)
            cases += block.size
    # Encode/decode ALL 17 low coordinate bits, and boundary legal sequence.
    for seq in (0,1,MAX_SEQ):
        for q in range(4):
            for l in range(32):
                for v in range(8):
                    for x in range(128):
                        e=entry(seq,q,l,v,x)
                        assert e>>17==seq and (e>>12)&31==l
                        assert (e>>8)&15==q*4+v//2 and (e>>7)&1==v%2 and e&127==x
    for count in (0,1,31,32,33,63,64,1024):
        got=csr([[entry(0,0,0,0,1)]*count,[]],2048)
        a,cap=got
        assert cap>count and cap%32==0 and a[count]==SENTINEL
        assert np.all(a[-32:]==SENTINEL)
    assert csr([[0]*32],32) is None
    for args in ((32767,0,0,0,1),(0,4,0,0,1),(0,0,32,0,1),(0,0,0,8,1),(0,0,0,0,128)):
        try: entry(*args)
        except ValueError: pass
        else: raise AssertionError("overflow accepted")
    # Simulate stock rq_fix register/half mapping and all 32-entry refill
    # boundaries with the full 65536-pattern corpus, base 58.
    for block in raw.reshape(-1,4,32,8):
        p,e=pack(block,58)
        partial=unpack(p,58,[])
        regs=partial.transpose(1,0,2).copy().reshape(32,16,2).astype(np.uint32)
        regs=regs[:,:,0]|(regs[:,:,1]<<16)
        arr,cap=csr([[entry(0,q,l,v,x) for q,l,v,x in e]])
        ei=0
        while int(arr[ei])!=SENTINEL:
            nx=int(arr[ei]); ln=(nx>>12)&31; reg=(nx>>8)&15
            regs[ln,reg]^=(nx&127)<<(8+16*((nx>>7)&1))
            ei+=1
        assert ei==len(e) and ei<cap
        halves=np.stack((regs&65535,regs>>16),-1).astype(np.uint16)
        assert np.array_equal(halves.reshape(32,4,8).transpose(1,0,2),block)
    assert serving_admissible([0,0x8000,1,0x7f7f],[0,0x8000,1,0x7f7f])
    assert not serving_admissible([0x7f80],[0x7f80])
    assert not serving_admissible([0x7fc1],[0x7fc1])
    assert not serving_admissible([1],[0])
    return {"exhaustive_bf16_base_pairs":cases,"coordinate_entry_patterns":3*4*32*8*128,
            "spread_selector_patterns":4096,"csr_capacity_tests":9,"csr_register_walk_bf16_patterns":65536,
            "finite_gate_tests":4}


def synthetic():
    rng=np.random.default_rng(5151)
    rows=[]
    for k in (1024,2048,3072,4096,8192):
        b=bf16_rne(rng.standard_normal(1<<20,dtype=np.float32)/np.sqrt(k))
        hist=np.bincount(((b>>8)&127).astype(np.int64),minlength=128)
        base=int(np.convolve(hist,np.ones(4,dtype=np.int64),'valid').argmax())
        bad=~((((b>>8)&127)>=base)&(((b>>8)&127)<base+4))
        n=len(b);esc=int(bad.sum())
        # Four warp lists with KC=256 units and KS=1. Group max drives
        # fixed-stride capacity; no actual GPU fixture/geometry claim.
        lists=bad.reshape(4,256,1024).sum((1,2))
        units=bad.reshape(-1,4,32,8)
        warp_counts=units.sum((1,2,3))
        lane_counts=units.sum((1,3))
        cap=int(((lists.max()+1+31)//32)*32)
        rows.append(dict(k=k,n=n,base=base,escapes=esc,escape_rate=esc/n,
            ideal_payload_plus_entries_bits=11+32*esc/n,
            four_group_fixed_stride_bits=11+8*(4*4*cap+128)/n,
            max_list=int(lists.max()),cap=cap,
            escapes_per_warp_unit=float(warp_counts.mean()),
            max_escapes_per_warp_unit=int(warp_counts.max()),
            mean_warp_max_lane_escapes=float(lane_counts.max(1).mean())))
    return rows


if __name__ == '__main__':
    result={"proof":prove(),"synthetic_not_gpu_fixture":synthetic()}
    print(json.dumps(result,indent=2))
    Path(__file__).with_name('cpu-results.json').write_text(json.dumps(result,indent=2)+'\n')

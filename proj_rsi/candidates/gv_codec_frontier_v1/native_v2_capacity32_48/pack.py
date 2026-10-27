"""Physical CPU pack of exact production GV fragments. No CUDA imports/calls.

Caller passes u16 numpy [T,KC,4,32,8], already from production _pack.
Returns None on any finite/capacity/byte-saving admission failure.
"""
import numpy as np


def pack(fr, base, geo, *, require_gain=True, capacity=64):
    fr = np.asarray(fr)
    assert fr.dtype == np.uint16 and fr.shape[2:] == (4, 32, 8)
    assert 0 <= base <= 124 and capacity <= 64
    if np.any(((fr >> 7) & 255) == 255):
        return None
    tiles, kc = fr.shape[:2]
    wr, wk, ku, splits, rt, _ = geo
    assert tiles % (wr*rt) == 0 and kc == wk*ku*splits
    units = tiles*kc
    b = fr.reshape(units, 4, 32, 8)
    hi = (b >> 8) & 127
    bad = (hi < base) | (hi >= base+4)
    count = bad.sum((1, 2, 3), dtype=np.int64)
    if count.max(initial=0) > capacity:
        return None
    padded_words = (count*2+15)//16*8
    begin = np.concatenate((np.array([0], np.int64), np.cumsum(padded_words)))
    if begin[-1] > np.iinfo(np.uint32).max:
        return None
    lane_count = bad.sum(3, dtype=np.int64)
    step_count = lane_count.sum(2, dtype=np.int64)
    before_step = np.cumsum(step_count, axis=1)-step_count
    desc = np.empty((units, 4, 2), dtype=np.uint32)
    desc[:, :, 0] = begin[:-1, None]+before_step
    desc[:, :, 1] = ((lane_count != 0).astype(np.uint64) << np.arange(32, dtype=np.uint64)[None,None,:]).sum(2)
    records = np.zeros(int(begin[-1]), dtype=np.uint16)
    u, q, lane, v = np.nonzero(bad)
    level = (np.cumsum(bad, axis=3)-1)[u,q,lane,v]
    order = np.argsort((((u*4+q)*8+level)*32+lane), kind='stable')
    u,q,lane,v,level = (a[order] for a in (u,q,lane,v,level))
    previous_units = np.cumsum(count)-count
    dest = begin[u]+np.arange(len(u))-previous_units[u]
    ent = v | ((hi[u,q,lane,v]^base) << 3) | ((lane_count[u,q,lane]>level+1).astype(np.uint16)<<15)
    records[dest] = ent.astype(np.uint16)
    lo = (b & 255).astype(np.uint8)
    p0 = lo[:, :2].transpose(0,2,1,3).reshape(units,32,16)
    p1 = lo[:, 2:].transpose(0,2,1,3).reshape(units,32,16)
    low = np.stack((p0,p1), axis=1)
    code = np.where(bad,0,hi-base).astype(np.uint32) | ((b.astype(np.uint32)>>15)<<2)
    header = np.zeros((units,3,32), dtype=np.uint32)
    for q in range(4):
        for v in range(8):
            bit = (q*8+v)*3
            word, shift = divmod(bit,32)
            header[:,word,:] |= code[:,q,:,v] << shift
            if shift > 29:
                header[:,word+1,:] |= code[:,q,:,v] >> (32-shift)
    total = low.nbytes+header.nbytes+desc.nbytes+records.nbytes
    groups = (tiles//(wr*rt))*splits*wr*wk
    old = units*1536+groups*128
    if require_gain and total >= old:
        return None
    return dict(low=low.reshape(tiles,kc,2,32,16), header=header.reshape(tiles,kc,3,32),
                desc=desc, records=records, units=units, words=int(begin[-1]),
                table=sum((base+i) << (8*i) for i in range(4)), base=base,
                total_bytes=total, incumbent_bytes=old, repairs=int(count.sum()),
                capacity=capacity, admitted=total < old)


def unpack(pk):
    tiles,kc = pk['low'].shape[:2]
    units = tiles*kc
    lo = pk['low'].reshape(units,2,32,16)
    rawlow = np.empty((units,4,32,8), np.uint16)
    rawlow[:,:2] = lo[:,0].reshape(units,32,2,8).transpose(0,2,1,3)
    rawlow[:,2:] = lo[:,1].reshape(units,32,2,8).transpose(0,2,1,3)
    header = pk['header'].reshape(units,3,32)
    out = rawlow.copy()
    for q in range(4):
        for v in range(8):
            word,shift = divmod((q*8+v)*3,32)
            code = header[:,word] >> shift
            if shift > 29:
                code |= header[:,word+1] << (32-shift)
            code &= 7
            out[:,q,:,v] |= ((pk['base']+(code&3)) << 8).astype(np.uint16) | ((code>>2)<<15).astype(np.uint16)
    for u in range(units):
        for q in range(4):
            pos,active = map(int,pk['desc'][u,q])
            for level in range(8):
                next_active = 0
                for lane in range(32):
                    if active & (1 << lane):
                        rank = (active & ((1<<lane)-1)).bit_count()
                        ent = int(pk['records'][pos+rank])
                        out[u,q,lane,ent&7] ^= ((ent>>3)&127)<<8
                        next_active |= (ent>>15)<<lane
                pos += active.bit_count()
                active = next_active
                if not active:
                    break
            assert not active
            if q < 3:
                assert pos == pk['desc'][u,q+1,0]
        end = pk['desc'][u+1,0,0] if u+1<units else pk['words']
        assert pos <= end and end-int(pk['desc'][u,0,0]) <= 64
    return out.reshape(tiles,kc,4,32,8)

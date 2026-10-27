"""KU8-only lossless escape re-bucketing. Units: uint32 words / uint16 records."""
KU=8
MASK=(7<<20)|(31<<15)|(15<<11)|(1<<10)|(1<<9)|127
def pack_group(old):
    if len(old)!=32:raise ValueError('exactly32 old uint32 words required')
    vals=[];ended=False
    for raw in old:
        x=int(raw)&0xffffffff
        if x==0xffffffff:ended=True;continue
        if ended or x&~MASK:raise ValueError('noncanonical old prefix/fields')
        if vals and x>>20 < vals[-1]>>20:raise ValueError('old units must be ordered')
        vals.append(x)
    if len(vals)>32:raise ValueError('overflow')
    headers=[];records=[]
    for u in range(KU):
        offsets=[]
        for q in range(4):
            offsets.append(len(records))
            for x in vals:
                if x>>20==u and ((x>>13)&3)==q:
                    records.append((x&127)|(((x>>10)&1)<<7)|(((x>>9)&1)<<8)|(((x>>11)&3)<<9)|(((x>>15)&31)<<11))
        offsets.append(len(records))
        headers.append(sum(o<<(6*q)for q,o in enumerate(offsets)))
    words=headers+[0]*24
    for i,r in enumerate(records):words[KU+i//2]|=r<<(16*(i&1))
    assert len(words)==32
    return words

def unpack_group(words):
    if len(words)!=32:raise ValueError('exact128B packet required')
    old=[];prev=0
    for u in range(KU):
        h=words[u];offsets=[(h>>(6*q))&63 for q in range(5)]
        if h>>30 or offsets[0]!=prev or any(a>b for a,b in zip(offsets,offsets[1:]))or offsets[-1]>32:raise ValueError('bad header')
        for q in range(4):
            for ep in range(offsets[q],offsets[q+1]):
                r=(words[KU+ep//2]>>(16*(ep&1)))&65535
                old.append((u<<20)|(((r>>11)&31)<<15)|((q*4+((r>>9)&3))<<11)|(((r>>7)&1)<<10)|(((r>>8)&1)<<9)|(r&127))
        prev=offsets[-1]
    return old

def pack_bank(old_words):
    """Host-side prototype adapter; pass the original accepted bank, never bypass overflow."""
    if old_words is None:return None  # original _gvesc overflow/fallback remains a fallback
    if len(old_words)%32:raise ValueError('bank must contain complete128B warpgroups')
    return [w for i in range(0,len(old_words),32) for w in pack_group(old_words[i:i+32])]

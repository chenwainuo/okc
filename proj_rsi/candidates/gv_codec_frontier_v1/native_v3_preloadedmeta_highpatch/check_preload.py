"""CPU ownership/address/group proof for both actual selected staging schemes."""
import importlib.util
import json
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pack_v2',D.with_name('native_v2_capacity32_48')/'pack.py')
pkmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(pkmod)
rng=np.random.default_rng(8492)
counts=0
for tiles,kc,geo,capacity,st in ((4,16,(2,2,8,1,1,0),32,1),(8,32,(1,2,16,1,1,2),48,2)):
    fr=(rng.integers(0,256,(tiles,kc,4,32,8),dtype=np.uint16)|
        (rng.integers(57,61,(tiles,kc,4,32,8),dtype=np.uint16)<<8))
    fr.reshape(-1)[::97]=120<<8
    flat=fr.reshape(-1,1024)
    flat[0]=57<<8;flat[0,:capacity]=120<<8
    flat[1]=57<<8
    pk=pkmod.pack(fr,57,geo,capacity=capacity)
    assert pk is not None and np.array_equal(pkmod.unpack(pk),fr)
    over=fr.copy().reshape(-1,1024);over[0,capacity]=120<<8
    assert pkmod.pack(over.reshape(fr.shape),57,geo,capacity=capacity)is None
    ku=geo[2];nwarp=geo[0]*geo[1]
    dbytes=ku*32+16;rbytes=capacity*2;wbytes=dbytes+st*rbytes
    assert dbytes%16==rbytes%16==wbytes%16==0
    for tile in range(tiles):
        for c0 in range(0,kc,ku):
            first=tile*kc+c0
            # All descriptor copies, plus lane0's boundary, precede wait0/sync.
            shared=pk['desc'][first:first+ku].reshape(-1).copy()
            boundary=int(pk['desc'][first+ku,0,0])if first+ku<pk['units']else pk['words']
            desc=np.concatenate((shared,np.array([boundary],np.uint32)))
            issued=list(range(st));landed=-1;owner=list(range(st))
            for u in range(ku):
                allowed=0 if st==1 or u+1==ku else 1
                while len(issued)>allowed:landed=issued.pop(0)
                assert landed>=u and owner[u%st]==u
                start,end=map(int,(desc[u*8],desc[(u+1)*8]))
                assert start==pk['desc'][first+u,0,0] and 0<=2*(end-start)<=rbytes
                assert start%8==end%8==0
                copied=pk['records'][start:end].copy()
                for q in range(4):
                    assert owner[u%st]==u
                    assert np.array_equal(desc[u*8+q*2:u*8+q*2+2],pk['desc'][first+u,q])
                    rel=int(desc[u*8+q*2])-start
                    assert 0<=rel<=len(copied)
                    counts+=1
                if u+st<ku:
                    owner[u%st]=u+st;issued.append(u+st)
            assert not issued
    # Every warp's persistent descriptors and record slots are disjoint.
    regions=[]
    for warp in range(nwarp):
        base=warp*wbytes
        regions.append((base,base+dbytes,'desc'))
        for slot in range(st):regions.append((base+dbytes+slot*rbytes,base+dbytes+(slot+1)*rbytes,'records'))
    assert all(a[1]<=b[0]for a,b in zip(regions,regions[1:]))
print(json.dumps(dict(state='CPU_PRELOAD_COPY_WAIT_OWNERSHIP_PASS_GPU_UNRUN',
    protected_unit_step_generations=counts,exact_physical_roundtrips=2,
    matrix_end_boundary_included=True,all_warp_regions_disjoint=True,
    original_record_slot_tail_waits_preserved=True),indent=2))

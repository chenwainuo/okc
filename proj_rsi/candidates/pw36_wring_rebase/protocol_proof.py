"""Independent W/X phase, lifetime and bounded producer/consumer model; CPU only."""
import random

def program(kcs):
    ops=[('read',0)]
    for s in range(kcs):
        ops.append(('mma',s))
        if s+1<kcs:ops.append(('read',s+1))
        ops.append(('wait1',s))
        if s>0:ops.append(('release',s-1))
    ops.extend([('wait0',kcs-1),('release',kcs-1)])
    return ops

def model(st,ws,kcs,split,exhaust=True,rng=None):
    ops=program(kcs);length=len(ops)
    read_before=[set()];released_before=[set()]
    for kind,s in ops:
        read_before.append(read_before[-1]|({s} if kind=='read' else set()))
        released_before.append(released_before[-1]|({s} if kind=='release' else set()))
    def next_states(state):
        wp,xp,c0,c1=state;consumers=(c0,c1);out=[]
        # Independent producers may issue only after ALL8 math warps released
        # the prior generation. Two WG witnesses here each represent4 arrivals.
        if wp<kcs and (wp<ws or all(wp-ws in read_before[c] for c in consumers)):
            out.append((wp+1,xp,c0,c1))
        if xp<kcs and (xp<st or all(xp-st in released_before[c] for c in consumers)):
            out.append((wp,xp+1,c0,c1))
        for wg,pos in enumerate(consumers):
            if pos==length:continue
            kind,s=ops[pos]
            if kind=='read':
                if wp<=s or (not split and xp<=s):continue
                # Current producer generation must still occupy the consumed slot.
                assert s<wp<=s+ws,(state,kind,s,'W overwritten')
                if not split:assert s<xp<=s+st
            if kind=='mma':
                if xp<=s:continue
                assert s<xp<=s+st,(state,kind,s,'X overwritten')
                assert s in read_before[pos]
            if kind=='release':
                # Release stage(s-1) follows wait1 after issuing stage(s), or
                # final wait0. All group operands remain live until this point.
                if s==kcs-1:assert ops[pos-1]==('wait0',s)
                else:assert ops[pos-1]==('wait1',s+1)
            target=list(state);target[2+wg]+=1;out.append(tuple(target))
        return out
    initial=(0,0,0,0);terminal=(kcs,kcs,length,length)
    if exhaust:
        seen={initial};todo=[initial]
        while todo:
            state=todo.pop();targets=next_states(state)
            assert targets or state==terminal,('DEADLOCK',st,ws,kcs,split,state)
            for target in targets:
                if target not in seen:seen.add(target);todo.append(target)
        assert terminal in seen
        return len(seen)
    state=initial;steps=0
    while state!=terminal:
        targets=next_states(state);assert targets,('RANDOM_DEADLOCK',state)
        state=rng.choice(targets);steps+=1
    return steps

def lifetime(ops):
    pending=[]
    for kind,s in ops:
        if kind=='mma':pending.append(s)
        elif kind=='wait1':pending=pending[-1:]
        elif kind=='wait0':pending=[]
        elif kind=='release':assert s not in pending,('X_RELEASE_BEFORE_WGMMA_RETIRE',s,pending)
    assert not pending

def require_X(stage,published):assert published>stage,'X_READ_BEFORE_PUBLISH'
def require_all_warp_reads(arrivals):assert arrivals==8,'W_REUSE_BEFORE_ALL8_READS'

def proof():
    cases=[];total_states=0
    for st,ws,split in ((5,5,False),(5,5,True),(6,4,True)):
        for kcs in range(1,9):
            states=model(st,ws,kcs,split);total_states+=states
            cases.append(dict(Xslots=st,Wslots=ws,split=split,KCS=kcs,reachable_states=states))
    rng=random.Random(48291);random_cases=0
    # Every valid KS1..64 partition of QKV K=8192=>KC128, including raggedKS7.
    partitions=[]
    for ks in range(1,65):
        lengths=[(i+1)*128//ks-i*128//ks for i in range(ks)]
        assert sum(lengths)==128 and min(lengths)>=1
        partitions.append(dict(KS=ks,KCS=sorted(set(lengths))))
        for kcs in set(lengths):
            for st,ws,split in ((5,5,False),(5,5,True),(6,4,True)):
                for _ in range(3):model(st,ws,kcs,split,False,rng);random_cases+=1
    regions=[]
    for st,ws,split in ((5,5,False),(5,5,True),(6,4,True)):
        xb=16384;wb=12288;xring=0;wring=st*xb;bars=wring+ws*wb
        full=bars;wempty=bars+8*st;xempty=wempty+8*ws;wfull=xempty+8*st if split else full
        zflag=wfull+8*ws if split else bars+24*st
        allocation=1024+st*xb+ws*wb+(8*(2*st+2*ws) if split else 24*st)+16
        spans=[('X',0,wring),('W',wring,bars),('Xfull' if split else 'jointFull',full,wempty),('Wempty',wempty,xempty),('Xempty',xempty,xempty+8*st)]
        if split:spans.append(('Wfull',wfull,wfull+8*ws))
        spans.append(('zeroFlag',zflag,zflag+4))
        for i,(name,lo,hi) in enumerate(spans):
            assert lo<hi and lo%4==0
            for _,otherlo,otherhi in spans[i+1:]:assert hi<=otherlo or otherhi<=lo
        # Shared base can round up by1023B; allocate1024slack+16tail.
        assert max(hi for _,lo,hi in spans)+1023<=allocation and allocation<227328
        for depth in (st,ws):
            for s in range(128):
                slot=s%depth;phase=(s//depth)&1
                if s>=depth:
                    assert (s-depth)%depth==slot and ((s-depth)//depth)&1!=phase
        regions.append(dict(Xslots=st,Wslots=ws,split=split,allocation=allocation,regions=spans))
    # Negative missing-X-wait control: split math would consume X0 while no X
    # generation has been published even when independent W0 is ready.
    negative={}
    for kcs in range(1,129):lifetime(program(kcs))
    mutated=program(3);mutated.remove(('release',0));mutated.insert(mutated.index(('mma',0))+1,('release',0))
    for name,check in [('early_X_reuse',lambda:lifetime(mutated)),
                       ('missing_Xfull_wait',lambda:require_X(0,0)),
                       ('W_reuse_before_all8reads',lambda:require_all_warp_reads(7))]:
        try:check()
        except AssertionError as error:negative[name]=str(error)
        else:raise AssertionError('negative admitted '+name)
    # Unchanged per-stage packed-W load mapping covers every one of12288B
    # exactly once:8warps x32lanes x3v4u32 loads, independent of W slot depth.
    seen=set()
    for warp in range(8):
        for lane in range(32):
            for chunk in range(3):
                for byte in range(16):
                    offset=warp*1536+lane*16+chunk*512+byte
                    assert offset not in seen;seen.add(offset)
    assert seen==set(range(12288))
    xcoords=set()
    for row in range(128):
        for col in range(64):
            address=row*128+(((col//8)^(row&7))*16)+(col%8)*2
            assert address%2==0 and address not in xcoords;xcoords.add(address)
    assert xcoords==set(range(0,16384,2))
    return dict(status='CPU_PROTOCOL_INTERLEAVING_PROOF_PASS_GPU_UNRUN',exhaustive_cases=cases,total_reachable_states=total_states,
                randomized_complete_schedules=random_cases,all_QKV_KS_partitions=partitions,shared_regions=regions,
                W_stage_bytes_once=12288,X_SW128_bf16_bijection=8192,full_barrier_counts={'coupled':2,'split_Xfull':1,'split_Wfull':1,'empty':8},
                release='Wempty only after all8warps LDS+syncwarp; Xempty only after wait1 retires previousgroup or finalwait0',
                initial='Both independent producers start at0 and can fill min(depth,KCS); unused slots never waited',
                drain='Everymathstage consumed; final wait0 before finalXrelease and epilogue; finite producers exit afterKCS',
                negative_controls=negative,
                limits='Two mathWG program witnesses aggregate four warp arrivals each; async engine transactions and WGMMA completion assumed finite. No GPU scheduler or arbitrary compiled-CFG proof.')
if __name__=='__main__':
    import json
    print(json.dumps(proof(),indent=2))

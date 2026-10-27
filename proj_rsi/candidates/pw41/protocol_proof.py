"""P1 late-ticket proof: ordered-launch epochs, publication DAG and full storage.

No CUDA imports. A source-bound abstract model, not an arbitrary GPU scheduler
or floating-point reduction-order proof. Unsupported shared-state streams HOLD.
"""
import json
import random
MAX=(1<<31)-1

def admit(even,odd,s,*,stream_owners=1,co_resident=True):
    if (type(s) is not int or not 2<=s<=16 or type(even) is not int or type(odd) is not int
            or even<0 or odd<0 or even%s or even//s!=odd or even+s>MAX or odd+1>MAX
            or stream_owners!=1 or co_resident is not True):
        raise ValueError('HOLD: unknown epoch/overflow/concurrent state ownership/progress precondition')

def sequences(s,x,late):
    seq=[]
    for cta in range(s):
        zero=[] if x or cta else ['zero','fence','publish']
        events=(zero+['math','partial','ticket'] if late else ['ticket']+zero+['math','partial'])
        seq.append(events+['acquire','cta_barrier','reduce'])
    if x:seq.append(['zero','fence','publish'])
    return seq

def transitions(state,seq,s,epoch,*,omit_acquire=False,early_publish=False):
    def done(actor,event):return event in seq[actor][:state[actor]]
    zeroer=s if len(seq)>s else 0
    published=done(zeroer,'publish');zeroed=done(zeroer,'zero');released=done(zeroer,'fence')
    result=[]
    for actor,pos in enumerate(state):
        if pos==len(seq[actor]):continue
        event=seq[actor][pos]
        if event=='ticket':
            count=sum(done(i,'ticket') for i in range(s))
            # For ANY permutation of the S tickets in one completed launch,
            # floor((E*S+k)/S)+1 == E+1 for every0<=k<S.
            ticket=epoch*s+count
            assert 0<=count<s and ticket//s+1==epoch+1
        elif event=='publish':
            if not early_publish:assert zeroed and released,'FLAG_BEFORE_ZERO_RELEASE'
        elif event=='acquire':
            assert done(actor,'ticket'),'ACQUIRE_WITHOUT_TICKET'
            if not published and not omit_acquire:continue
        elif event=='cta_barrier':
            assert done(actor,'partial') and done(actor,'acquire'),'BARRIER_BEFORE_ACQUIRE_OR_PARTIAL'
        elif event=='reduce':
            assert done(actor,'cta_barrier') and published and zeroed and released,'GLOBAL_WRITE_BEFORE_ZERO_ACQUIRE_BARRIER'
        target=list(state);target[actor]+=1;result.append(tuple(target))
    return result

def model(s,x,late,*,epoch=0,exhaust=True,rng=None,omit_acquire=False,early_publish=False):
    admit(epoch*s,epoch,s);seq=sequences(s,x,late)
    if early_publish:
        z=s if x else 0;seq[z].remove('publish');seq[z].insert(0,'publish')
    start=(0,)*len(seq);end=tuple(map(len,seq))
    if exhaust:
        seen={start};todo=[start]
        while todo:
            state=todo.pop();nexts=transitions(state,seq,s,epoch,omit_acquire=omit_acquire,early_publish=early_publish)
            assert nexts or state==end,'ABSTRACT_PROGRESS_FAILURE'
            for target in nexts:
                if target not in seen:seen.add(target);todo.append(target)
        assert end in seen
        return len(seen)
    state=start;steps=0
    while state!=end:
        nexts=transitions(state,seq,s,epoch,omit_acquire=omit_acquire,early_publish=early_publish)
        assert nexts,'ABSTRACT_PROGRESS_FAILURE';state=rng.choice(nexts);steps+=1
    return steps

def expected_storage(before,offset,tiles,s,calls):
    assert offset>=0 and offset+2*tiles<=len(before) and calls>=1
    after=list(before)
    for tile in range(tiles):
        i=offset+2*tile;admit(before[i],before[i+1],s)
        assert before[i]+s*calls<=MAX and before[i+1]+calls<=MAX
        after[i]+=s*calls;after[i+1]+=calls
    return after

def check_storage(before,after,offset,tiles,s,calls,identity_before,identity_after):
    assert identity_before==identity_after,'STORAGE_OFFSET_SHAPE_STRIDE_DTYPE_IDENTITY_CHANGED'
    assert after==expected_storage(before,offset,tiles,s,calls),'FULL_STORAGE_COUNTER_OR_DORMANT_ALIAS_CHANGED'

def proof():
    exhaustive=[];total=0
    for s in (2,3,4):
        for x in (False,True):
            for late in (False,True):
                n=model(s,x,late);total+=n
                exhaustive.append(dict(S=s,extra_zeroer=x,late_ticket=late,states=n))
    rng=random.Random(4101);schedules=0;epoch_checks=0
    for s in (4,8,16):
        for x in (False,True):
            for late in (False,True):
                for initial in (0,1,73,MAX//s-2):
                    for repeat in range(2):
                        epoch=initial+repeat
                        for _ in range(16):model(s,x,late,epoch=epoch,exhaust=False,rng=rng);schedules+=1
                        admit(epoch*s,epoch,s)
                        assert ((epoch*s+s)//s)==epoch+1
                        epoch_checks+=1
    negatives={}
    for label,call in [
        ('removed_acquire',lambda:model(2,True,True,omit_acquire=True)),
        ('flag_before_zero_release',lambda:model(2,True,True,early_publish=True)),
        ('misaligned_ticket_epoch',lambda:admit(3,0,4)),
        ('odd_epoch_mismatch',lambda:admit(8,1,4)),
        ('int32_ticket_wrap',lambda:admit((MAX//16)*16,MAX//16,16)),
        ('shared_state_concurrent_streams',lambda:admit(0,0,16,stream_owners=2)),
        ('unknown_residency_progress',lambda:admit(0,0,16,co_resident=False)),
    ]:
        try:call()
        except (AssertionError,ValueError) as error:negatives[label]=str(error)
        else:raise AssertionError('invalid protocol accepted '+label)
    full_storage_cases=0
    for tiles,s in ((16,16),(16,8),(64,4)):
        off=8;before=[314159+i for i in range(2064)]
        for tile in range(tiles):before[off+2*tile]=73*s;before[off+2*tile+1]=73
        identity=('device-proof',0x100000,len(before)*4,off,2048,1,'int32')
        for calls in (1,2,31):
            after=expected_storage(before,off,tiles,s,calls)
            check_storage(before,after,off,tiles,s,calls,identity,identity);full_storage_cases+=1
            # All active/dormant/guard/offset-alias words are compared, not just
            # the visible selected counter view or words guessed from a name.
            for label,index in [('even',off),('odd',off+1),('prefix_guard',0),('suffix_guard',len(before)-1),('dormant_alias',off+2*tiles+7)]:
                wrong=list(after);wrong[index]+=1
                try:check_storage(before,wrong,off,tiles,s,calls,identity,identity)
                except AssertionError:pass
                else:raise AssertionError('counter mutation accepted '+label)
            try:check_storage(before,after,off,tiles,s,calls,identity,identity[:-4]+(off+1,2048,1,'int32'))
            except AssertionError:pass
            else:raise AssertionError('counter offset identity mutation accepted')
    # Exact source zeroer and output tiling: each actual m<=TK and every row in
    # the16-wide tile is zeroed once before its global reductions. Ragged oddm
    # remain supported by the unchanged kernel body/dispatch.
    liveness=0
    for mt in (1,2,4):
        for m in range(1,8*mt+1):
            for threads in (32,256):
                points=[i for lane in range(threads) for i in range(lane,m*16,threads)]
                assert len(points)==len(set(points))==m*16 and set(points)==set(range(m*16));liveness+=1
    return dict(status='CPU_ORDERED_P1_EPOCH_AND_STORAGE_PASS_GPU_UNRUN',exhaustive_cases=exhaustive,total_states=total,
        randomized_complete_schedules=schedules,repeated_epoch_checks=epoch_checks,
        ticket_theorem='For arbitrary CTA ticket permutation k=0..S-1 in one ordered launch, floor((E*S+k)/S)+1=E+1. Placement changes no epoch, S tickets or one odd publication.',
        happens_before='all zero stores -> per-writer fence -> zeroer rendezvous -> odd publication -> successful acquire -> CTA barrier -> first GLOBAL reduction/output write',
        shared_partial_note='Private CTA red[] shared stores remain before acquire, as in parent; no external p.ws write is enabled in these P1 branches.',
        baseline_contract='At completed launch boundaries even=E*S, odd=E; active+dormant+alias/fullallocation identity exact; no harness reset; signed int32 wrap excluded.',
        negative_controls=negatives,full_storage_cases=full_storage_cases,storage_mutations_rejected=full_storage_cases*6,ragged_zero_tiling_cases=liveness,
        progress='Zeroer has finite bounded work; abstract wait does not block ready zeroer. Hardware all-block residency/fair progress and retained 2^24 poll trap require actual runtime validation; model is not a GPU scheduler proof.',
        unsupported='Concurrent launches/graphs/streams sharing one prepared counter/output state HOLD. Parent also requires this; no host-maintained replay epoch or new reset is introduced.',
        arithmetic='Exact parent partial computation/order retained in source. Cross-CTA floating reduction arrival order may vary and must pass fresh task/FP64/one-hot runtime gates.')
if __name__=='__main__':print(json.dumps(proof(),indent=2))

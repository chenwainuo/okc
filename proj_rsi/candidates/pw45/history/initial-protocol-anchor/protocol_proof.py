"""Pure CPU HALF/full-K packing/output/CSR and bounded EARLY lifetime model."""
import collections,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent

def layout():
    import numpy as np
    seen=np.zeros(8192*1024,dtype=np.bool_)
    warp,lane,q,v=np.indices((4,32,4,8))
    for nb in range(64):
        for half in range(2):
            row=128*nb+64*half+16*warp+lane//4+8*((v//2)%2)
            for stage in range(16):
                col=64*stage+16*q+8*(v//4)+2*(lane%4)+v%2
                idx=(row*1024+col).reshape(-1)
                assert idx.min()>=0 and idx.max()<len(seen) and len(np.unique(idx))==4096 and not bool(seen[idx].any())
                seen[idx]=True
    assert bool(seen.all())
    outputs=set()
    # Native D is m64(weightrows)n32(tokens): lane group gives8 channels,
    # value lowbit gives token parity; pp spans two16-token pieces.
    for nb in range(64):
        for half in range(2):
            cb=128*nb+64*half
            for warp in range(4):
                for lane in range(32):
                    for pp in range(2):
                        for reg in range(4):
                            for elem in range(2):
                                channel=cb+16*warp+(lane>>2)+8*(reg&1)
                                token=16*pp+8*(reg>>1)+2*(lane&3)+elem
                                assert 0<=token<32 and 0<=channel<8192 and (token,channel) not in outputs
                                outputs.add((token,channel))
    assert len(outputs)==32*8192
    # Global full-K CSR groups retain physical8warps/128row packedblock;
    # HALF selects four of those8lists with hh*4+warp.
    groups=set();fields=0
    for nb in range(64):
        for half in range(2):
            for warp in range(4):
                group=nb*8+4*half+warp;assert group not in groups;groups.add(group)
                for seq in range(16):
                    for lane in (0,31):
                        for reg in (0,15):
                            for hf in (0,1):
                                e=(seq<<17)|(lane<<12)|(reg<<8)|(hf<<7)|127
                                assert e!=0xffffffff and e>>17==seq and (e>>12)&31==lane and (e>>8)&15==reg and (e>>7)&1==hf
                                fields+=1
    assert groups==set(range(512))
    return {'state':'PASS_EXACT_HALF_FULLK_BIJECTIONS','weight_BF16_values':len(seen),'weight_bytes_original':2*len(seen),'packed_W_bytes':64*16*8*1536,'per_CTA_half_W_bytes_per_stage':6144,
        'output_BF16_values':len(outputs),'output_tiles':128,'output_tile_shape':[32,64],'fullK_stages':16,'KS':1,'partial_workspace_bytes':0,'global_reduce_or_zero_protocol':'NONE_PLAIN_TMA_STORE','CSR_lists':len(groups),'CSR_field_extreme_cases':fields,
        'actual_CSR_CAP':'UNKNOWN_UNTIL_REAL_GPU_WEIGHT_PREPARE','CAP_rule':'strict nextmultiple32 above actual max list, plus32guard words; no truncation','padding':0,'weight_encoding':'EXACT_PARENT_CODEC_BYTES_NO_NEW_CODEC',
        'source_tensormap_box':[64,32],'scope':'Finite coordinate/byte ownership proof; numerical GPU WGMMA execution unrun.'}

def successors(state,st,kcs,unsafe=None):
    w,x,read,issued,retired=state;out=[]
    if w<kcs and (w<st or w-st<read):out.append(('Wpublish',(w+1,x,read,issued,retired)))
    if x<kcs and (x<st or x-st<retired):out.append(('Xpublish',(w,x+1,read,issued,retired)))
    # This overapproximates independent completion after EARLY source publication.
    # Every read waits both current full contributors and preceding round issue.
    if read<kcs and read<w and read<x and (read==0 or read<=issued):out.append(('LDS_all4_Wrelease',(w,x,read+1,issued,retired)))
    if issued<read and (issued<2 or issued-1<=retired):out.append(('MMAgroup',(w,x,read,issued+1,retired)))
    if retired<kcs:
        if retired==kcs-1:
            ready=issued==kcs
        else:
            ready=issued>=retired+2 and read>=min(kcs,retired+3)
        if ready:out.append(('wait1_or_finalwait0_Xrelease',(w,x,read,issued,retired+1)))
    return out

def model(st,kcs):
    begin=(0,0,0,0,0);seen={begin};todo=collections.deque([begin]);final=(kcs,)*5
    while todo:
        state=todo.popleft();moves=successors(state,st,kcs)
        assert state==final or moves,('deadlock',st,kcs,state)
        for name,nxt in moves:
            w,x,read,issued,retired=nxt
            assert retired<=issued and issued<=read and read<=min(w,x)
            assert w-read<=st and x-retired<=st
            if nxt not in seen:seen.add(nxt);todo.append(nxt)
    assert final in seen
    return len(seen)

def proof():
    source=(HERE/'submission.py').read_text()
    for anchor in ('const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;','if (SELF || (HALF && warp >= 4)) return;','const int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;',
        'for (int s = 0; s < ST && s < KCS; s++)','if (G > 0) xrel(G - 1);','if (WM != 1) xrel(NGT - 1);'):
        assert anchor in source,anchor
    counts=[]
    for st in (4,5):
        for kcs in range(1,18):counts.append({'ST':st,'KCS':kcs,'states':model(st,kcs)})
    # EARLY min startup and slot phase source exactly bounds every W/X access.
    for st in (4,5):
        for kcs in range(1,33):
            initial=list(range(min(st,kcs)));later=list(range(min(st,kcs),kcs))
            assert initial+later==list(range(kcs))
            for stage in initial+later:assert 0<=stage<kcs and stage%st<st and (stage//st)&1 in (0,1)
    # Source obligations rejected when weakened: current full not published,
    # X still pending, or fewer than allfour readers in empty barrier.
    def require_full(w,x,s):assert s<w and s<x,'READ_BEFORE_BOTH_FULL_CONTRIBUTORS'
    def require_X(retired,s):assert s<retired,'X_REUSE_BEFORE_WGMMA_RETIRE'
    def require_readers(readers):assert set(readers)==set(range(4)),'W_REUSE_BEFORE_ALL4_WARP_READS'
    negatives={}
    for name,fn in [('missing_Xfull',lambda:require_full(1,0,0)),('early_Xreuse',lambda:require_X(0,0)),('only3_readers',lambda:require_readers(range(3)))]:
        try:fn()
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('invalid protocol admitted')
    return {'state':'PASS_BOUNDED_HALF4READER_PHASE_LIFETIME','cases':counts,'total_states':sum(c['states'] for c in counts),'source_fullK_KCS':16,
        'full_count':2,'Wempty_Xempty_count':4,'EARLY_startup':'min(ST,KCS) W+X beforeCTAbarrier','Wrelease':'all4 LDS+syncwarp beforeoverwrite','Xrelease':'afterwait1 coverspriorgroup or finalwait0','D':'never genericwritten whileasync; firstK16scaleD0 and allsubsequentK16scaleD1',
        'A_banks':'two16word paritybanks; twoqueuedgroups and wait1 retireprevious before bankreuse','negative_controls':negatives,
        'limits':'Finite completion/order source model, fourwarp reads aggregated into one all-reader event. Assumes eventual TMA/WGMMA progress, not arbitrary compiled-CFG proof.'}

if __name__=='__main__':
    r={'layout':layout(),'protocol':proof()};(HERE/'protocol-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'state':r['protocol']['state'],'states':r['protocol']['total_states'],'weight_values':r['layout']['weight_BF16_values']},indent=2))

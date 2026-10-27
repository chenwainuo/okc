"""Focused one-cell adaptation; no frozen parent edits."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent/'gv53_keyed_half_v1';edits=[]
s=(P/'driver.py').read_text()
def edit(old,new):
 global s
 assert s.count(old)==1,(old[:80],s.count(old));s=s.replace(old,new,1);edits.append({'old':old,'new':new})
edit('Two fixed GV53 cells','One fixed M8 quarter-stream cell')
edit('CAND=D','CAND=D.parent/"gv53_quarter_stream_v1"')
edit("LABELS=('control','keyed')","LABELS=('control','quarter')")
# Apply implementation label uniformly; timing/terminal schedule structure otherwise unchanged.
s=s.replace("'keyed'","'quarter'").replace("gv53_keyed_","gv53_quarter_").replace('gv_keyed<','gv_quarter<')
edit("'fixed_cells':2,'timed_calls':162,'warmup_calls':20,'own_terminal_sites':8,'mutations':20", "'fixed_cells':1,'timed_calls':81,'warmup_calls':10,'own_terminal_sites':4,'mutations':10")
edit("'source_sha256':sha(CAND/'submission.py')", "'source_sha256':sha(CAND/'pilot.cu')")
edit("time.monotonic()-started<120,'inner120s admission exhausted'", "time.monotonic()-started<90,'inner90s admission exhausted'")
edit("weights=reference.generate_weights(reference.WEIGHTS_SEED)", "w_seed=reference.WEIGHTS_SEED+193;weights=reference.generate_weights(w_seed)")
edit("immutable=[('pooler',weights['pooler'],weights['pooler'].clone()),('packed',packed,packed.clone()),('escape',escape,escape.clone())]", """sys.path.insert(0,str(CAND));from transport import pack_bank,unpack_group
        old_words=escape.detach().cpu().tolist()
        begin_transport=time.monotonic();new_words=pack_bank(old_words);assert new_words is not None
        assert len(new_words)==len(old_words)==131072
        record_count=0
        for offset in range(0,len(old_words),32):
            accepted=[int(v)&0xffffffff for v in old_words[offset:offset+32]if (int(v)&0xffffffff)!=0xffffffff]
            expected=sorted(accepted,key=lambda v:(v>>20,(v>>13)&3))
            assert unpack_group(new_words[offset:offset+32])==expected,('actual accepted escape mapping',offset)
            record_count+=len(accepted)
        quarter_escape=torch.tensor([v if v<2**31 else v-2**32 for v in new_words],dtype=torch.int32,device=escape.device)
        assert quarter_escape.data_ptr()!=escape.data_ptr() and quarter_escape.numel()==escape.numel()
        escapes={'control':escape,'quarter':quarter_escape}
        result['transport']={'bytes_per_warpgroup':128,'groups':len(old_words)//32,'capacity_per_group':32,'record_count':record_count,'actual_accepted_records_exact_after_stable_bucketing':True,'original_overflow_rejection_preserved':True,'parent_bank_sha256':hashlib.sha256(escape.cpu().numpy().tobytes()).hexdigest(),'quarter_bank_sha256':hashlib.sha256(quarter_escape.cpu().numpy().tobytes()).hexdigest(),'separate_bank_pointers':True,'outside_timed_region':True,'host_transform_and_actual_mapping_check_seconds':time.monotonic()-begin_transport}
        immutable=[('pooler',weights['pooler'],weights['pooler'].clone()),('packed',packed,packed.clone()),('parent_escape',escape,escape.clone()),('quarter_escape',quarter_escape,quarter_escape.clone())]""")
edit("'W_seed':reference.WEIGHTS_SEED", "'W_seed':w_seed")
edit('for m in (1,8):','for m in (8,):')
edit('ext.run(LABELS.index(arm),packed,escape,x,out,wb[16:-16],cb[16:-16],table)', 'ext.run(LABELS.index(arm),packed,escapes[arm],x,out,wb[16:-16],cb[16:-16],table)')
edit("'TWO_FIXED_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED'", "'ONE_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'")
# LABELS replacement is recorded as a broad exact stage separately for strict reproducibility.
(D/'driver.py').write_text(s)
binding=(P/'binding.cu').read_text().replace('gv_keyed<','gv_quarter<').replace('(x.size(0)==1||x.size(0)==8)','x.size(0)==8').replace('only pooler M1/M8,N4096,K8192','only pooler M8,N4096,K8192')
(D/'binding.cu').write_text(binding)
(D/'driver-derivation.json').write_text(json.dumps({'parent_driver':str(P/'driver.py'),'parent_sha256':hashlib.sha256((P/'driver.py').read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(s.encode()).hexdigest(),'exact_edits':edits,'uniform_label_replacement':{'old':'keyed','new':'quarter'},'retained_protocol':['schedule','sameXfixtures','guarded arenas','normal/outlier/zero/boundary oracle','own F1 and paired terminals','graph current-input mutation','actual attrs legal residency only','postscope drained BLAS cleanup'],'changed_scope':['M8 only','one prospective W seed offset193','actual oldescape stable quarter-bank transport+mapping outside timing','distinct perarm escape bank','90s inner'],'no_GPU':True},indent=2)+'\n')
print(hashlib.sha256(s.encode()).hexdigest())

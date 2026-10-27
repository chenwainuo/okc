"""Threearm same-workload adaptation with exact collected predecessor SO reuse."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent/'gv53_quarter_stream_trial_v1';R=D.parents[2]
COLL=Path('results/proj-rsi/scheduler/artifacts/research.gv53.quarter.stream.fixed.01/collection-1793140874912701666/logs/scheduler/gv53-quarter-stream-01/run')
meta=json.loads((R/COLL/'result.json').read_text());oldname='gv53_quarter_'+meta['build']['CUDA_sha256'][:20]
so=R/COLL/'actual.so';assert hashlib.sha256(so.read_bytes()).hexdigest()==meta['build']['SO_sha256']
assert 'PyInit_'+oldname in subprocess.check_output(['nm','-D',str(so)],text=True)
s=(P/'driver.py').read_text();edits=[]
def edit(old,new):
 global s
 assert s.count(old)==1,(old[:90],s.count(old));s=s.replace(old,new,1);edits.append({'old':old,'new':new})
edit('One fixed M8 quarter-stream cell','Threearm fixed M8 sparse-header cell')
edit('import argparse,ast,hashlib,json,statistics,sys,time,traceback','import argparse,ast,hashlib,importlib.util,itertools,json,statistics,sys,time,traceback')
edit('CAND=D.parent/"gv53_quarter_stream_v1"','CAND=D.parent/"gv53_quarter_sparse_header_v1"\nTRANSPORT=D.parent/"gv53_quarter_stream_v1"\nOLD_SO=ROOT/'+repr(str(COLL/'actual.so'))+'\nOLD_SO_SHA='+repr(meta['build']['SO_sha256'])+'\nOLD_INIT='+repr(oldname)+'\nOLD_CUDA_SHA='+repr(meta['build']['CUDA_sha256']))
edit("LABELS=('control','quarter')","LABELS=('control','quarter','sparse')\nCONTRASTS=(('control','quarter'),('control','sparse'),('quarter','sparse'))")
a=s.index('def schedule():');b=s.index('\ndef terminal(',a)
edit(s[a:b],'''def schedule():
    rows=[]
    for arm in LABELS:rows += [dict(arm=arm,seed=91000,mode='warmup')for _ in range(5)]
    for block,arm in enumerate(('control','quarter','sparse','sparse','quarter','control')):
        rows += [dict(arm=arm,seed=91100+i,mode='blocked',block=block)for i in range(7)]
    orders=list(itertools.permutations(LABELS))
    for i in range(24):
        rows += [dict(arm=arm,seed=91200+i,mode='triple',triple=i,order=list(orders[i%6]))for arm in orders[i%6]]
    assert len(rows)==129
    return rows
''')
a=s.index('    m=verify();rows=schedule();');b=s.index('\ndef run(args):',a)
edit(s[a:b],'''    m=verify();rows=schedule();assert sum(r['mode']!='warmup'for r in rows)==114
    triples=[rows[57+3*i:60+3*i]for i in range(24)]
    for triple in triples:assert len({r['seed']for r in triple})==1 and set(r['arm']for r in triple)==set(LABELS)
    for a,b in CONTRASTS:
        assert sum([r['arm']for r in t].index(a)<[r['arm']for r in t].index(b)for t in triples)==12
    for arm in LABELS:
        assert all(sum(t[pos]['arm']==arm for t in triples)==8 for pos in range(3))
    raw=[dict(arm=a,seed=i,mode='blocked')for i,a in enumerate(LABELS)];last=dict(zip(LABELS,range(3)))
    assert len(terminal(last,raw,lambda a,s:'PASS',lambda:None))==3
    caught=[]
    for name in ('stale_seed','missing_arm','invalid_output','counter_mutation'):
        state=dict(last)
        if name=='stale_seed':state['control']=99
        if name=='missing_arm':state.pop('quarter')
        def oracle(a,s):assert name!='invalid_output';return 'PASS'
        def integrity():assert name!='counter_mutation'
        try:terminal(state,raw,oracle,integrity)
        except AssertionError:caught.append(name)
        else:raise AssertionError(name)
    assert len(codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()))==7
    assert len(basis_columns())==8
    assert sha(OLD_SO)==OLD_SO_SHA
    return {'state':'PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN','pins_verified':True,'fixed_cells':1,'timed_calls':114,'warmup_calls':15,'own_terminal_sites':6,'mutations':15,'terminal_negatives':caught,'original_codec_AST_functions':5,'codec_literals':2,'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'three_arm_triples':24,'all3pairwise_orders':{'first_before_second':12,'second_before_first':12},'each_arm_position_count':8,'predecessor_SO_sha_verified':True,'resource_gate':'legal residency only; actual registers/local/static recorded'}
''')
edit("name='gv53_quarter_'+digest[:20]","name='gv53_sparse_'+digest[:20]")
edit("        CLEANUP_HOOK=ext.clear_blas_cache",'''        CLEANUP_HOOK=ext.clear_blas_cache
        assert sha(OLD_SO)==OLD_SO_SHA
        predecessor_load_start=time.monotonic()
        oldspec=importlib.util.spec_from_file_location(OLD_INIT,OLD_SO)
        assert oldspec is not None and oldspec.loader is not None
        oldext=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(oldext)
        assert sha(Path(oldext.__file__))==OLD_SO_SHA
        result['cost']['predecessor_SO_load_seconds']=time.monotonic()-predecessor_load_start
        modules={'control':(ext,0),'quarter':(oldext,1),'sparse':(ext,1)}
        result['predecessor_SO']={'path':str(OLD_SO.relative_to(ROOT)),'SO_sha256':OLD_SO_SHA,'PyInit_module_name':OLD_INIT,'source_CUDA_sha256':OLD_CUDA_SHA,'loaded_without_compile':True,'native_arm':1,'expected_symbol':'gv_quarter<1,1,1,2,16,8,1,1,0,3>'}
        result['arm_bindings']={arm:{'module':'collected_predecessor'if arm=='quarter'else'new_compile','native_arm':modules[arm][1]}for arm in LABELS}''')
edit("'native_symbols':['gv_predxor<1,1,1,2,16,8,1,1,0,3>','gv_quarter<1,1,1,2,16,8,1,1,0,3>'],'field_query_calls':16", "'native_symbols':['gv_predxor<1,1,1,2,16,8,1,1,0,3>','gv_sparse_quarter<1,1,1,2,16,8,1,1,0,3>'],'field_query_calls':24")
edit("{arm:{key:int(ext.attrs(i,j+1))for j,key in enumerate(fields)}for i,arm in enumerate(LABELS)}", "{arm:{key:int(modules[arm][0].attrs(modules[arm][1],j+1))for j,key in enumerate(fields)}for arm in LABELS}")
edit("a['arm_echo']==i", "a['arm_echo']==modules[arm][1]")
edit('sys.path.insert(0,str(CAND));from transport import pack_bank,unpack_group','sys.path.insert(0,str(TRANSPORT));from transport import pack_bank,unpack_group')
edit("escapes={'control':escape,'quarter':quarter_escape}","escapes={'control':escape,'quarter':quarter_escape,'sparse':quarter_escape}")
edit("len({v[0].data_ptr()for v in work.values()})==2 and len({v[1].data_ptr()for v in work.values()})==2", "len({v[0].data_ptr()for v in work.values()})==3 and len({v[1].data_ptr()for v in work.values()})==3")
edit('ext.run(LABELS.index(arm),packed,escapes[arm],x,out,wb[16:-16],cb[16:-16],table)', 'modules[arm][0].run(modules[arm][1],packed,escapes[arm],x,out,wb[16:-16],cb[16:-16],table)')
edit("if i==43:entry['terminals']['F1']=terminal(last,entry['raw'],oracle,integrity);save()", "if i==57:entry['terminals']['blocked']=terminal(last,entry['raw'],oracle,integrity);save()")
edit("entry['terminals']['paired']=terminal(last,entry['raw'],oracle,integrity)", "entry['terminals']['triples']=terminal(last,entry['raw'],oracle,integrity)")
a=s.index("            f1=entry['raw'][10:43]");b=s.index('            probe=torch.zeros_like(',a)
edit(s[a:b],'''            blocked=entry['raw'][15:57]
            med=[statistics.median(r['us']for r in blocked[i*7:(i+1)*7])for i in range(6)]
            entry['blocked_medians_us']=[{'arm':a,'us':us,'block':i}for i,(a,us)in enumerate(zip(('control','quarter','sparse','sparse','quarter','control'),med))]
            averages={'control':(med[0]+med[5])/2,'quarter':(med[1]+med[4])/2,'sparse':(med[2]+med[3])/2}
            triples=[{r['arm']:r['us']for r in entry['raw'][57+3*i:60+3*i]}for i in range(24)]
            entry['triples']=triples;entry['contrasts']={}
            for a,b in CONTRASTS:
                differences=[r[a]-r[b]for r in triples]
                entry['contrasts'][a+'_minus_'+b]={'paired_median_us':statistics.median(differences),'blocked_endpoint_average_us':averages[a]-averages[b],'second_arm_wins':sum(x>0 for x in differences),'first_arm_wins':sum(x<0 for x in differences),'ties':sum(x==0 for x in differences)}
            save()
''')
edit("'ONE_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'", "'THREE_ARM_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'")
(D/'driver.py').write_text(s)
b=(P/'binding.cu').read_text().replace('gv_quarter<','gv_sparse_quarter<');(D/'binding.cu').write_text(b)
(D/'driver-derivation.json').write_text(json.dumps({'parent_driver':str(P/'driver.py'),'parent_sha256':hashlib.sha256((P/'driver.py').read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(s.encode()).hexdigest(),'exact_edits':edits,'old_SO':str(COLL/'actual.so'),'old_SO_sha256':meta['build']['SO_sha256'],'PyInit_module_name':oldname,'protocol_changes':['3privatearena/grapharms','collectedoldSOarm1forquarter','newcompilecontrol0/sparse1','6palindromic7sampleblocks','24triplesallsixordersrepeated4','3pairwisesignedcontrasts'],'unchanged':['originalcodecAST','actualacceptedbankmapping','normal/outlier/zero/basisoracle','currentXguard/fullbackings','graphmutation','drainedpostscopecleanup'],'no_GPU':True},indent=2)+'\n')
print(hashlib.sha256(s.encode()).hexdigest())

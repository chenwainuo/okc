"""Two fixed GV53 cells, no full prepare/profiler/SASS campaign; root-only GPU run."""
import argparse,ast,hashlib,json,statistics,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[3]
CAND=D
LABELS=('control','frontier')
CELLS=(('pooler',1,4096,8192,(1,1,1,2,16,8,1,1,0,3),32),
       ('unpooler',16,8192,4096,(1,2,2,1,4,16,1,1,2,1),48))
DECL='std::vector<int64_t> frontier_info(int64_t,bool);\nvoid frontier_run(int64_t,bool,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def verify():
    m=json.loads((D/'manifest.json').read_text())
    for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
    return m

def codec_nodes(source):
    names={'_pack','_c12','_unc12','_gvesc','_frag_bits'};out=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef)and node.name in names:out.append(node)
        elif isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ('_OS','_SS')for t in node.targets):out.append(node)
    assert len(out)==7
    return out

def basis_columns(k=8192):
    columns=list(range(0,k,k//64))
    columns[:9]=[0,1,31,32,63,64,511,512,k-1]
    assert len(columns)==len(set(columns))==64
    return columns

def schedule():
    rows=[]
    for arm in LABELS:rows += [dict(arm=arm,seed=91000,mode='warmup')for _ in range(5)]
    for arm in ('control','frontier','control'):rows += [dict(arm=arm,seed=91100+i,mode='F1')for i in range(11)]
    for i in range(24):
        rows += [dict(arm=arm,seed=91200+i,mode='paired',pair=i)for arm in (LABELS if i%2==0 else LABELS[::-1])]
    assert len(rows)==91
    return rows

def terminal(last,raw,oracle,integrity):
    assert set(last)==set(LABELS)
    records=[]
    for arm in LABELS:
        seed=next(r['seed']for r in reversed(raw)if r['arm']==arm and r['mode']!='warmup')
        assert seed==last[arm]
        records.append(dict(arm=arm,seed=seed,oracle=oracle(arm,seed),before_any_replay_or_mutation=True))
    integrity();return records

def cpu_check():
    m=verify();rows=schedule();assert sum(r['mode']!='warmup'for r in rows)==81
    blocks=[rows[43+2*i:45+2*i]for i in range(24)]
    assert sum(b[0]['arm']=='control'for b in blocks)==12
    for b in blocks:assert len({r['seed']for r in b})==1
    raw=[dict(arm=a,seed=i,mode='F1')for i,a in enumerate(LABELS)];last=dict(zip(LABELS,range(2)))
    assert len(terminal(last,raw,lambda a,s:'PASS',lambda:None))==2
    caught=[]
    for name in ('stale_seed','missing_arm','invalid_output','counter_mutation'):
        state=dict(last)
        if name=='stale_seed':state['control']=99
        if name=='missing_arm':state.pop('frontier')
        def oracle(a,s):assert name!='invalid_output';return 'PASS'
        def integrity():assert name!='counter_mutation'
        try:terminal(state,raw,oracle,integrity)
        except AssertionError:caught.append(name)
        else:raise AssertionError(name)
    assert len(codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()))==7
    assert len(basis_columns())==64
    assert len(basis_columns(4096))==64 and max(basis_columns(4096))==4095
    selected=(ROOT/'proj_rsi/audits/selected-hotpaths-gv9916/report.md').read_text()
    stats=json.loads((D.parent/'actual-stats-proof.json').read_text())['layers']
    for layer,m,n,k,tpl,cap in CELLS:
        assert ','.join(map(str,tpl)) in selected
        assert list(tpl[3:9])==stats[layer]['geo'] and stats[layer]['C11_max_unit_repairs']<=cap
    return {'state':'PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN','pins_verified':True,'fixed_cells':2,'timed_calls':162,'warmup_calls':20,'own_terminal_sites':8,'mutations':20,'terminal_negatives':caught,'original_codec_AST_functions':5,'codec_literals':2,'profiler_calls':0,'full_prepare_calls':0}

def run(args):
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'native.cu'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':0,'full_prepare_calls':0}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<300,'inner300s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        import torch
        from torch.utils.cpp_extension import load_inline
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('compile_two_functions')
        cu='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n'+(D/'native.cu').read_text().replace('#include \"../parallel_fix.cuh\"',(D.parent/'parallel_fix.cuh').read_text())+'\n'+(D/'bindings.cuh').read_text()
        (args.output/'actual.cu').write_text(cu)
        digest=hashlib.sha256(cu.encode()).hexdigest();begin=time.monotonic()
        ext=load_inline(name='gv53_frontier_'+digest[:20],cpp_sources=DECL,cuda_sources=cu,functions=['frontier_run','frontier_info'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],verbose=True)
        result['cost']['compile_and_load_seconds']=time.monotonic()-begin
        __import__('shutil').copy2(ext.__file__,args.output/'actual.so')
        result['build']={'CUDA_sha256':digest,'CPP_declarations_sha256':hashlib.sha256(DECL.encode()).hexdigest(),'SO_sha256':sha(ext.__file__),'CUDA_flags':['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],'native_symbols':['two exact selected predxor controls','two exact C11 candidates'],'field_query_calls':16}
        # Exact compiler control identity is required before any launch/timing.
        import subprocess,re
        parser_nodes=[n for n in ast.parse((D/'verify_native.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='streams']
        pn={'re':re};exec(compile(ast.Module(body=parser_nodes,type_ignores=[]),'<native-stream-parser>','exec'),pn)
        sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',ext.__file__],check=True,capture_output=True,text=True,timeout=30).stdout
        (args.output/'actual.sass').write_text(sass)
        native=pn['streams'](sass);expected=pn['streams']((D/'native.sass').read_text())
        assert {symbol for symbol in native if 'gv_predxor' in symbol or 'gv_frontier' in symbol}==set(expected)
        for symbol,rows in expected.items():
            if 'gv_predxor' in symbol:assert native[symbol]==rows,('actual control codegen drift',symbol)
        result['actual_control_native_identity']='EXACT_PC_PREDICATE_ENCODING'
        phase('actual_task_weights')
        begin=time.monotonic();weights=reference.generate_weights(reference.WEIGHTS_SEED)
        env={'torch':torch};exec(compile(ast.Module(body=codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<exact-original-codec-AST>','exec'),env)
        import numpy as np,importlib.util
        ps=importlib.util.spec_from_file_location('frontier_physical_pack',D/'pack.py');pkmod=importlib.util.module_from_spec(ps);ps.loader.exec_module(pkmod)
        stats=json.loads((D.parent/'actual-stats-proof.json').read_text())['layers']
        immutable=[(key,t,t.clone())for key,t in weights.items()]
        result['cost']['task_weight_generation_seconds']=time.monotonic()-begin
        flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda',dtype=torch.float32)
        def immutables():
            for name,t,copy in immutable:assert torch.equal(t.view(torch.uint8),copy.view(torch.uint8)),('immutable changed',name)
        for cell,(layer,m,n,k,tpl,capacity) in enumerate(CELLS):
            geo=tpl[3:9]
            phase('fixed_'+layer+'_M'+str(m));config={'layer':layer,'m':m,'n':n,'k':k}
            entry=result['fixed_cells'][str(m)]={'config':config,'raw':[],'validation':{},'terminals':{},'mutations':[]};work={};arenas={}
            attributes={arm:list(map(int,ext.frontier_info(cell,arm=='frontier')))for arm in LABELS}
            result.setdefault('raw_attributes',{})[layer]=attributes;save()
            for arm,vals in attributes.items():
                regs,local,static,threads,dynamic,occupancy=vals
                assert regs==(64 if cell==0 else (110 if arm=='control' else 108)) and local==static==0 and threads==(1024 if cell==0 else 128) and occupancy>=1,(layer,arm,vals)
            assert attributes['frontier'][-1]>=attributes['control'][-1]
            raw=weights[layer].cpu();assert sha_bytes(raw.view(torch.uint8).numpy().tobytes())==stats[layer]['weight_sha256']
            cpu_fr=env['_pack'](raw).reshape(n//16,k//64,4,32,8)
            physical=pkmod.pack(cpu_fr.view(torch.int16).numpy().view(np.uint16),stats[layer]['C11_base'],geo,capacity=capacity)
            assert physical is not None and physical['total_bytes']==stats[layer]['C11_total_bytes']
            assert np.array_equal(pkmod.unpack(physical),cpu_fr.view(torch.int16).numpy().view(np.uint16))
            encoded=env['_c12'](cpu_fr,1);assert encoded is not None
            oldlow,oldtable,esc6=encoded
            assert torch.equal(env['_unc12'](oldlow,oldtable,esc6,1),env['_frag_bits'](cpu_fr))
            esc7=torch.cat((esc6,torch.full_like(esc6[:,:1],0)),1)
            oldesc=env['_gvesc'](esc7,n//16,geo);assert oldesc is not None
            header=torch.from_numpy(physical['header'].view(np.int32)).to('cuda')
            records=torch.from_numpy(physical['records']).to('cuda')
            frontlow=torch.from_numpy(physical['low']).to('cuda')
            frontdesc=torch.from_numpy(physical['desc'].view(np.int32)).to('cuda')
            packed=oldlow.to('cuda');escape=oldesc.to('cuda')
            packages={'control':(packed,header,escape,records,oldtable,physical['words']),
                      'frontier':(frontlow,header,frontdesc,records,physical['table'],physical['words'])}
            for label,tensor in [('packed',packed),('escape',escape),('low',frontlow),('header',header),('desc',frontdesc),('records',records)]:
                immutable.append((layer+'/'+label,tensor,tensor.clone()))
            entry['pack']={'W_seed':reference.WEIGHTS_SEED,'actual_W_bit_reconstruction':'EXACT_BOTH_FORMATS',
                'capacity':capacity,'repairs':physical['repairs'],'C11_bytes':physical['total_bytes'],
                'C12_bytes':physical['incumbent_bytes'],'geo':geo,'actual_fixture_sha':stats[layer]['weight_sha256']}
            fixtures={seed:reference.generate_input(m,layer,seed)['x']for seed in {r['seed']for r in schedule()}}
            for index,arm in enumerate(LABELS):
                xb=torch.full((m*k+32,),11.5,dtype=torch.bfloat16,device='cuda');x=xb[16:-16].reshape(m,k)
                ob=torch.full((m*n+32,),17.25,dtype=torch.bfloat16,device='cuda');out=ob[16:-16].reshape(m,n)
                cb=torch.full((65536+32,),0x19a75d3,dtype=torch.int32,device='cuda');wb=torch.full((65536+32,),17.25,dtype=torch.float32,device='cuda')
                arenas[arm]=(xb,ob,cb,wb,cb.clone(),wb.clone());work[arm]=(x,out)
            assert len({v[0].data_ptr()for v in work.values()})==2 and len({v[1].data_ptr()for v in work.values()})==2
            def integrity():
                immutables()
                for xb,ob,cb,wb,cb0,wb0 in arenas.values():
                    assert bool((xb[:16]==11.5).all())and bool((xb[-16:]==11.5).all())
                    assert bool((ob[:16]==17.25).all())and bool((ob[-16:]==17.25).all())
                    assert torch.equal(cb,cb0)and torch.equal(wb.view(torch.uint8),wb0.view(torch.uint8)),'S1 counter/scratch full backing changed'
            def invoke(arm):
                x,out=work[arm];_,_,cb,wb,_,_=arenas[arm]
                low_,header_,desc_,records_,table_,words_=packages[arm]
                ext.frontier_run(cell,arm=='frontier',low_,header_,desc_,records_,x,out,wb[16:-16],cb[16:-16],table_,words_)
            def oracle_x(arm,fresh):
                x,out=work[arm];assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
                ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(arm,m,msg)
                ref=fresh.double()@weights[layer].double().T
                rel=float((out.double()-ref).norm()/ref.norm().clamp(min=1e-30));row=float(((out.double()-ref).norm(dim=-1)/ref.norm(dim=-1).clamp(min=1e-30)).max())
                assert rel<=4e-3 and row<=2e-2,(arm,m,rel,row)
                return {'task':msg,'FP64_L2':rel,'worst_row':row,'current_X':'EXACT'}
            for arm in LABELS:
                x,out=work[arm];checks=[]
                for kind,fresh in [('normal',reference.generate_input(m,layer,91301)['x']),('outliers',reference.generate_input(m,layer,91303,dist='outliers')['x']),('zero',torch.zeros_like(x))]:
                    x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize();checks.append(dict(kind=kind,oracle=oracle_x(arm,fresh)));integrity();budget()
                columns=basis_columns(k)
                for start in range(0,64,m):
                    selected=columns[start:start+m];fresh=torch.zeros_like(x);fresh[torch.arange(m,device=x.device),torch.tensor(selected,device=x.device)]=8
                    x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize()
                    expected=(weights[layer][:,selected].T*8).contiguous()
                    assert torch.equal(out.float(),expected.float())and torch.equal(x,fresh),('selected basis',arm,m,selected)
                integrity();entry['validation'][arm]={'task':checks,'basis_columns':columns,'basis':'EXACT'};save();budget()
            live={}
            for arm in LABELS:
                x,out=work[arm];x.copy_(fixtures[91000]);g=torch.cuda.CUDAGraph();graphs.append(g);live[arm]=g
                with torch.cuda.graph(g):invoke(arm)
            integrity();last={}
            def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
            for i,r in enumerate(schedule()):
                if i==43:entry['terminals']['F1']=terminal(last,entry['raw'],oracle,integrity);save()
                arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);torch.sum(flush,0,out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();live[arm].replay();b.record();torch.cuda.synchronize()
                us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                row=dict(r,us=us);entry['raw'].append(row);last[arm]=r['seed']
                with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(row,m=m))+'\n');f.flush()
                budget()
            entry['terminals']['paired']=terminal(last,entry['raw'],oracle,integrity)
            f1=entry['raw'][10:43];med=[statistics.median(r['us']for r in f1[i*11:(i+1)*11])for i in range(3)]
            paired=[{r['arm']:r['us']for r in entry['raw'][43+2*i:45+2*i]}for i in range(24)]
            entry['F1_medians_us']=dict(zip(('control_before','frontier','control_after'),med));entry['paired']=paired
            entry['contrasts']={'paired_median_control_minus_new_us':statistics.median(r['control']-r['frontier']for r in paired),'F1_endpoint_average_control_minus_new_us':(med[0]+med[2])/2-med[1],'paired_new_wins':sum(r['control']>r['frontier']for r in paired)};save()
            probe=torch.zeros_like(fixtures[91000]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns(k)[i]for i in range(m)],device=probe.device)]=8
            mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,layer,91401)['x']),('outliers',reference.generate_input(m,layer,91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,layer,91407)['x'])]
            for kind,fresh in mutations:
                for arm in LABELS:
                    x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                    message=oracle_x(arm,fresh)
                    if kind=='zero':assert not int(torch.count_nonzero(out))
                    integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
            for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
            entry['state']='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['state']='TWO_FIXED_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED';result['performance_scope']='Fixed GV53 implementation comparison only, not selected policy or official score';phase('complete')
    except BaseException:
        result['state']='INVALID_NO_PERFORMANCE_ADMISSION';result['error']=traceback.format_exc();save();raise
    finally:
        cleanup=[]
        for g in graphs:
            try:torch.cuda.synchronize();g.reset();cleanup.append('RESET_AFTER_DRAIN')
            except BaseException as e:cleanup.append(repr(e));result['state']='INVALID_CLEANUP'
        result['cleanup']=cleanup;result['finished_at']=time.time();result['cost']['run_elapsed_seconds']=time.monotonic()-started;save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);args=p.parse_args()
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output and args.review
    error=None
    try:run(args)
    except BaseException as exc:error=repr(exc)
    # run() has returned/unwound, including its fixture/pack/closure aliases.
    import gc
    gc.collect()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
        cleanup={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'all_run_aliases_released':True}
        if (args.output/'result.json').exists():
            result=json.loads((args.output/'result.json').read_text());result['post_scope_cleanup']=cleanup
            if cleanup['allocated']!=0:result['state']='INVALID_CLEANUP'
            dump(args.output/'result.json',result)
        assert cleanup['allocated']==0,cleanup
    if error:raise RuntimeError(error)
if __name__=='__main__':main()

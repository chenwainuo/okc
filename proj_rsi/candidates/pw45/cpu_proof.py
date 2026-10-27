"""CPU-only PW45 source, finite layout/protocol, compiled controls/resources proof."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import apply,value,PARENT_SHA,PARENT_MANIFEST
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
NEW={(32,4,1,0,0,0,52),(32,5,1,0,0,0,52)}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def named(text):
    result={}
    for n in ast.parse(text).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):result[n.name]=ast.dump(n,include_attributes=False)
        elif isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):result[t.id]=ast.dump(n,include_attributes=False)
    return result

def productions(text):
    variants={v[0]:v for v in value(text,'RQV')}
    small=list(dict.fromkeys(n for choices in value(text,'RQSBAKE').values() for n,k in choices))
    return [(name,variants[name][1:8]) for name in value(text,'RQPROD')+small]

def factory(text):
    env={'RQCU':value(text,'RQCU')}
    for n in ast.parse(text).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_rqwg','_rqss','_rqsrc'):exec(compile(ast.Module(body=[n],type_ignores=[]),'<pure factory>','exec'),env)
    return env

def device(text):
    f=factory(text);cu=value(text,'RQCU').split('typedef CUresult (*RqEnc)')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu=cu.replace('@@WGMMA@@',''.join(f['_rqwg'](n) for n in (16,32,128,256))+f['_rqss'](128)+f['_rqss'](256))
    cu+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
    for name,args in productions(text):
        spec=','.join(map(str,args));cu+=f'rq<{spec}><<<dim3(18,7),384,RqC<{spec}>::SMEM>>>(p,x,o);\n'
    return cu+'}\n'

def tables(sass):return re.split(r'\n\s*Function : ',sass)[1:]
def streams(sass):
    return {tuple(map(int,re.findall(r'Li(\d+)E',t.splitlines()[0]))):[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)] for t in tables(sass)}
def resources(log):
    result={};current=None
    for line in log.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:current=tuple(map(int,re.findall(r'Li(\d+)E',m[1])))
        m=re.search(r'Used (\d+) registers',line)
        if m:assert current is not None;result[current]=int(m[1])
    return result

def source_proof():
    parent=HERE.parent/'pw38';assert sha(parent/'submission.py')==PARENT_SHA and sha(parent/'manifest.json')==PARENT_MANIFEST
    for name,pin in json.loads((parent/'manifest.json').read_text())['files'].items():assert sha(parent/name)==pin,name
    old=(parent/'submission.py').read_text();new=(HERE/'submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
    assert apply(old,delta)==new and apply(new,delta,True)==old
    a,b=named(old),named(new);changed=sorted(k for k in a if a[k]!=b.get(k));assert changed==['RQCU','RQSBAKE','RQV','_rqsbake'] and a.keys()==b.keys(),changed
    for key in a.keys()-set(changed):assert a[key]==b[key],key
    oldrv=value(old,'RQV');newrv=value(new,'RQV');assert [r for r in newrv if r[0] not in ('h32s4e','h32s5e')]==oldrv
    assert value(new,'RQSBAKE')[('o',32)]==value(old,'RQSBAKE')[('o',32)]+[('h32s4e',1),('h32s5e',1)]
    for key,choices in value(old,'RQSBAKE').items():
        if key!=('o',32):assert value(new,'RQSBAKE')[key]==choices
    oldcu,newcu=value(old,'RQCU'),value(new,'RQCU');guard=delta['edits'][2]
    assert newcu.replace(guard['after'],guard['before'])==oldcu
    assert set(args for _,args in productions(new))-set(args for _,args in productions(old))==NEW
    bake=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='_rqsbake')
    it=next(n.iter for n in ast.walk(bake) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['nm','KS'])
    enumerations=[]
    for (layer,m),ps in value(new,'RQSBAKE').items():
        chosen=eval(compile(ast.Expression(it),'<bake iterator>','eval'),{'layer':layer,'m':m,'ps':ps})
        assert chosen==ps[:(5 if (layer,m)==('o',32) else 4)]
        enumerations.append({'layer':layer,'m':m,'choices':chosen})
    from proj_rsi.static_check import check_source
    lint=check_source(new,str(HERE/'submission.py'));assert lint['ok'],lint
    negative=[]
    for e in delta['edits']:
        try:apply(old.replace(e['before'],e['before']+'\n'+e['before'],1),delta)
        except ValueError:negative.append(e['id'])
        else:raise AssertionError('duplicate anchor admitted')
    return {'state':'PASS_STRICT_DELTA_AND_ALL_EXISTING_AST_CONTROLS','parent_sha256':PARENT_SHA,'new_sha256':sha(HERE/'submission.py'),'changed_AST':changed,'kernel_body_codec_math_EH_unchanged':True,'all_other_shapes_and_existing_o32_choices_preserved':True,'actual_bake_enumerations':enumerations,'strict_anchor_negative_controls':negative,'lint':lint}

def compile_all():
    sources={'device':(HERE/'submission.py').read_text(),'control':(HERE.parent/'pw38/submission.py').read_text()};record={}
    for stem,text in sources.items():
        path=HERE/(stem+'_source.cu');path.write_text(device(text));cub=HERE/(stem+'.sm90a.cubin')
        r=subprocess.run(['/usr/local/cuda-12.8/bin/nvcc',*FLAGS,'-Xptxas=-v','-cubin',str(path),'-o',str(cub)],text=True,capture_output=True,timeout=240)
        log=r.stdout+r.stderr;(HERE/(stem+'-resource.log')).write_text(log);assert r.returncode==0,log
        assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log),log
        assert not re.search(r'C75|serializ',log,re.I),log
        regs=resources(log);assert len(regs)==len(productions(text)),(stem,len(regs),len(productions(text)))
        assert all(((n+7)//8)*8*384<=65536 for n in regs.values()),regs
        sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(cub)],text=True,timeout=30)
        (HERE/('sass.txt' if stem=='device' else 'control-sass.txt')).write_text(sass);record[stem]=regs
    candidate,control=streams((HERE/'sass.txt').read_text()),streams((HERE/'control-sass.txt').read_text())
    assert set(candidate)==set(control)|NEW
    for args,instructions in control.items():assert candidate[args]==instructions and record['device'][args]==record['control'][args],('control SASS drift',args)
    return {'existing_control_streams':len(control),'equality':'ALL_PC_PREDICATE_INSTRUCTION_OPERAND_BRANCH_TARGETS','registers':{stem:{','.join(map(str,k)):v for k,v in r.items()} for stem,r in record.items()}}

def remaining_proofs():
    from protocol_proof import layout,proof
    import physical_proof
    from learning_proof import proof as lessons
    protocol={'layout':layout(),'protocol':proof()}
    sass=(HERE/'sass.txt').read_text();physical=physical_proof.check(sass);negative={}
    table=next(t for t in tables(sass) if tuple(map(int,re.findall(r'Li(\d+)E',t.splitlines()[0])))==(32,4,1,0,0,0,52))
    mutations={
        'remove_wait1':table.replace('WARPGROUP.DEPBAR.LE gsb0, 0x1','NOP'),
        'plain_store_to_reduce':table.replace('UTMASTG.2D','UTMAREDG.2D'),
        'wrong_reader_barrier':table.replace('BAR.SYNC.DEFER_BLOCKING 0x1, 0x80','BAR.SYNC.DEFER_BLOCKING 0x1, 0x100'),
    }
    for name,altered in mutations.items():
        try:physical_proof.check(sass.replace(table,altered,1))
        except AssertionError:negative[name]='REJECTED'
        else:raise AssertionError('negative admitted '+name)
    # New host specialization admits exact o32/fullK only. Existing tensor
    # validation remains untouched and precedes this new specialization guard.
    def admitted(m,n,k,kc,ks,alias):return (m,n,k,kc,ks,alias)==(32,8192,1024,16,1,0)
    assert admitted(32,8192,1024,16,1,0)
    invalid=[(16,8192,1024,16,1,0),(33,8192,1024,16,1,0),(32,4096,1024,16,1,0),(32,8192,4096,64,1,0),(32,8192,1024,16,2,0),(32,8192,1024,16,1,1)]
    assert not any(admitted(*case) for case in invalid)
    learning=lessons()
    from host_admission_proof import proof as host_proof
    host_admission=host_proof()
    for filename,value_ in [('protocol-proof.json',protocol),('physical-proof.json',physical),('learning-proof.json',learning)]:
        (HERE/filename).write_text(json.dumps(value_,indent=2)+'\n')
    resource={str(st):{'registers':168,'threads':384,'uniform_rounded_register_pool':64512,'dynamic_worst_case_pool':64512,
        'dynamic_shared_bytes':1024+st*10240+24*st+16,'stack_spills_local':0,'actual_GPU_occupancy':None,
        'static_upper_bound_under_65536_register_SM':1,'runtime_gate':'Before candidate math query actual numRegs/local/shared/max_dynamic/occupancy; require supported allocation and occupancy>=1. HALF exit does not prove twoCTAs/SM.','counter_protocol':'INACTIVE_ALL_STORAGE_UNCHANGED'} for st in (4,5)}
    return {'host_admission':host_admission,'logical_payload':{'old_ST6_KS2_X_bytes':128*8*4096,'new_HALF_KS1_X_bytes':128*16*4096,'old_W_bytes':128*8*12288,'new_W_bytes':128*16*6144,'output_bytes':32*8192*2,'scope':'Source logical transaction payload only, not measured DRAM/cache traffic/time or claimed speed.'},'protocol':protocol['protocol'],'layout':protocol['layout'],'physical':physical,'learning':learning,'new_resources':resource,'physical_negative_controls':negative,'invalid_new_shape_controls':len(invalid)}

if __name__=='__main__':
    result={'state':'CPU_PW45_STARTED_GPU_UNRUN'}
    try:
        result.update(source=source_proof(),compile=compile_all(),proofs=remaining_proofs(),state='CPU_PW45_HALF_EARLY_PASS_GPU_UNRUN_REVIEW_REQUIRED',GPU_calls=0,prepare_calls=0,API_calls=0,queue_changes=0)
    except Exception as error:
        result.update(state='CPU_PW45_FAILURE_RETAINED',error=repr(error));raise
    finally:
        (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('state','error','GPU_calls') if k in result},indent=2))

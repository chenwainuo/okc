"""Pure CPU source/JIT resource helper construction; imports no torch or production module."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CAND=ROOT/'proj_rsi/candidates/rq32_predxor'
PARENT=CAND.parent/'pw45'
TARGET={(32,5,1,0,0,0,35),(32,6,1,0,0,0,35)}
ARTIFACT_ROOT=None
SOURCE_SHA='cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210'
MANIFEST_SHA='31b0079efc89c2b35cbddf7c156c558f621871f0693f9ed8bc6a736f918c9188'
SOURCE_GO='proj_rsi/scheduling/reviews/rq32-predxor-source-01.md'
SOURCE_GO_SHA='dc4be34d67cb26efb0039fc4d445ab6c2cf6d5869dec545811ef9d5b781ec051'
FIXED={'parent5':('pw45',(32,5,1,0,0,0,35)),'new5':('rq32_predxor',(32,5,1,0,0,0,35)),'parent6':('pw45',(32,6,1,0,0,0,35)),'new6':('rq32_predxor',(32,6,1,0,0,0,35))}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_args(cu):
    result={tuple(map(int,x.split(','))) for x in re.findall(r'rq_go<([0-9,]+)>',re.sub(r'\s+','',cu))}
    assert result and all(len(x)==7 for x in result);return sorted(result)
def append_attrs(cu):
    args=source_args(cu)
    helper='''\ntemplate<int NT,int ST,int GS,int PRB,int WM,int INS,int FL>
static int64_t pw45_query(int field){
 using C=RqC<NT,ST,GS,PRB,WM,INS,FL>;auto k=rq<NT,ST,GS,PRB,WM,INS,FL>;
 if(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess)return -1;
 cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -2;
 if(field==1)return a.numRegs;if(field==2)return a.localSizeBytes;if(field==3)return C::SMEM;
 if(field==4){int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,384,C::SMEM)!=cudaSuccess)return -3;return n;}
 if(field==5)return a.maxDynamicSharedSizeBytes;if(field==6)return a.maxThreadsPerBlock;return -4;
}
int64_t pw45_labattrs(int64_t index,int64_t field){switch(index){
'''
    for i,a in enumerate(args):helper+='case %d:return pw45_query<%s>((int)field);\n'%(i,','.join(map(str,a)))
    helper+='default:return -5;}}\n'
    return cu+helper,args

def streams(sass):
    result={}
    for table in re.split(r'\n\s*Function : ',sass)[1:]:
        assert table.strip(),'empty SASS function'
        symbol=table.splitlines()[0].strip()
        if not re.search(r'rqILi',symbol) or 'HGMMA.' not in table:continue
        args=tuple(map(int,re.findall(r'Li(\d+)E',symbol)));assert len(args)==7 and args not in result
        result[args]={'symbol':symbol,'table':table,'stream':[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]}
    return result

def resource(resources,symbol):
    start=resources.find(symbol);assert start>=0,('missing actual function resources',symbol)
    block=resources[start+len(symbol):].split('Function',1)[0]
    assert re.search(r'REG\s*[:=]\s*168\b',block),('REG',symbol,block[:200])
    assert re.search(r'LOCAL\s*[:=]\s*0\b',block),('LOCAL',symbol,block[:200])

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def canonical_gate(original_cu,sass,resources):
    expected=set(source_args(original_cu));actual=streams(sass);is_new='void rq_fix_pred(' in original_cu;canonical=streams(((CAND if is_new else PARENT)/'sass.txt').read_text())
    assert set(actual)==expected and expected<=set(canonical),('exact dispatched RQ set',expected,set(actual))
    for args,entry in actual.items():
        assert entry['stream']==canonical[args]['stream'],('actual RQ PC/predicate/instruction drift',args)
        resource(resources,entry['symbol'])
    physical=None
    new={(32,4,1,0,0,0,52),(32,5,1,0,0,0,52)}
    if expected&new:
        assert new<=expected
        previous={name:sys.modules.get(name) for name in ('cpu_proof','build_patch')};old_path=list(sys.path)
        try:
            sys.path.insert(0,str(PARENT));sys.modules['build_patch']=load(PARENT/'build_patch.py','pw45_frozen_build_patch')
            sys.modules['cpu_proof']=load(PARENT/'cpu_proof.py','pw45_frozen_cpu_proof')
            checker=load(PARENT/'physical_proof.py','pw45_frozen_physical')
            physical=checker.check(sass)
        finally:
            sys.path[:]=old_path
            for name,value in previous.items():
                if value is None:sys.modules.pop(name,None)
                else:sys.modules[name]=value
    predicated={}
    if is_new:
        env={'re':re}
        for node in ast.parse((CAND/'proof.py').read_text()).body:
            if isinstance(node,ast.FunctionDef) and node.name=='native_gate':exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-predxor-native-gate>','exec'),env)
        for geometry in TARGET & expected:predicated[','.join(map(str,geometry))]=env['native_gate'](actual[geometry]['stream'])
    return dict(args=sorted(actual),source_kind='predxor' if is_new else 'PW45_parent',exact_all_PC_predicate_instructions=True,registers=168,local=0,new_plain_fullK_physical=physical,predicated_physical_A=predicated)

def check_attrs(args,attrs):
    nt,st,gs,prb,wm,ins,fl=args
    shared=1024+st*(nt*128+(6144 if fl&16 else 12288))+24*st+16
    assert attrs['regs']==168 and attrs['local']==0 and attrs['shared']==shared and attrs['occupancy']>=1 and attrs['max_dynamic']>=shared and attrs['max_threads']>=384,(args,attrs)
    assert ((attrs['regs']+7)//8)*8*384==64512
    return dict(args=list(args),**attrs,threads=384,rounded_register_pool=64512)

def counter_expected(before,mode,offset=0,reset_indices=(),changes=None,calls=0):
    result=list(before)
    if mode=='reset0':
        for relative in reset_indices:
            index=offset+relative;assert 0<=index<len(result);result[index]=0
    elif mode=='progress':
        for relative,delta in (changes or {}).items():
            index=offset+relative;assert 0<=index<len(result);result[index]+=delta*calls;assert result[index]<2**31
    else:assert mode=='unchanged'
    return result

def cpu_factory(label='rq32_predxor'):
    text=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text();tree=ast.parse(text)
    literals={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RQCU','RQV','RQPROD','RQSBAKE'):literals[target.id]=ast.literal_eval(node.value)
    env={'RQCU':literals['RQCU']}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss','_rqsrc'):exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure-original-source-factory>','exec'),env)
    names=literals['RQPROD']+list(dict.fromkeys(name for options in literals['RQSBAKE'].values() for name,ks in options))
    variants={v[0]:v for v in literals['RQV']}
    return env['_rqsrc']([variants[name] for name in names])

def synthetic_resources(sass):
    return '\n'.join('Function '+entry['symbol']+'\n REG:168 LOCAL:0\n' for entry in streams(sass).values())

class ServingStatus:
    """Transparent serving-only status admission; no extra extension/GPU call."""
    def __init__(self,ext):self.ext=ext
    def __getattr__(self,name):
        fn=getattr(self.ext,name)
        if name!='lt_mm':return fn
        def checked(*args,**kwargs):
            code=fn(*args,**kwargs);assert code==0,('serving extension declined',name,code);return code
        return checked

def bytes_receipt(payload,metadata):
    assert ARTIFACT_ROOT is not None
    digest=hashlib.sha256(payload).hexdigest();ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True);path=ARTIFACT_ROOT/(digest+'.bin')
    if not path.exists():path.write_bytes(payload)
    else:assert sha(path)==digest
    return dict(metadata,sha256=digest,bytes=len(payload),artifact='physical-storage/'+path.name)
def arena_receipt(arena):
    import array
    out=[]
    for identity,entry in arena.items():
        words=array.array('i',entry['values']);assert words.itemsize==4
        if sys.byteorder!='little':words.byteswap()
        out.append(bytes_receipt(words.tobytes(),{'identity':list(identity),'name':entry['name'],'dtype':'int32','word_count':len(words)}))
    return out
def sink_receipt(A,state,torch):
    out=[];seen=set()
    for name,t in A.tensors(state):
        identity=A.tensor_id(t)
        if 'sink' in name and t.numel() and identity not in seen:
            seen.add(identity);physical=t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)
            out.append(bytes_receipt(bytes(physical.view(torch.uint8).detach().cpu().tolist()),{'identity':list(identity),'name':name,'dtype':str(t.dtype)}))
    return out

def policy_order(i):
    assert 0<=i<32
    return ((0,1),(1,0))[i%2]
def policy_plan():
    rows=[]
    for label in ('pw45','rq32_predxor','pw45'):rows.extend({'mode':'F1','label':label,'seed':9000+i} for i in range(21))
    for i in range(32):rows.extend({'mode':'paired','label':('pw45','rq32_predxor')[j],'seed':9020+i} for j in policy_order(i))
    assert len(rows)==127;return rows

def expected_edge_rows():
    rows=[[0,0,q,lane,v,24] for q in range(4) for lane in range(32) for v in range(8)]
    return rows+[[0,1,0,0,0,24],[0,31,3,31,7,24]]
def verify_edge_rows(rows,table):
    assert type(table) is int and table==56*0x01010101
    expected={tuple(row) for row in expected_edge_rows()};assert len(rows)==1026 and {tuple(row) for row in rows}==expected
    owners={(row[3],4*row[2]+row[4]//2,row[4]%2) for row in rows if row[1]==0}
    assert owners=={(lane,word,half) for lane in range(32) for word in range(16) for half in range(2)}
    return {'entries':1026,'stage0_unique_lane_word_half_owners':len(owners),'empty_other_tile_lists':True,'stages':[0,1,31],'xor7':24}

def elapsed_gate(us,metadata,on_invalid):
    import math
    if not math.isfinite(us) or us<=0:
        record=dict(metadata,state='INVALID_ELAPSED_MEASUREMENT',elapsed_us_repr=repr(us))
        on_invalid(record)
        raise AssertionError(record)
    return us

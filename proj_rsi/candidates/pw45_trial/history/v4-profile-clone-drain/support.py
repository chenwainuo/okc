"""Pure CPU source/JIT resource helper construction; imports no torch or production module."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CAND=ROOT/'proj_rsi/candidates/pw45'
SOURCE_SHA='835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3'
MANIFEST_SHA='988214460a032f11d620801f2661c32cbe9ca9572fe4f40ab0bbe931aa61ae02'
SOURCE_GO='proj_rsi/scheduling/reviews/pw45-source-01.md'
SOURCE_GO_SHA='3b3310d866c44be669c555b5d4e96a317478eeb082f1ebda3ffc2771ab043cfc'
FIXED={'old_st5_early_zi':((32,5,1,0,0,0,34),2),'half_st4_fullk':((32,4,1,0,0,0,52),1),'half_st5_fullk':((32,5,1,0,0,0,52),1)}
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
    expected=set(source_args(original_cu));actual=streams(sass);canonical=streams((CAND/'sass.txt').read_text())
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
            sys.path.insert(0,str(CAND));sys.modules['build_patch']=load(CAND/'build_patch.py','pw45_frozen_build_patch')
            sys.modules['cpu_proof']=load(CAND/'cpu_proof.py','pw45_frozen_cpu_proof')
            checker=load(CAND/'physical_proof.py','pw45_frozen_physical')
            physical=checker.check(sass)
        finally:
            sys.path[:]=old_path
            for name,value in previous.items():
                if value is None:sys.modules.pop(name,None)
                else:sys.modules[name]=value
    return dict(args=sorted(actual),exact_all_PC_predicate_instructions=True,registers=168,local=0,new_plain_fullK_physical=physical)

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

def cpu_factory(label='pw45'):
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

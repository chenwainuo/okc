"""CPU-only source and actual-binary bindings; imports no submission or CUDA."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CAND=HERE.parent
LABELS=('parent2','parent3','eh2','eh3')
FLAGS={'parent2':2,'parent3':3,'eh2':65538,'eh3':65539}
SOURCES={'parent':'pw36/submission.py','eh':'pw38/submission.py'}
PINS={'parent':'2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf','eh':'29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'}
SHAPES={'qkv_gate':(2304,8192,7),'pooler':(4096,8192,4),'unpooler':(8192,4096,2),'o':(8192,1024,2)}
CUDA_FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def value(source,name):
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)
def source_for(role):
    path=CAND/SOURCES[role];assert sha(path)==PINS[role],('source drift',role)
    return path.read_text()
def cuda_source(role):
    text=source_for(role);ns={'RQCU':value(text,'RQCU')}
    for n in ast.parse(text).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_rqwg','_rqss','_rqsrc'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<trusted source factory>','exec'),ns)
    fls=[2,3] if role=='parent' else [65538,65539]
    variants=[('fixed'+str(fl),128,5,1,0,0,0,fl) for fl in fls]
    cuda=ns['_rqsrc'](variants)
    # Read-only research attrs; device kernel bodies and dispatch cases unchanged.
    cuda+='''\ntemplate<int FL> static int64_t ehinfo(int info){
      using C=RqC<128,5,1,0,0,0,FL>;auto k=rq<128,5,1,0,0,0,FL>;
      if(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess)return -1;
      cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -2;
      if(info==1)return a.numRegs;if(info==2)return a.localSizeBytes;if(info==3)return C::SMEM;
      if(info==4){int o=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o,k,384,C::SMEM)!=cudaSuccess)return -3;return o;}
      if(info==5)return a.maxThreadsPerBlock;return -4;
    }
    int64_t labattrs(int64_t variant,int64_t info){
'''
    for idx,fl in enumerate(fls):cuda+=f'if(variant=={idx})return ehinfo<{fl}>((int)info);\n'
    return cuda+'return -5;}\n'
def load_module(role):
    path=CAND/SOURCES[role];assert sha(path)==PINS[role]
    spec=importlib.util.spec_from_file_location('eh_fixed_'+role,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def table(sass,fl):
    hits=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if re.search(r'rqILi128ELi5ELi1ELi0ELi0ELi0ELi'+str(fl)+r'EEv3RqP',t.splitlines()[0])]
    assert len(hits)==1,(fl,'expected exactly one kernel',len(hits));return hits[0]
def instructions(t):return [(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)]
def binary_gate(role,sass):
    canonical=(CAND/'rs32_eh_backport'/'sass.txt').read_text();fls=[2,3] if role=='parent' else [65538,65539]
    result={}
    for fl in fls:
        actual,expected=instructions(table(sass,fl)),instructions(table(canonical,fl))
        assert actual==expected,(role,fl,'actual JIT differs from CPU canonical')
        result[str(fl)]={'all_PC_predicate_instructions':'EXACT','instructions':len(actual),'EH':bool(fl&65536)}
    return result
def verify_frozen():
    for role in SOURCES:source_for(role)
    pins=json.loads((HERE/'pins.json').read_text())
    for path,expected in pins.items():assert sha(ROOT/path)==expected,('dependency drift',path)
    return pins

def verify_lab():
    data=json.loads((HERE/'manifest.json').read_text())
    for path,expected in data['files'].items():assert sha(ROOT/path)==expected,('lab drift',path)
    return data

"""CPU-only exact source checks and sm90a compilation; no production import."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import apply,value,BASES,TRAIT,PIN,clone_kernel
HERE=Path(__file__).resolve().parent
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-lineinfo','-std=c++17']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def named(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for t in node.targets:
                if isinstance(t,ast.Name):result[t.id]=ast.dump(node,include_attributes=False)
    return result

def source_proof():
    parent=HERE.parent/'pw38';assert sha(parent/'manifest.json')==PIN
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha(parent/name)==digest,name
    original=(parent/'submission.py').read_text();source=(HERE/'submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
    assert apply(original,delta)==source and apply(source,delta,True)==original
    old,clone,edits=clone_kernel(value(original,'GVCU'))
    assert apply(clone,edits,True)==old and old in value(source,'GVCU')
    a,b=named(original),named(source);changed=sorted(k for k in a if a[k]!=b.get(k))
    assert changed==['GVCU','_gvprep'] and set(b)-set(a)=={'GV_EMPTY_BASES','GV_EMPTY_TRAIT','_gvee_choices'},changed
    for key in ('prepare','kernel_fn','_gvcall','_gvesc','_c12','_unc12','_gvst','_gvinst','_gvext','_gvok','_time_us','RQCU','RWCU','CUDA','LTCPP','GVC','RQFLAGS','RQPROD','RQBAKE','RQSBAKE','PREPARE_BUDGET_S','MARGIN'):
        assert a[key]==b[key],key
    assert value(source,'GV_EMPTY_BASES')==tuple(BASES)
    assert 'if (entry_nx == ~0u)' in clone and '__shfl_sync(~0u, entry_e, 0)' in clone
    assert clone in value(source,'GVCU')
    # Appended GVI entries do not reorder old instance IDs, and trait masks are
    # disjoint from every old geometry/cache/reduction field for these bases.
    for args in BASES:
        v=args[-1];flagged=v|TRAIT
        for mask in (1,2,4,8,256,0x7000,0x18000,0x1c000000,0x3c000000):assert flagged&mask==v&mask
        assert (flagged & ~TRAIT)==v
    negatives=[]
    for edit in delta['edits']:
        try:apply(original.replace(edit['before'],edit['before']+'\n'+edit['before'],1),delta)
        except ValueError:negatives.append(edit['id'])
        else:raise AssertionError('duplicate source anchor admitted '+edit['id'])
    from proj_rsi.static_check import check_source
    lint=check_source(source,str(HERE/'submission.py'));assert lint['ok'],lint
    return dict(lint=lint,status='PASS_ENTIRE_PARENT_INVERSE_AND_GENERIC_KERNEL_VERBATIM',candidate_sha256=sha(HERE/'submission.py'),parent_source_sha256=sha(parent/'submission.py'),
       changed_named_AST=changed,appended_GVI=2,original_generic_kernel_sha256=hashlib.sha256(old.encode()).hexdigest(),clone_inverse='ENTIRE_ORIGINAL_KERNEL_EXACT',
       approved_bases=BASES,trait=TRAIT,strict_anchor_negatives=negatives,
       scope='Actual first prepared escape sentinel checked perwarp at entry; no host empty histogram or CPU/CUDA-weight equivalence assumption')

def device(source,twins):
    cu=value(source,'GVCU');end=cu.index('\n__global__ void p1zero')
    cu=cu[:end].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu+='\nvoid instantiate(G p){\n'
    for args in BASES:
        spec=','.join(map(str,args));wr,wk=args[3:5];threads=wr*wk*32;smem=(wk+1)*wr*32*16
        cu+=f'gv<{spec}><<<dim3(1,2304/(16*{wr}),1),{threads},{smem}>>>(p);\n'
        if twins:cu+=f'gv_empty_guard<{spec}><<<dim3(1,2304/(16*{wr}),1),{threads},{smem}>>>(p);\n'
    return cu+'}\n'

def tables(sass):
    result={}
    for table in re.split(r'\n\s*Function : ',sass)[1:]:
        name=table.splitlines()[0];kind='empty' if 'gv_empty_guard' in name else 'generic'
        args=tuple(map(int,re.findall(r'Li(\d+)E',name)))
        result[(kind,args)]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
    return result

def resource(log):
    result={};kind=None;args=None
    for line in log.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:kind='empty' if 'gv_empty_guard' in m[1] else 'generic';args=tuple(map(int,re.findall(r'Li(\d+)E',m[1])))
        n=re.search(r'Used (\d+) registers',line)
        if n:result[(kind,args)]=int(n[1])
    return result

def main():
    result=dict(state='CPU_PW42_STARTED_GPU_UNRUN')
    try:
        result['source']=source_proof();source=(HERE/'submission.py').read_text();old=(HERE.parent/'pw38/submission.py').read_text();nvcc='/usr/local/cuda-12.8/bin/nvcc';regs={}
        for stem,text,twins in [('device',source,True),('control',old,False)]:
            path=HERE/(stem+'_source.cu');path.write_text(device(text,twins))
            c=subprocess.run([nvcc,*FLAGS,'-Xptxas=-v','-cubin',str(path),'-o',str(HERE/(stem+'.sm90a.cubin'))],text=True,capture_output=True,timeout=180)
            log=c.stdout+c.stderr;(HERE/(stem+'-resource.log')).write_text(log);assert c.returncode==0,log
            assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log),'Register/code-size feasibility failure: retain version and diagnose'
            regs[stem]=resource(log);assert len(regs[stem])==(4 if twins else 2)
            assert all(r<=64 for r in regs[stem].values()),regs[stem]
            sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/(stem+'.sm90a.cubin'))],text=True,timeout=30)
            (HERE/('sass.txt' if twins else 'control-sass.txt')).write_text(sass)
        actual,control=tables((HERE/'sass.txt').read_text()),tables((HERE/'control-sass.txt').read_text())
        for args in BASES:assert actual[('generic',args)]==control[('generic',args)],('GENERIC_CONTROL_CODEGEN_DRIFT',args)
        counts={kind+':'+','.join(map(str,args)):dict(instructions=len(rows),HMMA=sum('HMMA.' in i for p,i in rows),SHFL=sum('SHFL.' in i for p,i in rows)) for (kind,args),rows in actual.items()}
        result.update(state='CPU_PW42_COMPILE_PASS_REVIEW_REQUIRED_GPU_UNRUN',registers={stem:{kind+':'+','.join(map(str,args)):r for (kind,args),r in values.items()} for stem,values in regs.items()},
            instruction_counts=counts,existing_generic_controls='EXACT_ALL_PC_PREDICATE_INSTRUCTION_OPERAND_BRANCH_TARGETS',GPU='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE')
    except Exception as error:result.update(state='CPU_PW42_FAILURE_RETAINED',error=repr(error));raise
    finally:
        (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ('state','error','registers')},indent=2))
if __name__=='__main__':main()

"""CPU-only source, protocol, nvcc resource and exact control SASS gates."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from build_patch import apply,sha,value
HERE=Path(__file__).resolve().parent
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
ARMS={'coupled_x5w5':(128,5,1,0,0,0,3),'split_x5w5':(128,5,1,0,0,0,1283),'split_x6w4':(128,6,1,0,0,0,1027)}
CONTROLS={'coupled_x5w5':ARMS['coupled_x5w5'],'current_x5w5_nohint':(128,5,1,0,0,0,2),'current_x4w4_hint':(128,4,1,0,0,0,3)}
CPU_ARMS={**ARMS,**CONTROLS}
def factories(source):
    ns={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss','_rqsrc'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<trusted-source-factory>','exec'),ns)
    ns['RQCU']=value(source,'RQCU');return ns

def extension_source(source):
    ns=factories(source)
    vs=[(name,*args) for name,args in ARMS.items()]
    cu=ns['_rqsrc'](vs)
    cu+='\n'
    cu+='template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL> int64_t wr_info(int info){using C=RqC<NT,ST,GS,PRB,WM,INS,FL>;auto k=rq<NT,ST,GS,PRB,WM,INS,FL>;cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;if(info==4){if(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess)return -6;int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,384,C::SMEM)!=cudaSuccess)return -7;return n;}return info==1?a.numRegs:info==2?(int)a.localSizeBytes:info==3?C::SMEM:info==5?a.maxDynamicSharedSizeBytes:a.maxThreadsPerBlock;}\n'
    cu+='int64_t wrinfo(int64_t variant,int64_t info){switch(variant){\n'
    for i,args in enumerate(ARMS.values()):
        cu+=f"case {i}:return wr_info<{','.join(map(str,args))}>(info);\n"
    return cu+'}return -2;}\n' 

def device(source,arms=CPU_ARMS):
    ns=factories(source);cu=value(source,'RQCU').split('typedef CUresult (*RqEnc)')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu=cu.replace('@@WGMMA@@',ns['_rqwg'](128)+ns['_rqss'](128))
    cu+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
    for args in arms.values():
        a=','.join(str(x) for x in args)
        cu+=f'rq<{a}><<<dim3(18,7),384,RqC<{a}>::SMEM>>>(p,x,o);\n'
    return cu+'}\n'

def tables(sass):return re.split(r'\n\s*Function : ',sass)[1:]
def streams(sass):
    result={}
    for table in tables(sass):
        name=table.splitlines()[0]
        args=tuple(map(int,re.findall(r'Li(\d+)E',name)))
        result[args]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
    return result

def physical(sass,expected=5):
    result=[]
    for args,rows in streams(sass).items():
        st=args[1];fl=args[-1];ws=(fl>>8)&255 or st
        hgmma=[p for p,i in rows if 'HGMMA.' in i]
        stsm=[p for p,i in rows if 'STSM.' in i]
        reduce=[p for p,i in rows if 'UTMAREDG.' in i]
        final=[p for p,i in rows if 'WARPGROUP.DEPBAR.LE' in i and '0x0' in i]
        assert len(hgmma)>=8 and len(stsm)==8 and len(reduce)==1,(args,len(hgmma),len(stsm),len(reduce))
        assert final and max(hgmma)<max(final)<min(stsm)
        assert any(p>max(reduce) and i.startswith('DEPBAR.LE') for p,i in rows)
        assert not any(re.search(r'\b(?:LDL|STL)\b',i) for p,i in rows)
        assert any('TRAP' in i for p,i in rows),'bounded zero poll lost trap'
        assert any('USETMAXREG.TRY_ALLOC.CTAPOOL' in i and '0xe8' in i for p,i in rows)
        assert any('USETMAXREG.DEALLOC' in i and '0x28' in i for p,i in rows)
        result.append(dict(args=args,Xslots=st,Wslots=ws,HGMMA=len(hgmma),output_STSM=len(stsm),reduce_issues=1,
                           final_wait0_before_output=True,TMA_read_drain=True,bounded_poll_trap=True,
                           limits='Source order binds independent ring protocol; actual JIT must match this exact stream. No arbitrary CFG/register allocator proof.'))
    assert len(result)==expected
    return result

def main():
    result={'state':'CPU_STARTED_GPU_UNRUN'}
    try:
        source=(HERE/'submission.py').read_text();parent=(HERE.parent/'pw36/submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
        assert sha(parent)==delta['parent_source_sha256']
        assert apply(parent,delta)==source and apply(source,delta,True)==parent
        old,new=value(parent,'RQCU'),value(source,'RQCU')
        # Entire zeroing+counter protocol and entire output suffix must stay exact.
        za='    } else if (ZI && warp >= 10)';zb='  // ---------------- math warpgroups:'
        assert old[old.index(za):old.index(zb)]==new[new.index(za):new.index(zb)]
        out='  if (INS) pc[0] = clock64() - tq0;'
        assert old[old.index(out):]==new[new.index(out):]
        for key in ('RQPROD','RQBAKE','RQKS','RQFLAGS'):assert value(parent,key)==value(source,key)
        negatives=[]
        for edit in delta['edits']:
            try:apply(parent.replace(edit['before'],edit['before']+'\n'+edit['before'],1),delta)
            except ValueError:negatives.append(edit['id'])
            else:raise AssertionError('Duplicate anchor accepted '+edit['id'])
        (HERE/'extension_source.cu').write_text(extension_source(source))
        nvcc='/usr/local/cuda-12.8/bin/nvcc'
        for stem,src,arms in [('device',source,CPU_ARMS),('control',parent,CONTROLS)]:
            path=HERE/(stem+'_source.cu');path.write_text(device(src,arms))
            command=[nvcc,*FLAGS,'-Xptxas=-v','-cubin',str(path),'-o',str(HERE/(stem+'.sm90a.cubin'))]
            c=subprocess.run(command,text=True,capture_output=True,timeout=90)
            (HERE/(stem+'-resource.log')).write_text(c.stdout+c.stderr);assert c.returncode==0,c.stderr
            assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',c.stderr)
            assert not re.search(r'C75|serializ',c.stderr,re.I)
            regs=[int(x) for x in re.findall(r'Used (\d+) registers',c.stderr)]
            assert len(regs)==len(arms) and all(((r+7)//8)*8*384<=65536 for r in regs),regs
            sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/(stem+'.sm90a.cubin'))],text=True,timeout=30)
            (HERE/('sass.txt' if stem=='device' else 'control-sass.txt')).write_text(sass)
            result[stem+'_registers']=regs
        baseline=streams((HERE/'control-sass.txt').read_text());candidate=streams((HERE/'sass.txt').read_text())
        for control in CONTROLS.values():assert baseline[control]==candidate[control],('Non-Wring current control SASS drift',control)
        control=ARMS['coupled_x5w5']
        import protocol_proof
        protocol=protocol_proof.proof();(HERE/'protocol-proof.json').write_text(json.dumps(protocol,indent=2)+'\n')
        result.update(state='CPU_WRING_REBASE_PASS_GPU_UNRUN_REVIEW_REQUIRED',candidate_source_sha256=sha(source),extension_source_sha256=sha(extension_source(source)),
                      exact_control_instruction_count=len(baseline[control]),existing_control_SASS='EXACT_PC_PREDICATE_OPERAND_BRANCH_TARGET_EQUAL',
                      physical=physical((HERE/'sass.txt').read_text()),strict_duplicate_anchor_negatives=negatives,
                      original_zero_counter_and_output_source='EXACT_UNCHANGED',runtime='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE')
    except Exception as error:
        result.update(state='CPU_WRING_REBASE_FAILURE_RETAINED',error=repr(error));raise
    finally:
        (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

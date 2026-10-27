"""CPU-only PW40 composition, full RQ control SASS/resource and lifetime proofs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import apply,value,PW38_MANIFEST,WRING_MANIFEST
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def frozen(folder,pin):
    root=HERE.parent/folder;assert sha(root/'manifest.json')==pin
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(root/name)==expected,(folder,name)
    return manifest

def named(text):
    result={}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):result[target.id]=ast.dump(node,include_attributes=False)
    return result

def productions(source):
    rv={v[0]:v for v in value(source,'RQV')};large=value(source,'RQPROD')
    small=list(dict.fromkeys(name for choices in value(source,'RQSBAKE').values() for name,ks in choices))
    return [(name,rv[name][1:8]) for name in large+small]

def factories(source):
    env={'RQCU':value(source,'RQCU')}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss','_rqsrc'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-pure-CUDA-factory>','exec'),env)
    return env

def device(source):
    env=factories(source);cu=value(source,'RQCU').split('typedef CUresult (*RqEnc)')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu=cu.replace('@@WGMMA@@',''.join(env['_rqwg'](n) for n in (16,32,128,256))+env['_rqss'](128)+env['_rqss'](256))
    cu+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
    for name,args in productions(source):
        spec=','.join(map(str,args));cu+=f'rq<{spec}><<<dim3(18,7),384,RqC<{spec}>::SMEM>>>(p,x,o);\n'
    return cu+'}\n'

def tables(sass):return re.split(r'\n\s*Function : ',sass)[1:]
def streams(sass):
    result={}
    for table in tables(sass):
        args=tuple(map(int,re.findall(r'Li(\d+)E',table.splitlines()[0])))
        result[args]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
    return result

def source_proof():
    parent_manifest=frozen('pw38',PW38_MANIFEST);frozen('pw36_wring_rebase',WRING_MANIFEST)
    frozen('rs32_eh_backport',parent_manifest['EH_backport_manifest_sha256'])
    frozen('pw39','72b5298233bbef7ef4c02e225695618b9a5f7fce9dae51db99372c766f6157b0')
    old=(HERE.parent/'pw38/submission.py').read_text();research=(HERE.parent/'pw36_wring_rebase/submission.py').read_text()
    source=(HERE/'submission.py').read_text();base=(HERE.parent/'pw36/submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text());eh=json.loads((HERE.parent/'pw38/delta.json').read_text())
    core,policy=delta['kernel_catalog_edits'],delta['host_policy_edits']
    assert apply(old,delta)==source and apply(source,delta,True)==old
    ew=apply(apply(base,eh),core);we=apply(apply(base,core),eh)
    assert ew==we and apply(ew,policy)==source
    # Entire research ancestor recovers through the explicit negative-catalog bridge.
    bridge=delta['alternate_ancestor_bridge'];portable=research.replace(bridge['remove_unused_negative_equal_catalog_row'],'',1)
    assert portable==apply(base,core) and apply(apply(portable,eh),policy)==source
    restored=apply(apply(source,policy,True),eh,True)
    assert restored.replace(bridge['split_row'],bridge['remove_unused_negative_equal_catalog_row']+bridge['split_row'],1)==research
    # No normalization or arbitrary source rewriting in the commuting paths.
    spans=[]
    for group,edits in [('wring',core),('EH',eh['edits'])]:
        for edit in edits:
            assert base.count(edit['before'])==1
            lo=base.index(edit['before']);hi=lo+len(edit['before']);spans.append(dict(group=group,id=edit['id'],lo=lo,hi=hi))
    for a in spans:
        for b in spans:
            if a['group']!=b['group']:assert a['hi']<=b['lo'] or b['hi']<=a['lo'],(a['id'],b['id'])
    a,b=named(old),named(source);changed=sorted(key for key in a if a[key]!=b.get(key))
    assert changed==['RQCU','RQPROD','RQV','_rqbake'] and set(a)==set(b),changed
    for key in ('prepare','kernel_fn','_time_us','_rqstart','_rqsbake','RQBAKE','RQSBAKE','RQPROD2','RQKS','RQFLAGS','MARGIN','PREPARE_BUDGET_S','RWCU','_rsbake'):
        assert a[key]==b[key],key
    assert '_rqconfirm' not in b
    assert value(source,'RQPROD')==value(old,'RQPROD')+['wring_x6w4hz']
    variants={v[0]:v for v in value(source,'RQV')};assert 'wring_x5w5hz' not in variants
    assert variants['wring_x6w4hz']==('wring_x6w4hz',128,6,1,0,0,0,1027)
    assert value(source,'RQKS')['qkv_gate']==7
    cu_old,cu_new=value(old,'RQCU'),value(source,'RQCU')
    start='    } else if (ZI && warp >= 10)';end='  // ---------------- math warpgroups:'
    assert cu_old[cu_old.index(start):cu_old.index(end)]==cu_new[cu_new.index(start):cu_new.index(end)]
    start='  if (INS) pc[0] = clock64() - tq0;';assert cu_old[cu_old.index(start):]==cu_new[cu_new.index(start):]
    # Only QKV128 receives one extra validated contender; all existing order/KS survive.
    bake=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_rqbake')
    iterator=next(n.iter for n in ast.walk(bake) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='nm')
    choices=[]
    for layer in value(source,'RQLAYERS'):
        for m in (128,256):
            original=value(source,'RQBAKE')[m]
            trial=eval(compile(ast.Expression(body=iterator),'<actual-bake-iterator>','eval'),{'RQBAKE':value(source,'RQBAKE'),'layer':layer,'m':m})
            assert trial[:len(original)]==original
            choices.append(dict(layer=layer,m=m,candidates=trial,KS=value(source,'RQKS')[layer]))
    assert sum('wring_x6w4hz' in x['candidates'] for x in choices)==1
    duplicates=[]
    for edit in delta['edits']:
        try:apply(old.replace(edit['before'],edit['before']+'\n'+edit['before'],1),delta)
        except ValueError:duplicates.append(edit['id'])
        else:raise AssertionError('duplicate anchor admitted '+edit['id'])
    # PW39 remains unintegrated. Establish only typed source compatibility in memory.
    pw39=json.loads((HERE.parent/'pw39/delta.json').read_text())
    assert apply(apply(old,pw39),delta)==apply(apply(old,delta),pw39)
    from proj_rsi.static_check import check_source
    lint=check_source(source,str(HERE/'submission.py'));assert lint['ok'],lint
    return dict(status='PASS_EXACT_FULL_SOURCE_COMMUTATIVITY_AND_ANCESTOR_INVERSES',parent_sha256=digest(old),research_parent_sha256=digest(research),common_base_sha256=digest(base),
       kernel_catalog_EH_commuting_sha256=digest(ew),full_candidate_sha256=digest(source),normalizations=0,disjoint_base_edit_spans=spans,
       changed_named_AST=changed,optional_bake_choices=choices,strict_anchor_negatives=duplicates,
       future_PW39='IN_MEMORY_SOURCE_COMMUTATIVITY_ONLY_NOT_INTEGRATED_NOT_VALIDATED',
       invariant='codec/math/zero-trap/counter/output/D-retirement/existing-EH/default-controls unchanged outside seven Wring layout+publication edits',lint=lint)

def resources(log):
    out={};current=None
    for line in log.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:current=tuple(map(int,re.findall(r'Li(\d+)E',m[1])))
        r=re.search(r'Used (\d+) registers',line)
        if r:assert current is not None;out[current]=int(r[1])
    return out

def execute_pure(path,names,env):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-pure-proof>','exec'),env)
    return env

def main():
    result=dict(state='CPU_PW40_STARTED_GPU_UNRUN')
    try:
        proof=source_proof()
        from proj_rsi.candidates.pw40.learning_proof import proof as lessons
        learning=lessons();(HERE/'learning-proof.json').write_text(json.dumps(learning,indent=2)+'\n')
        source=(HERE/'submission.py').read_text();old=(HERE.parent/'pw38/submission.py').read_text();nvcc='/usr/local/cuda-12.8/bin/nvcc'
        resource={}
        for stem,text in [('device',source),('control',old)]:
            path=HERE/(stem+'_source.cu');path.write_text(device(text))
            command=[nvcc,*FLAGS,'-Xptxas=-v','-cubin',str(path),'-o',str(HERE/(stem+'.sm90a.cubin'))]
            run=subprocess.run(command,text=True,capture_output=True,timeout=180);log=run.stdout+run.stderr;(HERE/(stem+'-resource.log')).write_text(log)
            assert run.returncode==0,log
            assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log)
            assert not re.search(r'C75|serializ',log,re.I)
            regs=resources(log);assert len(regs)==len(productions(text)),(stem,len(regs),len(productions(text)))
            assert all(((n+7)//8)*8*384<=65536 for n in regs.values()),regs
            sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/(stem+'.sm90a.cubin'))],text=True,timeout=30)
            (HERE/('sass.txt' if stem=='device' else 'control-sass.txt')).write_text(sass);resource[stem]=regs
        candidate,control=streams((HERE/'sass.txt').read_text()),streams((HERE/'control-sass.txt').read_text())
        assert len(candidate)==20 and len(control)==19
        for args,instructions in control.items():
            assert candidate[args]==instructions,('EXISTING_EH_OR_NONWRING_CODEGEN_DRIFT',args)
            assert resource['device'][args]==resource['control'][args]
        wring_args=(128,6,1,0,0,0,1027)
        validated=streams((HERE.parent/'pw36_wring_rebase/sass.txt').read_text())
        assert candidate[wring_args]==validated[wring_args],'VALIDATED_WRING_SPLIT_CODEGEN_DRIFT'
        assert resource['device'][wring_args]==168
        # Correct frozen physical gates for inheritedEH and validated split.
        selected=[]
        for table in tables((HERE/'sass.txt').read_text()):
            args=tuple(map(int,re.findall(r'Li(\d+)E',table.splitlines()[0])))
            if args in [(128,5,1,0,0,0,f) for f in (2,3,65538,65539)]:selected.append('\n Function : '+table)
        env=execute_pure(HERE.parent/'rs32_eh_backport/preflight.py',{'tables','physical'},{'re':re})
        EH=env['physical'](''.join(selected))
        split_table=next(table for table in tables((HERE/'sass.txt').read_text()) if tuple(map(int,re.findall(r'Li(\d+)E',table.splitlines()[0])))==wring_args)
        env=execute_pure(HERE.parent/'pw36_wring_rebase/preflight.py',{'tables','streams','physical'},{'re':re})
        split=env['physical']('\n Function : '+split_table,expected=1)
        env=execute_pure(HERE.parent/'pw36_wring_rebase/protocol_proof.py',{'program','model','lifetime','require_X','require_all_warp_reads','proof'},{'random':__import__('random')})
        protocol=env['proof']();(HERE/'protocol-proof.json').write_text(json.dumps(protocol,indent=2)+'\n')
        result.update(state='CPU_PW40_OPTIONAL_WRING_STACK_PASS_GPU_UNRUN_REVIEW_REQUIRED',source=proof,applied_lessons=[r['ref'] for r in learning['applied']],existing_control_SASS=dict(functions=19,equal='ALL_PC_PREDICATE_INSTRUCTION_OPERAND_BRANCH_TARGETS',
           instruction_counts={','.join(map(str,k)):len(v) for k,v in control.items()}),validated_split_SASS='EXACT_FROZEN_WRING_ACTUAL_VALIDATED_MAIN',EH_physical=EH,split_physical=split,
           registers={stem:{','.join(map(str,k)):v for k,v in rows.items()} for stem,rows in resource.items()},
           new_split_resource=dict(registers=168,threads=384,uniform_pool=64512,dynamic_pool=64512,shared_bytes=148656,local_stack_spills=0),
           CUDA_calls=0,production_imports=0,prepare_calls=0,extension_builds=0,performance='UNMEASURED_AGAINST_PW38_NO_ADDITIVE_CLAIM')
    except Exception as error:result.update(state='CPU_PW40_FAILURE_RETAINED',error=repr(error));raise
    finally:
        (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ('state','error','CUDA_calls','prepare_calls')},indent=2))
if __name__=='__main__':main()

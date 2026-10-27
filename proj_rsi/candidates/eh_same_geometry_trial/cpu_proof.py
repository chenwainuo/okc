"""CPU source, cubin and adversarial terminal tests; never imports GPU lab."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import torch
import support as S
HERE=Path(__file__).resolve().parent
LAB=S.ROOT/'proj_rsi/tools/eh_same_geometry_lab.py'

def terminal_proof():
    text=LAB.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal_gate')
    calls=[n.func.attr for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'replay','copy_','fill_','reset','capture_begin','capture_end'}
    ns=dict(torch=torch,LABELS=S.LABELS,reference=SimpleNamespace(check_implementation=lambda d,o,w:(torch.equal(d['x'].float(),o),'CPU synthetic exact recorded fixture')))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(LAB),'exec'),ns)
    def case(bad=None):
        fixtures={i:torch.arange(8).reshape(2,4)+100*i for i in range(4)};last=dict(zip(S.LABELS,range(4)))
        x=fixtures[3].clone();outputs={l:fixtures[last[l]].float().clone() for l in S.LABELS}
        w=torch.tensor([1.]);lay=torch.tensor([2.]);esc=torch.tensor([3.]);immutable=[(a,a.clone()) for a in [w,lay,esc]]
        counters={l:torch.zeros(4,dtype=torch.int32) for l in S.LABELS}
        if bad=='output':outputs['parent2'][0,0]+=1
        if bad=='NaN':outputs['eh2'][0,0]=float('nan')
        if bad=='currentX':x[0,0]+=1
        if bad=='W':w[0]+=1
        if bad=='packedW':lay[0]+=1
        if bad=='escape':esc[0]+=1
        if bad=='counter':counters['eh3'][0]=1
        if bad=='wrong_recorded_seed':last['parent2']=3
        if bad=='output_alias':outputs['parent2']=outputs['eh3']
        return ns['terminal_gate'](outputs,x,{'layer':'o'},w,fixtures,last,{'input_seed':3},immutable,counters)
    assert len(case())==4
    rejected={}
    for bad in ['output','NaN','currentX','W','packedW','escape','counter','wrong_recorded_seed','output_alias']:
        try:case(bad)
        except AssertionError as e:rejected[bad]=str(e)
        else:raise AssertionError('admitted '+bad)
    measure=text[text.index('def measure('):text.index('def validate(')]
    assert measure.index("checks={'F1':strict")<measure.index('for repetition in range(32):')
    assert measure.index("checks['paired']=strict")<measure.index('    return dict(')
    main=text[text.index('def main():'):]
    assert main.index("record['timing']=measure(")<main.index("record['mutations']=mutation_checks(")<main.index('from torch.profiler import')
    assert main.index("result['all_candidate_JIT_and_resources_before_first_math']")<main.index('weights=reference.generate_weights')
    assert 'P.prepare(' not in text and 'B.prepare(' not in text
    return dict(status='CPU_ADVERSARIAL_TERMINAL_PASS',negative_controls=rejected,distinct_last_fixtures='PASS',phase_order='F1 then strict4 then paired then strict4 then mutations/profile',no_gate_mutation_replay=True)

def proof():
    S.verify_frozen()
    records={}
    for role in S.SOURCES:
        cu=S.cuda_source(role);(HERE/(role+'.extension.cu')).write_text(cu)
        device=cu.split('typedef CUresult (*RqEnc)')[0].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
        flags=[2,3] if role=='parent' else [65538,65539]
        device+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
        for fl in flags:device+=f'rq<128,5,1,0,0,0,{fl}><<<128,384,RqC<128,5,1,0,0,0,{fl}>::SMEM>>>(p,x,o);\n'
        device+='}\n';path=HERE/(role+'.device.cu');path.write_text(device)
        out=HERE/(role+'.cubin')
        command=['/usr/local/cuda-12.8/bin/nvcc',*S.CUDA_FLAGS,'-Xptxas=-v','-cubin',str(path),'-o',str(out)]
        c=subprocess.run(command,text=True,capture_output=True,timeout=90);(HERE/(role+'.resource.log')).write_text(c.stdout+c.stderr)
        assert c.returncode==0,c.stderr
        sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(out)],text=True)
        (HERE/(role+'.sass')).write_text(sass);records[role]=S.binary_gate(role,sass)
    wrong={}
    for role,other in [('parent','eh'),('eh','parent')]:
        try:S.binary_gate(role,(HERE/(other+'.sass')).read_text())
        except AssertionError as e:wrong[role+'_wrong_source']=repr(e)
        else:raise AssertionError('wrong source admitted')
        damaged=(HERE/(role+'.sass')).read_text().replace('HGMMA.','BROKEN_HGMMA.',1)
        try:S.binary_gate(role,damaged)
        except AssertionError as e:wrong[role+'_instruction_drift']=repr(e)
        else:raise AssertionError('instruction drift admitted')
    result=dict(status='CPU_EH_SAME_GEOMETRY_LAB_PASS_GPU_UNRUN',sources=S.PINS,shapes=S.SHAPES,
                canonical_all_PC_gates=records,wrong_source_and_instruction_rejections=wrong,terminal=terminal_proof(),
                per_shape_counts=dict(F1=126,paired=128,warmup=20,all_raw=274,terminal_oracles=8,mutations=20),
                total_four_shapes=dict(all_raw=1096,terminal_oracles=32,mutations=80),
                scope='FIXED_GEOMETRY_RESEARCH_NOT_SELECTED_POLICY_OR_FULLSTACK_WIN',GPU='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE')
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':proof()

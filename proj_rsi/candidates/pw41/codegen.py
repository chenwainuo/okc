"""Offline sm90a codegen only. Never imports torch or loads a CUDA extension."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from build_patch import HERE,PARENT,TARGETS,LATE,value,catalog

def device(source,args):
    cu=value(source,'GVCU').split('\n__global__ void p1zero(')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu+='\nvoid instantiate(G p){\n'
    for a in args:
        cu+='  gv<'+','.join(map(str,a))+'><<<dim3(1,16,16),256,32768>>>(p);\n'
    return cu+'}\n'

def streams(text):
    result={}
    for block in text.split('Function : ')[1:]:
        symbol=block.splitlines()[0].strip()
        if not symbol.startswith('_Z2gvI'):continue
        args=tuple(map(int,re.findall(r'Li(\d+)E',symbol)))
        assert len(args)==10 and args not in result
        result[args]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',block)]
    return result

def resources(text):
    result={};current=None
    for line in text.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:current=tuple(map(int,re.findall(r'Li(\d+)E',m[1])))
        m=re.search(r'Used (\d+) registers',line)
        if m:result[current]=int(m[1])
    return result

def details(text):
    out={};current=None
    for line in text.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:
            current=tuple(map(int,re.findall(r'Li(\d+)E',m[1])));out[current]={}
        m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
        if m:out[current].update(zip(('stack','spill_stores','spill_loads'),map(int,m.groups())))
        m=re.search(r'Used (\d+) registers',line)
        if m:out[current]['registers']=int(m[1])
    return out

def build(full=False):
    source=(HERE/'submission.py').read_text();old=PARENT.read_text()
    folder=HERE/('full-codegen-v1' if full else 'prototype-v1');folder.mkdir(exist_ok=False)
    oldargs=catalog(old)['GVI'] if full else TARGETS
    newargs=oldargs+[a[:-1]+(a[-1]|LATE,) for a in TARGETS]
    result={'status':'STARTED','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'GPU_calls':0,'mode':'full' if full else 'target-prototype','commands':[]}
    try:
        for stem,text,args in [('control',old,oldargs),('candidate',source,newargs)]:
            cu=folder/(stem+'.cu');cu.write_text(device(text,args))
            command=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-arch=sm_90a','-O3','-lineinfo','-Xptxas=-v','-cubin',str(cu),'-o',str(folder/(stem+'.cubin'))]
            result['commands'].append(command)
            process=subprocess.run(command,capture_output=True,text=True,timeout=240)
            log=process.stdout+process.stderr;(folder/(stem+'.log')).write_text(log)
            assert process.returncode==0,('compile failed',stem,process.returncode)
            sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(folder/(stem+'.cubin'))],text=True,timeout=30)
            (folder/(stem+'.sass')).write_text(sass)
        controls=streams((folder/'control.sass').read_text());new=streams((folder/'candidate.sass').read_text())
        assert len(controls)==len(oldargs) and len(new)==len(newargs)
        for args,stream in controls.items():assert new[args]==stream,('old control SASS changed',args)
        oldregs=resources((folder/'control.log').read_text());newregs=resources((folder/'candidate.log').read_text())
        assert all(newregs[a]==r for a,r in oldregs.items())
        olddetails=details((folder/'control.log').read_text());newdetails=details((folder/'candidate.log').read_text())
        assert all(newdetails[a]==r for a,r in olddetails.items()),'old resource contract changed'
        rows=[]
        for args in TARGETS:
            late=args[:-1]+(args[-1]|LATE,);items=new[late];prior=controls[args]
            assert all(newdetails[late][key]==0 for key in ('stack','spill_stores','spill_loads'))
            def positions(ins,token):return [pc for pc,i in ins if token in i]
            integer_atomics=[(pc,ins) for pc,ins in items if 'ATOMG.E.ADD.STRONG.GPU' in ins]
            tickets=[pc for pc,ins in integer_atomics if not re.search(r'\bPT, RZ,',ins)]
            flags=[pc for pc,ins in integer_atomics if re.search(r'\bPT, RZ,',ins)]
            mma=positions(items,'HMMA.');loads=positions(items,'LDG.E.128')+positions(items,'LDG.E.NA.128')
            assert len(tickets)==len(flags)==1 and max(mma)<tickets[0],'late ticket did not remain after math'
            # The odd flag increment's out-of-line zeroer is another ATOMG.
            # For x types it occurs in the early-return block, not this ticket.
            rows.append(dict(args=args,late_args=late,registers=newregs[late],control_registers=oldregs[args],instructions=len(items),control_instructions=len(prior),ticket_atomic_PC=tickets[0],zeroer_flag_atomic_PC=flags[0],integer_atomic_PCs=positions(items,'ATOMG.E.ADD.STRONG.GPU'),HMMA_PCs=mma,first_vector_load_PC=min(loads),control_integer_atomic_PCs=positions(prior,'ATOMG.E.ADD.STRONG.GPU'),stack_spills=newdetails[late]))
        result.update(status='COMPILE_AND_OLD_CONTROL_IDENTITY_PASS',old_control_count=len(controls),new_function_count=4,targets=rows)
    except BaseException as error:
        result.update(status='FAILURE_RETAINED',failure=repr(error));raise
    finally:
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args();build(a.full)

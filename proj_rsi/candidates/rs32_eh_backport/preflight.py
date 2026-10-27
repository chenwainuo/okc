"""AST/source/bitmap + offline nvcc/cubin proofs. No torch/CUDA import or context."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from build_patch import apply,sha,value
HERE=Path(__file__).resolve().parent
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']

def device(source,flags=(2,3,65538,65539)):
    tree=ast.parse(source);ns={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<trusted-source-factory>','exec'),ns)
    cu=value(source,'RQCU').split('typedef CUresult (*RqEnc)')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu=cu.replace('@@WGMMA@@',ns['_rqwg'](128)+ns['_rqwg'](256)+ns['_rqss'](128)+ns['_rqss'](256))
    cu+='\nvoid instantiate(RqP p,RqTmap x,RqTmap o){\n'
    for fl in flags:
        cu+=f'rq<128,5,1,0,0,0,{fl}><<<128,384,RqC<128,5,1,0,0,0,{fl}>::SMEM>>>(p,x,o);\n'
    return cu+'}\n'

def bitmap():
    # Parent stmatrix.trans layout: pp writes16tokens acrossall64channels.
    # Input D fragments follow canonical m64n128 register-D map; each8x8
    # matrix transposes channelrow/token pair coordinates before sharedSW128.
    allslots=set();halves=[]
    for g in range(2):
        for half in range(2):
            slab=set()
            for pp in range(4*half,4*half+4):
                for w in range(4):
                    for lane in range(32):
                        # stmatrix.trans.x4 output belongs to four8x8matrices:
                        # i0/i0+1 token8blocks, channel16*w+0/8.
                        for reg in range(4):
                            for elem in range(2):
                                ch=64*g+16*w+(lane>>2)+8*(reg&1)
                                tok=16*pp+8*(reg>>1)+2*(lane&3)+elem
                                # Corresponds to original accumulator D[8pp+2reg+elem].
                                coord=(tok,ch)
                                assert coord not in allslots
                                allslots.add(coord);slab.add(coord)
            assert len(slab)==4096
            assert {t for t,c in slab}==set(range(64*half,64*half+64))
            assert {c for t,c in slab}==set(range(64*g,64*g+64))
            halves.append(dict(WG=g,half=half,bytes=8192,source_range=[g*16384+half*8192,g*16384+(half+1)*8192],global_token_range=[half*64,(half+1)*64],global_channel_range=[g*64,(g+1)*64]))
    assert allslots=={(t,c) for t in range(128) for c in range(128)}
    assert all(h['source_range'][0]%1024==0 for h in halves)
    return dict(status='PASS_16384_OUTPUT_BF16_ONCE',half_slabs=halves,first_TMA_vs_second_STSM='DISJOINT_8KiB_RANGES_PER_WG',descriptor_box=[64,64],no_alias_until_both_TMA_reads_complete=True)

def tables(sass):return re.split(r'\n\s*Function : ',sass)[1:]
def physical(sass):
    results=[]
    for table in tables(sass):
        name=table.splitlines()[0];flag=int(re.search(r'Li(\d+)EEv3RqP',name)[1]);eh=bool(flag&65536)
        rows=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
        hgmma=[p for p,i in rows if 'HGMMA.' in i];assert hgmma
        stsm=[p for p,i in rows if 'STSM.' in i];assert len(stsm)==8
        reduce=[p for p,i in rows if 'UTMAREDG.' in i];assert len(reduce)==(2 if eh else 1),(flag,reduce)
        readwait=[p for p,i in rows if 'UTMA' in i and 'WAIT' in i]
        # TMA read drain lowers to DEPBAR.LE SB0,0 in SASS; require it afterallissues.
        depbar=[p for p,i in rows if i.startswith('DEPBAR.LE')]
        assert depbar and max(depbar)>max(reduce)
        wgwait0=[p for p,i in rows if 'WARPGROUP.DEPBAR.LE' in i and '0x0' in i]
        assert wgwait0 and min(stsm)>max(wgwait0)
        assert max(hgmma)<max(wgwait0)<min(stsm)
        assert not re.search(r'\b(?:LDL|STL)\b',table)
        if eh:
            assert max(stsm[:4])<reduce[0]<min(stsm[4:])<reduce[1]
            # Check actual quartet publicationfences andWGbarriers beforeeachissue.
            for group,issue in [(stsm[:4],reduce[0]),(stsm[4:],reduce[1])]:
                between=[i for p,i in rows if max(group)<p<issue]
                assert any('FENCE.VIEW.ASYNC.S' in i for i in between)
                assert any('BAR.SYNC' in i for i in between)
        results.append(dict(flag=flag,EH=eh,static_STSM=len(stsm),static_reduce_issues=len(reduce),WGMMA_drain_before_STSM='PASS',both_TMA_reads_drain_before_exit='PASS',half0_issue_before_half1_STSM=eh))
    assert sorted(r['flag'] for r in results)==[2,3,65538,65539]
    return results

def main():
    result={'state':'CPU_STARTED_GPU_UNRUN'}
    try:
        delta=json.loads((HERE/'delta.json').read_text());src=(HERE/'submission.py').read_text()
        rs=(HERE.parent/'rs32/submission.py').read_text();pw=(HERE.parent/'pw36/submission.py').read_text()
        assert apply(rs,delta)==src and apply(src,delta,True)==rs
        assert value(apply(pw,delta),'RQCU')==value(src,'RQCU')
        # Current math/producer prefix and all fullK/D-retirement fences stayexact.
        before=value(rs,'RQCU');after=value(src,'RQCU')
        prefix='    const unsigned long long ce0 = INS ? clock64() : 0;'
        a=before[:before.index(prefix)];b=after[:after.index(prefix)]
        guard=delta['edits'][1];b=b.replace(guard['after'],guard['before']).replace('    if constexpr (FL & 65536) {\n','')
        assert a==b
        assert 'if (!f) asm volatile("trap;");' in after
        assert 'constexpr bool EH = (FL & 65536) != 0, EP = false;' in after
        assert 'rq_facc<ND>(D);' in b
        assert 'wgmma.wait_group.sync.aligned 0;' in b
        # Negative anchor mutations reject patching; no best-effort textual merge.
        for e in delta['edits']:
            try:apply(rs.replace(e['before'],e['before']+'\n'+e['before'],1),delta)
            except ValueError:pass
            else:raise AssertionError('duplicate anchor admitted '+e['id'])
        cu=device(src);(HERE/'device_source.cu').write_text(cu)
        nvcc='/usr/local/cuda-12.8/bin/nvcc'
        cmd=[nvcc,*FLAGS,'-Xptxas=-v','-cubin',str(HERE/'device_source.cu'),'-o',str(HERE/'device.sm90a.cubin')]
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=90);(HERE/'resource.log').write_text(p.stdout+p.stderr)
        assert p.returncode==0,p.stderr
        assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',p.stderr)
        assert not re.search(r'C75|serializ',p.stderr,re.I)
        regs=[int(x) for x in re.findall(r'Used (\d+) registers',p.stderr)]
        assert len(regs)==4 and all(((r+7)//8)*8*384<=65536 for r in regs),regs
        sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/'device.sm90a.cubin')],text=True)
        (HERE/'sass.txt').write_text(sass)
        # Preserve an exact-current control cubin and compare both existingflags.
        control=device(rs,(2,3));(HERE/'control_source.cu').write_text(control)
        cmd=[nvcc,*FLAGS,'-Xptxas=-v','-cubin',str(HERE/'control_source.cu'),'-o',str(HERE/'control.sm90a.cubin')]
        cp=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
        (HERE/'control-resource.log').write_text(cp.stdout+cp.stderr);assert cp.returncode==0,cp.stderr
        cs=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/'control.sm90a.cubin')],text=True)
        (HERE/'control-sass.txt').write_text(cs)
        def streams(text):
            out={}
            for t in tables(text):
                flag=int(re.search(r'Li(\d+)EEv3RqP',t.splitlines()[0])[1])
                out[flag]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)]
            return out
        original_streams,candidate_streams=streams(cs),streams(sass)
        codegen={fl:dict(equal=original_streams[fl]==candidate_streams[fl],original_instructions=len(original_streams[fl]),candidate_instructions=len(candidate_streams[fl])) for fl in (2,3)}
        assert all(r['equal'] for r in codegen.values()), 'Existing non-EH codegen drift'
        result.update(state='CPU_EH_BACKPORT_PASS_GPU_UNRUN_REVIEW_REQUIRED',registers=regs,bitmap=bitmap(),physical=physical(sass),
                      existing_no_EH_actual_SASS_comparison=codegen,candidate_sha256=sha(src),RQCU_sha256=sha(after),current_rs32_PW36_RQCU_identical=True,
                      producer_math_fullK_and_generic_D_retirement='EXACT_UNCHANGED_PREFIX',
                      flags=dict(EH=65536,EP=131072,EP_ported=False,HALF=16,EARLY=32),
                      source_domains=['M128/ST5/GS1/PRB0/WM0/INS0/FL65538','M128/ST5/GS1/PRB0/WM0/INS0/FL65539'],
                      runtime='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE; both EH variants must selfvalidate and win actual preparation bake; no gain admission')
    except Exception as error:
        result.update(state='CPU_EH_BACKPORT_FAILURE_RETAINED',error=repr(error));raise
    finally:
        (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ('state','error','registers','candidate_sha256')},indent=2))
if __name__=='__main__':main()

"""CPU compiler/proof only; imports no extension and creates no CUDA context."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import prototype as P
import proofs
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=dict(state='CPU_PREFLIGHT_STARTED_GPU_UNRUN')
    try:
        parent=HERE.parent/'reverse_tma_x'
        assert sha(parent/'extension_source.cu')=='ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377'
        source=P.source();device=P.device_source()
        assert 'rx_pack' not in source and 'px_ldg' not in source
        assert 'int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);' in source
        assert 'cuuint32_t box[2] = {64, (cuuint32_t)box_outer};' in source
        assert 'p.XP =' not in source and 'xp.data_ptr' not in source
        assert source.count('k<<<dim3(')==1
        (HERE/'extension_source.cu').write_text(source)
        (HERE/'device_source.cu').write_text(device)
        nvcc='/usr/local/cuda-12.8/bin/nvcc'
        cmd=[nvcc,*P.S.RQFLAGS,'-Xptxas=-v','-cubin',str(HERE/'device_source.cu'),'-o',str(HERE/'device.sm90a.cubin')]
        c=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
        (HERE/'resource.log').write_text(c.stdout+c.stderr)
        assert c.returncode==0,c.stderr
        assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',c.stderr)
        assert not re.search(r'warning.*C75|wgmma.*serial',c.stderr,re.I)
        regs=[int(r) for r in re.findall(r'Used (\d+) registers',c.stderr)]
        assert regs==[122]
        assert ((regs[0]+7)//8)*8*512==65536
        sass=subprocess.check_output([str(Path(nvcc).with_name('cuobjdump')),'--dump-sass',str(HERE/'device.sm90a.cubin')],text=True,timeout=20)
        (HERE/'sass.txt').write_text(sass)
        all_proofs=proofs.all_proofs()
        (HERE/'proofs.json').write_text(json.dumps(all_proofs,indent=2)+'\n')
        px=HERE.parent/'operand_reverse_packed_x'/'preflight.py'
        spec=importlib.util.spec_from_file_location('retained_px_math_proofs',px)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        result.update(weight_bits=old.wproof(),output_map=old.outputproof(),math_A4_lifetime=old.lifetime())
        result.update(state='CPU_FEASIBILITY_PASS_REVIEW_REQUIRED_GPU_UNRUN',registers=regs,
                      rounded_register_pool=65536,threads=512,dynamic_shared_bytes=170016,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),proofs='proofs.json',
                      host='fresh current x.data_ptr tensor map per call; single main launch; xp argument ignored')
    except Exception as e:
        result.update(state='CPU_FEASIBILITY_FAILURE_RETAINED',error=repr(e));raise
    finally:
        (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k in ('state','error','registers','dynamic_shared_bytes','source_sha256')},indent=2))
    files=['prototype.py','kernel_body.cuh','extension_source.cu','device_source.cu','device.sm90a.cubin','sass.txt','resource.log','proofs.py','proofs.json','liveness.py','preflight.py','preflight.json','layout_proof.py','protocol_proof.py','composition_proof.py','README.md','actual_binary_gate.py','actual-binary-selfcheck.json','checker-history/initial-preflight.json','checker-history/initial-proofs.py','checker-history/initial-composition-proof.py']
    manifest=dict(version='reverse_tma_x4_stmatrix_v1',state=result['state'],GPU='UNRUN',
                  predecessor_source_sha256='ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377',
                  parent_x4_manifest_sha256='d47ed1663eac87f5bcc05b5f17df8fa62e7c9e91780a13d02ce9b95ac0f05c8a',
                  parent_stmatrix_manifest_sha256='2c132b82a854e476f433335462b2122f21bc3cb490dec6f2b691c01cd9775045',
                  cpu_compile_flags=P.S.RQFLAGS,files={n:sha(HERE/n) for n in files})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()

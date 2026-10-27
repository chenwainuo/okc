"""Source/models/SM90a device gate only. Never loads an extension or uses CUDA."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import prototype as P
import liveness
from schedule_proof import proof

HERE=P.HERE
PARENT=HERE.parent/"operand_reverse_dual_producer"
PARENT_SOURCE="97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b"


def sourceproof():
    assert hashlib.sha256((PARENT/"manifest.json").read_bytes()).hexdigest()=="281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241"
    assert hashlib.sha256((HERE.parent/"operand_reverse_packed_x/preflight.py").read_bytes()).hexdigest()=="a0e154236f66fc9d8194261c17310d20e0aeb1d1ef453cfe2f9306d0569cd67b"
    body=(HERE/"kernel_body.cuh").read_text();old=(PARENT/"kernel_body.cuh").read_text()
    lo="  float d[32]={};";hi='  asm volatile("bar.sync 1,256;\\n":::"memory");'
    restored=body[:body.index(lo)]+old[old.index(lo):old.index(hi)]+body[body.index(hi):]
    restored=restored.replace("rq_px_g8(","rq_px_dp(")
    restored=restored.replace("uint32_t q0[32],q1[32];","uint32_t q0[16],q1[16],q2[16],q3[16];")
    restored=restored.replace("int s,uint32_t* q","int s,uint32_t (&q)[16]")
    restored=restored.replace("prefetch(0,q0);prefetch(1,q0+16);prefetch(2,q1);prefetch(3,q1+16);",
                            "prefetch(0,q0);prefetch(1,q1);prefetch(2,q2);prefetch(3,q3);")
    assert restored==old,"Unexpected producer/pack/output/metadata change"
    original=P.source().replace(body,old).replace("rq_px_g8<","rq_px_dp<")
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SOURCE
    assert "KC=64,WST=6,BST=8,WB=6144,BB=8192" in body
    assert "__launch_bounds__(512,1)" in body and "+16*BST;" in body
    assert "k,512,C::SMEM" in P.source() and "dim3(512)" in P.source()
    return {"restores_exact_parent":PARENT_SOURCE,"W6_B8_KC64_source_identical":True,
            "pack_producers_barriers_weight_and_output_source_identical":True,
            "same_256_K16_MMA_order":True,"SMEM_bytes":103600,"threads":512}


def physical_negative_controls(sass):
    bad_path=HERE.parent/"operand_reverse_packed_x/failed-q4a2/sass.txt"
    table=next(t for t in re.split(r"\n\s*Function : ",bad_path.read_text())[1:] if "rq_px" in t.splitlines()[0])
    def reject(text,strict,error):
        try:liveness.check_one(text,strict=strict)
        except AssertionError as e:
            assert error in str(e),(error,str(e))
            return str(e)
        raise AssertionError("Bad binary unexpectedly passed "+error)
    controls={"old_real_premature_A_overwrite":reject(table,False,"GENERIC_ASYNC_OPERAND_ACCESS_BEFORE_RETIRE")}
    table=next(t for t in re.split(r"\n\s*Function : ",sass)[1:] if "rq_px_g8" in t.splitlines()[0])
    controls["removed_initial_fence"]=reject(table.replace("WARPGROUP.ARRIVE","NOP",1),False,"UNFENCED_GENERIC_A_D_ACCESS")
    controls["predicated_aligned_fence"]=reject(table.replace("WARPGROUP.ARRIVE","@P0 WARPGROUP.ARRIVE",1),True,"Predicated aligned WGMMA/fence/wait forbidden")
    rows=liveness.parse(table);first_wait=min(pc for pc,ins in rows if "WARPGROUP.DEPBAR.LE" in ins)
    _,raw=next((pc,ins) for pc,ins in rows if pc>first_wait and liveness.plain(ins).startswith("LDG.") and ".128" in ins)
    first_a=min(liveness.check_one(table)["A_register_sets"][0])
    bad=re.sub(r"(LDG\.[^ ]+ )R\d+",lambda m:m[1]+"R"+str(first_a),raw,count=1)
    controls["new_A_bank_overwrite_before_retire"]=reject(table.replace(raw,bad,1),True,"GENERIC_ASYNC_OPERAND_ACCESS_BEFORE_RETIRE")
    hs=[pc for pc,ins in rows if "HGMMA." in ins]
    _,raw=next((pc,ins) for pc,ins in rows if hs[0]<pc<hs[1] and not liveness.plain(ins).startswith("HGMMA."))
    d=min(liveness.check_one(table)["D_registers"])
    controls["generic_pending_D_write"]=reject(table.replace(raw,f"IMAD.MOV.U32 R{d}, RZ, RZ, RZ",1),True,"GENERIC_ASYNC_OPERAND_ACCESS_BEFORE_RETIRE")
    return {"status":"PASS_NEGATIVE_CONTROLS","rejections":controls}


def main():
    result={"state":"CPU_PREFLIGHT_STARTED_GPU_UNRUN","source":sourceproof(),"schedule":proof()}
    try:
        spec=importlib.util.spec_from_file_location("retained_group8_models",HERE.parent/"operand_reverse_packed_x/preflight.py")
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        result["W_bit_map"]=old.wproof();result["output_map"]=old.outputproof();result["X_pack_map"]=old.xproof()
        spec=importlib.util.spec_from_file_location("retained_group8_parity",PARENT/"protocol_proof.py")
        par=importlib.util.module_from_spec(spec);spec.loader.exec_module(par)
        result["unchanged_parity_producer_protocol"]=par.proof()
        source=P.source();(HERE/"extension_source.cu").write_text(source);(HERE/"device_source.cu").write_text(P.device_source())
        cmd=["/usr/local/cuda-12.8/bin/nvcc",*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),"-o",str(HERE/"device.sm90a.cubin")]
        run=subprocess.run(cmd,text=True,capture_output=True,timeout=90);log=run.stdout+run.stderr
        (HERE/"resource.log").write_text(log);assert run.returncode==0,log
        assert not re.search(r"\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)",log),log
        assert not re.search(r"C75\d\d|wgmma.*serial",log,re.I),log
        regs=[int(n) for n in re.findall(r"Used (\d+) registers",log)]
        assert len(regs)==2 and 0<regs[0]<=128 and 0<regs[1]<=64,regs
        pool=((regs[0]+7)//8)*8*512;assert pool<=65536
        sass=subprocess.check_output(["/usr/local/cuda-12.8/bin/cuobjdump","--dump-sass",str(HERE/"device.sm90a.cubin")],text=True,timeout=20)
        (HERE/"sass.txt").write_text(sass)
        assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b",sass)
        result["physical_A2x32_and_fence_coverage"]=liveness.check(sass)
        result["physical_negative_controls"]=physical_negative_controls(sass)
        result.update(state="CPU_DEVICE_PASS_GPU_UNRUN_REVIEW_REQUIRED",registers=regs,rounded512_register_pool=pool,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      device_source_sha256=hashlib.sha256(P.device_source().encode()).hexdigest(),gpu_calls=0)
    except Exception as e:
        result.update(state="FAILED_CPU_VERSION_RETAIN_DIAGNOSE",error=repr(e));raise
    finally:
        (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k in ("state","registers","rounded512_register_pool","source_sha256","error","gpu_calls")},indent=2))


if __name__=="__main__":main()

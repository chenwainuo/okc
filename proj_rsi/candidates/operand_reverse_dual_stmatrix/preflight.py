"""CPU-only composition proof and one SM90a compile; no extension/GPU launch."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import prototype as P
import layout_proof as L
import liveness
from protocol_proof import proof

HERE=P.HERE
PARENT=HERE.parent/"operand_reverse_dual_producer"
PARENT_SOURCE="97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b"


def store_span(body):
    start=body.index("#pragma unroll",body.index("rq_fix(a,s,"))
    stop=body.index('      asm volatile("fence.proxy.async',start)
    return start,stop


def sourceproof():
    old=(PARENT/"kernel_body.cuh").read_text()
    assert hashlib.sha256(old.encode()).hexdigest()=="05c009c93c22404273ec6bb761aee8cfbd0dafc6f787d921e76724352a2645b0"
    body=(HERE/"kernel_body.cuh").read_text()
    a,b=store_span(body);x,y=store_span(old)
    restored=(body[:a]+old[x:y]+body[b:]).replace("rq_px_dp_stmx(","rq_px_dp(")
    assert restored==old
    source=P.source()
    full=source.replace(body,old).replace("rq_px_dp_stmx<","rq_px_dp<")
    assert hashlib.sha256(full.encode()).hexdigest()==PARENT_SOURCE
    assert source.count("__launch_bounds__(512,1)")==1
    assert "k,512,C::SMEM" in source and "dim3(512)" in source
    assert body.count("stmatrix.sync.aligned.m8n8.x4.shared.b16")==1
    assert 'if(owner==0)asm volatile("bar.sync 3,128;' in body
    assert 'else asm volatile("bar.sync 4,128;' in body
    return {"status":"EXACT_DP_PARENT_WHOLE_SOURCE_EXCEPT_STORES_AND_NAME",
            "restored_parent_sha256":PARENT_SOURCE,
            "unchanged":"512threads, parity/CSR skip, W6/B8, proxy/bar3+4, Xpack/A4/cache, output"}


def ownerproof():
    coverage={}
    for s in range(64):
        owner=s%2
        lanes=[]
        for w in range(4):
            warp=8+4*owner+w
            assert warp&3==w and (warp-8)//4==owner
            lanes.extend(range(32*warp,32*warp+32))
        assert lanes==list(range(256+128*owner,384+128*owner))
        coverage[s]={"owner":owner,"barrier":3+owner,"threads":len(lanes)}
    return {"all64stages_exactly_once":True,"owner0_stages":32,"owner1_stages":32,
            "same4warp_store_map_for_each_owner":True,"stage_ownership":coverage}


def main():
    result={"state":"CPU_PREFLIGHT_STARTED_GPU_UNRUN"}
    try:
        result["source_identity"]=sourceproof()
        result["protocol"]=proof()
        result["store_owners"]=ownerproof()
        result["weight_bits_and_crosslane_store_map"]=L.wproof()
        result["Xpack_inverse_map"]=L.xproof()
        result["output_map"]=L.outputproof()
        result["math_A4_lifetime"]=L.lifetime()
        source=P.source();device=P.device_source()
        (HERE/"extension_source.cu").write_text(source)
        (HERE/"device_source.cu").write_text(device)
        nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        compiled=subprocess.run([nvcc,*P.S.RQFLAGS,"-Xptxas=-v","-cubin",str(HERE/"device_source.cu"),
                                 "-o",str(HERE/"device.sm90a.cubin")],text=True,capture_output=True,timeout=90)
        log=compiled.stdout+compiled.stderr;(HERE/"resource.log").write_text(log)
        assert compiled.returncode==0,log
        assert not re.search(r"\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)",log)
        assert not re.search(r"warning.*C75|wgmma.*serial",log,re.I)
        regs=[int(n) for n in re.findall(r"Used (\d+) registers",log)]
        assert len(regs)==2 and max(regs)<=128,regs
        result["registers"]=regs
        result["max_rounded_512_thread_pool"]=((max(regs)+7)//8)*8*512
        assert result["max_rounded_512_thread_pool"]<=65536
        sass=subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")),"--dump-sass",str(HERE/"device.sm90a.cubin")],text=True,timeout=20)
        (HERE/"sass.txt").write_text(sass)
        assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b",sass)
        result["physical_A4_retirement"]=liveness.check(sass)
        result["store_form"]=liveness.store_form(sass,"rq_px_dp_stmx",True,dual=True)
        # Both controls must also satisfy the same actual-JIT store/proxy gates.
        result["DP_scalar_store_control"]=liveness.store_form((PARENT/"sass.txt").read_text(),"rq_px_dp",False,dual=True)
        bad=(HERE.parent/"operand_reverse_packed_x/failed-q4a2/sass.txt").read_text()
        try:liveness.check(bad)
        except AssertionError as error:
            assert "ASYNC_A_WRITE_BEFORE_RETIRE" in str(error),repr(error)
            result["known_bad_copy_negative_control"]="REJECTED_ASYNC_A_WRITE_BEFORE_RETIRE"
        else:raise AssertionError("known physical alias hazard was not rejected")
        result["source_sha256"]=hashlib.sha256(source.encode()).hexdigest()
        result["device_sha256"]=hashlib.sha256(device.encode()).hexdigest()
        result["state"]="CPU_DEVICE_FEASIBILITY_PASS_GPU_UNRUN_REVIEW_REQUIRED"
    except Exception as error:
        result.update(state="CPU_FEASIBILITY_FAILED_RETAIN_AND_DIAGNOSE",error=repr(error));raise
    finally:
        (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k in ("state","error","registers","max_rounded_512_thread_pool","source_sha256")},indent=2))


if __name__=="__main__":main()

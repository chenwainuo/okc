"""Root CPU/device feasibility; never loads an extension or launches CUDA."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import prototype as P
import liveness
from protocol_proof import proof

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "operand_reverse_packed_x"


def restore_body(body):
    body = body.replace("rq_px_dp(", "rq_px(").replace("__launch_bounds__(512,1)", "__launch_bounds__(384,1)")
    body = body.replace("    const int owner=(warp-8)/4;\n", "", 1)
    body = body.replace("for(int s=owner;s<KC;s+=2){", "for(int s=0;s<KC;++s){", 1)
    new = '''rq_unit(x0,x1,x2,p.Tb,a);
      // Each independent warp cursor skips the other producer's earlier stages.
      while(nxs<(uint32_t)s){
        ++ei;if((ei&31)==0)ec=__ldg(el+ei+lane);
        nx=__shfl_sync(0xffffffffu,ec,ei&31);nxs=nx>>17;
      }
      rq_fix(a,s,nx,nxs,ei,ec,el,lane);'''
    assert new in body
    body = body.replace(new, "rq_unit(x0,x1,x2,p.Tb,a);rq_fix(a,s,nx,nxs,ei,ec,el,lane);", 1)
    new = '''asm volatile("fence.proxy.async.shared::cta;\\n":::"memory");
      if(owner==0)asm volatile("bar.sync 3,128;\\n":::"memory");
      else asm volatile("bar.sync 4,128;\\n":::"memory");'''
    assert new in body
    body = body.replace(new, 'asm volatile("fence.proxy.async.shared::cta;\\nbar.sync 3,128;\\n":::"memory");', 1)
    return body.replace("      if(tid==256+128*owner){", "      if(tid==256){", 1)


def main():
    result = {"state": "CPU_PREFLIGHT_STARTED_GPU_UNRUN", "protocol": proof()}
    try:
        body = (HERE / "kernel_body.cuh").read_text()
        original = (PARENT / "kernel_body.cuh").read_text()
        assert restore_body(body) == original
        source = P.source()
        restored = source.replace(body, original).replace("rq_px_dp<", "rq_px<")
        restored = restored.replace("k,512,C::SMEM", "k,384,C::SMEM").replace("dim3(512)", "dim3(384)")
        restored=restored.replace("    if(info==6)return a.maxThreadsPerBlock;\n", "", 1)
        assert hashlib.sha256(restored.encode()).hexdigest() == "2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a"
        assert "k,512,C::SMEM" in source and "dim3(512)" in source
        assert source.count("__launch_bounds__(512,1)") == 1
        result["source_restore_to_exact_parent"] = "PASS"
        # These retained mathematical models apply to source-proven unchanged
        # Wdecode/output/Xpack/math lifetimes; no GPU or extension API is called.
        spec = importlib.util.spec_from_file_location("retained_px_cpu_proofs", PARENT / "preflight.py")
        previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
        result["weight_bits"] = previous.wproof()
        result["output_map"] = previous.outputproof()
        result["math_A4_lifetime"] = previous.lifetime()
        result["Xpack_inverse_map"] = previous.xproof()
        (HERE / "extension_source.cu").write_text(source)
        (HERE / "device_source.cu").write_text(P.device_source())
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        command = [nvcc, *P.S.RQFLAGS, "-Xptxas=-v", "-cubin", str(HERE / "device_source.cu"),
                   "-o", str(HERE / "device.sm90a.cubin")]
        compiled = subprocess.run(command, text=True, capture_output=True, timeout=90)
        (HERE / "resource.log").write_text(compiled.stdout + compiled.stderr)
        assert compiled.returncode == 0, compiled.stderr
        assert not re.search(r"\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)", compiled.stderr)
        assert not re.search(r"warning.*C75|wgmma.*serial", compiled.stderr, re.I)
        regs = [int(n) for n in re.findall(r"Used (\d+) registers", compiled.stderr)]
        assert len(regs) == 2 and max(regs) <= 128, regs
        result["registers"] = regs
        result["max_rounded_512_thread_pool"] = ((max(regs) + 7) // 8) * 8 * 512
        assert result["max_rounded_512_thread_pool"] <= 65536
        sass = subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")), "--dump-sass",
                                        str(HERE / "device.sm90a.cubin")], text=True, timeout=20)
        (HERE / "sass.txt").write_text(sass)
        assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b", sass)
        result["physical_A4_retirement"] = liveness.check(sass)
        result["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        result["state"] = "CPU_DEVICE_FEASIBILITY_PASS_GPU_UNRUN_REVIEW_REQUIRED"
    except Exception as error:
        result.update(state="CPU_FEASIBILITY_FAILED_RETAIN_AND_DIAGNOSE", error=repr(error))
        raise
    finally:
        (HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k in ("state", "error", "registers", "max_rounded_512_thread_pool", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()

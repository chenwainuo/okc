"""compile single bh variants with patches and report ptxas C7514 (run inside the container from /okc)"""
import subprocess, sys, sysconfig
sys.path.insert(0, "proj_rsi/candidates/pw24h")
sys.argv = ["x", "--prepare-only"]
import submission as S
from torch.utils.cpp_extension import include_paths

V = ("rsk2", 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2)
EPI0 = """    if (PRB == 0) {
      const int g = lane >> 2, t = lane & 3;"""
CVT_OLD = """          const __nv_bfloat16 v0 = __float2bfloat16(D[4 * i]), v1 = __float2bfloat16(D[4 * i + 1]), v2 = __float2bfloat16(D[4 * i + 2]), v3 = __float2bfloat16(D[4 * i + 3]);"""
CVT_NEW = """          uint32_t p0, p1;
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p0) : "f"(D[4 * i]), "f"(D[4 * i + 2]));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p1) : "f"(D[4 * i + 1]), "f"(D[4 * i + 3]));
          const unsigned short v0 = p0 & 0xffff, v1 = p1 & 0xffff, v2 = p0 >> 16, v3 = p1 >> 16;"""
PATCHES = {
    "base": [],
    "no_named_bar": [('asm volatile("bar.sync 1, %0;\\n" :: "n"(NCT) : "memory");', "__syncwarp();")],
    "cvt_asm": [(CVT_OLD, CVT_NEW)],
    "f32_smem": [(EPI0, """    if (PRB == 0) {
#pragma unroll
      for (int i = 0; i < NR; i++) asm volatile("st.shared.f32 [%0], %1;" :: "r"(xring + (i * 128 + tid) * 4), "f"(D[i]) : "memory");
    }
    if (PRB == 0 && false) {
      const int g = lane >> 2, t = lane & 3;""")],
}
for name, pats in PATCHES.items():
    old = S.BHV
    S.BHV = [V]
    src = S._src()
    S.BHV = old
    for a, b in pats:
        assert src.count(a) >= 1, (name, a[:60])
        src = src.replace(a, b)
    f = "/tmp/rsb_%s.cu" % name
    open(f, "w").write(src)
    inc = sum((["-I", p] for p in include_paths() + [sysconfig.get_paths()["include"]]), [])
    r = subprocess.run(["nvcc", "-cubin", "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-std=c++17", "-o", "/tmp/rsb.cubin", f] + inc,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    regs = [l for l in out.splitlines() if "Used" in l and "registers" in l]
    print("%-18s C7514=%s %s rc=%d" % (name, "C7514" in out, regs[-1].strip() if regs else "", r.returncode), flush=True)
    if r.returncode:
        print(out[-1500:])

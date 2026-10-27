"""Generate exactly two incumbent and two C11 native kernels. CPU/offline only."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[3]
SOURCE = ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
PIN = '501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
cu = next(ast.literal_eval(n.value) for n in ast.parse(SOURCE.read_text()).body
          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GVCU' for t in n.targets))
prefix = 'template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__'
common = cu[:cu.index(prefix)]
common = common.replace('#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>',
                        '#include <cuda_runtime.h>\n#include <type_traits>')
helper_start = cu.index('__device__ __forceinline__ void gv_fix_pred(')
start = cu.index(prefix, helper_start)
end = cu.index('\n__global__ void p1zero', start)
control = cu[start:end].strip()
helper = cu[helper_start:start]
candidate = control.replace(' gv_predxor(G p) {', ' gv_frontier(G p, Frontier fp) {', 1)
changes = []
def replace(old, new, count=1):
    global candidate
    assert candidate.count(old) == count, (old[:100], candidate.count(old), count)
    candidate = candidate.replace(old, new)
    changes.append({'old': old, 'new': new, 'occurrences': count})

# Existing G.W points to two original 512B low planes. Header is separate.
replace('  const size_t wo = ', '  const size_t wo = ', 1)  # recorded anchor only
ld0 = candidate.index('  auto ld = [&](int s, int u) {')
ld1 = candidate.index('\n    if (!XS && LX)', ld0)
old_load = candidate[ld0:ld1]
new_load = r'''  constexpr int FCAP = MT == 1 ? 32 : 48, FSLOT = 32 + FCAP * 2;
  const uint32_t fs = (uint32_t)__cvta_generic_to_shared(smem) + L::SMEM;
  static_assert(RT == 1 && NA == 1 && !L::BK && S == 1 && (ST == 1 || ST == 2), "two-cell frontier scope");
  auto ld = [&](int s, int u) {
    const unsigned unit = ((bx * WR + wr) * p.KC + c0 + u);
    wb[s][0][0][0] = gnc(p.W + ((size_t)unit * 2) * 32 + lane);
    wb[s][0][0][1] = gnc(p.W + ((size_t)unit * 2 + 1) * 32 + lane);
    const uint32_t h0 = frontier_ldg(fp.header + ((size_t)unit * 3) * 32 + lane);
    const uint32_t h1 = frontier_ldg(fp.header + ((size_t)unit * 3 + 1) * 32 + lane);
    const uint32_t h2 = frontier_ldg(fp.header + ((size_t)unit * 3 + 2) * 32 + lane);
    wb[s][0][0][2] = make_uint4(h0, __funnelshift_r(h0,h1,24), __funnelshift_r(h1,h2,16), h2 >> 8);
    const uint32_t begin = __ldg(p.esc + (size_t)unit * 8);
    const uint32_t finish = unit + 1 < fp.units ? __ldg(p.esc + (size_t)(unit + 1) * 8) : fp.words;
    const uint32_t dest = fs + (w * ST + s) * FSLOT;
    if (lane < 2) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dest + lane * 16), "l"(p.esc + (size_t)unit * 8 + lane * 4));
    }
    if (lane >= 2 && lane < 2 + FCAP / 8 && (lane - 2) * 16 < (finish - begin) * 2) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dest + 32 + (lane - 2) * 16), "l"(fp.records + (size_t)begin * 2 + (lane - 2) * 16));
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");'''
replace(old_load, new_load)
replace('  auto step = [&](const int u, const int s) {', r'''  auto step = [&](const int u, const int s) {
    // Exactly ST initial groups and one group per refill, including tails.
    if constexpr (ST == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if (u + 1 < KU) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();''')
# The unchanged old escape stream must never be read or applied.
replace('if ((C12 || CE) && LDEC)', 'if constexpr (false)', 3)
replace('dec1(q[', 'frontier_dec(q[', 4)
anchor = '    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router'
insert = r'''    // Consume every repair before ld() can overwrite this slot.
    const uint32_t* desc = (const uint32_t*)(smem + L::SMEM + (w * ST + s) * FSLOT);
    const uint32_t first = desc[0];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const unsigned short* fixes = (const unsigned short*)(smem + L::SMEM + (w * ST + s) * FSLOT + 32 + 2*(desc[2*q] - first));
      frontier_fix(*reinterpret_cast<unsigned (*)[4]>(A[0][0] + 4*q), fixes, desc[2*q+1], lane);
    }
    __syncwarp();
'''
replace(anchor, insert+anchor)
scope = r'''  static_assert((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
                (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1), "exact selected two-cell scope");
'''
replace('  // pw14 probes only', scope+'  // pw14 probes only')
frontier_helpers = r'''
struct Frontier { const uint32_t* header; const uint8_t* records; uint32_t units, words; };
__device__ __forceinline__ uint32_t frontier_ldg(const uint32_t* p) {
  uint32_t x; asm("ld.global.nc.L1::no_allocate.u32 %0,[%1];" : "=r"(x) : "l"(p)); return x;
}
__device__ __forceinline__ unsigned frontier_spread(unsigned c) {
  const unsigned x=(c&0x3fu)|((c&0xfc0u)<<2);
  return (x&0x0707u)|((x&0x3838u)<<1);
}
__device__ __forceinline__ void frontier_dec(unsigned l0,unsigned l1,unsigned c,unsigned table,unsigned,unsigned* a) {
  const unsigned hi0=prmt(table,table|0x80808080u,frontier_spread(c));
  const unsigned hi1=prmt(table,table|0x80808080u,frontier_spread(c>>12));
  a[0]=prmt(l0,hi0,0x5140); a[1]=prmt(l0,hi0,0x7362);
  a[2]=prmt(l1,hi1,0x5140); a[3]=prmt(l1,hi1,0x7362);
}
'''
text = common + helper + '\n#include "../parallel_fix.cuh"\n' + frontier_helpers + control + '\n' + candidate
templates = [(1,1,1,2,16,8,1,1,0,3),(1,2,2,1,4,16,1,1,2,1)]
for tpl in templates:
    args = ','.join(map(str,tpl))
    text += '\ntemplate __global__ void gv_predxor<'+args+'>(G);'
    text += '\ntemplate __global__ void gv_frontier<'+args+'>(G,Frontier);'
(D/'native.cu').write_text(text+'\n')
(D/'delta.json').write_text(json.dumps(dict(parent_sha256=PIN, changes=changes, templates=templates,
    status='OFFLINE_NATIVE_PROTOTYPE_GPU_UNRUN', additional_shared_bytes={'pooler':3072,'unpooler':1024}),indent=2)+'\n')
print('Generated exact two-cell controls/candidates; launch wrapper/physical pack still required.')

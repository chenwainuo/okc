# okc: problem=muse_bf16_proj
"""pw24h (H100 research copy of pw24's bf16 "bg" GEMM; NOT a judge candidate): why does the wgmma step cost +8 us?

One templated kernel bh<ST, BM, BN, CL, PP, WN, CG, RS, FN, PRB>: out[tok][n] = x[tok] . W[n], W bf16 as given.
CTA tile = BM W rows (64 | 128) x BN tokens (64 | 128) over the full K (no split-K); both operands written by 2D TMA
(128B swizzle) into two rings (W ring, x ring) fed by two producer lanes (warps 4*NC and 4*NC+1).
  CL  1: cluster (2,1,1) over token tiles, W multicast halves (pw24 bg12c); BM = BN = 64 only.
  PP  1: two consumer warpgroups ping-pong on alternate stages, own accumulators, summed in smem at the end.
      0: cooperative, NC = BM / 64 consumer warpgroups, warpgroup c owns W rows 64c .. 64c+63 of every stage.
  WN  wgmma units in flight: after issuing unit u, wait_group(WN) and release unit u - WN.
  CG  0: one commit per stage; 1: one commit per k16 wgmma (4 groups / stage); 2: one commit per 2 stages.
  RS  1: A (W) fragments by ldmatrix from the swizzled smem into registers, wgmma RS (WN = 1, CG = 0 only).
  FN  1: wgmma.fence only before the first unit (accumulators are never touched between units).
  PRB probes (no epilogue): 1 W only | 2 W + x (waits / releases only) | 3 + wgmma | 4 wgmma only (no TMA, no
      barriers, stale slot 0) | 5 producers unthrottled (self-paced on their own full barriers) + independent wgmma on
      stale slot 0 (smem-port contention test) | 6 ring flow as 3 but wgmma on stale slot 0.
prepare() builds the extension (compile only under --prepare-only), runs the ladder and bake (BAKE / PROBE lines),
and kernel_fn serves the picked shapes; everything else (and every non-picked shape) is torch."""
import os
import sys
import time
import torch
from torch.utils.cpp_extension import load_inline

_T0 = time.time()


def _wg(n, rs):
    r = n // 2
    regs = ",".join("%%%d" % i for i in range(r))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(r))
    if not rs:
        return ('__device__ __forceinline__ void bh_wg(float (&D)[%d], uint64_t da, uint64_t db) {\n'
                '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"\n'
                '    "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n"\n'
                '    : %s : "l"(da), "l"(db), "r"(1));\n}\n') % (r, r + 2, n, regs, r, r + 1, outs)
    return ('__device__ __forceinline__ void bh_wgr(float (&D)[%d], const uint32_t (&a)[4], uint64_t db) {\n'
            '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"\n'
            '    "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, %%%d, p, 1, 1, 0;\\n}\\n"\n'
            '    : %s : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));\n}\n') % (
                r, r + 5, n, regs, r, r + 1, r + 2, r + 3, r + 4, outs)


# (name, ST, BM, BN, CL, PP, WN, CG, RS, FN, PRB). PRB = 0 entries are bake paths; the rest are ladder probes.
BHV = [  # (name, ST, BM, BN, CL, PP, WN, CG, RS, FN, PRB[, INS[, SY[, KS]]])
    ("b12", 12, 64, 64, 0, 0, 1, 0, 0, 0, 0),
    ("b12.L1w", 12, 64, 64, 0, 0, 1, 0, 0, 0, 1),
    ("b12.L1wx", 12, 64, 64, 0, 0, 1, 0, 0, 0, 2),
    ("b12.L3", 12, 64, 64, 0, 0, 1, 0, 0, 0, 3),
    ("k2s1.L1w", 7, 64, 64, 0, 0, 1, 0, 0, 1, 1, 0, 1, 2),
    ("k2s1.L1wx", 7, 64, 64, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2),
    ("k2s1.L3", 7, 64, 64, 0, 0, 1, 0, 0, 1, 3, 0, 1, 2),
    ("k2s1.L4mma", 7, 64, 64, 0, 0, 1, 0, 0, 1, 4, 0, 1, 2),
    ("k2s1", 7, 64, 64, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2),
    ("k2x", 7, 64, 64, 0, 0, 1, 0, 0, 1, 0, 0, 9, 2),
    ("k2x6", 6, 64, 64, 0, 0, 1, 0, 0, 1, 0, 0, 9, 2),
    ("rsk2.L3", 7, 64, 64, 0, 0, 1, 0, 1, 0, 3, 0, 1, 2),
    ("rsk2", 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2),
    ("rsk2x", 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 9, 2),
    ("n128s1", 9, 64, 128, 0, 0, 1, 0, 0, 1, 0, 0, 1),
    ("n128x", 9, 64, 128, 0, 0, 1, 0, 0, 1, 0, 0, 9),
    ("n128x8", 8, 64, 128, 0, 0, 1, 0, 0, 1, 0, 0, 9),
    ("n128rsx", 9, 64, 128, 0, 0, 1, 0, 1, 0, 0, 0, 9),
    ("n128x.L1wx", 9, 64, 128, 0, 0, 1, 0, 0, 1, 2, 0, 9),
    ("n128x.L3", 9, 64, 128, 0, 0, 1, 0, 0, 1, 3, 0, 9),
]
SLACK = True  # slack.py (generated from this file by tools/mkslack.py) sets True: P4 decode-slack curve only
if SLACK:
    BHV = [("rsk2x", 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 9, 2)]
    for f, fn in enumerate(("ldsonly", "dec1", "orfp", "orimadfp", "oraligned", "prmtonly")):
        BHV += [("rsdec-%s" % fn, 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 9, 2, 50000 + f)]
    for r in (2, 3, 4):
        BHV += [("rsdec-dec1x%d" % r, 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 9, 2, 50009 + r)]
    for t, tn in ((1, "lop3"), (2, "hfma2"), (3, "ffma")):  # dec1 + R rounds x 16 ops / thread / k64
        for r in (2, 4, 8, 12):
            BHV += [("rsdec1+%s%d" % (tn, 16 * r), 7, 64, 64, 0, 0, 1, 0, 1, 0, 0, 0, 9, 2, 50000 + 20 * t + r)]
BHI = {v[0]: i for i, v in enumerate(BHV)}

BHCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
struct alignas(64) BgTmap { unsigned long long v[16]; };
__device__ __forceinline__ void bh_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nBH_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra BH_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bh_spin(uint32_t b, uint32_t ph) {  // non-suspending poll
  asm volatile("{\n.reg .pred p;\nBH_S_%=:\nmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra BH_S_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bh_tile(uint32_t dst, const BgTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void bh_tile_mc(uint32_t dst, const BgTmap* tm, int k0, int r0, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ uint64_t bh_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void bh_csync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void bh_wgwait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void bh_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void bh_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
@@WG@@
template <int NR> __device__ __forceinline__ void bh_facc(float (&D)[NR]) {
#pragma unroll
  for (int i = 0; i < NR; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
template <int NA> __device__ __forceinline__ void bh_fa(uint32_t (&A)[NA]) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}

// P4c synthetic op of type OPT on chain value x (step parity i selects the runtime operand pair, so no two consecutive
// ops can be merged or re-associated by ptxas): 0 IMAD | 1 LOP3 (majority) | 2 PRMT (runtime selector) | 3 SHF (funnel)
// | 4 FFMA | 5 HFMA2.BF16 | 6 FMUL | 7 HMUL2.BF16
template <int OPT> __device__ __forceinline__ void bh_op(uint32_t& x, int i, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  const uint32_t p = (i & 1) ? c : a, q = (i & 1) ? d : b;
  if constexpr (OPT == 0) asm volatile("mad.lo.u32 %0, %0, %1, %2;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 1) asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 2) asm volatile("prmt.b32 %0, %0, %1, %2;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 3) asm volatile("shf.l.wrap.b32 %0, %1, %0, %2;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 4) asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 5) asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;\n" : "+r"(x) : "r"(p), "r"(q));
  else if constexpr (OPT == 6) asm volatile("mul.rn.f32 %0, %0, %1;\n" : "+r"(x) : "r"(p));
  else asm volatile("mul.rn.bf16x2 %0, %0, %1;\n" : "+r"(x) : "r"(p));
}
__device__ __forceinline__ uint32_t bh_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t bh_hmul2(uint32_t a, uint32_t b) { uint32_t d; asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b)); return d; }
// P4c decode formulations (8 weights from L0, L1 = low bytes, C = sign + 3-bit exponent offset nibbles) -> a[0..3] bf16x2
template <int F> __device__ __forceinline__ void bh_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t K8, uint32_t Sc, uint32_t* a) {
  if constexpr (F == 0) {  // no decode: the loaded words go to the wgmma as they are (data path floor)
    a[0] = L0; a[1] = L1; a[2] = C; a[3] = L0 ^ C;
  } else if constexpr (F == 1) {  // dec1 (pw10/r6): LOP3 + IADD, SHF + LOP3 + IADD, 4 PRMT = 9 INT / 8 weights
    const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
    a[0] = bh_prmt(L0, hl, 0x5140); a[1] = bh_prmt(L0, hl, 0x7362); a[2] = bh_prmt(L1, hh, 0x5140); a[3] = bh_prmt(L1, hh, 0x7362);
  } else if constexpr (F == 2) {  // OR-rebase + FP scale: (C & m) | K8 (one LOP3: fixed exponent bias 8), SHF + LOP3, 4 PRMT
    // = 7 INT; then 4 HMUL2.BF16 by 2^(2B-16) restore the window base on the FP pipe (exact: power-of-two scaling)
    const uint32_t hl = (C & 0x87878787u) | K8, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) | K8;
    a[0] = bh_hmul2(bh_prmt(L0, hl, 0x5140), Sc); a[1] = bh_hmul2(bh_prmt(L0, hl, 0x7362), Sc);
    a[2] = bh_hmul2(bh_prmt(L1, hh, 0x5140), Sc); a[3] = bh_hmul2(bh_prmt(L1, hh, 0x7362), Sc);
  } else if constexpr (F == 3) {  // as 2 with the nibble shift by IMAD (C * 16, values 4-7 stored one nibble lower): 6 INT + 1 IMAD + 4 HMUL2
    const uint32_t hl = (C & 0x87878787u) | K8, hh = ((C * 16u) & 0x87878787u) | K8;
    a[0] = bh_hmul2(bh_prmt(L0, hl, 0x5140), Sc); a[1] = bh_hmul2(bh_prmt(L0, hl, 0x7362), Sc);
    a[2] = bh_hmul2(bh_prmt(L1, hh, 0x5140), Sc); a[3] = bh_hmul2(bh_prmt(L1, hh, 0x7362), Sc);
  } else if constexpr (F == 4) {  // aligned window (B % 8 == 0, codec change): (C & m) | Tb, SHF + LOP3, 4 PRMT = 7 INT, no FP
    const uint32_t hl = (C & 0x87878787u) | Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) | Tb;
    a[0] = bh_prmt(L0, hl, 0x5140); a[1] = bh_prmt(L0, hl, 0x7362); a[2] = bh_prmt(L1, hh, 0x5140); a[3] = bh_prmt(L1, hh, 0x7362);
  } else if constexpr (F >= 20) {  // dec1, then R = F % 20 rounds of 4 independent ops of type T = LOP3 / HFMA2 / FFMA
    // (F / 20 = 1 / 2 / 3) on the four outputs: high-ILP pipe test (16 independent ops per thread per k64 per round)
    constexpr int T = F / 20, NR = F % 20;
    bh_dec<1>(L0, L1, C, Tb, K8, Sc, a);
#pragma unroll
    for (int r = 0; r < NR; r++) {
#pragma unroll
      for (int i = 0; i < 4; i++) bh_op<T == 1 ? 1 : T == 2 ? 5 : 4>(a[i], r, Sc, K8, Tb, Sc ^ K8);
    }
  } else if constexpr (F >= 10) {  // dec1 applied F - 9 times in sequence (each round decodes the previous round's output):
    // real decode ILP at 2x / 4x / 8x the instruction count, to find the slack of a high-ILP decode
    uint32_t l0 = L0, l1 = L1, c = C;
#pragma unroll
    for (int r = 0; r < F - 9; r++) {
      bh_dec<1>(l0, l1, c, Tb, K8, Sc, a);
      l0 = a[0] ^ a[1]; l1 = a[2] ^ a[3]; c = a[0] ^ a[3];
    }
  } else {  // merge floor: 4 PRMT per 8 weights (0.5 INT / weight), high bytes taken as stored
    a[0] = bh_prmt(L0, C, 0x5140); a[1] = bh_prmt(L0, C, 0x7362); a[2] = bh_prmt(L1, C, 0x5140); a[3] = bh_prmt(L1, C, 0x7362);
  }
}

template <int ST, int BM, int BN, int CL, int PP, int WN, int CG, int RS, int FN, int PRB, int INS, int SY, int KS, int ALU>
__global__ void __launch_bounds__((PP ? 2 : BM / 64) * 128 + 64, 1)
bh(const __grid_constant__ BgTmap tw, const __grid_constant__ BgTmap tx, __nv_bfloat16* out, int m, int N, int K, unsigned long long* prof) {
  // INS: clock64 phase counters per CTA: [0] consumer total, [1] wait_in, [2] fence+issue+commit, [3] wait_group, [4] release,
  // [5] W producer empty waits, [6] x producer empty waits, [7] W producer total, [8] consumer epilogue
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define BH_T0() do { if (INS) tq = clock64(); } while (0)
#define BH_T1(i) do { if (INS) { unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)

  constexpr int NC = PP ? 2 : BM / 64, NCT = NC * 128;
  constexpr int AB = ALU % 100000, OPT = ALU / 100000;  // P4c: synthetic op type (critical-path rungs); AB >= 50000: RS decode F
  // KS: k64 sub-tiles per ring slot (2 = BLOCK_K 128: one barrier round and one commit per 8 wgmma)
  constexpr int WT1 = BM * 128, XT1 = BN * 128, WTB = WT1 * KS, XTB = XT1 * KS, NR = BN / 2;
  constexpr bool XON = PRB != 1, MMA = PRB == 0 || (PRB >= 3 && PRB <= 6), WAITS = PRB != 4 && PRB != 5, TMAON = PRB != 4;
  constexpr bool STALE = PRB >= 4 && PRB <= 6;
  constexpr int SPG = CG == 2 ? 2 : 1, WAITG = CG == 1 ? 4 * WN : WN;
  constexpr int CW = PP ? 4 : 4 * NC;  // consumer warps that release one stage (per CTA)
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t wring = sbase, xring = sbase + ST * WTB, bars = xring + ST * XTB;
  // SY bit 0: one full + one empty barrier per slot shared by both rings (full counts both producers' arrive.expect_tx);
  // bit 1: consumer polls with test_wait instead of the suspending try_wait; bit 2: producers poll too
  constexpr bool MB = SY & 1;
  const uint32_t wfull = bars, wempty = bars + 8 * ST, xfull = MB ? wfull : bars + 16 * ST, xempty = MB ? wempty : bars + 24 * ST;
  auto cwait = [&](uint32_t b, uint32_t ph) { if (SY & 2) bh_spin(b, ph); else bh_wait(b, ph); };
  auto pwait = [&](uint32_t b, uint32_t ph) { if (SY & 4) bh_spin(b, ph); else bh_wait(b, ph); };
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tt = blockIdx.x, nb = blockIdx.y, KC = K / (64 * KS);
  const int t0 = tt * BN, n0 = nb * BM;
  uint32_t rank = 0;
  if (CL == 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  if (tid == 0) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wfull + 8 * i), "r"(MB && XON && TMAON ? 2 : 1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(CL == 1 ? 2 * CW : CW));
      if (!MB) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(xfull + 8 * i));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(CW));
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (CL == 1) bh_csync();
  if (warp == 4 * NC) {  // W producer
    if (lane == 0 && TMAON) {
      unsigned long long tp0 = INS ? clock64() : 0;
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) {
          BH_T0();
          if (PRB == 5) pwait(wfull + 8 * slot, ((s / ST) - 1) & 1);
          else pwait(wempty + 8 * slot, ((s / ST) - 1) & 1);
          BH_T1(5);
        }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(wfull + 8 * slot), "n"(WTB) : "memory");
#pragma unroll
        for (int kk = 0; kk < KS; kk++) {
          const uint32_t d = wring + slot * WTB + kk * WT1;
          const int k0 = (s * KS + kk) * 64;
          if (CL == 1) bh_tile_mc(d + rank * 4096, &tw, k0, n0 + 32 * (int)rank, wfull + 8 * slot, (uint16_t)3);
          else if (CL == 2) { bh_tile(d, &tw, k0, n0, wfull + 8 * slot); bh_tile(d + 4096, &tw, k0, n0 + 32, wfull + 8 * slot); }
          else {
#pragma unroll
            for (int j = 0; j < BM / 64; j++) bh_tile(d + j * 8192, &tw, k0, n0 + 64 * j, wfull + 8 * slot);
          }
        }
      }
      if (PRB == 5) for (int s = (KC > ST ? KC - ST : 0); s < KC; s++) bh_wait(wfull + 8 * (s % ST), (s / ST) & 1);
      if (INS) {
        const int b = (blockIdx.y * gridDim.x + blockIdx.x) * 16;
        prof[b + 5] = pc[5]; prof[b + 7] = clock64() - tp0;
      }
    }
  } else if (warp == 4 * NC + 1) {  // x producer
    if (lane == 0 && XON && TMAON) {
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) {
          BH_T0();
          if (PRB == 5) pwait(xfull + 8 * slot, ((s / ST) - 1) & 1);
          else pwait(xempty + 8 * slot, ((s / ST) - 1) & 1);
          BH_T1(6);
        }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(xfull + 8 * slot), "n"(XTB) : "memory");
#pragma unroll
        for (int kk = 0; kk < KS; kk++) bh_tile(xring + slot * XTB + kk * XT1, &tx, (s * KS + kk) * 64, t0, xfull + 8 * slot);
      }
      if (PRB == 5) for (int s = (KC > ST ? KC - ST : 0); s < KC; s++) bh_wait(xfull + 8 * (s % ST), (s / ST) & 1);
      if (INS) prof[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + 6] = pc[6];
    }
  } else {  // consumer warpgroup c (warps 4c .. 4c+3); warp w owns rows 16w .. 16w+15 of the warpgroup's 64
    const int c = warp >> 2, w = warp & 3;
    float D[NR];
#pragma unroll
    for (int i = 0; i < NR; i++) D[i] = 0.f;
    if (INS) { tq0 = clock64(); tq = tq0; }
    const int s0 = PP ? c : 0, ds = PP ? 2 : 1;
    const int nst = PP ? (KC - c + 1) / 2 : KC;
    const uint32_t aoff = PP ? 0u : (uint32_t)c * 8192u;
    uint32_t alu[4] = {(uint32_t)tid, (uint32_t)tid + 1u, (uint32_t)tid + 2u, (uint32_t)tid + 3u}, dzz = 0u;
    const uint32_t alu_a = ((uint32_t)K * 2654435761u) | 1u, alu_b = (uint32_t)m * 40503u + (uint32_t)N;  // runtime: no folding
    const uint32_t alu_c = (alu_a ^ (alu_b << 1)) | 1u, alu_d = alu_b * 5u + alu_a;
    auto wait_in = [&](int s) {
      cwait(wfull + 8 * (s % ST), (s / ST) & 1);
      if (XON && !MB) cwait(xfull + 8 * (s % ST), (s / ST) & 1);
    };
    auto release = [&](int s) {
      __syncwarp();
      if (lane == 0) {
        const uint32_t we = wempty + 8 * (s % ST);
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(we) : "memory");
        if (CL == 1) {
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(we), "r"(rank ^ 1u));
          asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" :: "r"(ra) : "memory");
        }
        if (XON && !MB) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(xempty + 8 * (s % ST)) : "memory");
      }
    };
    auto issue = [&](int s) {
      const int slot = STALE ? 0 : s % ST;
#pragma unroll
      for (int kk = 0; kk < KS; kk++) {
        const uint64_t da = bh_desc(wring + slot * WTB + kk * WT1 + aoff + dzz), db = bh_desc(xring + slot * XTB + kk * XT1);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          bh_wg(D, da + 2 * q, db + 2 * q);
          if (CG == 1) bh_commit();
        }
      }
    };
    if (MMA && !RS && AB >= 4000 && AB < 50000) {  // pipelined synthetic decode: slot s+1 is waited, read (3 LDS.128 per k64) and decoded
      // after stage s's commit and before its wait_group, overlapping wgmma(s); the result feeds stage s+1's W descriptor
      constexpr int NA = AB % 1000, IL = (AB / 1000) & 1 ? 4 : 1, NI = NA * KS / IL;
      auto decode = [&](int s) -> uint32_t {
        const uint32_t base = wring + (s % ST) * WTB + (uint32_t)(tid & 127) * 16u;
        uint32_t sd = 0;
#pragma unroll
        for (int kk = 0; kk < KS; kk++) {
#pragma unroll
          for (int j = 0; j < 3; j++) {
            uint32_t a0, a1, a2, a3;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(base + kk * WT1 + j * 2048) : "memory");
            sd ^= a0 ^ a1 ^ a2 ^ a3;
          }
        }
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++) alu[c4] ^= sd;
#pragma unroll
        for (int i = 0; i < NI; i++) {
#pragma unroll
          for (int c4 = 0; c4 < IL; c4++) asm volatile("mad.lo.u32 %0, %0, %1, %2;\n" : "+r"(alu[c4]) : "r"(alu_a), "r"(alu_b));
        }
        return ((alu[0] ^ alu[1] ^ alu[2] ^ alu[3]) == 0u && m < 0) ? 1024u : 0u;
      };
      // first and last stages peeled: a branch in the loop body puts ptxas's injected warpgroup.arrive on a divergent
      // path and serializes the wgmma (C7520)
      auto mma_stage = [&](int s) { bh_facc(D); bh_fence(); issue(s); bh_commit(); bh_facc(D); };
      wait_in(0);
      dzz = decode(0);
      mma_stage(0);
      wait_in(1);
      dzz = decode(1);
#pragma unroll 1
      for (int s = 1; s < KC - 1; s++) {
        mma_stage(s);
        if (AB >= 8000) {  // release-first: free slot s-1 (wgmma s-1 done), then wait + decode s+1 while wgmma s runs
          bh_wgwait<1>();
          release(s - 1);
          wait_in(s + 1);
          dzz = decode(s + 1);
        } else {  // decode-first (as specified): slot s-1 is held until s+1 has landed and been decoded
          wait_in(s + 1);
          const uint32_t dzn = decode(s + 1);
          bh_wgwait<1>();
          release(s - 1);
          dzz = dzn;
        }
      }
      mma_stage(KC - 1);
      bh_wgwait<1>();
      release(KC - 2);
      bh_wgwait<0>();
      bh_facc(D);
      release(KC - 1);
    } else if (MMA && !RS) {
      const int nu = nst / SPG;
      if (FN) bh_fence();
#pragma unroll 1
      for (int u = 0; u < nu; u++) {
        BH_T0();
        if (WAITS) {
#pragma unroll
          for (int j = 0; j < SPG; j++) wait_in(s0 + (u * SPG + j) * ds);
        }
        BH_T1(1);
        if (AB % 1000 && AB < 4000) {  // synthetic decode load: ALU % 1000 int ops per thread per k64 stage; bit 1000: 4 chains;
          // bit 2000: the chains feed this unit's W descriptor (an unprovable zero), so ptxas cannot sink them past the wgmma
          constexpr int NA = AB % 1000, IL = (AB / 1000) & 1 ? 4 : 1, NI = NA * KS * SPG / IL;
          if (AB >= 2000) {  // the decode reads the landed slot: seed the chains with a word of this unit's W tile
            uint32_t wv;
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(wv) : "r"(wring + ((s0 + u * SPG * ds) % ST) * WTB + aoff + lane * 4) : "memory");
#pragma unroll
            for (int c4 = 0; c4 < 4; c4++) alu[c4] ^= wv;
          }
#pragma unroll
          for (int i = 0; i < NI; i++) {
#pragma unroll
            for (int c4 = 0; c4 < IL; c4++) bh_op<OPT>(alu[c4], i, alu_a, alu_b, alu_c, alu_d);
          }
          if (AB >= 2000) dzz = ((alu[0] ^ alu[1] ^ alu[2] ^ alu[3]) == 0u && m < 0) ? 1024u : 0u;
        }
        bh_facc(D);
        if (!FN) bh_fence();
#pragma unroll
        for (int j = 0; j < SPG; j++) issue(s0 + (u * SPG + j) * ds);
        if (CG != 1) bh_commit();
        bh_facc(D);
        BH_T1(2);
        if (u >= WN) {
          bh_wgwait<WAITG>();
          BH_T1(3);
          if (WAITS) {
#pragma unroll
            for (int j = 0; j < SPG; j++) release(s0 + ((u - WN) * SPG + j) * ds);
          }
          BH_T1(4);
        }
      }
      bh_wgwait<0>();
      bh_facc(D);
      if (WAITS) {
        for (int u = (nu > WN ? nu - WN : 0); u < nu; u++)
          for (int j = 0; j < SPG; j++) release(s0 + (u * SPG + j) * ds);
      }
    } else if (MMA && RS) {  // register-A: ldmatrix.x4 per k16 step from the 128B-swizzled W tile; WN = 1, CG = 0
      constexpr int NA = 16 * KS;
      uint32_t A0[NA], A1[NA];
      const int lrow = 16 * w + (lane & 7) + ((lane >> 3) & 1) * 8, lkh = lane >> 4;
      const uint32_t dec_tb = (alu_b & 0x3fu) * 0x01010101u, dec_k8 = 0x08080808u | (alu_b & 0x0u) | ((uint32_t)(m < 0) << 4),
                     dec_sc = 0x3f803f80u ^ ((uint32_t)(m < 0) << 20);  // runtime-opaque window base / bias / scale
      auto step = [&](uint32_t (&A)[NA], uint32_t (&Ap)[NA], int s, bool first) {
        BH_T0();
        if (WAITS) wait_in(s);
        BH_T1(1);
        const int slot = STALE ? 0 : s % ST;
        if constexpr (AB >= 50000) {  // P4c: 3 LDS.128 per k64 (the 12-bit payload: 32 weights / thread) -> decode F -> A
#pragma unroll
          for (int kk = 0; kk < KS; kk++) {
            const uint32_t base = wring + slot * WTB + kk * WT1 + (uint32_t)(tid & 127) * 16u;
            uint32_t wd[12];
#pragma unroll
            for (int j = 0; j < 3; j++)
              asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(wd[4 * j]), "=r"(wd[4 * j + 1]), "=r"(wd[4 * j + 2]), "=r"(wd[4 * j + 3]) : "r"(base + j * 2048) : "memory");
#pragma unroll
            for (int g = 0; g < 4; g++) bh_dec<AB - 50000>(wd[g], wd[4 + g], wd[8 + g], dec_tb, dec_k8, dec_sc, &A[16 * kk + 4 * g]);
          }
        } else
#pragma unroll
        for (int kk = 0; kk < KS; kk++) {
          const uint32_t wb = wring + slot * WTB + kk * WT1 + aoff + lrow * 128;
#pragma unroll
          for (int q = 0; q < 4; q++)
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(A[16 * kk + 4 * q]), "=r"(A[16 * kk + 4 * q + 1]), "=r"(A[16 * kk + 4 * q + 2]), "=r"(A[16 * kk + 4 * q + 3])
                         : "r"(wb + ((((2 * q + lkh) ^ (lrow & 7))) << 4)) : "memory");
        }
        bh_facc(D);
        bh_fence();
#pragma unroll
        for (int kk = 0; kk < KS; kk++) {
          const uint64_t db = bh_desc(xring + slot * XTB + kk * XT1);
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const uint32_t a4[4] = {A[16 * kk + 4 * q], A[16 * kk + 4 * q + 1], A[16 * kk + 4 * q + 2], A[16 * kk + 4 * q + 3]};
            bh_wgr(D, a4, db + 2 * q);
          }
        }
        bh_commit();
        bh_facc(D);
        BH_T1(2);
        if (!first) {  // A of stage s - 1 stays live (and unaliased) until the wait covering it
          bh_wgwait<1>();
          bh_fa(Ap);
          BH_T1(3);
          if (WAITS) release(s - 1);
          BH_T1(4);
        }
      };
      step(A0, A1, 0, true);
#pragma unroll 1
      for (int s = 1; s + 1 < KC; s += 2) { step(A1, A0, s, false); step(A0, A1, s + 1, false); }
      if (KC % 2 == 0) step(A1, A0, KC - 1, false);
      bh_wgwait<0>();
      bh_facc(D);
      bh_fa(A0); bh_fa(A1);
      if (WAITS) release(KC - 1);
    } else if (WAITS) {  // PRB 7: release s - 1 after stage s landed (wait_group(1)'s hold, no wgmma); 8: hold SPIN clk
#pragma unroll 1
      for (int i = 0; i < nst; i++) {
        wait_in(s0 + i * ds);
        if (PRB == 7) { if (i > 0) release(s0 + (i - 1) * ds); }
        else {
          if (PRB == 8) { const long long t_ = clock64(); while (clock64() - t_ < 240) { } }
          release(s0 + i * ds);
        }
      }
      if (PRB == 7) release(s0 + (nst - 1) * ds);
    }
    BH_T0();
    if (PRB == 0) {
      const int g = lane >> 2, t = lane & 3;
      if (PP) {  // warpgroup 1's partial sums to smem (fragment order), warpgroup 0 adds them
        if (c == 1) {
#pragma unroll
          for (int i = 0; i < NR / 4; i++)
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "r"(wring + (i * 128 + (tid & 127)) * 16), "f"(D[4 * i]), "f"(D[4 * i + 1]), "f"(D[4 * i + 2]), "f"(D[4 * i + 3]) : "memory");
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NCT) : "memory");
        if (c == 0) {
#pragma unroll
          for (int i = 0; i < NR / 4; i++) {
            float a, b, cc, d;
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(a), "=f"(b), "=f"(cc), "=f"(d) : "r"(wring + (i * 128 + (tid & 127)) * 16) : "memory");
            D[4 * i] += a; D[4 * i + 1] += b; D[4 * i + 2] += cc; D[4 * i + 3] += d;
          }
        }
      }
      constexpr bool SX = (SY & 8) && !PP;  // SY bit 3: stmatrix.trans into a chunk-swizzled tile (chunk ^= tok & 7)
      if (SX) {
        const int j = lane >> 3, rr = lane & 7, nc = 64 * c + 16 * w + 8 * (j & 1);
#pragma unroll
        for (int i = 0; i < NR / 4; i += 2) {
          uint32_t q0, q1, q2, q3;
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(q0) : "f"(D[4 * i]), "f"(D[4 * i + 1]));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(q1) : "f"(D[4 * i + 2]), "f"(D[4 * i + 3]));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(q2) : "f"(D[4 * i + 4]), "f"(D[4 * i + 5]));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(q3) : "f"(D[4 * i + 6]), "f"(D[4 * i + 7]));
          const int tk = 8 * (i + (j >> 1)) + rr;
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n"
                       :: "r"(xring + tk * BM * 2 + (((nc >> 3) ^ (tk & 7)) << 4)), "r"(q0), "r"(q1), "r"(q2), "r"(q3) : "memory");
        }
      } else if (!PP || c == 0) {  // bf16 tile [BN tok][BM n] in the drained x ring
        const int r = (PP ? 0 : 64 * c) + 16 * w + g;
#pragma unroll
        for (int i = 0; i < NR / 4; i++) {
          const int tk = 8 * i + 2 * t;
          // inline-asm cvt: __float2bfloat16(D[i]) here makes ptxas serialize the RS kernel's wgmma pipeline (C7514)
          uint32_t p0, p1;
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(p0) : "f"(D[4 * i]), "f"(D[4 * i + 2]));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(p1) : "f"(D[4 * i + 1]), "f"(D[4 * i + 3]));
          asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(xring + (tk * BM + r) * 2), "h"((unsigned short)(p0 & 0xffffu)) : "memory");
          asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(xring + ((tk + 1) * BM + r) * 2), "h"((unsigned short)(p1 & 0xffffu)) : "memory");
          asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(xring + (tk * BM + r + 8) * 2), "h"((unsigned short)(p0 >> 16)) : "memory");
          asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(xring + ((tk + 1) * BM + r + 8) * 2), "h"((unsigned short)(p1 >> 16)) : "memory");
        }
      }
      asm volatile("bar.sync 1, %0;\n" :: "n"(NCT) : "memory");
      constexpr int CPR = BM / 8;  // 16-B chunks per token row
      for (int e = tid; e < BN * CPR; e += NCT) {
        const int tk = e / CPR, cc = e % CPR;
        if (t0 + tk < m) {
          uint4 v;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(xring + tk * BM * 2 + ((SX ? (cc ^ (tk & 7)) : cc) << 4)));
          *(uint4*)(out + (size_t)(t0 + tk) * N + n0 + cc * 8) = v;
        }
      }
      BH_T1(8);
    } else if (D[0] == 1.2345f) out[0] = __float2bfloat16(0.f);
    if (AB % 1000 && (alu[0] ^ alu[1] ^ alu[2] ^ alu[3]) == 0x9e3779b9u && D[1] == 1.2345f) out[1] = __float2bfloat16(0.f);
    if (INS && tid == 0) {
      pc[0] = clock64() - tq0;
      for (int i = 0; i < 5; i++) prof[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + i] = pc[i];
      prof[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + 8] = pc[8];
    }
  }
  if (CL == 1) bh_csync();
}

typedef CUresult (*BgEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                               const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                               CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(BgTmap), "CUtensorMap size");
static int bh_map(BgTmap* tm, const void* p, int64_t rows, int64_t K, int box_rows) {
  static BgEncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (BgEncodeFn)fn;
  }
  if (((uintptr_t)p) % 16) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int ST, int BM, int BN, int CL, int PP, int WN, int CG, int RS, int FN, int PRB, int INS, int SY, int KS, int ALU>
static int bh_go(torch::Tensor x, torch::Tensor W, torch::Tensor out, torch::Tensor prof, int info) {
  constexpr int NC = PP ? 2 : BM / 64;
  constexpr int SM = 1024 + ST * (BM + BN) * 128 * KS + 32 * ST;
  auto k = bh<ST, BM, BN, CL, PP, WN, CG, RS, FN, PRB, INS, SY, KS, ALU>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    return info == 1 ? a.numRegs : a.localSizeBytes;
  }
  const int m = (int)x.size(0), N = (int)out.size(1), K = (int)x.size(1);
  if (N % BM || K % 64 || m < 1) return -4;
  BgTmap tw, tx;
  int r = bh_map(&tw, W.data_ptr(), N, K, CL ? 32 : 64);
  if (INS && prof.numel() < (int64_t)16 * ((m + BN - 1) / BN + 1) * (N / BM)) return -11;
  if (r) return r;
  r = bh_map(&tx, x.data_ptr(), m, K, BN);
  if (r) return r;
  int TT = (m + BN - 1) / BN;
  if (CL == 1 && TT % 2) TT++;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(TT, N / BM); cfg.blockDim = dim3(NC * 128 + 64); cfg.dynamicSmemBytes = SM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = CL == 1 ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, k, tw, tx, (__nv_bfloat16*)out.data_ptr(), m, N, K, INS ? (unsigned long long*)prof.data_ptr() : (unsigned long long*)nullptr);
  if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
  return 0;
}
int64_t bhrun(int64_t v, torch::Tensor x, torch::Tensor W, torch::Tensor out, torch::Tensor prof, int64_t info) {
  if (!info && (x.dim() != 2 || W.dim() != 2 || x.size(1) != W.size(1) || out.size(0) != x.size(0) || out.size(1) > W.size(0))) return -10;
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""


def _src():
    cases = "\n".join("    case %d: return bh_go<%s>(x, W, out, prof, (int)info);" % (i, ", ".join(str(a) for a in tuple(v[1:]) + (0, 0, 1, 0)[len(v) - 11:]))
                      for i, v in enumerate(BHV))
    wg = _wg(64, False) + _wg(128, False) + _wg(64, True) + _wg(128, True)
    return BHCU.replace("@@WG@@", wg).replace("@@CASES@@", cases)


_mod = None
_E = None
FLAGS = ["-O3", "-lineinfo", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a"]


def _ext():
    global _mod
    if _mod is None:
        src = _src()
        pv = os.environ.get("BH_PTXV") == "1"  # ptxas -v / serialisation warnings in the build log
        _mod = load_inline("bh24h_%x%s" % (hash(src) & 0xffffffff, "v" if pv else ""), cpp_sources="int64_t bhrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                           cuda_sources=src, functions=["bhrun"], extra_cuda_cflags=FLAGS + (["-Xptxas=-v"] if pv else []), verbose=pv)
    return _mod


def _time_us(fn, x, out, flush, sink, it=31):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x, out); fn(x, out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x, out)
    times = []
    for _ in range(it):
        torch.sum(flush, 0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    return times[len(times) // 2]


_EYE = {}


def _validate(fn, w, m, n, gen, m_odd):
    """one-hot rows x8 bit-exact against W; random rows within 4e-3 rel L2 of FP64, at m and an odd m"""
    K = w.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
    exact = w[:n].t()
    if K not in _EYE:
        _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
    eye = _EYE[K]
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
        code = fn(x, out)
        if code:
            return False, "launch %d" % code
        got = out[: min(m, K - start)]
        want = exact[start:start + got.shape[0]] * 8.0
        if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
            return False, "one-hot mismatch at column %d" % start
    for mm in (m, m_odd):
        x = torch.randn((mm, K), generator=gen, device=w.device).to(torch.bfloat16)
        o = torch.empty((mm, n), dtype=torch.bfloat16, device=w.device)
        fn(x, o)
        ref = x.double() @ w[:n].double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random m=%d rel L2 %.2e row %.2e" % (mm, err, row)
    return True, "ok"


BAKE_SHAPES_ALL = [("pooler", 128), ("qkv_gate", 128), ("unpooler", 128), ("o", 128), ("pooler", 256), ("qkv_gate", 256), ("unpooler", 256), ("o", 256)]


BAKE_SHAPES = [("pooler", 128), ("pooler", 256)] if SLACK else BAKE_SHAPES_ALL


def prepare(weights):
    st = {"raw": dict(weights), "sel": {}}
    t0 = time.time()
    try:
        ext = _ext()
    except Exception as error:
        print("PREP bh build failed: %s" % str(error)[-2000:], flush=True)
        return st
    print("PREP bh compiled in %.0fs" % (time.time() - t0), flush=True)
    for i, v in enumerate(BHV):
        e = torch.empty(0)
        print("REGS %-14s regs=%d local=%d" % (v[0], ext.bhrun(i, e, e, e, e, 1), ext.bhrun(i, e, e, e, e, 2)), flush=True)
    if "--prepare-only" in sys.argv:  # compile only: no GPU timing outside the lock
        return st
    global _E
    dev = weights["pooler"].device
    _E = torch.empty(0, device=dev)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(2424)
    only = [s for s in os.environ.get("BH_ONLY", "").split(",") if s]
    names = [v[0] for v in BHV if not only or v[0].split(".")[0] in only]
    # PROBE ladder: pooler m = 128
    w = weights["pooler"]
    n, K = w.shape
    x = torch.randn((128, K), generator=gen, device=dev).to(torch.bfloat16)
    out = torch.empty((128, n), dtype=torch.bfloat16, device=dev)
    cub = lambda xx, oo, w=w: torch.mm(xx, w.t(), out=oo)
    print("PROBE ladder pooler m=128 cublas %.2f" % _time_us(cub, x, out, flush, sink), flush=True)
    for name in names:
        i = BHI[name]
        if BHV[i][10] == 0 and not (len(BHV[i]) > 11 and BHV[i][11]):
            continue
        if len(BHV[i]) > 11 and BHV[i][11]:  # instrumented: phase cycles per stage, mean over CTAs (cold L2, 5th run)
            prof = torch.zeros((16 * 4 * 64,), dtype=torch.int64, device=dev)
            for _ in range(5):
                torch.sum(flush, 0, out=sink)
                code = ext.bhrun(i, x, w, out, prof, 0)
            torch.cuda.synchronize()
            if code:
                print("PROBE ins %s launch-error %d" % (name, code), flush=True)
                continue
            v = BHV[i]
            ncta = ((128 + v[3] - 1) // v[3]) * (n // v[2])
            pr = prof[: 16 * ncta].view(ncta, 16).double()
            nst = (K // 64) / (2 if v[5] else 1)  # per 64-k
            mn, mx = pr.mean(0) / nst, pr.max(0).values / nst
            print("PROBE ins %-14s per-stage clk mean(max): total %.0f(%.0f) wait_in %.0f(%.0f) issue %.0f(%.0f) wgwait %.0f(%.0f) release %.0f(%.0f) | W-prod empty-wait %.0f x-prod empty-wait %.0f W-prod total %.0f | epilogue %.0f clk" % (
                name, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], mn[3], mx[3], mn[4], mx[4], mn[5] * nst / (K // 64), mn[6] * nst / (K // 64), mn[7] * nst / (K // 64), mn[8] * nst), flush=True)
            continue
        fn = lambda xx, oo, i=i: ext.bhrun(i, xx, w, oo, _E, 0)
        if SLACK and BHV[i][3] == 128:  # slack mode: the 64x128 tile's W+x rung at its own shape, m = 256
            x2 = torch.randn((256, K), generator=gen, device=dev).to(torch.bfloat16)
            o2 = torch.empty((256, n), dtype=torch.bfloat16, device=dev)
            if fn(x2, o2) == 0:
                print("PROBE ladder %-14s m=256 %.2f" % (name, _time_us(fn, x2, o2, flush, sink)), flush=True)
            continue
        code = fn(x, out)
        torch.cuda.synchronize()
        if code:
            print("PROBE ladder %s launch-error %d" % (name, code), flush=True)
            continue
        print("PROBE ladder %-14s %.2f" % (name, _time_us(fn, x, out, flush, sink)), flush=True)
    for layer, m in BAKE_SHAPES:
        w = weights[layer]
        n, K = w.shape
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        cub = lambda xx, oo, w=w: torch.mm(xx, w.t(), out=oo)
        best = ("cublas", None, _time_us(cub, x, out, flush, sink))
        print("BAKE %s m=%d cublas %.2f" % (layer, m, best[2]), flush=True)
        for name in names:
            i = BHI[name]
            v = BHV[i]
            if v[10] != 0 or (len(v) > 11 and v[11]) or n % v[2] or (SLACK and (m == 128) != (v[3] == 64)):
                continue
            fn = lambda xx, oo, i=i, w=w: ext.bhrun(i, xx, w, oo, _E, 0)
            try:
                ok, msg = _validate(fn, w, m, n, gen, m - 7) if not (SLACK and len(v) > 14 and v[14]) else (fn(x, out) == 0, "unvalidated")  # noqa
                if not ok:
                    print("BAKE %s m=%d %s INVALID %s" % (layer, m, name, msg), flush=True)
                    continue
                us = _time_us(fn, x, out, flush, sink)
            except Exception as error:
                print("BAKE %s m=%d %s ERROR %s" % (layer, m, name, str(error).splitlines()[-1][:160]), flush=True)
                continue
            print("BAKE %s m=%d %-10s %.2f" % (layer, m, name, us), flush=True)
            if us < best[2] * 0.99:
                best = (name, i, us)
        if best[1] is not None:
            st["sel"][(layer, m)] = best[1]
        print("PICK %s m=%d %s %.2f" % (layer, m, best[0], best[2]), flush=True)
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    w = state["raw"][layer]
    i = state["sel"].get((layer, x.shape[0]))
    if i is not None and n == w.shape[0]:
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        if _ext().bhrun(i, x.contiguous(), w, out, _E, 0) == 0:
            return out
    if layer == "router":
        return x.float() @ w[:n].float().t()
    return x @ w[:n].t()

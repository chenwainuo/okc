template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv_sparse_quarter(G p) {
  static_assert((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1), "GVpredxor only seven reviewed format1 templates");
  static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"quarter prototype exact tuple");
  if(p.m!=8 || p.N!=4096 || p.K!=8192) return;
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane];
  const uint32_t emask=__ballot_sync(~0u,lane<KU && ((e>>24)&63)!=(e&63));
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    // Snapshot every current payload word before prefetch overwrites wb[s].
    const uint4 q0=wb[s][0][0][0], q1=wb[s][0][0][1], q2=wb[s][0][0][2];
    const uint4 b0=xb[s % XB][0][0], b1=xb[s % XB][0][1];
    const uint32_t lo[8]={q0.x,q0.y,q0.z,q0.w,q1.x,q1.y,q1.z,q1.w};
    const uint32_t codes[4]={q2.x,q2.y,q2.z,q2.w};
    const uint32_t B[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
    const bool has_esc=((emask>>u)&1)!=0;
    uint32_t header=0;
    if(has_esc) header=__shfl_sync(~0u,e,u);
    if (u+ST<KU) ld(s,u+ST);
#pragma unroll
    for (int q=0;q<4;q++) {
      uint32_t A4[4];
      dec1(lo[2*q],lo[2*q+1],codes[q],p.T0,p.T1,A4);
      if(has_esc) {
      const int begin=(header>>(6*q))&63, end=(header>>(6*(q+1)))&63;
#pragma unroll 1
      for (int ep4=begin;ep4<end;ep4++) {
        const uint32_t pair=__shfl_sync(~0u,e,KU+(ep4>>1));
        const uint32_t r16=(pair>>(16*(ep4&1)))&65535u;
        const uint32_t xr=(r16&127u)<<(8+16*((r16>>7)&1));
        gv_fix_quarter(A4,(r16>>9)&3,xr,lane,r16>>11,(r16>>8)&1);
      }
      }
      hmma(acc[0][0][q % NC],A4,B[2*q],B[2*q+1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}


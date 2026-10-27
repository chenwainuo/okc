  auto step = [&](const int u, const int s) {
    // Snapshot every current payload word before prefetch overwrites wb[s].
    const uint4 q0=wb[s][0][0][0], q1=wb[s][0][0][1], q2=wb[s][0][0][2];
    const uint4 b0=xb[s % XB][0][0], b1=xb[s % XB][0][1];
    const uint32_t lo[8]={q0.x,q0.y,q0.z,q0.w,q1.x,q1.y,q1.z,q1.w};
    const uint32_t codes[4]={q2.x,q2.y,q2.z,q2.w};
    const uint32_t B[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
    const uint32_t header=__shfl_sync(~0u,e,u);
    if (u+ST<KU) ld(s,u+ST);
#pragma unroll
    for (int q=0;q<4;q++) {
      uint32_t A4[4];
      dec1(lo[2*q],lo[2*q+1],codes[q],p.T0,p.T1,A4);
      const int begin=(header>>(6*q))&63, end=(header>>(6*(q+1)))&63;
#pragma unroll 1
      for (int ep4=begin;ep4<end;ep4++) {
        const uint32_t pair=__shfl_sync(~0u,e,KU+(ep4>>1));
        const uint32_t r16=(pair>>(16*(ep4&1)))&65535u;
        const uint32_t xr=(r16&127u)<<(8+16*((r16>>7)&1));
        gv_fix_quarter(A4,(r16>>9)&3,xr,lane,r16>>11,(r16>>8)&1);
      }
      hmma(acc[0][0][q % NC],A4,B[2*q],B[2*q+1]);
    }
  };

// Appended only by the explicitly launched two-cell trial. No standalone run.
template <bool F, int MT, int ST, int WR, int WK, int KU, int XS, int V>
std::vector<int64_t> frontier_properties() {
  constexpr int nt=WR*WK*32;
  constexpr int oldsm=GL<1,MT,WR,WK,KU,1,1,XS,V>::SMEM;
  constexpr int cap=MT==1?32:48;
  constexpr int dynamic=oldsm+(F?WR*WK*ST*(32+cap*2):0);
  cudaFuncAttributes a; int blocks=0; cudaError_t e;
  if constexpr(F) {
    auto k=gv_frontier<1,MT,ST,WR,WK,KU,1,1,XS,V>;
    e=cudaFuncGetAttributes(&a,k); TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
    e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,nt,dynamic);
  } else {
    auto k=gv_predxor<1,MT,ST,WR,WK,KU,1,1,XS,V>;
    e=cudaFuncGetAttributes(&a,k); TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
    e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,nt,dynamic);
  }
  TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,nt,dynamic,blocks};
}
std::vector<int64_t> frontier_info(int64_t cell,bool f) {
  TORCH_CHECK(cell==0||cell==1);
  if(cell==0) return f?frontier_properties<true,1,1,2,16,8,0,3>():frontier_properties<false,1,1,2,16,8,0,3>();
  return f?frontier_properties<true,2,2,1,4,16,2,1>():frontier_properties<false,2,2,1,4,16,2,1>();
}
template <bool F, int MT, int ST, int WR, int WK, int KU, int XS, int V>
void frontier_launch(G p,Frontier f,cudaStream_t stream) {
  constexpr int oldsm=GL<1,MT,WR,WK,KU,1,1,XS,V>::SMEM;
  constexpr int dynamic=oldsm+(F?WR*WK*ST*(32+(MT==1?32:48)*2):0);
  if constexpr(F) gv_frontier<1,MT,ST,WR,WK,KU,1,1,XS,V><<<dim3(1,p.N/(WR*16),1),WR*WK*32,dynamic,stream>>>(p,f);
  else gv_predxor<1,MT,ST,WR,WK,KU,1,1,XS,V><<<dim3(1,p.N/(WR*16),1),WR*WK*32,dynamic,stream>>>(p);
}
void frontier_run(int64_t cell,bool front,torch::Tensor low,torch::Tensor header,torch::Tensor desc,
                  torch::Tensor records,torch::Tensor x,torch::Tensor out,torch::Tensor ws,
                  torch::Tensor cnt,int64_t table,int64_t words) {
  TORCH_CHECK(cell==0||cell==1);
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&out.is_cuda()&&out.is_contiguous());
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&out.scalar_type()==torch::kBFloat16);
  TORCH_CHECK((cell==0&&x.size(0)==1&&x.size(1)==8192&&out.size(1)==4096)||
              (cell==1&&x.size(0)==16&&x.size(1)==4096&&out.size(1)==8192));
  G p{}; p.W=(const uint4*)low.data_ptr();p.W2=p.W;p.x=(const __nv_bfloat16*)x.data_ptr();
  p.out=out.data_ptr();p.esc=(const uint32_t*)desc.data_ptr();p.ws=(float4*)ws.data_ptr();p.cnt=cnt.data_ptr<int>();
  p.m=x.size(0);p.N=out.size(1);p.K=x.size(1);p.KC=p.K/64;p.T0=(uint32_t)table;p.T1=(uint32_t)((uint64_t)table>>32);
  Frontier f{(const uint32_t*)header.data_ptr(),(const uint8_t*)records.data_ptr(),(uint32_t)(p.N/16*p.KC),(uint32_t)words};
  auto stream=at::cuda::getCurrentCUDAStream();
  if(cell==0) { if(front)frontier_launch<true,1,1,2,16,8,0,3>(p,f,stream);else frontier_launch<false,1,1,2,16,8,0,3>(p,f,stream); }
  else { if(front)frontier_launch<true,2,2,1,4,16,2,1>(p,f,stream);else frontier_launch<false,2,2,1,4,16,2,1>(p,f,stream); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

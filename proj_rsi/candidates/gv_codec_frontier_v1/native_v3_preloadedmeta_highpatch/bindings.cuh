// Host-only entrypoints appended by the bounded three-arm driver.
#include <ATen/cuda/CUDAContextLight.h>
template<class K>std::vector<int64_t> frontier_query(K k,int threads,int dynamic) {
  cudaFuncAttributes a;int blocks=0;
  auto e=cudaFuncGetAttributes(&a,k);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
  e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,threads,dynamic);
  TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,threads,dynamic,blocks};
}
template<int ARM,int MT,int ST,int WR,int WK,int KU,int XS,int V>
std::vector<int64_t> frontier_properties() {
  constexpr int cap=MT==1?32:48;
  constexpr int oldsm=GL<1,MT,WR,WK,KU,1,1,XS,V>::SMEM;
  constexpr int extra=ARM==0?0:ARM==1?WR*WK*ST*(32+cap*2):WR*WK*(KU*32+16+ST*cap*2);
  if constexpr(ARM==0)return frontier_query(gv_predxor<1,MT,ST,WR,WK,KU,1,1,XS,V>,WR*WK*32,oldsm+extra);
  else if constexpr(ARM==1)return frontier_query(gv_frontier<1,MT,ST,WR,WK,KU,1,1,XS,V>,WR*WK*32,oldsm+extra);
  else return frontier_query(gv_frontier_preload<1,MT,ST,WR,WK,KU,1,1,XS,V>,WR*WK*32,oldsm+extra);
}
std::vector<int64_t> frontier_info(int64_t cell,int64_t arm) {
  TORCH_CHECK((cell==0||cell==1)&&arm>=0&&arm<3);
  if(cell==0){
    if(arm==0)return frontier_properties<0,1,1,2,16,8,0,3>();
    if(arm==1)return frontier_properties<1,1,1,2,16,8,0,3>();
    return frontier_properties<2,1,1,2,16,8,0,3>();
  }
  if(arm==0)return frontier_properties<0,2,2,1,4,16,2,1>();
  if(arm==1)return frontier_properties<1,2,2,1,4,16,2,1>();
  return frontier_properties<2,2,2,1,4,16,2,1>();
}
template<int ARM,int MT,int ST,int WR,int WK,int KU,int XS,int V>
void frontier_launch(G p,Frontier f,cudaStream_t stream) {
  constexpr int cap=MT==1?32:48;
  constexpr int oldsm=GL<1,MT,WR,WK,KU,1,1,XS,V>::SMEM;
  constexpr int extra=ARM==0?0:ARM==1?WR*WK*ST*(32+cap*2):WR*WK*(KU*32+16+ST*cap*2);
  const dim3 grid(1,p.N/(WR*16),1);
  if constexpr(ARM==0)gv_predxor<1,MT,ST,WR,WK,KU,1,1,XS,V><<<grid,WR*WK*32,oldsm+extra,stream>>>(p);
  else if constexpr(ARM==1)gv_frontier<1,MT,ST,WR,WK,KU,1,1,XS,V><<<grid,WR*WK*32,oldsm+extra,stream>>>(p,f);
  else gv_frontier_preload<1,MT,ST,WR,WK,KU,1,1,XS,V><<<grid,WR*WK*32,oldsm+extra,stream>>>(p,f);
}
void frontier_run(int64_t cell,int64_t arm,torch::Tensor low,torch::Tensor header,torch::Tensor desc,
                  torch::Tensor records,torch::Tensor x,torch::Tensor out,torch::Tensor ws,
                  torch::Tensor cnt,int64_t table,int64_t words) {
  TORCH_CHECK((cell==0||cell==1)&&arm>=0&&arm<3);
  for(const auto& t:{low,header,desc,records,x,out,ws,cnt})
    TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==x.device());
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&out.scalar_type()==torch::kBFloat16&&
    low.scalar_type()==torch::kUInt8&&header.scalar_type()==torch::kInt32&&
    desc.scalar_type()==torch::kInt32&&records.scalar_type()==torch::kUInt16&&
    ws.scalar_type()==torch::kFloat32&&cnt.scalar_type()==torch::kInt32);
  TORCH_CHECK(x.dim()==2&&out.dim()==2&&out.size(0)==x.size(0));
  TORCH_CHECK((cell==0&&x.size(0)==1&&x.size(1)==8192&&out.size(1)==4096)||
              (cell==1&&x.size(0)==16&&x.size(1)==4096&&out.size(1)==8192));
  const int64_t units=out.size(1)/16*(x.size(1)/64);
  TORCH_CHECK(low.numel()==units*(arm==0?1536:1024)&&header.numel()==units*96);
  TORCH_CHECK(desc.numel()==(arm==0?(cell==0?131072:65536):units*8)&&
              records.numel()==words&&words>0&&ws.numel()>=65536&&cnt.numel()>=65536);
  G p{};p.W=(const uint4*)low.data_ptr();p.W2=p.W;p.x=(const __nv_bfloat16*)x.data_ptr();
  p.out=out.data_ptr();p.esc=(const uint32_t*)desc.data_ptr();p.ws=(float4*)ws.data_ptr();p.cnt=cnt.data_ptr<int>();
  p.m=x.size(0);p.N=out.size(1);p.K=x.size(1);p.KC=p.K/64;p.T0=(uint32_t)table;p.T1=(uint32_t)((uint64_t)table>>32);
  Frontier f{(const uint32_t*)header.data_ptr(),(const uint8_t*)records.data_ptr(),(uint32_t)units,(uint32_t)words};
  auto stream=at::cuda::getCurrentCUDAStream();
  if(cell==0){
    if(arm==0)frontier_launch<0,1,1,2,16,8,0,3>(p,f,stream);
    else if(arm==1)frontier_launch<1,1,1,2,16,8,0,3>(p,f,stream);
    else frontier_launch<2,1,1,2,16,8,0,3>(p,f,stream);
  }else{
    if(arm==0)frontier_launch<0,2,2,1,4,16,2,1>(p,f,stream);
    else if(arm==1)frontier_launch<1,2,2,1,4,16,2,1>(p,f,stream);
    else frontier_launch<2,2,2,1,4,16,2,1>(p,f,stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Called only after every graph/stream is drained and run-scope tensors released.
std::vector<int64_t> frontier_clear_blas_cache() {
  const auto before=at::cuda::cublas_handle_stream_to_workspace().size();
  const auto bytes=at::cuda::getChosenWorkspaceSize();
  at::cuda::clearCublasWorkspaces();
  const auto after=at::cuda::cublas_handle_stream_to_workspace().size();
  return {(int64_t)before,(int64_t)bytes,(int64_t)after};
}

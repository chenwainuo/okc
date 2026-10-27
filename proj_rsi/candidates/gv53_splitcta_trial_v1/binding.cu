// Appended after frozen pilot.cu; only host code, no extra device kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
static int gv53_smem(int64_t arm){return arm==0?17408:8704;}
static int gv53_threads(int64_t arm){return arm==0?1024:512;}
int64_t attrs(int64_t arm, int64_t field) {
  if (arm < 0 || arm > 1) return -90;
  auto k = arm == 0 ? gv_predxor<1,1,1,2,16,8,1,1,0,3> : gv_predxor<1,1,1,1,16,8,1,1,0,8195>;
  cudaError_t rc = cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,gv53_smem(arm));
  if (rc != cudaSuccess) return -1000-(int)rc;
  cudaFuncAttributes a; rc=cudaFuncGetAttributes(&a,k);
  if (rc != cudaSuccess) return -2000-(int)rc;
  if(field==1)return a.numRegs; if(field==2)return a.localSizeBytes; if(field==3)return a.sharedSizeBytes;
  if(field==4)return a.maxDynamicSharedSizeBytes; if(field==5)return a.maxThreadsPerBlock;
  if(field==6){int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,gv53_threads(arm),gv53_smem(arm));return rc==cudaSuccess?n:-3000-(int)rc;}
  if(field==7){int dev=0,n=0;rc=cudaGetDevice(&dev);if(rc!=cudaSuccess)return -4000-(int)rc;rc=cudaDeviceGetAttribute(&n,cudaDevAttrReservedSharedMemoryPerBlock,dev);return rc==cudaSuccess?n:-5000-(int)rc;}
  if(field==8)return arm;
  return -91;
}
int64_t run(int64_t arm,torch::Tensor W,torch::Tensor esc,torch::Tensor x,torch::Tensor out,torch::Tensor ws,torch::Tensor cnt,int64_t table){
  if(arm<0||arm>1)return -90;
  TORCH_CHECK(W.is_cuda()&&esc.is_cuda()&&x.is_cuda()&&out.is_cuda()&&ws.is_cuda()&&cnt.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(W.device()==x.device()&&esc.device()==x.device()&&out.device()==x.device()&&ws.device()==x.device()&&cnt.device()==x.device(),"device mismatch");
  TORCH_CHECK(W.is_contiguous()&&esc.is_contiguous()&&x.is_contiguous()&&out.is_contiguous()&&ws.is_contiguous()&&cnt.is_contiguous(),"contiguous tensors required");
  TORCH_CHECK(W.scalar_type()==at::kByte&&esc.scalar_type()==at::kInt&&x.scalar_type()==at::kBFloat16&&out.scalar_type()==at::kBFloat16&&ws.scalar_type()==at::kFloat&&cnt.scalar_type()==at::kInt,"dtype mismatch");
  TORCH_CHECK(x.dim()==2&&(x.size(0)==1||x.size(0)==8)&&x.size(1)==8192&&out.dim()==2&&out.size(0)==x.size(0)&&out.size(1)==4096,"only pooler M1/M8,N4096,K8192");
  TORCH_CHECK(W.numel()==50331648&&esc.numel()==131072&&ws.numel()>=65536&&cnt.numel()>=65536,"backing size mismatch");
  G p; p.W=(const uint4*)W.data_ptr();p.W2=p.W;p.esc=(const uint32_t*)esc.data_ptr();p.x=(const __nv_bfloat16*)x.data_ptr();p.out=out.data_ptr();p.ws=(float4*)ws.data_ptr();p.cnt=cnt.data_ptr<int>();
  p.m=x.size(0);p.N=4096;p.K=8192;p.KC=128;p.T0=(uint32_t)table;p.T1=(uint32_t)((uint64_t)table>>32);p.U0=p.T0;p.U1=p.T1;
  auto k=arm==0?gv_predxor<1,1,1,2,16,8,1,1,0,3>:gv_predxor<1,1,1,1,16,8,1,1,0,8195>;
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((p.m+7)/8,arm==0?128:256,1);cfg.blockDim=dim3(gv53_threads(arm));cfg.dynamicSmemBytes=gv53_smem(arm);cfg.stream=at::cuda::getCurrentCUDAStream();
  cudaError_t rc=cudaLaunchKernelEx(&cfg,k,p);if(rc!=cudaSuccess)cudaGetLastError();return (int64_t)rc;
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
static void qcheck(torch::Tensor t, torch::Tensor x, torch::ScalarType dt, int64_t n, const char* name){
 TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.is_contiguous() && t.scalar_type()==dt && t.numel()==n, name," device/dtype/contiguity/size");
 TORCH_CHECK((uintptr_t)t.data_ptr()%16==0,name," alignment");
}
int64_t qkv_info(int64_t which,int64_t field){
 cudaFuncAttributes a; cudaError_t e;
 if(which==0)e=cudaFuncGetAttributes(&a,rx_pack);
 else if(which==1)e=cudaFuncGetAttributes(&a,rq_qkv_g8_sk4<128,8,1,0,0,0,20>);
 else {TORCH_CHECK(which==2,"kernel selector");e=cudaFuncGetAttributes(&a,qkv_reduce);}
 C10_CUDA_CHECK(e);
 if(field==0)return a.numRegs;if(field==1)return a.localSizeBytes;
 if(field==2)return a.sharedSizeBytes;if(field==3)return a.maxThreadsPerBlock;
 if(field==4)return a.maxDynamicSharedSizeBytes;
 TORCH_CHECK(which==1 && field==5,"info selector");
 C10_CUDA_CHECK((cudaFuncSetAttribute(rq_qkv_g8_sk4<128,8,1,0,0,0,20>,cudaFuncAttributeMaxDynamicSharedMemorySize,RqC<128,8,1,0,0,0,20>::SMEM)));
 int n=0;C10_CUDA_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,rq_qkv_g8_sk4<128,8,1,0,0,0,20>,512,RqC<128,8,1,0,0,0,20>::SMEM)));return n;
}
int64_t qkv_run(torch::Tensor W,torch::Tensor x,torch::Tensor out,torch::Tensor esc,torch::Tensor xp,torch::Tensor partial,int64_t tab){
 TORCH_CHECK(x.is_cuda(),"CUDA X required");c10::cuda::CUDAGuard guard(x.device());
 qcheck(x,x,torch::kBFloat16,128*8192,"X");TORCH_CHECK(x.dim()==2 && x.size(0)==128 && x.size(1)==8192,"X shape");
 qcheck(W,x,torch::kUInt8,18*128*8*1536,"W");TORCH_CHECK(W.dim()==4 && W.size(0)==18 && W.size(1)==128 && W.size(2)==8 && W.size(3)==1536,"W shape");
 qcheck(out,x,torch::kBFloat16,128*2304,"out");TORCH_CHECK(out.dim()==2 && out.size(0)==128 && out.size(1)==2304,"out shape");
 qcheck(xp,x,torch::kInt32,524288,"XP");qcheck(partial,x,torch::kFloat32,4*128*2304,"partial");
 TORCH_CHECK(esc.numel()>=576*32+32 && (esc.numel()-32)%576==0,"CSR size");
 const int cap=(esc.numel()-32)/576;TORCH_CHECK(cap%32==0 && cap<=32800,"CSR capacity bound");qcheck(esc,x,torch::kInt32,576LL*cap+32,"esc");
 // Reject every overlap, including views into a common allocation.
 torch::Tensor ts[]={W,x,out,esc,xp,partial};
 for(int i=0;i<6;i++)for(int j=0;j<i;j++){
  uintptr_t a=(uintptr_t)ts[i].data_ptr(),b=(uintptr_t)ts[j].data_ptr();
  TORCH_CHECK(a+ts[i].nbytes()<=b || b+ts[j].nbytes()<=a,"overlapping tensor storage");
 }
 TORCH_CHECK(tab>=0 && tab<=0xffffffffLL && ((tab&255)*0x01010101LL)==tab && (tab&255)<=120,"codec table");
 RqP p={};p.X=(const __nv_bfloat16*)x.data_ptr();p.XP=(const uint32_t*)xp.data_ptr();p.partial=partial.data_ptr<float>();p.W=W.data_ptr<uint8_t>();p.esc=(const uint32_t*)esc.data_ptr();p.ecap=cap;p.out=(__nv_bfloat16*)out.data_ptr();p.m=128;p.N=2304;p.K=8192;p.KC=128;p.NB=18;p.Tb=(uint32_t)tab;
 RqTmap tx={},to={};auto stream=c10::cuda::getCurrentCUDAStream();
 // No tensor map is consumed by this exact register-A/direct-store specialization.
 // Dynamic SMEM opt-in is performed by qkv_info(1,5) before graph capture.
 rx_pack<<<512,256,0,stream>>>((const uint32_t*)p.X,(uint32_t*)p.XP);C10_CUDA_KERNEL_LAUNCH_CHECK();
 rq_qkv_g8_sk4<128,8,1,0,0,0,20><<<dim3(36,4),512,RqC<128,8,1,0,0,0,20>::SMEM,stream>>>(p,tx,to);C10_CUDA_KERNEL_LAUNCH_CHECK();
 qkv_reduce<<<1152,256,0,stream>>>(p.partial,p.out);C10_CUDA_KERNEL_LAUNCH_CHECK();return 0;
}

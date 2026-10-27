// Append to frozen target-only pilot.cu. No Torch/ATen/Python headers.
static auto gv53_kernel(int arm) {
 return arm==0 ? gv_predxor<1,1,1,2,16,8,1,1,0,3> : arm==1 ? gv_keyed<1,1,1,2,16,8,1,1,0,3> : gv_earlyload<1,1,1,2,16,8,1,1,0,3>;
}
extern "C" int gv53_attrs(int arm,int field,int64_t* result) {
 if(arm<0||arm>2||!result)return -90;
 auto k=gv53_kernel(arm);cudaError_t rc=cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,17408);
 if(rc!=cudaSuccess)return int(rc);
 cudaFuncAttributes a;rc=cudaFuncGetAttributes(&a,k);if(rc!=cudaSuccess)return int(rc);
 switch(field){
 case 1:*result=a.numRegs;break;case 2:*result=a.localSizeBytes;break;
 case 3:*result=a.sharedSizeBytes;break;case 4:*result=a.maxDynamicSharedSizeBytes;break;
 case 5:*result=a.maxThreadsPerBlock;break;
 case 6:{int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,1024,17408);*result=n;break;}
 case 7:{int dev=0,n=0;rc=cudaGetDevice(&dev);if(rc==cudaSuccess)rc=cudaDeviceGetAttribute(&n,cudaDevAttrReservedSharedMemoryPerBlock,dev);*result=n;break;}
 case 8:*result=arm;break;default:return -91;
 }
 return int(rc);
}
extern "C" int gv53_run(int arm,const void* W,const void* esc,const void* x,void* out,void* ws,void* cnt,uint64_t table,int m,void* stream) {
 if(arm<0||arm>2||(m!=1&&m!=8)||(arm==1&&m!=1)||(arm==2&&m!=8)||!W||!esc||!x||!out||!ws||!cnt)return -90;
 G p{};p.W=(const uint4*)W;p.W2=p.W;p.esc=(const uint32_t*)esc;p.x=(const __nv_bfloat16*)x;p.out=out;p.ws=(float4*)ws;p.cnt=(int*)cnt;
 p.m=m;p.N=4096;p.K=8192;p.KC=128;p.T0=uint32_t(table);p.T1=uint32_t(table>>32);p.U0=p.T0;p.U1=p.T1;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((m+7)/8,128,1);cfg.blockDim=dim3(1024);cfg.dynamicSmemBytes=17408;cfg.stream=(cudaStream_t)stream;
 auto k=gv53_kernel(arm);cudaError_t rc=cudaLaunchKernelEx(&cfg,k,p);
 // Same immediate error return/clear as the original binding. Async failures
 // still propagate at the harness's mandatory Torch synchronization/terminal.
 if(rc!=cudaSuccess)cudaGetLastError();return int(rc);
}
extern "C" const char* gv53_error(int code){return code<0?"thin ABI argument error":cudaGetErrorString((cudaError_t)code);}

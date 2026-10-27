
template<class K> int64_t query(K k,int64_t f){
 auto rc=cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,99488);if(rc!=cudaSuccess)return -1000-int(rc);
 cudaFuncAttributes a;rc=cudaFuncGetAttributes(&a,k);if(rc!=cudaSuccess)return -2000-int(rc);
 if(f==1)return a.numRegs;if(f==2)return a.localSizeBytes;if(f==3)return a.sharedSizeBytes;if(f==4)return a.maxDynamicSharedSizeBytes;if(f==5)return a.maxThreadsPerBlock;
 if(f==6){int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,384,99488);return rc==cudaSuccess?n:-3000-int(rc);}return -9;
}
int64_t attrs(int64_t arm,int64_t f){if(arm==0)return query(rq_control::rq<32,6,1,0,0,0,35>,f);if(arm==1)return query(rq_offset::rq<32,6,1,0,0,0,35>,f);return -90;}
int64_t run(int64_t arm,torch::Tensor W,torch::Tensor x,torch::Tensor out,torch::Tensor esc,torch::Tensor cnt,int64_t KS,int64_t Tb,torch::Tensor prof,torch::Tensor sink){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.size(0)==32,"M32 CUDA input required");
 TORCH_CHECK((out.size(1)==4096&&x.size(1)==8192&&KS==4)||(out.size(1)==8192&&x.size(1)==4096&&KS==2),"only scored selected pooler/unpooler cells");
 if(arm==0)return rq_control::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);
 if(arm==1)return rq_offset::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);return -90;
}

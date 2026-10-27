"""Small delta from reviewed two-tile header; original CUDA math stays included unchanged."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
P=R/'serving/experiments/moe_rsi_20261027/anchor_marlin_tiles_v1/two_tiles.cuh';PS='515ba06d70d29fb0537de3069e5f1f5134a9e631628b88269de99a4ff1f4e12c'
OUT=R/'serving/experiments/moe_rsi_20261027/anchor_marlin_four_blocks_v1/four_blocks.cuh'
INFO='''
namespace sglang {
// Query only; no kernel launch. Every CUDA error is an unsupported receipt.
int64_t anchor_block_info(int64_t field) {
  int dev=0;
  if(cudaGetDevice(&dev)!=cudaSuccess)return -100;
  int regs_sm=0,shared_sm=0,shared_block=0,threads_sm=0,blocks_sm=0;
  if(cudaDeviceGetAttribute(&regs_sm,cudaDevAttrMaxRegistersPerMultiprocessor,dev)!=cudaSuccess)return -101;
  if(cudaDeviceGetAttribute(&shared_sm,cudaDevAttrMaxSharedMemoryPerMultiprocessor,dev)!=cudaSuccess)return -102;
  if(cudaDeviceGetAttribute(&shared_block,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev)!=cudaSuccess)return -103;
  if(cudaDeviceGetAttribute(&threads_sm,cudaDevAttrMaxThreadsPerMultiProcessor,dev)!=cudaSuccess)return -104;
  if(cudaDeviceGetAttribute(&blocks_sm,cudaDevAttrMaxBlocksPerMultiprocessor,dev)!=cudaSuccess)return -105;
  auto kernel=device::marlin_moe::get_marlin_kernel<__nv_bfloat16,true,false>(host::kFE2M1f,1,8,4,true,false,false,1,128,false);
  if(kernel==device::marlin_moe::MarlinDefault)return -106;
  cudaFuncAttributes attr{};
  if(cudaFuncGetAttributes(&attr,kernel)!=cudaSuccess)return -107;
  const int reserve=device::marlin_moe::kSharedMemoryLaunchReserve;
  const int dyn3=shared_block/3-reserve, dyn4=shared_block/4-reserve;
  if(dyn3<0 || dyn4<0)return -108;
  int occupancy3=0,occupancy4=0;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn3)!=cudaSuccess)return -109;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy3,kernel,128,dyn3)!=cudaSuccess)return -110;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn4)!=cudaSuccess)return -111;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy4,kernel,128,dyn4)!=cudaSuccess)return -112;
  const int cache=device::marlin_moe::get_kernel_cache_size({64,128,128},true,1,8,8192,4096,4,16,false,true,false,false);
  const int64_t regs4=int64_t(attr.numRegs)*128*4;
  const int64_t shared4=(int64_t(dyn4)+attr.sharedSizeBytes)*4;
  const bool legal=attr.maxThreadsPerBlock>=128 && attr.numRegs>0 && regs4<=regs_sm && shared4<=shared_sm &&
      threads_sm>=4*128 && blocks_sm>=4 && dyn4+attr.sharedSizeBytes<=shared_block &&
      cache+device::marlin_moe::kSharedMemoryValidityMargin<=dyn4 && occupancy4>=4;
  const int64_t values[]={legal,attr.numRegs,int64_t(attr.sharedSizeBytes),int64_t(attr.localSizeBytes),
      attr.maxThreadsPerBlock,regs_sm,shared_sm,shared_block,threads_sm,blocks_sm,dyn3,dyn4,cache,
      occupancy3,occupancy4,regs4,shared4,reserve};
  if(field<0 || field>=int(sizeof(values)/sizeof(values[0])))return -113;
  return values[field];
}
} // namespace sglang
'''
def edits():
 return [dict(name='query-declaration',before='// Diagnostic host controls only. Graphs retain their captured native launch.',after='int64_t anchor_block_info(int64_t field);\n// Diagnostic host controls only. Graphs retain their captured native launch.'),dict(name='new-single-mode',before='if(command>=0 && command<=2)',after='if(command>=0 && command<=3)'),dict(name='four-block-requery-before-force',before='  if(anchor_tile_mode){host::RuntimeCheck(forced_cfg.blocks_per_sm>0,"forced existing tile unsupported");exec_cfg=forced_cfg;}\n',after='''  if(anchor_tile_mode==3){
    host::RuntimeCheck(exec_cfg.tb_cfg.thread_k==64 && exec_cfg.tb_cfg.thread_n==128 && exec_cfg.tb_cfg.num_threads==128,
                       "four-block diagnostic requires current K64/N128/128t tile");
    host::RuntimeCheck(anchor_block_info(0)==1,"four-block runtime device/kernel occupancy unsupported");
    exec_cfg.blocks_per_sm=4;
  } else if(anchor_tile_mode){host::RuntimeCheck(forced_cfg.blocks_per_sm>0,"forced existing tile unsupported");exec_cfg=forced_cfg;}
'''),dict(name='query-only-export',before='\n}  // namespace sglang\n',after='\n}  // namespace sglang\n'+INFO)]
def apply(text,reverse=False):
 for e in reversed(edits())if reverse else edits():
  a,b=(e['after'],e['before'])if reverse else(e['before'],e['after']);assert text.count(a)==1,(e['name'],text.count(a));text=text.replace(a,b,1)
 return text
def check():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==PS
 s=OUT.read_text();assert apply(P.read_text())==s and apply(s,True)==P.read_text()
 return {'state':'CPU_FOCUSED_HOST_DELTA_INVERSE_PASS_NATIVE_GPU_UNRUN','reviewed_parent_sha256':PS,'variant_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'strict_forward_inverse':True,'unchanged_math_proof_reused':'tools/moe_marlin_anchor_tiles_v1/source-proof.json','native_factory_and_existing_two_instantiations_unchanged':True,'new_math':False,'new_query_kernel_launches':0,'actual_kernel_query':'cudaFuncGetAttributes; both3/4cudaOccupancyMaxActiveBlocksPerMultiprocessor; actualCUDAdevicecaps','forced_scope':'currentK64N128128t exactM8EP8up, mode3 only; requerybeforeforce','dynamic_policy':'same parent maxSharedPerBlockOptin/blocks minus1024 reserve; changes A-cache capacity','GPU_compile_calls':0}
if __name__=='__main__':
 OUT.write_text(apply(P.read_text()));(D/'delta.json').write_text(json.dumps({'edits':edits()},indent=2)+'\n');(D/'source-proof.json').write_text(json.dumps(check(),indent=2)+'\n');print(json.dumps(check(),indent=2))

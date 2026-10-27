#include "kernel.cuh"
template __global__ void moe_ss_lut_v2_fused::project<8,1>(const uint32_t*,const uint8_t*,const moe_ss_lut_v2_fused::B*,const moe_ss_lut_v2_fused::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_lut_v2_fused::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_lut_v2_fused::project<8,8>(const uint32_t*,const uint8_t*,const moe_ss_lut_v2_fused::B*,const moe_ss_lut_v2_fused::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_lut_v2_fused::B*,int,int,int,int,int,int,int,int);

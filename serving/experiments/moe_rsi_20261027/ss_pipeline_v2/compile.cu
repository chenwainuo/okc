#include "kernel.cuh"
template __global__ void moe_ss_v2::project<8,1>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<8,8>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<16,1>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<16,8>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<32,1>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<32,8>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<64,1>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);
template __global__ void moe_ss_v2::project<64,8>(const uint32_t*,const uint8_t*,const moe_ss_v2::B*,const moe_ss_v2::B*,const int32_t*,const int32_t*,const int32_t*,moe_ss_v2::B*,int,int,int,int,int,int,int,int);

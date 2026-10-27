#include <array>
#include <cstdint>
#include <cstdio>
#include <cassert>
using A=std::array<uint32_t,16>;
void original(A& a,uint32_t nx,int lane){
 if(lane==int((nx>>15)&31)){
  uint32_t r=(nx>>11)&15,hm=(nx>>9)&1,xr=(nx&127)<<(8+((nx>>10)&1)*16);
  for(int h=0;h<1;h++)for(int i=0;i<16;i++)if(i==int(r)&&h==int(hm))a[i]^=xr;
 }
}
// Interpreter of checked helper PTX: copied ri/mask, pl, pa and 16 setp.and/xor pairs.
void interpreted(A& a,uint32_t nx,int lane){
 uint32_t ri=(nx>>11)&15,mask=(nx&127)<<(8+((nx>>10)&1)*16);
 bool pl=lane==int((nx>>15)&31),pa=(((nx>>9)&1)==0)&&pl;
 for(int i=0;i<16;i++){bool pr=(ri==uint32_t(i))&&pa;if(pr)a[i]^=mask;}
}
int main(){
 uint64_t cases=0,chain=0;
 for(int lane=0;lane<32;lane++)for(int target=0;target<32;target++)for(int r=0;r<16;r++)for(int hm=0;hm<2;hm++)for(int half=0;half<2;half++)for(int x=0;x<128;x++){
  uint32_t nx=(uint32_t(target)<<15)|(r<<11)|(half<<10)|(hm<<9)|x;
  A a,b;for(int i=0;i<16;i++)a[i]=b[i]=0xdeadbeefu^(0x9e3779b9u*(i+lane+1));
  original(a,nx,lane);interpreted(b,nx,lane);assert(a==b);cases++;
 }
 // All legal 0..32-entry cursor lengths, duplicate writes and sequential XORs.
 for(int len=0;len<=32;len++)for(int seed=0;seed<128;seed++)for(int lane=0;lane<32;lane++){
  A a,b;for(int i=0;i<16;i++)a[i]=b[i]=0xcafebabeu^(i*0x12345u);
  uint32_t entries[32];for(int i=0;i<len;i++)entries[i]=(3u<<20)|((seed%32)<<15)|((seed%16)<<11)|((i&1)<<10)|((i%7==0)<<9)|((seed+i)%128);
  // Hardware shfl chooses identical warp-uniform entry; sentinel ends the run.
  uint32_t ep=0,nx=len?entries[0]:~0u;
  while((nx>>20)==3){original(a,nx,lane);interpreted(b,nx,lane);nx=++ep<uint32_t(len)?entries[ep]:~0u;}
  assert(a==b&&ep==uint32_t(len));chain++;
 }
 printf("{\"scalar_cases\":%llu,\"cursor_chain_cases\":%llu,\"state\":\"PASS\"}\n",(unsigned long long)cases,(unsigned long long)chain);
}

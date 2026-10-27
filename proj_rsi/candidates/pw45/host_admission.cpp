#include <cassert>
#include <initializer_list>
struct P{int m,N,K,KC,walias;};
template<int NT,int ST,int GS,int FL> int admission(P p,int KS){
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  return 0;
}
int main(){P p{32,8192,1024,16,0};assert((admission<32,4,1,52>(p,1)==0));assert((admission<32,5,1,52>(p,1)==0));assert((admission<32,4,1,52>(p,2)==-45));assert((admission<32,6,1,35>(p,2)==0));
for(int m:{16,31,33}){P q=p;q.m=m;assert((admission<32,4,1,52>(q,1)!=0));}
P q=p;q.N=4096;assert((admission<32,5,1,52>(q,1)==-45));q=p;q.K=4096;q.KC=64;assert((admission<32,5,1,52>(q,1)==-45));q=p;q.walias=1;assert((admission<32,5,1,52>(q,1)==-45));assert((admission<32,5,1,52>(p,0)==-4));assert((admission<32,5,1,52>(p,65)==-4));}

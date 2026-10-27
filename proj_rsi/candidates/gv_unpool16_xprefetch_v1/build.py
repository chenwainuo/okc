import hashlib,json,pathlib
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2]
prior=R/'proj_rsi/candidates/gv53_earlyload_v1'
delta=json.loads((prior/'delta.json').read_text());old=delta['old_kernel']
assert hashlib.sha256((R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_bytes()).hexdigest()==delta['parent_sha256']
new=old.replace(' gv_predxor(G p) {',' gv_xprefetch(G p) {',1)
a=new.index('  static_assert(');b=new.index('  // pw14 probes only',a)
new=new[:a]+'  static_assert(FMT==1 && MT==2 && ST==2 && WR==1 && WK==4 && KU==16 && S==1 && RT==1 && XS==2 && V==1,"only selected unpooler M16");\n'+new[b:]
anchor='  auto step = [&](const int u, const int s) {'
prime='''  uint4 next_x[MT][2];
#pragma unroll
  for (int n=0;n<MT;n++) { next_x[n][0]=gxp<(V >> 15) & 3>(xp[n]); next_x[n][1]=gxp<(V >> 15) & 3>(xp[n]+8); }
'''
assert new.count(anchor)==1;new=new.replace(anchor,prime+anchor,1)
block='''    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }'''
replace='''    if (XS == 2 && LX && !((V >> 25) & 1)) {
#pragma unroll
      for (int n=0;n<MT;n++) { xd[n][0]=next_x[n][0]; xd[n][1]=next_x[n][1]; }
      if (u+1<KU) {
#pragma unroll
        for (int n=0;n<MT;n++) { next_x[n][0]=gxp<(V >> 15) & 3>(xp[n]+(u+1)*64); next_x[n][1]=gxp<(V >> 15) & 3>(xp[n]+(u+1)*64+8); }
      }
    }'''
assert new.count(block)==1;new=new.replace(block,replace,1)
pilot=(prior/'pilot.cu').read_text();prefix=pilot[:pilot.index(old)]
cu=prefix+old+'\n'+new+'\n'
for f in ['gv_predxor','gv_xprefetch']:cu+='template __global__ void '+f+'<1,2,2,1,4,16,1,1,2,1>(G);\n'
(D/'pilot.cu').write_text(cu)
(D/'delta-v1.json').write_text(json.dumps({'parent_sha256':delta['parent_sha256'],'source':'one X slot primed with u0; current snapshot then load u+1 before decode/repair; W schedule unchanged','old_block':block,'new_block':replace,'prime':prime,'scope':[1,2,2,1,4,16,1,1,2,1]},indent=2)+'\n')

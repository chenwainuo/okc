"""W-only early refill, separate version; never edits frozen81fd or parent501d."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
P=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py';E=R/'proj_rsi/candidates/gv53_earlyload_v1'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def value(s):return next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
def build():
 base=P.read_text();assert sha(base)=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9';assert sha((E/'submission.py').read_text())=='81fdc825bc21a568db8082873892982fb16eff5fbbc894baed2211a58aef1791'
 cu=value(base);delta=json.loads((E/'delta.json').read_text());old=delta['new_kernel'];new=old;edits=[]
 def change(a,b):
  nonlocal new
  assert new.count(a)==1,a[:100];pos=new.index(a);new=new.replace(a,b,1);edits.append([a,b,pos])
 change(' gv_earlyload(G p) {',' gv_earlyload_wonly(G p) {')
 start=new.index('  auto ld = [&](int s, int u) {');end=new.index('  uint32_t bbar = 0, bring = 0;',start);ld=new[start:end]
 marker='    if (!XS && LX) {';wi=ld.index(marker)
 # Exact W and X expressions/loops preserved in separate same-signature helpers.
 wbody=ld[len('  auto ld = [&](int s, int u) {\n'):wi]
 xbody=ld[wi:-len('  };\n')]
 helpers='  auto ldw = [&](int s, int u) {\n'+wbody+'  };\n  auto ldx = [&](int s, int u) {\n'+xbody+'  };\n  auto ld = [&](int s, int u) { ldw(s,u); ldx(s,u); };\n'
 change(ld,helpers)
 change('    uint4 early_w[RT][NA][NL], early_x[MT][2];','    uint4 early_w[RT][NA][NL];')
 change('''#pragma unroll
    for (int n=0;n<MT;n++) { early_x[n][0]=xb[s%XB][n][0]; early_x[n][1]=xb[s%XB][n][1]; }
    if (u+ST<KU) ld(s,u+ST);''','    if (u+ST<KU) ldw(s,u+ST);')
 change('else { b0 = early_x[n][0]; b1 = early_x[n][1]; }','else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }')
 change('    if ((C12 || CE) && LDEC) {\n#pragma unroll\n      for (int j = 0; j < RT; j++)\n        while', '    if (u + ST < KU) ldx(s, u + ST);\n    if ((C12 || CE) && LDEC) {\n#pragma unroll\n      for (int j = 0; j < RT; j++)\n        while')
 back=new
 for a,b,pos in reversed(edits):assert back[pos:pos+len(b)]==b;back=back[:pos]+a+back[pos+len(b):]
 assert back==old
 insertion='\n'+new;route=delta['route'].replace('gv_earlyload<','gv_earlyload_wonly<');anchor='    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;\n'
 nc=cu.replace('\n__global__ void p1zero',insertion+'\n__global__ void p1zero',1).replace(anchor,anchor+route,1)
 out=base.replace('GVCU = r"""'+cu+'"""','GVCU = r"""'+nc+'"""',1).replace('load_inline("gvp45_predxor_v2_%x"','load_inline("gv53_earlyload_wonly_v1_%x"',1)
 inverse=out.replace('load_inline("gv53_earlyload_wonly_v1_%x"','load_inline("gvp45_predxor_v2_%x"',1).replace(insertion,'',1).replace(route,'',1);assert inverse==base
 pilot=(E/'pilot.cu').read_text()+'\n'+new+'\ntemplate __global__ void gv_earlyload_wonly<1,1,1,2,16,8,1,1,0,3>(G);\n'
 (D/'submission.py').write_text(out);(D/'pilot.cu').write_text(pilot);(D/'delta.json').write_text(json.dumps({'parent_sha256':sha(base),'W_X_parent_sha256':sha((E/'submission.py').read_text()),'candidate_sha256':sha(out),'W_only_from_W_X_kernel_edits':edits,'native_kernel':new,'route':route,'whole_file_inverse_to_501d':True,'native_inverse_to_W_X':True},indent=2)+'\n')
 print(sha(out))
if __name__=='__main__':build()

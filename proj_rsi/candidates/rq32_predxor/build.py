"""Isolated PW45 descendant; CPU source construction only."""
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw45'
PARENT_SHA='835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3'
PARENT_MANIFEST='988214460a032f11d620801f2661c32cbe9ca9572fe4f40ab0bbe931aa61ae02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def apply(text,edits,reverse=False):
 for e in reversed(edits) if reverse else edits:
  a,b=(e['after'],e['before']) if reverse else (e['before'],e['after'])
  assert text.count(a)==1,e['id'];text=text.replace(a,b,1)
 return text

def build():
 assert sha(PARENT/'submission.py')==PARENT_SHA and sha(PARENT/'manifest.json')==PARENT_MANIFEST
 old=(PARENT/'submission.py').read_text()
 anchor='__device__ __forceinline__ unsigned long long rq_gt()'
 helper='''// rq32_predxor: keep entry order/window; predicate the target register XOR directly.
__device__ __forceinline__ void rq_fix_pred(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    const uint32_t target = (nx >> 12) & 31, reg = (nx >> 8) & 15;
    const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
    asm volatile("{\\n.reg .pred pl, pr;\\n.reg .u32 mask, ri;\\nmov.u32 mask, %16;\\nmov.u32 ri, %19;\\nsetp.eq.u32 pl, %17, %18;\\n"
'''
 for i in range(16):helper+='      "setp.eq.and.u32 pr, ri, %d, pl; @pr xor.b32 %%%d, %%%d, mask;\\n"\n'%(i,i,i)
 helper+='''      "}\\n"
      : '''+', '.join('"+r"(A[%d])'%i for i in range(16))+'''
      : "r"(xr), "r"(lane), "r"(target), "r"(reg));
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
'''
 call='          rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);'
 replacement='''          if constexpr (NT == 32 && (ST == 5 || ST == 6) && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && FL == 35)
            rq_fix_pred(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);'''
 edits=[{'id':'append-distinct-predicated-helper','before':anchor,'after':helper+anchor},{'id':'selected-NT32-types-only','before':call,'after':replacement}]
 new=apply(old,edits);assert apply(new,edits,True)==old;ast.parse(new)
 (HERE/'submission.py').write_text(new)
 (HERE/'delta.json').write_text(json.dumps({'parent_source_sha256':PARENT_SHA,'parent_manifest_sha256':PARENT_MANIFEST,'edits':edits},indent=2)+'\n')
 return {'source_sha256':sha(HERE/'submission.py'),'parent_sha256':PARENT_SHA,'changed_scope':'only escape application in NT32/ST5-or6/FL35/GS1/WM0/INS0/PRB0 types; selectors untouched'}
if __name__=='__main__':print(json.dumps(build(),indent=2))

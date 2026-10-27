"""Strict narrow separate-W-ring rebase, CPU source operations only."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw36/submission.py'
HISTORICAL=HERE.parent/'rs128b/submission.py'
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)
def apply(text,delta,reverse=False):
    for edit in reversed(delta['edits']) if reverse else delta['edits']:
        before,after=(edit['after'],edit['before']) if reverse else (edit['before'],edit['after'])
        if text.count(before)!=1:raise ValueError('nonunique/missing anchor '+edit['id'])
        text=text.replace(before,after,1)
    return text

def build():
    source=PARENT.read_text();historical=HISTORICAL.read_text();cu=value(source,'RQCU');edits=[]
    def edit(name,before,after):
        assert source.count(before)==1,(name,source.count(before))
        edits.append(dict(id=name,before=before,after=after))
    edit('ring-config',
      '  static constexpr int SMEM = 1024 + ST * (XB + WB) + 24 * ST + 16;',
      '  static constexpr bool SPLITF = ((FL >> 8) & 255) != 0;\n  static constexpr int WS = SPLITF ? ((FL >> 8) & 255) : ST;\n  static constexpr int SMEM = SPLITF ? 1024 + ST * XB + WS * WB + 8 * (2 * ST + 2 * WS) + 16 : 1024 + ST * (XB + WB) + 24 * ST + 16;')
    edit('approved-domain','  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;',
      '  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;\n  constexpr bool SPLITF = C::SPLITF;\n  constexpr int WS = C::WS;\n  static_assert(!SPLITF || (NT == 128 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && (FL & ~65280) == 3 && ((ST == 5 && WS == 5) || (ST == 6 && WS == 4))), "Wring rebase only x5w5/x6w4 M128 HINT+ZI; no HALF/EARLY/EH/EP");')
    edit('ring-bases','  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;\n  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;',
      '  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + WS * WB;\n  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = wempty + 8 * WS, wfull0 = xempty + 8 * ST;\n  const uint32_t wfull = SPLITF ? wfull0 : full, zflag = SPLITF ? wfull0 + 8 * WS : bars + 24 * ST;')
    start='    for (int i = 0; i < ST; i++) {';stop='    asm volatile("st.shared.u32 [%0], 0;'
    a=cu.index(start);b=cu.index(stop,a);before=cu[a:b]
    after='''    if constexpr (SPLITF) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\\n" :: "r"(full + 8 * i));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      for (int i = 0; i < WS; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\\n" :: "r"(wfull0 + 8 * i));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(wempty + 8 * i), "r"(NW));
      }
    } else {
'''+before+'    }\n'
    edit('split-full-init',before,after)
    a=cu.index('    if (warp == 8 && lane == 0 && WON) {');b=cu.index('    } else if (warp == 9',a);before=cu[a:b]
    import re
    after=re.sub(r'\bST\b','WS',before);after=re.sub(r'\bfull\b','wfull',after)
    after=after.replace('if (SELF && !XON)','if (SELF && (!XON || SPLITF))')
    edit('W-producer-independent-slot-phase',before,after)
    a=cu.index('  auto ldr =');b=cu.index('  auto xrel =',a);before=cu[a:b]
    after=before.replace('s % ST','s % WS').replace('(s / ST)','(s / WS)').replace('rq_wait(full +','rq_wait(wfull +').replace(') % ST',') % WS')
    edit('W-read-and-release-independent-ring',before,after)
    before='    constexpr int NS = decltype(NSC)::value;\n    rq_facc<ND>(D);'
    after='''    constexpr int NS = decltype(NSC)::value;
    if constexpr (SPLITF && !RES) {
#pragma unroll
      for (int j = 0; j < NS; j++) { const int s = GS * G + j; rq_wait(full + 8 * (s % ST), (s / ST) & 1); }
    }
    rq_facc<ND>(D);'''
    edit('X-full-wait-before-math',before,after)
    edit('lab-selectable-only-variants',']\nRQI = {v[0]: i for i, v in enumerate(RQV)}',
      '    ("wring_x5w5hz", 128, 5, 1, 0, 0, 0, 1283),\n    ("wring_x6w4hz", 128, 6, 1, 0, 0, 0, 1027),\n]\nRQI = {v[0]: i for i, v in enumerate(RQV)}')
    delta=dict(version='pw36_wring_rebase_v1',parent_source_sha256=sha(source),historical_source_sha256=sha(historical),historical_RQCU_sha256=sha(value(historical,'RQCU')),edits=edits,
      arms={'coupled_x5w5':[128,5,1,0,0,0,3],'split_x5w5':[128,5,1,0,0,0,1283],'split_x6w4':[128,6,1,0,0,0,1027]},
      semantics='FL bits8..15 WS, independent Wfull/Wempty; Xfull/Xempty retains ST. Full-output HINT+ZI only. RQPROD/RQBAKE/prepare/selection unchanged; experimental variants only explicitly selected by separate lab.')
    candidate=apply(source,delta);assert apply(candidate,delta,True)==source
    (HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n');(HERE/'submission.py').write_text(candidate)
    return delta
if __name__=='__main__':build()

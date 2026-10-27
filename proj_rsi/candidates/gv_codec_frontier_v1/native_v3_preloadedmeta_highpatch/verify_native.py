"""Read-only six-function source inverse/native baseline audit. CPU only."""
import json
import re
from pathlib import Path

D=Path(__file__).resolve().parent
V2=D.with_name('native_v2_capacity32_48')
HIGH=D.with_name('native_v3_highpatch')
prefix='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__'
base=(V2/'native.cu').read_text();high=(HIGH/'native.cu').read_text();native=(D/'native.cu').read_text()
assert native.startswith(base+'\n')
a=native.rindex(prefix);b=native.index('\ntemplate __global__ void gv_frontier_preload',a)
candidate=native[a:b].strip()
delta=json.loads((D/'delta.json').read_text())
for change in reversed(delta['changes']):
    assert candidate.count(change['new'])==1
    candidate=candidate.replace(change['new'],change['old'])
candidate=candidate.replace(' gv_frontier_preload(G p, Frontier fp)',' gv_frontier(G p, Frontier fp)',1).replace('frontier_dec_preloaded(','frontier_dec(')
a=high.rindex(prefix);b=high.index('\ntemplate __global__ void gv_predxor',a)
highkernel=high[a:b].strip()
assert candidate==highkernel
# Restore the high-byte-patch placement into exact v2 source, independently.
a=base.rindex(prefix);b=base.index('\ntemplate __global__ void gv_predxor',a)
old=base[a:b].strip()
restore=highkernel.replace('    const uint32_t* desc=(const uint32_t*)(smem+L::SMEM+(w*ST+s)*FSLOT);\n    const unsigned short* fixes=(const unsigned short*)(smem+L::SMEM+(w*ST+s)*FSLOT+32);\n','',1)
for q in range(4):restore=restore.replace(', desc, fixes, '+str(q)+', lane);',');',1)
a=old.index('    // Consume every repair before ld() can overwrite this slot.')
b=old.index('    if constexpr (((V >> 25) & 1) != 0)',a)
restore=restore.replace('    __syncwarp();\n    if constexpr (((V >> 25) & 1) != 0)',old[a:b]+'    if constexpr (((V >> 25) & 1) != 0)',1)
assert restore==old

def streams(text):
    return {sym:[(int(m[1],16),re.sub(r'\s+',' ',m[2].strip()),m[3],m[4])for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',body)]for sym,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S)}

actual=streams((D/'native.sass').read_text());previous=streams((V2/'native.sass').read_text())
assert len(actual)==6 and len(previous)==4
for sym,rows in previous.items():assert actual[sym]==rows,sym
print(json.dumps(dict(status='SIX_KERNEL_BASELINES_AND_PRELOAD_HIGH_SOURCE_INVERSE_PASS_GPU_UNRUN',
    four_current_and_v2_native_controls_exact=True,all_arithmetic_input_reduction_output_unchanged=True,
    new_shared_transport_changes=len(delta['changes']),new_high_field_patch_scalar_proof='native_v3_highpatch/cpu-proof.json'),indent=2))

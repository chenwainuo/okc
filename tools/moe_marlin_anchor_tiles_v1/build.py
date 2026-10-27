"""Source-only narrow original-Marlin host chooser; exactly two existing math instantiations."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
P=R/'serving/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh'
OUT=R/'serving/experiments/moe_rsi_20261027/anchor_marlin_tiles_v1/two_tiles.cuh'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
CONTROL='''
// Diagnostic host controls only. Graphs retain their captured native launch.
static int64_t anchor_tile_mode=0, anchor_tile_default=0, anchor_tile_selected=0, anchor_tile_hits=0;
int64_t anchor_tile_control(int64_t command) {
  if(command>=0 && command<=2){anchor_tile_mode=command;anchor_tile_default=anchor_tile_selected=anchor_tile_hits=0;return command;}
  if(command==-1)return anchor_tile_default;
  if(command==-2)return anchor_tile_selected;
  if(command==-3)return anchor_tile_hits;
  return -99;
}
'''
def edits():
 s=P.read_text();start=s.index('  COMMON_GET_IF(host::kU4)\n',s.index('MarlinFuncPtr get_marlin_kernel('));end=s.index('\n  return kernel;',start)
 return [dict(name='host-control',before='namespace sglang {\n',after='namespace sglang {\n'+CONTROL),dict(name='only-two-existing-NVFP4-math-instantiations',before=s[start:end],after='''  _GET_IF(host::kFE2M1f, 1, 8, 8, true, 1, 256, false)
  _GET_IF(host::kFE2M1f, 1, 8, 4, true, 1, 128, false)
'''),dict(name='tiny-host-scope',before='  int count = 0;\n  constexpr int device_max_reg_size',after='''  const bool anchor_scope = std::is_same_v<scalar_t, __nv_bfloat16> && kIsEP && !kHasBias &&
      q_type==host::kFE2M1f && group_size==16 && prob_m==8 && prob_n==8192 && prob_k==4096 && top_k==8 &&
      thread_m_blocks==1 && m_block_size_8 && !has_act_order && is_k_full && !has_zp && !is_zp_float;
  host::RuntimeCheck(anchor_scope, "two-tile diagnostic only accepts exact M8 EP8 gate/up NVFP4 path");
  exec_config_t forced_cfg = exec_config_t{0, thread_config_t{-1,-1,-1}};
  int count = 0;
  constexpr int device_max_reg_size'''),dict(name='reuse-original-validity-attrs-and-residency',before='    if (allow_count > count) {\n',after='''    if (anchor_tile_mode==i+1) forced_cfg={allow_count,th_config};
    if (allow_count > count) {
'''),dict(name='record-default-before-force',before='  return exec_cfg;\n}\n\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nvoid marlin_mm(',after='''  auto encode=[](exec_config_t c)->int64_t{return int64_t(c.blocks_per_sm)*1000000000LL+
      int64_t(c.tb_cfg.num_threads)*1000000LL+int64_t(c.tb_cfg.thread_n)*1000LL+c.tb_cfg.thread_k;};
  anchor_tile_default=encode(exec_cfg);
  if(anchor_tile_mode){host::RuntimeCheck(forced_cfg.blocks_per_sm>0,"forced existing tile unsupported");exec_cfg=forced_cfg;}
  anchor_tile_selected=encode(exec_cfg);++anchor_tile_hits;
  return exec_cfg;
}

template <typename scalar_t, bool kIsEP, bool kHasBias>
void marlin_mm(''')]
def apply(text,reverse=False):
 for e in reversed(edits())if reverse else edits():
  a,b=(e['after'],e['before'])if reverse else(e['before'],e['after']);assert text.count(a)==1,(e['name'],text.count(a));text=text.replace(a,b,1)
 return text
def check():
 old=P.read_text();new=OUT.read_text();assert apply(old)==new and apply(new,True)==old
 assert new.count('  _GET_IF(host::kFE2M1f, 1, 8,')==2
 # The math is included from the original, source-pinned unedited headers.
 assert '#include "kernel.h"'in new and '#include "marlin_template.h"'in new
 return {'state':'CPU_SOURCE_INVERSE_PASS_GPU_UNRUN','parent':str(P.relative_to(R)),'parent_sha256':sha(P),'variant':str(OUT.relative_to(R)),'variant_sha256':sha(OUT),'exact_forward_inverse':True,'math_headers_unchanged':[{'path':str(p.relative_to(R)),'sha256':sha(p)}for p in(P.with_name('kernel.h'),P.with_name('marlin_template.h'))],'native_Marlin_instantiations':2,'only_scope':['BF16','EP','no_bias','NVFP4_group16','m8','n8192','k4096','topk8','blockM8','no_act_order','no_zp','fullK'],'original_residency_and_legal_config_filters':True,'same_return_ABI':True,'compile_GPU_calls':0}
if __name__=='__main__':
 OUT.write_text(apply(P.read_text()));(D/'delta.json').write_text(json.dumps({'edits':edits()},indent=2)+'\n');(D/'source-proof.json').write_text(json.dumps(check(),indent=2)+'\n');print(json.dumps(check(),indent=2))

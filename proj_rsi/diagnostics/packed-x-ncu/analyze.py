"""Reproduce packed-X PC/memory accounting; CPU-only, no GPU or source mutations."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re

P=Path(__file__).resolve().parent
PREFIX='packed-x-ncu01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(v):
    try:return float(v.replace(',',''))
    except (ValueError,AttributeError):return None

source=[]
for row in csv.reader(io.StringIO((P/(PREFIX+'.source.csv')).read_text())):
    if row and row[0]=='Kernel Name':source.append({'name':row[1],'rows':[]});continue
    if not row or not source:continue
    if row[0]=='Address':source[-1]['header']=row;continue
    if row[0].startswith('0x'):source[-1]['rows'].append(dict(zip(source[-1]['header'],row)))
assert len(source)==2
raw_text=(P/(PREFIX+'.raw.csv')).read_text();raw_text=raw_text[raw_text.index('"ID"'):]
raw_rows=list(csv.DictReader(io.StringIO(raw_text)));units=[r for r in raw_rows if not r['ID']][0]
raw={r['ID']:r for r in raw_rows if r['ID'] in ('0','1')};assert len(raw)==2
jit=(P/(PREFIX+'-driver.jit.sass')).read_text()
jit_tables={('main' if 'rq_px' in x.splitlines()[0] else 'packer'):x for x in re.split(r'\n\s*Function : ',jit)[1:]}

def plain(x):return re.sub(r'\s+',' ',x.strip().replace('.reuse',''))
def jit_rows(text):return {int(m[1],16):m[2].strip() for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',text)}

def total(rows):
    out={'all':sum(r['all'] for r in rows),'not_issued':sum(r['ni'] for r in rows),
         'execution_counts':sum(r['exe'] for r in rows),'PC_count':len(rows)}
    for key in source[0]['header']:
        if key.startswith('stall_') and 'Not Issued' not in key:out[key]=sum(int(r['raw'][key]) for r in rows)
    out['omitted_all_reason_residual']=out['all']-sum(v for k,v in out.items() if k.startswith('stall_'))
    out['omitted_not_issued_reason_residual']=out['not_issued']-sum(int(r['raw'][k]) for r in rows for k in source[0]['header'] if k.startswith('stall_') and 'Not Issued' in k)
    return out

math_ranges=[(0xa70,0x1da0),(0x2ee0,0x2fd0)]
producer_ranges=[(0x1db0,0x2ed0),(0x2fe0,0x3140)]
def inside(pc,ranges):return any(lo<=pc<=hi for lo,hi in ranges)
bfull={0xc50,0xc90,0xf80,0x1000,0x11f0,0x1260,0x14c0,0x1540}
wfull={0x2140,0x2160};bempty={0x21f0,0x2200}

def role(pc):
    if inside(pc,math_ranges):return 'math'
    if inside(pc,producer_ranges):return 'producer'
    return 'common_setup_unreachable'

def phase(pc,ins):
    rr=role(pc)
    if rr=='math':
        if pc in bfull or 0x2ee0<=pc<=0x2fd0:return 'math_Bfull_wait'
        if 'LDG.E.NA.128' in ins:return 'math_direct_X_LDG'
        if 'WARPGROUP.ARRIVE' in ins:return 'math_register_proxy_fence'
        if 'HGMMA.' in ins:return 'math_HGMMA'
        if 'WARPGROUP.DEPBAR.LE' in ins:return 'math_wait_group'
        if 0xa70<=pc<0xb80:return 'math_setup'
        if pc>=0x1870:return 'math_drain_output'
        return 'math_other_loop_address_release_control'
    if rr=='producer':
        if pc in wfull or 0x3050<=pc<=0x3080:return 'producer_Wfull_wait'
        if pc in bempty or 0x3090<=pc<=0x30c0:return 'producer_Bempty_wait'
        if 0x1db0<=pc<=0x2030 or 0x2fe0<=pc<=0x3040:return 'producer_setup'
        if 0x2040<=pc<=0x2150:return 'producer_W_index_address'
        if 0x2170<=pc<=0x21e0:return 'producer_Bempty_index_address'
        if 0x2210<=pc<=0x29d0 or 0x30d0<=pc<=0x3140:return 'producer_LDS_decode_escape'
        if 0x29e0<=pc<=0x2c00:return 'producer_store_address_STS'
        if 0x2c10<=pc<=0x2c70:return 'producer_store_proxy_bar3_convergence'
        if 0x2c80<=pc<=0x2e90:return 'producer_publish_refill'
        if 0x2ea0<=pc<=0x2ed0:return 'producer_loop_control'
        raise AssertionError(('unclassified producer',hex(pc),ins))
    return 'common_setup_unreachable'

kernels={}
for tab in source:
    label='main' if 'rq_px' in tab['name'] else 'packer';rid='1' if label=='main' else '0'
    base=min(int(r['Address'],16) for r in tab['rows']);jr=jit_rows(jit_tables[label]);parsed=[];mismatches=[]
    for r in tab['rows']:
        pc=int(r['Address'],16)-base;ins=r['Source'].strip()
        rel=re.sub(r'0x[0-9a-f]+',lambda m:hex(int(m[0],16)-base) if base<=int(m[0],16)<=base+max(jr) else m[0],ins)
        if plain(rel)!=plain(jr[pc]):mismatches.append({'PC':pc,'source':rel,'JIT':jr[pc]})
        assert r['# Samples']==r['Warp Stall Sampling (All Samples)']
        parsed.append({'PC':pc,'instruction':ins,'all':int(r['# Samples']),'ni':int(r['Warp Stall Sampling (Not-issued Samples)']),
                       'exe':int(r['Instructions Executed']),'raw':r})
    assert not mismatches,mismatches[:5]
    totals=total(parsed);assert totals['all']==int(raw[rid]['smsp__pcsamp_sample_count'])
    assert totals['omitted_all_reason_residual']==int(raw[rid]['smsp__pcsamp_warps_issue_stalled_warpgroup_arrive'])
    assert totals['omitted_not_issued_reason_residual']==int(raw[rid]['smsp__pcsamp_warps_issue_stalled_warpgroup_arrive_not_issued'])
    for r in parsed:
        r['role']=role(r['PC']) if label=='main' else 'packer'
        r['phase']=phase(r['PC'],r['instruction']) if label=='main' else 'packer'
    roles={name:total([r for r in parsed if r['role']==name]) for name in sorted({r['role'] for r in parsed})}
    phases={name:total([r for r in parsed if r['phase']==name]) for name in sorted({r['phase'] for r in parsed})}
    assert sum(r['all'] for r in roles.values())==totals['all'];assert sum(r['all'] for r in phases.values())==totals['all']
    main_keys=['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
       'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum',
       'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio','dram__bytes_read.sum','dram__bytes_write.sum',
       'lts__t_sectors_srcunit_tex_op_read.sum','lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum','lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum',
       'smsp__inst_executed.sum','smsp__sass_inst_executed_op_global_ld.sum','smsp__sass_inst_executed_op_shared_ld.sum','smsp__sass_inst_executed_op_shared_st.sum',
       'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio','smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
       'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio','smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio',
       'smsp__warps_eligible.avg.per_cycle_active','smsp__pcsamp_aggregated_passes','smsp__pcsamp_buffer_size_bytes','smsp__pcsamp_dropped_bytes',
       'smsp__pcsamp_interval_cycles','gpu__time_duration.sum']
    metrics={k:{'value':number(raw[rid][k]),'raw':raw[rid][k],'unit':units[k]} for k in main_keys if k in raw[rid]}
    top=[]
    for r in sorted(parsed,key=lambda x:x['all'],reverse=True)[:40]:
        top.append({'offset':hex(r['PC']),'instruction':r['instruction'],'role':r['role'],'phase':r['phase'],
                    **total([r])})
    kernels[label]={'ID':rid,'name':tab['name'],'absolute_function_base':hex(base),'PC_count':len(parsed),
       'canonical_source_matches_actual_JIT':True,'totals':totals,'roles':roles,'phases':phases,'top_PCs':top,'metrics':metrics}

main=jit_rows(jit_tables['main'])
# Validate mapped role split, main branch slowpaths, producer slowpaths, and return targets.
assert plain(main[0xa60])=='@P2 BRA 0x1db0'
expected={0xc90:0x2ee0,0x1000:0x2f20,0x1260:0x2f60,0x1540:0x2fa0,0x2160:0x3050,0x2200:0x3090,0x1de0:0x2fe0,0x2970:0x30d0}
for pc,target in expected.items():assert main[pc].endswith(hex(target)),(pc,main[pc])
returns={0x2f10:0xca0,0x2f50:0x1010,0x2f90:0x1270,0x2fd0:0x1550,0x3040:0x1e00,0x3080:0x2170,0x30c0:0x2210,0x3140:0x2990}
for pc,target in returns.items():assert main[pc]=='BRA '+hex(target)
for pc in (0xc50,0xf80,0x11f0,0x14c0):assert '+0x19030]' in main[pc]
assert '+0x19000]' in main[0x2140] and '+0x19070]' in main[0x21f0]

out={'state':'VALID_REPRODUCIBLE_PC_ROLE_PHASE_ACCOUNTING','purpose':'COUNTERS_PC_ASSOCIATION_ONLY_NO_TIME_DECOMPOSITION',
 'analysis_program_sha256':sha(Path(__file__)),'artifacts':{name:sha(P/name) for name in [PREFIX+'.raw.csv',PREFIX+'.source.csv',PREFIX+'.session.txt',PREFIX+'.host.log',PREFIX+'.ncu-rep',PREFIX+'-driver.json',PREFIX+'-driver.jit.sass']},
 'source_sha256':'2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a',
 'role_ranges_relative_to_main_base':{'math':math_ranges,'producer':producer_ranges,'common':'all other offsets'},
 'CFG_branch_targets_verified':{hex(k):hex(v) for k,v in expected.items()},'CFG_return_targets_verified':{hex(k):hex(v) for k,v in returns.items()},
 'kernels':kernels,'sample_gate':{'role_all_min':200,'phase_all_min':30},
 'decision':{'immediate_A6':'HOLD; mathBfull handoff is largest namedphase, registerfence is not uniquelyX','next':'One root-admitted producerstore/handoff causaltest; unchanged bytes/barriers; no guarantee'},
 'limits':['34metricpasses with2PCaggregatedpasses, uncontrolledclocks; periodiccorrelatedsamples','WARPGROUP.ARRIVE fencesregisters includingA/acc; longSB there is not uniquelyX','Sourceexecutioncounts ignorepredicateactive-lane semantics and are not samplecounts','PackerDRAMwrite0 does not mean missingpackstores; full1MiBdriverbyteoraclePASS','NoautomaticA6implementation or additionalprofile from thisanalysis']}
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'state':out['state'],'kernels':{k:{'totals':v['totals'],'roles':v['roles'],'phases':v['phases']} for k,v in kernels.items()}},indent=2))

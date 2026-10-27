"""CPU-only four-node DP/G8 actual-JIT association accounting; never time decomposition."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

DRIVER_SHA = '245b78898a16c2087b00924cd809cd33fb74bfd7c19c9012bd7d040005485702'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(x):
    if x in ('', '-', 'n/a', 'N/A', None): return None
    try: value = float(x.replace(',', ''))
    except ValueError: return None
    return value if math.isfinite(value) else None

def source_sections(path):
    sections=[]
    for row in csv.reader(path.open()):
        if not row: continue
        if row[0]=='Kernel Name': sections.append({'kernel':row[1],'header':None,'rows':[]})
        elif row[0]=='Address': sections[-1]['header']=row
        elif row[0].startswith('0x'):
            assert len(row)==len(sections[-1]['header'])
            sections[-1]['rows'].append(dict(zip(sections[-1]['header'],row)))
    assert len(sections)==4
    assert all(token in section['kernel'] for token,section in zip(['rx_pack','rq_px_dp','rx_pack','rq_px_g8'],sections))
    return sections

def instruction_rows(sass, token):
    tables=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if token in t.splitlines()[0]]
    assert len(tables)==1
    return {int(m[1],16):m[2].strip() for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',tables[0])}

def bind_section(section, instructions):
    base=min(int(r['Address'],16) for r in section['rows'])
    rows={int(r['Address'],16)-base:r for r in section['rows']}
    def canonical(text, relocate=False):
        if relocate:
            text=re.sub(r'0x[0-9a-f]+',lambda m:hex(int(m[0],16)-base) if base<=int(m[0],16)<=base+max(instructions)+16 else m[0],text)
        return re.sub(r'\s+','',text.replace('.reuse','').strip().rstrip(';'))
    assert set(rows)==set(instructions)
    for pc,r in rows.items(): assert canonical(r['Source'],True)==canonical(instructions[pc]), (hex(pc), r['Source'], instructions[pc])
    return rows,base

def main_analysis(section, entry, instructions):
    rows,base=bind_section(section,instructions)
    m=entry['PC_map'][0]
    roles={key:{int(p,16) for p in m['roles'][key+'_PCs']} for key in ['math','producer','common_setup_unreachable']}
    assert not roles['math']&roles['producer'] and set.union(*roles.values())==set(rows)
    reasons=[k for k in section['header'] if k.startswith('stall_') and '(Not Issued)' not in k]
    def stats(pcs):
        get=lambda key:sum(number(rows[pc].get(key)) or 0 for pc in pcs)
        out={key:get(key) for key in ['# Samples','Warp Stall Sampling (Not-issued Samples)','Instructions Executed','Thread Instructions Executed','Predicated-On Thread Instructions Executed']+reasons}
        out['samples']=out.pop('# Samples');out['PC_count']=len(pcs)
        out['unexported_reason_residual']=out['samples']-sum(out[k] for k in reasons)
        return out
    phases={}
    bfull=set()
    branch_records=[]
    def wait_record(q):
        record={kind:{'PC':q[kind],'instruction':instructions[int(q[kind],16)],**stats({int(q[kind],16)})} for kind in ['fast_try_PC','failed_entry_branch_PC','slow_try_PC','failed_retry_branch_PC']}
        a,b,c=[record[k] for k in ['failed_entry_branch_PC','failed_retry_branch_PC','slow_try_PC']]
        denom=a['Thread Instructions Executed'];a['failed_lane_fraction']=a['Predicated-On Thread Instructions Executed']/denom if denom else None
        record['slow_test_lane_flow_residual']=c['Thread Instructions Executed']-a['Predicated-On Thread Instructions Executed']-b['Predicated-On Thread Instructions Executed']
        return record
    for q in m['math_Bfull_waits']:
        bfull.update(int(q[k],16) for k in ['fast_try_PC','failed_entry_branch_PC'])
        bfull.update(int(r['PC'],16) for r in q['outlined_instructions'])
        branch_records.append(wait_record(q))
    phases['math_Bfull']=bfull
    phases['math_fence']={int(p,16) for p in m['math_fence_PCs']}
    phases['math_wait1']={int(p,16) for p in m['math_retirement_wait1_PCs']}
    phases['math_tail_wait0']={pc for pc in roles['math'] if 'WARPGROUP.DEPBAR.LE' in instructions[pc]}-phases['math_wait1']
    phases['math_HGMMA']={pc for pc in roles['math'] if 'HGMMA.' in instructions[pc]}
    phases['math_loop_X_LDG128']={int(p,16) for p in m['math_loop_X_LDG128_PCs']}
    phases['math_loop_head_MOV']={pc for pc in roles['math'] if instructions[pc]=='MOV R0, 0x400'}
    assert len(phases['math_loop_head_MOV'])==1
    phases['math_other']=roles['math']-set.union(*phases.values())
    # Source-bound barrier-address constants distinguish Wfull19000 and Bempty19070.
    producer_waits={}
    for label,offset in [('Wfull','0x19000'),('Bempty','0x19070')]:
        fast=[pc for pc in roles['producer'] if 'TRYWAIT' in instructions[pc] and offset in instructions[pc]]
        # One fast path, one outlined retry: fast entry target points to outlined head.
        candidates=[]
        for pc in fast:
            pred=re.search(r'TRYWAIT (P\d+),',instructions[pc])[1]
            branch=next(((p,re.fullmatch(r'@!'+pred+r' BRA 0x([0-9a-f]+)',instructions[p])) for p in sorted(rows) if pc<p<=pc+64 and re.fullmatch(r'@!'+pred+r' BRA 0x([0-9a-f]+)',instructions[p])),None)
            if branch and int(branch[1][1],16)>branch[0]:candidates.append((pc,branch[0],int(branch[1][1],16)))
        assert len(candidates)==1,(label,candidates)
        fastpc,entrypc,target=candidates[0];block=[]
        for pc in sorted(rows):
            if pc<target:continue
            block.append(pc)
            ret=re.fullmatch(r'BRA 0x([0-9a-f]+)',instructions[pc])
            if ret:
                assert int(ret[1],16)==entrypc+16
                break
        slow=[pc for pc in block if 'TRYWAIT' in instructions[pc]]
        retry=[pc for pc in block if re.fullmatch(r'@!P\d+ BRA 0x'+format(target,'x'),instructions[pc])]
        assert len(slow)==len(retry)==1
        q=dict(zip(['fast_try_PC','failed_entry_branch_PC','slow_try_PC','failed_retry_branch_PC'],map(hex,[fastpc,entrypc,slow[0],retry[0]])))
        producer_waits[label]=wait_record(q)
        phases['producer_'+label]={fastpc,entrypc}|set(block)
    used=set.union(*phases.values())
    for label,test in [('proxy_barriers',lambda ins:'FENCE.VIEW.ASYNC' in ins or 'BAR.SYNC' in ins or 'WARPSYNC' in ins or 'BSYNC' in ins),('scalar_STS',lambda ins:bool(re.search(r'\bSTS(?:\.| )',ins))),('LDS_decode_loads',lambda ins:bool(re.search(r'\bLDS(?:\.| )',ins)))]:
        pcs={pc for pc in roles['producer']-used if test(instructions[pc])}
        phases['producer_'+label]=pcs;used|=pcs
    phases['producer_other_addresses_decode_publish_control']=roles['producer']-used
    phases['common_setup_unreachable']=roles['common_setup_unreachable']
    seen=set()
    for pcs in phases.values():assert not seen&pcs;seen|=pcs
    assert seen==set(rows)
    role_stats={k:stats(v) for k,v in roles.items()};phase_stats={k:stats(v) for k,v in phases.items()}
    for k,v in phase_stats.items():
        role='math' if k.startswith('math_') else 'producer' if k.startswith('producer_') else 'common_setup_unreachable'
        denom=role_stats[role]['samples'];v['fraction_of_role_samples']=v['samples']/denom if denom else None
    failed=sum(q['failed_entry_branch_PC']['Predicated-On Thread Instructions Executed'] for q in branch_records)
    tested=sum(q['failed_entry_branch_PC']['Thread Instructions Executed'] for q in branch_records)
    top=sorted(rows,key=lambda pc:number(rows[pc].get('# Samples')) or 0,reverse=True)[:32]
    return {'source_all_rows_equal_actual_JIT':True,'source_row_count':len(rows),'source_absolute_base':hex(base),'totals':stats(set(rows)),'roles':role_stats,'phases':phase_stats,'Bfull_failed_lane_records':branch_records,'Bfull_failed_lane_fraction':failed/tested if tested else None,'producer_wait_records':producer_waits,
            'top32_PCs':[{'PC':hex(pc),'instruction':instructions[pc],**stats({pc})} for pc in top],
            'sufficiency':{'math_min200':role_stats['math']['samples']>=200,'producer_min200':role_stats['producer']['samples']>=200,'leading_phase_min30':{k:v['samples']>=30 for k,v in phase_stats.items() if k in ['math_fence','math_Bfull','producer_Bempty','math_loop_head_MOV']}},
            'limits':'Samples associate exposure with PCs, not time or intrinsic latency. Failed predicates count lanes, TRYWAIT may block, no exact warp count from /32.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--prefix',default='group8-compare-ncu01');p.add_argument('--output',type=Path);args=p.parse_args()
    here=args.directory;prefix=args.prefix
    driver=json.loads((here/(prefix+'-driver.json')).read_text())
    assert driver['driver_sha256']==DRIVER_SHA and driver['status']=='COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED'
    assert driver['graph_cleanup']==['RESET_AFTER_DRAIN']*2
    sections=source_sections(here/(prefix+'.source.csv'))
    rr=list(csv.reader((here/(prefix+'.raw.csv')).open()));assert len(rr)==6
    units=dict(zip(rr[0],rr[1]));raw=[dict(zip(rr[0],r)) for r in rr[2:]]
    assert [r['ID'] for r in raw]==['0','1','2','3']
    result={'state':'VALID_FOUR_NODE_ACTUAL_JIT_CPU_ACCOUNTING','purpose':'COUNTERS_PC_ASSOCIATIONS_NOT_PERFORMANCE','analysis_sha256':sha(__file__),'artifacts':{f.name:sha(f) for f in here.iterdir() if f.is_file() and f.suffix in ['.csv','.sass','.log'] or f.is_file() and f.name==prefix+'-driver.json'},'variants':{},'raw_kernel_metrics':[]}
    for i,(variant,token) in enumerate([('DP','rq_px_dp'),('G8','rq_px_g8')]):
        entry=driver['variants'][variant]
        expected_pins={'DP':('97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b','281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241'),'G8':('e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e','2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7')}
        assert (entry['source_sha256'],entry['manifest_sha256'])==expected_pins[variant]
        assert entry['selected_oracles']['XP']=='EXACT_ALL_1MiB' and entry['selected_oracles']['cnt']=='ZERO' and entry['selected_oracles']['W_pack_esc_X']=='IMMUTABLE'
        jit=here/Path(entry['JIT_artifact']).name;assert sha(jit)==entry['JIT_sha256']
        sass=jit.read_text()
        pack=instruction_rows(sass,'rx_pack');packrows,packbase=bind_section(sections[2*i],pack)
        result['variants'][variant]=main_analysis(sections[2*i+1],entry,instruction_rows(sass,token))
        result['variants'][variant]['pack_binding']={'source_all_rows_equal_actual_JIT':True,'source_row_count':len(packrows),'base':hex(packbase),'totals':{k:sum(number(row.get(k)) or 0 for row in packrows.values()) for k in ['# Samples','Instructions Executed','Thread Instructions Executed','Predicated-On Thread Instructions Executed']}}
    for i,r in enumerate(raw):
        assert ['rx_pack','rq_px_dp','rx_pack','rq_px_g8'][i] in r['Kernel Name']
        assert any(':profile:' in v for k,v in r.items() if 'Domain:' in k)
        vals={k:number(v) for k,v in r.items() if k in units and number(v) is not None}
        if i%2:
            assert vals['launch__block_dim_x']==512 and vals['launch__registers_per_thread']==122 and vals['launch__grid_dim_x']==128
            assert vals['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum']==263168 and vals['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']==4198400
        result['raw_kernel_metrics'].append({'name':r['Kernel Name'],'values':vals,'units':units})
    prior=here.parent/'dual-producer-ncu/analysis.json'
    if prior.is_file():
        old=json.loads(prior.read_text());result['prior_DP_capture']={'analysis_sha256':sha(prior),'totals':old['totals'],'roles':old['roles'],'phases':old['phases'],'limit':'Historical capture; retain its own denominators, no raw-sample subtraction as timing.'}
    result['limits']=['Cold per-node profiling differs from fresh XP consumption. No duration benchmark; no sample fraction to microseconds.','Raw/source execution totals belong to separate replay passes and variable polling; retain both without forcing identity.','Compare DP/G8 and historical parent denominators separately. This single capture gives associations, not causal proof or independent repetitions.']
    output=args.output or here/'analysis.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{'roles':{k:d['samples'] for k,d in r['roles'].items()},'phases':{k:d['samples'] for k,d in r['phases'].items()},'Bfull_failed_lane_fraction':r['Bfull_failed_lane_fraction'],'sufficiency':r['sufficiency']} for v,r in result['variants'].items()},indent=2))
if __name__=='__main__':main()

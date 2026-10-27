"""Read-only PW41 source/current-JIT/codegen/protocol gates; no compiler/GPU."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import HERE,PARENT,PARENT_SHA,TARGETS,LATE,apply,catalog,value
from codegen import streams,details
from protocol_proof import proof as protocol

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def named(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for t in node.targets:
                if isinstance(t,ast.Name):result[t.id]=ast.dump(node,include_attributes=False)
    return result

def source_proof():
    old=PARENT.read_text();new=(HERE/'submission.py').read_text();d=json.loads((HERE/'delta.json').read_text())
    assert sha(PARENT)==PARENT_SHA==d['parent_source_sha256']
    assert apply(old,d['edits'])==new and apply(new,d['edits'],True)==old
    a,b=named(old),named(new);assert sorted(k for k in a if a[k]!=b.get(k))==['GVCU']
    assert set(b)-set(a)=={'_P1LT','P1LATE'}
    for key in ('prepare','kernel_fn','_gvcall','_gxbake','_gvprep','_gvext','_time_us','_rqstart','RQCU','RWCU','RQPROD','RQSBAKE','MARGIN','PREPARE_BUDGET_S'):
        assert a[key]==b[key],key
    oc,nc=catalog(old),catalog(new);assert len(oc['GVI'])==95 and len(nc['GVI'])==99
    assert nc['GVI'][:95]==oc['GVI'] and nc['GXP'][:len(oc['GXP'])]==oc['GXP']
    assert nc['GVI'][95:]==[t[:-1]+(t[-1]|LATE,) for t in TARGETS]
    cells=[]
    for row in nc['P1LATE']:
        name,layer,ms,geo,v=row
        for m in ms:cells.append([layer,m])
    assert sorted(cells)==sorted([['router',8],['router',16],['router',32],['unpooler',1],['unpooler',8]])
    negatives=[]
    for e in d['edits']:
        try:apply(old.replace(e['before'],e['before']+'\n'+e['before'],1),d['edits'])
        except AssertionError:negatives.append(e['id'])
        else:raise AssertionError('duplicate source anchor accepted')
    from proj_rsi.static_check import check_source
    lint=check_source(new,str(HERE/'submission.py'));assert lint['ok'],lint
    return dict(parent_sha256=PARENT_SHA,candidate_sha256=sha(HERE/'submission.py'),strict_inverse='PASS',changed_existing_AST=['GVCU'],old_GVI_prefix=95,new_GVI_total=99,old_GXP_prefix=len(oc['GXP']),new_unique_types=4,new_bake_cells=cells,strict_anchor_negatives=negatives,lint=lint,
        invariants='All original functions/old catalog order/indices/prepare-clock/budget/fallbacks unchanged; only compile-time ticket placement and append-only optional P1 catalog.')

def machine_proof():
    folder=HERE/'full-codegen-v1';result=json.loads((folder/'result.json').read_text());assert result['status']=='COMPILE_AND_OLD_CONTROL_IDENTITY_PASS'
    old=streams((folder/'control.sass').read_text());new=streams((folder/'candidate.sass').read_text())
    assert len(old)==95 and len(new)==99
    for a,s in old.items():assert new[a]==s
    oldres=details((folder/'control.log').read_text());newres=details((folder/'candidate.log').read_text())
    assert all(newres[a]==r for a,r in oldres.items())
    prior=json.loads((HERE/'current-JIT-proof.json').read_text());p=HERE.parents[2]/prior['actual_JIT_path']
    assert sha(p)==prior['actual_JIT_sha256'];actual=streams(p.read_text());assert len(actual)==95
    assert all(actual[a]==s for a,s in old.items())
    rows=[]
    for a in TARGETS:
        late=a[:-1]+(a[-1]|LATE,);items=new[late];before=old[a]
        assert all(newres[late][k]==0 for k in ('stack','spill_stores','spill_loads'))
        ticket=[pc for pc,i in items if 'ATOMG.E.ADD.STRONG.GPU' in i and not re.search(r'\bPT, RZ,',i)]
        flags=[pc for pc,i in items if 'ATOMG.E.ADD.STRONG.GPU' in i and re.search(r'\bPT, RZ,',i)]
        mma=[pc for pc,i in items if 'HMMA.' in i];oldmma=[pc for pc,i in before if 'HMMA.' in i]
        acquire=[pc for pc,i in items if 'LDG.E.STRONG.GPU' in i]
        barriers=[pc for pc,i in items if 'BAR.SYNC' in i and ticket[0]<pc]
        reductions=[pc for pc,i in items if 'ATOMG.E.ADD.F32' in i or 'REDG.E.ADD.BF16x2' in i]
        assert len(ticket)==len(flags)==1 and len(mma)==len(oldmma)
        assert max(mma)<ticket[0]<min(acquire)<max(acquire)<min(barriers)<min(reductions)
        assert any('BPT.TRAP' in i for pc,i in items) and any('MEMBAR.SC.GPU' in i for pc,i in items)
        assert len(reductions)==4
        # Source P1 guard masks are unchanged by the new high bit.
        assert ((a[-1]>>27)&5)==((late[-1]>>27)&5) and ((a[-1]>>26)&15)==((late[-1]>>26)&15)
        _,mt,_,wr,wk,_,s,rt,_,v=a;threads=wr*wk*32;regs=newres[late]['registers'];rounded=(regs+7)//8*8
        blocks=min(65536//(rounded*threads),2048//threads,32)
        tiles=16 if a[0]==2 else 64;grid=tiles*(s+bool((v>>29)&1));shared=(wk+1)*(wr*rt*mt*32)*16
        rows.append(dict(args=a,late_args=late,ticket_PC=hex(ticket[0]),last_HMMA_PC=hex(max(mma)),first_acquire_PC=hex(min(acquire)),pre_reduction_CTA_barrier_PC=hex(min(barriers)),first_global_reduction_PC=hex(min(reductions)),HMMA_count=len(mma),integer_atomics=2,global_reduction_sites=4,registers=regs,control_registers=oldres[a]['registers'],rounded_registers=rounded,register_pool=rounded*threads,shared_bytes=shared,threads=threads,static_blocks_per_SM_UPPER_BOUND_register_thread_limits_only=blocks,grid_CTAs=grid,necessary_SM_count_from_that_upper_bound=(grid+blocks-1)//blocks))
    return dict(existing_GV_SASS_exact=95,existing_resource_records_exact=95,current_actual_JIT_GV_exact=95,targets=rows,
        limits='Linear PC/source-qualified gate, not arbitrary divergent-CFG or allocator proof. Zeroer stores/publication can be in an out-of-line block; ticket return value distinguishes it from odd flag. Runtime exact-JIT/occupancy/oracles remain mandatory.')

def proof():
    source=source_proof();machine=machine_proof();p=protocol();retained=json.loads((HERE/'protocol-proof.json').read_text());assert p==retained
    return dict(status='CPU_PW41_SOURCE_MACHINE_EPOCH_STORAGE_PASS_GPU_UNRUN',source=source,machine=machine,protocol=p,GPU_contexts=0,extension_loads=0,prepare_calls=0,performance='UNMEASURED; ticket can move onto final acquire critical path or increase counter contention; no atomic-count gain.')
if __name__=='__main__':print(json.dumps(proof(),indent=2))

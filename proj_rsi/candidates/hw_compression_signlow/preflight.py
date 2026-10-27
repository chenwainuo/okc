"""CPU all-bit exact XOR mapping + frozen-reader/protocol dependency + four-case parser gates."""
import ast
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def imported(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def function_ast(path,name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    assert not list(HERE.glob('*.cu')) and not list(HERE.glob('*.cpp')), 'reader must be reused, no new CUDA/C++'
    subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
    D=imported(HERE/'driver.py','hw_signlow_cpu_driver');reader=D.load_reader()
    L=imported(D.NATIVE,'hw_signlow_cpu_native');L.verify_helper_binding(L.HELPER_DIR,L.HELPER_MANIFEST_SHA)
    fixed=imported(D.PATTERNS/'cpu_patterns.py','hw_signlow_frozen_bit_proof')
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16)
    b=bits.view(torch.uint8);enc=D.signlow_encode(b);back=D.signlow_decode(enc)
    assert torch.equal(back,b) and torch.equal(D.signlow_encode(D.signlow_decode(b)),b)
    assert np.unique(enc.view(torch.int16).numpy().view(np.uint16)).size==65536
    assert np.array_equal(enc.view(torch.int16).numpy().view(np.uint16),fixed.encode_xor(np.arange(65536,dtype=np.uint32)))
    assert torch.equal(back.view(torch.int16),bits)
    assert bool(torch.all(torch.bitwise_not(enc)!=enc))
    byte_rejections=0
    for bad in [torch.bitwise_not(enc),enc.clone()]:
        if torch.equal(bad,enc):bad[-1]^=1
        try:reader.exact_bytes(bad,enc)
        except AssertionError:byte_rejections+=1
        else:raise AssertionError('byte oracle accepted bad output')
    P=imported(HERE/'parse_report.py','hw_signlow_cpu_parser')
    # Core counter/schema guards are AST-identical to frozen protocol except fixed new case/payload names.
    for name in ['number','parse_text','validate_run']:
        assert function_ast(HERE/'parse_report.py',name).replace('signlow','planes')==function_ast(D.READER/'parse_report.py',name)
    assert (HERE/'metrics.txt').read_bytes()==(D.READER/'metrics.txt').read_bytes()
    assert (HERE/'metrics.txt').read_text().strip().split(',')==P.METRICS
    shell=(HERE/'run.sh').read_text().replace('hw_compression_signlow','hw_compression_byteplanes').replace('hw-signlow01','hw-byteplanes01')
    shell=shell.replace('signlow_generic/','planes_generic/').replace('signlow_none/','planes_none/')
    assert shell==(D.READER/'run.sh').read_text(), 'changed profiler/cache/clock/launch policy'
    driver=(HERE/'driver.py').read_text();tree=ast.parse(driver)
    attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'Case' in attrs and 'copy_extension' in attrs and not {'prepare','lt_add','_ext','load_inline','measure','_time_us'} & set(attrs)
    assert 'decoded_gpu = signlow_decode(signlow)' in driver and 'decoded_gpu.cpu(), native_snapshot' in driver
    header=['ID','Kernel Name','thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg']+P.METRICS
    units=dict.fromkeys(header,'')
    for k in P.PHYSICAL:units[k]='byte' if 'bytes' in k else 'sector'
    for k in P.L2[:2]:units[k]='sector'
    units[P.L2[2]]='%';units[P.ILC[4]]='%';units[P.LAUNCH[6]]='register/thread';units[P.LAUNCH[7]]='byte/block'
    rows=[]
    for i,label in enumerate(P.LABELS):
        r=dict.fromkeys(header,'0');r.update({'ID':str(i),'Kernel Name':P.KERNEL,header[2]:'<default domain>:'+label+':none:none:none:none:none:none'})
        for direction in ['read','write']:
            sectors=1000 if direction=='read' else 100
            if label=='signlow_generic' and direction=='read':sectors=750
            r[f'dram__sectors_{direction}.sum']=str(sectors);r[f'dram__bytes_{direction}.sum']=str(32*sectors)
        for k,v in zip(P.LAUNCH,[128,1,1,256,1,1,32,1024]):r[k]=str(v)
        for k in P.ILC[3:]:r[k]='n/a'
        rows.append(r)
    def csv_text(u,rs,cols=header):
        out=io.StringIO();w=csv.DictWriter(out,fieldnames=cols,quoting=csv.QUOTE_ALL)
        w.writeheader();w.writerow({k:u[k] for k in cols});w.writerows([{k:r[k] for k in cols} for r in rs]);return out.getvalue()
    assert P.parse_text('Docker banner\n'+csv_text(units,rows))['ratio_of_allocation_ratios_signlow_over_raw']==.75
    negatives=[csv_text(dict(units,**{'dram__bytes_write.sum':'Mbyte'}),rows),csv_text(units,rows[:3]),csv_text(units,rows+[rows[0]]),
               csv_text(units,[rows[0],dict(rows[1],**{header[2]:rows[0][header[2]]})]+rows[2:]),
               csv_text(units,[dict(rows[0],**{'Kernel Name':'different'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'dram__bytes_read.sum':'NaN'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'dram__bytes_read.sum':'32001'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'launch__registers_per_thread':'31'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'launch__grid_dim_x':'127'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'lrc__ilc_input_sectors.sum':'n/a'})]+rows[1:]),
               csv_text(units,rows,[k for k in header if k!='lts__t_sectors.sum']),
               csv_text(units,rows[:2]+[dict(rows[2],**{header[2]:rows[2][header[2]].replace('signlow','planes')})]+rows[3:])]
    rejects=0
    for bad in negatives:
        try:P.parse_text(bad)
        except(AssertionError,KeyError):rejects+=1
        else:raise AssertionError('accepted invalid report')
    # Actual prior successful schema, renamed for new payload, supplies a CPU-only validation fixture.
    # It is a fixture only, explicitly not a new GPU result or evidence that GPU XOR was tested.
    unknown={'retain_result':1,'compression':None,'meaning':'UNKNOWN'}
    run={'status':'DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED','profile_order':list(P.LABELS),'allocation_spans_disjoint':True,
         'postprofile_sources_immutable':True,'cleanup':{'status':'CHECKED_CLOSE_PASS','counts_after_close':[{'Region_owners':0,'tensor_defined':False} for _ in P.LABELS]},
         'copy_identity':{'kernel':P.KERNEL,'bytes':67108864,'source_sha256':D.digest(D.READER/'copy_kernel.cu'),'bindings_sha256':D.digest(D.READER/'bindings.cpp')},
         'copy_resources':{'localSizeBytes':0,'sharedSizeBytes':0,'occupancy_blocks_per_SM':1,'numRegs':32},
         'jit_resources':{'registers':32,'stack':0,'shared':0,'local':0},
         'payloads':{'raw':{'sha256':D.WT_SHA},'signlow':{'inverse_to_raw_exact':True,'GPU_full_payload_inverse_to_raw_exact':True}},
         'reader_manifest_sha256':D.READER_MANIFEST_SHA,'CPU_patterns_manifest_sha256':D.PATTERNS_MANIFEST_SHA,
         'format':{'name':'SIGNLOW_XOR16','BASE':D.BASE,'arithmetic_precision_change':False},
         'checks':[{'case':label,'complement_poison_overwritten':True,'stage':'IMMEDIATELY_AFTER_SELECTED_ACTUAL_GRAPH','full64MiB_bytes_exact':True,'input_immutable':True,'repair_replay':False} for label in P.LABELS],
         'preprofile_checks':[{'case':label,'complement_poison_overwritten':True,'eager_and_actual_graph_bytes_exact':True,'input_immutable':True} for label in P.LABELS],
         'fresh_output_checks':[{'case':label,'complement_poison_overwritten':True,'full64MiB_bytes_exact':True,'output_properties':unknown} for label in P.LABELS],
         'allocations':[{'label':label,'info':{'actual_compression':int(label.endswith('_generic')),'requested_compression':int(label.endswith('_generic')),'bytes':67108864},'properties':{'compression':int(label.endswith('_generic')),'retain_result':0}} for label in P.LABELS],
         'outputs':{label:{'properties':unknown} for label in P.LABELS}}
    P.validate_run(run);P.validate_signlow_metadata(run)
    bads=[copy.deepcopy(run) for _ in range(6)]
    bads[0]['checks'][0]['repair_replay']=True;bads[1]['cleanup']['counts_after_close'][0]['Region_owners']=1;bads[2]['fresh_output_checks'].pop()
    bads[3]['payloads']['signlow']['GPU_full_payload_inverse_to_raw_exact']=False;bads[4]['format']['BASE']=0;bads[5]['reader_manifest_sha256']='0'*64
    for bad in bads:
        try:P.validate_run(bad);P.validate_signlow_metadata(bad)
        except AssertionError:pass
        else:raise AssertionError('accepted invalid protocol/format record')
    result={'state':'CPU_ALL_BITS_FROZEN_READER_PARSER_PROTOCOL_PASS_GPU_UNRUN','CUDA_API_calls':0,
            'all65536bit_patterns_both_directions':True,'same_frozen_NumPy_XOR_encoding':True,
            'no_new_CUDA_CPP':True,'frozen_Case_copy_extension_reused':True,'parser_core_AST_equal_except_fixed_names':True,
            'same21metrics_profiler_policy':True,'byte_oracle_negatives':byte_rejections,'CSV_negatives':rejects,
            'graph_lifetime_format_schema_negatives':6,'reader_manifest_sha256':D.READER_MANIFEST_SHA,
            'CPU_patterns_manifest_sha256':D.PATTERNS_MANIFEST_SHA,'GPU_fullpayload_inverse':'REQUIRED_FIRST_RUNTIME_GATE_UNRUN'}
    (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    names=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json')
    manifest={'state':'ONE_SIGNLOW_XOR_FOUR_NODE_COUNTER_FROZEN_FOR_REVIEW_GPU_UNRUN',
              'dependencies':{'reader_manifest_sha256':D.READER_MANIFEST_SHA,'CPU_patterns_manifest_sha256':D.PATTERNS_MANIFEST_SHA,
                              'reference_sha256':D.digest(D.REFERENCE),'native_lab_sha256':D.NATIVE_SHA,'helper_manifest_sha256':L.HELPER_MANIFEST_SHA},
              'sha256':{name:D.digest(HERE/name) for name in names}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('MANIFEST',D.digest(HERE/'manifest.json'))


if __name__=='__main__':main()

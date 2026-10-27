"""CPU-only complete-bit/layout proof, local compile/SASS, four-case parser/unit controls."""
from __future__ import annotations
import ast
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CUDA = Path('/usr/local/cuda-12.8')
REAL = HERE.parent / 'hw_compression_counters'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def imported(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    for p in HERE.glob('*.py'): ast.parse(p.read_text())
    subprocess.run(['bash', '-n', str(HERE/'run.sh')], check=True)
    D = imported(HERE/'driver.py', 'hw_planes_cpu_driver'); P = imported(HERE/'parse_report.py', 'hw_planes_cpu_parser')
    assert sha(D.NATIVE) == D.NATIVE_SHA
    L = imported(D.NATIVE, 'hw_planes_cpu_frozen_native')
    L.verify_helper_binding(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    assert (HERE/'metrics.txt').read_bytes() == (REAL/'metrics.txt').read_bytes()
    assert (HERE/'metrics.txt').read_text().strip().split(',') == P.METRICS and len(P.METRICS) == 21
    # Every u16 bit pattern; no floating conversion of NaN payloads.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    b = bits.view(torch.uint8)
    encoded = D.byteplanes64(b)
    assert torch.equal(D.unplanes64(encoded), b)
    assert torch.equal(D.byteplanes64(D.unplanes64(b)), b)
    assert torch.equal(D.unplanes64(encoded).view(torch.int16), bits)
    indices = np.arange(128).reshape(64, 2).T.copy().reshape(-1)
    assert np.array_equal(np.sort(indices), np.arange(128))
    for value in range(64):
        for plane in range(2): assert indices[plane*64+value] == 2*value+plane
    # Enumerate every16B load/store address: one-to-one full64MiB coverage, no tail.
    vectors = np.arange(128*256, dtype=np.uint32)[:, None] + np.arange(128, dtype=np.uint32)[None, :] * (128*256)
    assert vectors.size == 4194304 and vectors.max() == 4194303
    assert np.array_equal(np.sort(vectors.reshape(-1)), np.arange(4194304, dtype=np.uint32))
    poisoned = torch.bitwise_not(b)
    assert bool(torch.all(poisoned != b))
    oracle_rejections = 0
    for corrupted in (poisoned, b.clone()):
        if torch.equal(corrupted, b): corrupted[-1] ^= 1
        try: D.exact_bytes(corrupted, b)
        except AssertionError: oracle_rejections += 1
        else: raise AssertionError('full byte oracle accepted corruption')
    kernel = (HERE/'copy_kernel.cu').read_text()
    assert kernel.count('ld.volatile.global.v4.u32') == 1 and kernel.count('st.volatile.global.v4.u32') == 1
    assert 'kVectors = 67108864u / 16u' in kernel
    command = [str(CUDA/'bin/nvcc'), '-O3', '-std=c++17', '-lineinfo', '-arch=sm_90a', '--ptxas-options=-v',
               '--cubin', str(HERE/'copy_kernel.cu'), '-o', str(HERE/'copy.sm90a.cubin')]
    build = subprocess.run(command, check=True, text=True, capture_output=True)
    (HERE/'build.txt').write_text(build.stdout+build.stderr)
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in build.stderr
    resource = subprocess.run([str(CUDA/'bin/cuobjdump'), '--dump-resource-usage', str(HERE/'copy.sm90a.cubin')],
                              check=True, text=True, capture_output=True).stdout
    match = re.search(r'Function okc_hwc_copy64m:\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', resource)
    assert match and list(map(int, match.groups())) == [32, 0, 0, 0]
    sass = subprocess.run([str(CUDA/'bin/cuobjdump'), '--dump-sass', str(HERE/'copy.sm90a.cubin')],
                          check=True, text=True, capture_output=True).stdout
    (HERE/'copy.sass').write_text(sass)
    assert 'LDG.E.128.STRONG.SYS' in sass and 'STG.E.128.STRONG.SYS' in sass
    assert sass.count('LDG.E.128.STRONG.SYS') == sass.count('STG.E.128.STRONG.SYS') == 5
    assert not re.search(r'\b(?:LDL|STL)\b', sass)
    subprocess.run([str(CUDA/'bin/nvcc'), '-O3', '-std=c++17', '-lineinfo', '-arch=sm_90a', '--ptx',
                    str(HERE/'copy_kernel.cu'), '-o', str(HERE/'copy.ptx')], check=True, capture_output=True)
    ptx = (HERE/'copy.ptx').read_text()
    assert 'ld.volatile.global.v4.u32' in ptx and 'st.volatile.global.v4.u32' in ptx
    # Compile only host syntax against installed CPU Torch headers and CUDA12.8; no link/load/API call.
    from torch.utils.cpp_extension import include_paths
    cxx = ['g++', '-fsyntax-only', '-O2', '-std=c++17', '-DTORCH_EXTENSION_NAME=hw_planes_cpu_syntax']
    for include in include_paths() + [str(CUDA/'include'), sysconfig.get_paths()['include']]: cxx += ['-I', include]
    host = subprocess.run(cxx + [str(HERE/'bindings.cpp')], text=True, capture_output=True)
    (HERE/'host-syntax.txt').write_text(host.stdout+host.stderr)
    assert host.returncode == 0, host.stderr
    proof = {'state':'CPU_COMPILE_PASS_GPU_UNRUN', 'command':command, 'registers':32, 'stack_bytes':0,
             'spill_load_bytes':0, 'spill_store_bytes':0, 'shared_bytes':0, 'local_bytes':0,
             'kernel_source_sha256':sha(HERE/'copy_kernel.cu'), 'cubin_sha256':sha(HERE/'copy.sm90a.cubin'),
             'ptx_sha256':sha(HERE/'copy.ptx'), 'sass_sha256':sha(HERE/'copy.sass'),
             'host_syntax_pass':True, 'SASS_volatile_LDG128_sites':5, 'SASS_volatile_STG128_sites':5,
             'note':'Compiler peels/unrolls loop; each volatile vector load is stored. All4194304addresses covered exactly. Actual CUDA JIT/runtime attrs and launch remain GPU gates.'}
    (HERE/'build-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    header = ['ID', 'Kernel Name', 'thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'] + P.METRICS
    units = dict.fromkeys(header, '')
    for k in P.PHYSICAL: units[k] = 'byte' if 'bytes' in k else 'sector'
    for k in P.L2[:2]: units[k] = 'sector'
    units[P.L2[2]] = '%'; units[P.ILC[4]] = '%'
    units[P.LAUNCH[6]] = 'register/thread'; units[P.LAUNCH[7]] = 'byte/block'
    rows=[]
    for i,label in enumerate(P.LABELS):
        r=dict.fromkeys(header,'0'); r['ID']=str(i); r['Kernel Name']=P.KERNEL
        r[header[2]]='<default domain>:'+label+':none:none:none:none:none:none'
        for direction in ['read','write']:
            sectors = (1000 if direction == 'read' else 100)
            if label == 'planes_generic' and direction == 'read': sectors=750
            r[f'dram__sectors_{direction}.sum']=str(sectors); r[f'dram__bytes_{direction}.sum']=str(sectors*32)
        for k,v in zip(P.LAUNCH,[128,1,1,256,1,1,32,1024]): r[k]=str(v)
        for k in P.ILC[3:]: r[k]='n/a'
        rows.append(r)
    def csv_text(u,rs,cols=header):
        out=io.StringIO(); w=csv.DictWriter(out,fieldnames=cols,quoting=csv.QUOTE_ALL)
        w.writeheader();w.writerow({k:u[k] for k in cols});w.writerows([{k:r[k] for k in cols} for r in rs]);return out.getvalue()
    assert P.parse_text('Docker banner\n'+csv_text(units,rows))['ratio_of_allocation_ratios_planes_over_raw']==.75
    negatives=[csv_text(dict(units,**{'dram__bytes_write.sum':'Mbyte'}),rows),
               csv_text(units,rows[:3]),csv_text(units,rows+[rows[0]]),
               csv_text(units,[rows[0],dict(rows[1],**{header[2]:rows[0][header[2]]})]+rows[2:]),
               csv_text(units,[dict(rows[0],**{'Kernel Name':'different'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'dram__bytes_read.sum':'NaN'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'dram__bytes_read.sum':'32001'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'launch__registers_per_thread':'31'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'launch__grid_dim_x':'127'})]+rows[1:]),
               csv_text(units,[dict(rows[0],**{'lrc__ilc_input_sectors.sum':'n/a'})]+rows[1:]),
               csv_text(units,rows,[k for k in header if k!='lts__t_sectors.sum'])]
    rejected=0
    for bad in negatives:
        try:P.parse_text(bad)
        except (AssertionError,KeyError):rejected+=1
        else:raise AssertionError('parser accepted invalid schema/units/node/identity')
    unknown={'retain_result':1,'compression':None,'meaning':'UNKNOWN'}
    run={'status':'DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED','profile_order':list(P.LABELS),
         'allocation_spans_disjoint':True,'postprofile_sources_immutable':True,
         'cleanup':{'status':'CHECKED_CLOSE_PASS','counts_after_close':[{'Region_owners':0,'tensor_defined':False} for _ in P.LABELS]},
         'copy_identity':{'kernel':P.KERNEL,'bytes':67108864},
         'copy_resources':{'localSizeBytes':0,'sharedSizeBytes':0,'occupancy_blocks_per_SM':1,'numRegs':32},
         'jit_resources':{'registers':32,'stack':0,'shared':0,'local':0},
         'payloads':{'raw':{},'planes':{'inverse_to_raw_exact':True}},
         'checks':[{'case':label,'complement_poison_overwritten':True,'stage':'IMMEDIATELY_AFTER_SELECTED_ACTUAL_GRAPH',
                    'full64MiB_bytes_exact':True,'input_immutable':True,'repair_replay':False} for label in P.LABELS],
         'preprofile_checks':[{'case':label,'complement_poison_overwritten':True,'eager_and_actual_graph_bytes_exact':True,
                              'input_immutable':True} for label in P.LABELS],
         'fresh_output_checks':[{'case':label,'complement_poison_overwritten':True,'full64MiB_bytes_exact':True,
                                 'output_properties':unknown} for label in P.LABELS],
         'allocations':[{'label':label,'info':{'actual_compression':int(label.endswith('_generic')),
                                              'requested_compression':int(label.endswith('_generic')),'bytes':67108864},
                         'properties':{'compression':int(label.endswith('_generic')),'retain_result':0}} for label in P.LABELS],
         'outputs':{label:{'properties':unknown} for label in P.LABELS}}
    P.validate_run(run)
    bad_runs=[copy.deepcopy(run) for _ in range(3)]
    bad_runs[0]['checks'][0]['repair_replay']=True
    bad_runs[1]['cleanup']['counts_after_close'][0]['Region_owners']=1
    bad_runs[2]['fresh_output_checks'].pop()
    for bad in bad_runs:
        try:P.validate_run(bad)
        except AssertionError:pass
        else:raise AssertionError('parser accepted invalid graph/lifetime record')
    shell=(HERE/'run.sh').read_text()
    assert all('--nvtx-include '+label+'/' in shell for label in P.LABELS)
    assert all(flag in shell for flag in ['--launch-count 4','--cache-control all','--clock-control none','--kill no'])
    result={'state':'CPU_PROOF_COMPILE_PARSER_PASS_GPU_UNRUN','CUDA_API_calls':0,
            'all65536u16_bitpatterns_roundtrip':True,'byte_index_map_bijection':True,'full64MiB_vector_coverage':True,
            'byte_oracle_negative_controls':oracle_rejections,'four_node_parser_negative_controls':rejected,
            'actual_graph_lifetime_schema_negative_controls':3,'UNKNOWN_output_compression_preserved':True,
            'same_supported_metrics':21,'registers':32,'stack_spill_local_shared_bytes':0,'host_syntax':True,
            'capture_stream':'actual torch.cuda.current_stream inside each capture','formats':['raw','byteplanes64']}
    (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    names=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json')
    manifest={'state':'FOUR_NODE_BYTEPLANE_COUNTER_DIAGNOSTIC_FROZEN_FOR_REVIEW_GPU_UNRUN',
              'dependencies':{'native_lab_sha256':D.NATIVE_SHA,'helper_manifest_sha256':L.HELPER_MANIFEST_SHA,
                              'reference_sha256':sha(D.REFERENCE),'metrics_parent_manifest_sha256':sha(REAL/'manifest.json')},
              'sha256':{n:sha(HERE/n) for n in names}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('MANIFEST',sha(HERE/'manifest.json'))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Isolated runtime Marlin execution-config sweep; never modifies serving sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from optimize_loop import ROOT, SOURCE, adapter, atomic, gate

CSRC = ROOT / 'serving/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe'
OPS = ROOT / 'serving/sglang/python/sglang/kernels/ops/moe/moe_wna16_marlin.py'
BATCHES = [8, 32, 128, 256, 512, 753]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected unique anchor: {old!r}')
    return source.replace(old, new)


def generate(directory, narrow=False):
    source = (CSRC / 'moe_wna16_marlin.cuh').read_text()
    if narrow:
        old = """thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128}};"""
        source = replace_once(source, old, old.replace('{64, 128, 128}};', '{64, 128, 128}, {128, 64, 128}};'))
        lines = source.splitlines()
        index = next(i for i,line in enumerate(lines) if line.strip().startswith('NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128)'))
        lines.insert(index+1, '  NVFP4_GET_IF_M1(W_TYPE, 4, 8, 128)    ' + chr(92))
        source = '\n'.join(lines) + '\n'
        # Preserve the automatic control: it still searches only original entries.
        source = replace_once(source, 'i < thread_configs_size;', 'i < (config_index < 0 ? min(thread_configs_size, 2) : thread_configs_size);')
    for header in ('kernel.h', 'marlin_template.h'):
        source = replace_once(source, f'#include "{header}"', f'#include "{CSRC / header}"')
    source = replace_once(source, '    int sms) {',
                          '    int sms, int config_index, int block_limit) {')
    source = replace_once(source, '    thread_config_t th_config = thread_configs[i];',
                          '    if (config_index >= 0 && i != config_index) continue;\n'
                          '    thread_config_t th_config = thread_configs[i];')
    source = replace_once(source, '    if (allow_count > count) {',
                          '    if (block_limit > 0) allow_count = min(allow_count, block_limit);\n'
                          '    if (allow_count > count) {')
    anchor = '    bool use_fp32_reduce,\n    bool is_zp_float) {'
    if source.count(anchor) != 2:
        raise ValueError('Expected internal and exported signatures')
    source = source.replace(anchor, '    bool use_fp32_reduce,\n'
                            '    bool is_zp_float, int64_t config_index, int64_t block_limit) {')
    source = replace_once(source, '        sms);', '        sms, config_index, block_limit);')
    before, after = source.rsplit('      is_zp_float);', 1)
    source = before + '      is_zp_float, config_index, block_limit);' + after
    header = directory / 'execution.cuh'
    header.write_text(source)
    ops = OPS.read_text()
    ops = replace_once(ops, '"moe_wna16_marlin",', '"okc_execution_sweep",')
    ops = replace_once(ops, 'cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],',
                       f'cuda_files=[{str(header)!r}],')
    ops = replace_once(ops, '    is_zp_float: bool = False,',
                       '    is_zp_float: bool = False,\n    config_index: int = -1,\n    block_limit: int = -1,')
    ops = replace_once(ops, '        is_zp_float,\n    )',
                       '        is_zp_float,\n        config_index,\n        block_limit,\n    )')
    (directory / 'execution_variant.py').write_text(ops)
    return {'cuda_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'ops_sha256': hashlib.sha256(ops.encode()).hexdigest()}


def candidate_source(source, directory, up, down, blocks, max_tokens=512):
    source = replace_once(source,
        'from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm',
        'from execution_variant import moe_wna16_marlin_gemm')
    anchor = '            is_zp_float=False,\n'
    if source.count(anchor) != 2:
        raise ValueError('Expected two GEMMs')
    # Leave prefill and all precision flags unchanged. Runtime args permit one JIT
    # compilation for all policies, while baseline still loads the original JIT.
    parts = source.split(anchor)
    source = parts[0] + anchor + f'            config_index={up} if num_tokens <= {max_tokens} else -1,\n            block_limit={blocks} if num_tokens <= {max_tokens} else -1,\n' + parts[1] + anchor + f'            config_index={down} if num_tokens <= {max_tokens} else -1,\n            block_limit={blocks} if num_tokens <= {max_tokens} else -1,\n' + parts[2]
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes', type=float, default=45)
    p.add_argument('--pairs', type=int, default=40)
    p.add_argument('--narrow', action='store_true', help='Compile new N64 NVFP4 specialization; apply only through M256')
    p.add_argument('--seed', type=int, default=14009)
    p.add_argument('--directory', type=Path, default=ROOT / 'results/execution-sweep')
    args = p.parse_args()
    if args.pairs < 20 or args.pairs % 2:
        p.error('--pairs must be even and at least 20')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (directory / 'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = generate(directory, args.narrow)
    source = SOURCE.read_text()
    adapter(directory / 'incumbent.py', source)
    state = dict(status='running',jit=metadata,source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=[])
    statefile = directory / 'state.json'
    if statefile.exists():
        previous = json.loads(statefile.read_text())
        atomic(directory / f'previous-state-{time.time_ns()}.json', previous)
    state.update(seed=args.seed,pairs=args.pairs)
    deadline = time.monotonic() + args.minutes * 60
    policies = ([(-1,-1,-1),(2,2,-1),(2,2,1),(2,2,2),(2,1,-1),(1,2,-1)] if args.narrow else
                [(-1,-1,-1),(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1),(-1,-1,1),(-1,-1,2)])
    state['narrow'] = args.narrow
    try:
        for up, down, blocks in policies:
            name = f'up{up}-down{down}-blocks{blocks}'
            candidate = directory / f'{name}.py'
            changed = candidate_source(source, directory, up, down, blocks, max_tokens=256 if args.narrow else 512)
            adapter(candidate, changed)
            candidate.write_text(f'import sys\nsys.path.insert(0, {str(directory)!r})\n' + candidate.read_text())
            output = directory / f'{name}-seed{args.seed}-pairs{args.pairs}.json'
            state['current'] = name
            atomic(statefile, state)
            remaining = deadline-time.monotonic()
            if remaining <= 0 or (directory / 'STOP').exists():
                state['status'] = 'stopped'
                break
            with output.with_suffix('.log').open('w') as log:
                proc = subprocess.run([sys.executable,str(ROOT / 'benchmarks/moe_local.py'),
                    '--baseline',str(directory / 'incumbent.py'),'--candidate',str(candidate),
                    '--output',str(output),'--seed',str(args.seed),'--pairs',str(args.pairs),
                    '--batches',*map(str,BATCHES)], stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=remaining)
            if proc.returncode:
                state['results'].append(dict(name=name,status='failed',returncode=proc.returncode))
                # A compile or correctness failure needs agent inspection; do not burn
                # through a queue of dependent broken variants.
                state['status']='failed_closed'
                break
            report = json.loads(output.read_text())
            passed = gate(report, BATCHES, [8,32,128,256,512], require_gain=(up,down,blocks)!=(-1,-1,-1))
            state['results'].append(dict(name=name,status='screen_pass' if passed else 'rejected',
                measurements=[{k:r[k] for k in ('bs','balanced_speedup','bootstrap_95','assessment')} for r in report['results']]))
            print(json.dumps(state['results'][-1]),flush=True)
            if (up,down,blocks)==(-1,-1,-1) and not passed:
                state['status']='control_noninferiority_failed'
                break
        else:
            state['status']='screen_complete_pending_agent_review'
    except subprocess.TimeoutExpired:
        state['status']='timeout'
    finally:
        atomic(statefile,state)


if __name__=='__main__':
    main()

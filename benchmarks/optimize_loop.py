#!/usr/bin/env python3
"""Persistent, bounded experiment queue. Run inside the pinned GPU container."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py'
FULL = [1, 8, 16, 32, 64, 128, 256, 257, 288, 320, 321, 384, 385, 512, 513, 753, 3012, 16384]


def gate(report, expected, targets, require_gain=True):
    rows = report.get('results', [])
    if report.get('status') != 'complete' or {r['bs'] for r in rows} != set(expected):
        return False
    if report.get('pairs', 0) < 20:
        return False
    # Require positive evidence of noninferiority, not merely absence of significance.
    if any(not r.get('bootstrap_95') or r['bootstrap_95'][0] < 1 / 1.02 for r in rows):
        return False
    return not require_gain or any(r['bs'] in targets and r['bootstrap_95'][0] > 1.02 for r in rows)


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def adapter(path, source):
    path.write_text('from types import ModuleType\nfrom serving_moe import prepare_layer, kernel_fn\n'
                    f'SOURCE = {source!r}\n'
                    'def prepare(weights):\n'
                    '    module = ModuleType("loop_expert")\n'
                    '    exec(compile(SOURCE, __file__, "exec"), module.__dict__)\n'
                    '    return prepare_layer(weights, module.MuseSpark1xFp4Experts)\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes', type=float, default=45)
    p.add_argument('--directory', type=Path, default=ROOT / 'results/optimization-loop')
    args = p.parse_args()
    if args.minutes <= 0:
        p.error('--minutes must be positive')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = SOURCE.read_text()
    diff = subprocess.check_output(['git', '-C', str(ROOT / 'serving/sglang'), 'diff', 'HEAD'])
    fingerprint = hashlib.sha256(diff).hexdigest()
    statefile = directory / 'state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {
        'schema_version': 1, 'serving_diff_sha256': fingerprint, 'iterations': {},
        'promotion': 'agent_review_required; full_serving_validation_still_required'}
    if state['serving_diff_sha256'] != fingerprint:
        raise SystemExit('Incumbent changed: use a new --directory; do not mix evidence')
    deadline = time.monotonic() + args.minutes * 60
    incumbent = directory / 'incumbent.py'
    adapter(incumbent, source)

    def measure(folder, name, candidate, baseline, batches, seed, profile=False):
        output = folder / f'{name}.json'
        if output.exists():
            report = json.loads(output.read_text())
            if report.get('status') == 'complete':
                return report
        if (directory / 'STOP').exists():
            raise InterruptedError('STOP requested')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Session budget exhausted')
        command = [sys.executable, str(ROOT / 'benchmarks/moe_local.py'),
                   '--candidate', str(candidate), '--baseline', str(baseline),
                   '--output', str(output), '--pairs', '20', '--seed', str(seed),
                   '--batches', *map(str, batches)]
        if profile:
            command.append('--profile')
        state['current_measurement'] = str(output)
        atomic(statefile, state)
        with (folder / f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining, check=True)
        return json.loads(output.read_text())

    try:
        for cutoff in (320, 384):
            name = f'extend-tile16-to-{cutoff}'
            if state['iterations'].get(name, {}).get('status') in ('rejected', 'validated_candidate'):
                continue
            folder = directory / name
            folder.mkdir(exist_ok=True)
            old = 'if num_tokens <= 256:'
            if source.count(old) != 1:
                raise ValueError('Expected unique tile policy anchor')
            changed = source.replace(old, f'if num_tokens <= {cutoff}:')
            candidate = folder / 'candidate.py'
            adapter(candidate, changed)
            (folder / 'candidate-source.py').write_text(changed)
            targets = [m for m in (257, 288, 320, 384) if m <= cutoff]
            row = {'status': 'screening', 'hypothesis': 'Tile16 reduces padding near the current 256-token boundary',
                   'target_batches': targets, 'candidate_source_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                   'evidence': []}
            state['iterations'][name] = row
            atomic(statefile, state)
            passed = True
            for seed in (8801, 9901):
                tag = f'screen-{seed}'
                report = measure(folder, tag, candidate, incumbent, [256, 257, 288, 320, 384, 512, 753], seed)
                row['evidence'].append(tag)
                if not gate(report, [256, 257, 288, 320, 384, 512, 753], targets):
                    passed = False
                    break
            if passed:
                row['status'] = 'validating'
                for seed in (11003, 12007):
                    for baseline_name, baseline in [('incumbent', incumbent), ('stock', ROOT / 'benchmarks/serving_stock_moe.py')]:
                        tag = f'validate-{baseline_name}-{seed}'
                        report = measure(folder, tag, candidate, baseline, FULL, seed,
                                         profile=seed == 12007 and baseline_name == 'incumbent')
                        row['evidence'].append(tag)
                        if not gate(report, FULL, targets, require_gain=baseline_name == 'incumbent'):
                            passed = False
                            break
                    if not passed:
                        break
            row['status'] = 'validated_candidate' if passed else 'rejected'
            atomic(statefile, state)
            print(json.dumps({'iteration': name, 'status': row['status']}), flush=True)
        state['status'] = 'queue_exhausted_agent_review_required'
    except (TimeoutError, subprocess.TimeoutExpired, InterruptedError) as exc:
        state['status'] = 'stopped_resumable'
        state['reason'] = str(exc)
    except Exception as exc:
        state['status'] = 'failed_closed'
        state['reason'] = repr(exc)
        raise
    finally:
        state['session_elapsed_seconds'] = args.minutes * 60 - max(0, deadline - time.monotonic())
        atomic(statefile, state)


if __name__ == '__main__':
    main()

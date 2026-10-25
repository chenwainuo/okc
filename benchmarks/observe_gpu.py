#!/usr/bin/env python3
"""Sample host GPU process ownership during one named-container experiment.

An observational supplement to the shared flock, not continuous proof: jobs
shorter than the polling interval can escape sampling. Never launches CUDA work.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 3600:
        raise ValueError('Observation deadline must be within one hour')
    if args.output.exists():
        raise ValueError('Do not overwrite observation evidence')
    result = {'status': 'observing', 'poll_interval_seconds': 1,
              'scope': 'sampled ownership only; cooperative device lock still required', 'samples': []}
    deadline = time.monotonic() + args.seconds
    container_id = None
    known = {}
    while time.monotonic() < deadline:
        state = json.loads(args.process_file.read_text()) if args.process_file.exists() else {}
        name = state.get('container')
        if name and container_id is None:
            probe = subprocess.run(['docker', 'inspect', '-f', '{{.Id}}', name],
                                   capture_output=True, text=True, timeout=5)
            if probe.returncode == 0:
                container_id = probe.stdout.strip()
                result.update(container=name, container_id=container_id)
        row = {'time_unix': time.time(), 'owned': [], 'foreign': [], 'unresolved': []}
        probe = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5)
        if probe.returncode:
            row['query_error'] = probe.stderr
        else:
            for item in probe.stdout.splitlines():
                try:
                    pid = int(item.strip())
                    group = Path(f'/proc/{pid}/cgroup').read_text()
                    owner = 'owned' if container_id and container_id in group else 'foreign'
                    known[pid] = owner
                except ValueError:
                    row['unresolved'].append(item)
                    continue
                except FileNotFoundError:
                    owner = known.get(pid, 'unresolved')
                row[owner].append(pid)
        result['samples'].append(row)
        if state.get('status') == 'terminal':
            result['status'] = 'terminal_observed'
            break
        time.sleep(1)
    else:
        result['status'] = 'observation_deadline'
    result['no_foreign_process_observed'] = bool(container_id) and all(
        not r['foreign'] and not r['unresolved'] and not r.get('query_error') for r in result['samples'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))


if __name__ == '__main__':
    main()

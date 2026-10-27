"""Read-only metadata extraction; never imports/locks/repairs QueueStore.

Direct checkpoint view + fixed initial WAL prefix. Per-record raw byte SHA is
an evidence pointer, not replay validation of the WAL payload/checksum/schema.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('.json')

def utc(value):
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat() if type(value) in (int, float) else value

def main():
    wal = ROOT / 'results/proj-rsi/scheduler/events.jsonl'
    size = wal.stat().st_size
    events = []
    offset = 0
    started = time.monotonic()
    with wal.open('rb') as stream:
        while offset < size:
            raw = stream.readline(size - offset)
            if not raw.endswith(b'\n'):
                break
            before, separator, after = raw.partition(b'"snapshot":')
            if not separator:
                raise ValueError('no snapshot')
            metadata = json.loads(before.rstrip().rstrip(b',') + b'}}')['payload']
            timestamps = re.findall(rb',\s*"timestamp":\s*([0-9.eE+-]+)}[,}]', after[-300:])
            stamp = float(timestamps[-1])
            data = metadata.get('data', {})
            events.append({k: metadata[k] for k in ('sequence', 'action', 'actor', 'reason')} | {
                'timestamp': stamp, 'timestamp_utc': utc(stamp),
                'job_id': data.get('id') if isinstance(data, dict) else None,
                'source': {'path': str(wal.relative_to(ROOT)), 'offset': offset,
                           'bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest()}})
            offset += len(raw)
    checkpoint = ROOT / 'results/proj-rsi/scheduler/checkpoint.json'
    raw = checkpoint.read_bytes()
    snapshot = json.loads(raw)
    rows = []
    for job_id, job in snapshot['jobs'].items():
        if job['resource'] != 'gpu':
            continue
        observations = [o.get('observation', o) for o in job.get('observations', [])]
        handles = [o['dispatch'] for o in observations if 'dispatch' in o]
        handle = handles[-1] if handles else {}
        collections = [o for o in observations if 'collection' in o]
        terminal = None
        collected = None
        inventory = None
        if collections:
            row = collections[-1]
            collected = row['timestamp']
            inventory = row['collection']['inventory']
            stamps = re.findall(r'^2026-10-27T\d\d:\d\d:\d\dZ$', row['collection']['cleanup_observation'].get('host_log_tail', ''), re.M)
            if stamps:
                terminal = datetime.datetime.fromisoformat(stamps[-1].replace('Z', '+00:00')).timestamp()
        clean = []
        transitions = []
        for observation in observations:
            probe = observation.get('probe', {})
            if probe.get('terminal_exit') is not None and not probe.get('compute_pids') and not probe.get('guard_pids') and probe.get('container_exists') is False and probe.get('gpu_memory_mib') == 0:
                clean.append(observation['timestamp'])
            if 'transition' in observation:
                transitions.append({'timestamp_utc': utc(observation['timestamp']),
                    **{key: observation['transition'][key] for key in ('from', 'to', 'reason')}})
        registrations = [e['timestamp'] for e in events if e['action'] == 'register' and e['job_id'] == job_id]
        row = {'id': job_id, 'state_at_read': job['state'], 'descriptor_created_at': job.get('created_at'),
            'registered_at': registrations[0] if registrations else None,
            'claimed_at': handle.get('claimed_at'), 'launch_requested_at': handle.get('launch_requested_at'),
            'host_terminal_timestamp': terminal, 'first_clean_terminal_observed_at': min(clean) if clean else None,
            'collected_at': collected, 'transitions': transitions, 'inventory': inventory,
            'review_path': job.get('review', {}).get('path')}
        for field in ('descriptor_created_at', 'registered_at', 'claimed_at', 'launch_requested_at',
                      'host_terminal_timestamp', 'first_clean_terminal_observed_at', 'collected_at'):
            row[field + '_utc'] = utc(row[field])
        rows.append(row)
    result = {'status': 'READ_ONLY_DESCRIPTIVE_METADATA_NOT_WAL_REPLAY', 'observed_at_utc': utc(time.time()),
        'wal_initial_bytes': size, 'complete_record_count': len(events), 'scan_seconds': time.monotonic() - started,
        'checkpoint_sequence': snapshot['sequence'], 'checkpoint_bytes': len(raw),
        'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
        'limitations': ['Direct checkpoint read is a view, not authoritative replay.',
            'WAL metadata and raw byte hashes extracted without payload/checksum replay.',
            'Fixed prefix and later checkpoint may span concurrently appended events.',
            'Terminal timestamps come from retained collected host logs; point idle observations do not prove continuous idle hardware.',
            'No live QueueStore, lock, SSH, GPU, API, recovery or process probe performed.'],
        'gpu_jobs': sorted(rows, key=lambda row: row['launch_requested_at'] or 1e99), 'event_metadata': events}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('complete_record_count', 'wal_initial_bytes', 'checkpoint_sequence', 'scan_seconds')}))

if __name__ == '__main__':
    main()

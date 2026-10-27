"""Read existing dispatcher observations; never contact or probe hardware.

Run once: python3 -m proj_rsi.dashboard.observe_hardware --gpu-name H100
Follow existing WAL evidence: add --interval 10. Root owns process lifecycle.
Only the local dashboard hardware JSON is atomically replaced. Scheduler files
are read under the existing shared lock and are never created or repaired.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from proj_rsi.dashboard.snapshot import ROOT, STALE_SECONDS, finite, read_scheduler, safe_text, timestamp, write_snapshot


def _epoch(value):
    observed = timestamp(value)
    return observed if observed is not None and observed > 0 else None


def _pids(value):
    return isinstance(value, list) and all(type(pid) is int and pid > 0 for pid in value)


def classify_probe(probe):
    """Positive actual signals prove busy; complete explicit absence proves idle.

    A malformed present field invalidates the observation. Optional status/log
    fields never become device evidence, and nothing reads process IDs itself.
    """
    if not isinstance(probe, dict):
        return 'unknown', None, None, 'invalid_probe'
    pids = probe.get('compute_pids')
    memory = probe.get('gpu_memory_mib')
    exists = probe.get('container_exists')
    process_count = len(pids) if _pids(pids) else None
    memory_used = memory if finite(memory) and memory >= 0 else None
    invalid = (
        ('compute_pids' in probe and not _pids(pids))
        or ('gpu_memory_mib' in probe and memory_used is None)
        or ('container_exists' in probe and type(exists) is not bool)
        or ('guard_pids' in probe and not _pids(probe['guard_pids']))
        or ('containers' in probe and (not isinstance(probe['containers'], list) or not all(isinstance(item, str) and item for item in probe['containers'])))
        or ('cleanup_failed' in probe and type(probe['cleanup_failed']) is not bool)
    )
    if invalid:
        return 'unknown', None, None, 'inconsistent_probe'
    if process_count or exists is True or (memory_used is not None and memory_used > 0):
        return 'busy', memory_used, process_count, 'actual_busy_signal'
    # Match the dispatcher's complete clean observation, not queue state or
    # terminal-exit inference. Guard/container inventory omissions are unknown.
    if (exists is False and pids == [] and memory_used == 0
            and probe.get('guard_pids') == [] and probe.get('containers') == []
            and probe.get('cleanup_failed') is False):
        return 'idle', memory_used, process_count, 'complete_actual_absence'
    return 'unknown', memory_used, process_count, 'incomplete_probe'


def derive_hardware(state, *, now, gpu_name=None, source_status='ok'):
    if not finite(now):
        raise ValueError('now must be finite')
    result = {'status': 'unknown', 'observed_at': None, 'gpu_name': safe_text(gpu_name, 100),
              'memory_used_mb': None, 'process_count': None, 'observation_source': None,
              'reason': 'no_actual_observation'}
    if source_status != 'ok' or not isinstance(state, dict) or not isinstance(state.get('jobs'), dict):
        result['reason'] = 'scheduler_unavailable'
        return result
    candidates, superseding, unknown_at = [], [], []
    untimed_claim = False
    untimed_observation = False
    for job in state['jobs'].values():
        if not isinstance(job, dict) or job.get('resource') != 'gpu':
            continue
        lease = job.get('lease')
        claim = _epoch(lease.get('claimed_at')) if isinstance(lease, dict) else None
        if claim is not None:
            superseding.append(claim)
        elif job.get('state') == 'RUNNING':
            untimed_claim = True
        observations = job.get('observations', [])
        if not isinstance(observations, list):
            untimed_observation = True
            continue
        # A collection attaches an earlier probe dictionary. Resolve its exact
        # matching direct observation so collection time cannot freshen it.
        direct = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            at = _epoch(observation.get('timestamp'))
            probe = observation.get('probe')
            if isinstance(probe, dict) and at is not None:
                original = _epoch(probe.get('observed_at')) or _epoch(probe.get('timestamp')) or at
                direct.append((original, probe))
                candidates.append((original, probe, 'probe'))
            elif 'probe' in observation:
                untimed_observation = True
            dispatch = observation.get('dispatch')
            if isinstance(dispatch, dict):
                for key in ('claimed_at', 'launch_requested_at'):
                    event = _epoch(dispatch.get(key))
                    if event is not None:
                        superseding.append(event)
                if dispatch.get('phase') in ('LAUNCH_REQUESTED', 'LAUNCH_UNKNOWN') and at is not None:
                    superseding.append(at)
            if observation.get('status') == 'UNKNOWN' and at is not None:
                unknown_at.append(at)
            elif observation.get('status') == 'UNKNOWN':
                untimed_observation = True
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            collection = observation.get('collection')
            if not isinstance(collection, dict) or not isinstance(collection.get('cleanup_observation'), dict):
                continue
            probe = collection['cleanup_observation']
            recorded = _epoch(observation.get('timestamp'))
            explicit = _epoch(probe.get('observed_at')) or _epoch(probe.get('timestamp'))
            matches = [at for at, previous in direct if previous == probe and (recorded is None or at <= recorded)]
            # Bare copied cleanup evidence without a matching direct probe or
            # explicit original time is not allowed to acquire attach-time age.
            original = explicit or (max(matches) if matches else None)
            if original is not None:
                candidates.append((original, probe, 'collection.cleanup_observation'))
            else:
                untimed_observation = True
    if not candidates:
        return result
    latest = max(item[0] for item in candidates)
    newest = [item for item in candidates if item[0] == latest]
    status, memory, count, reason = classify_probe(newest[0][1])
    if any(classify_probe(item[1]) != (status, memory, count, reason) for item in newest[1:]):
        status, memory, count, reason = 'unknown', None, None, 'conflicting_latest_observations'
    result.update(status=status, observed_at=latest, memory_used_mb=memory, process_count=count,
                  observation_source=newest[0][2], reason=reason)
    if latest > now + 30:
        result.update(status='unknown', reason='future_observation')
    elif now - latest > STALE_SECONDS:
        result.update(status='unknown', reason='stale_observation')
    elif untimed_claim or any(event > latest for event in superseding):
        result.update(status='unknown', reason='claim_or_launch_after_observation')
    elif untimed_observation:
        result.update(status='unknown', reason='untimed_observation')
    elif any(event > latest for event in unknown_at):
        result.update(status='unknown', reason='later_probe_unavailable')
    return result


def observe_once(*, scheduler_dir=None, output=None, gpu_name=None, now=None):
    """No QueueStore instance, SSH, subprocess, GPU library or hardware call."""
    directory = Path(scheduler_dir) if scheduler_dir is not None else ROOT / 'results/proj-rsi/scheduler'
    destination = Path(output) if output is not None else ROOT / 'results/proj-rsi/dashboard/hardware.json'
    state, _last_event, status = read_scheduler(directory)
    hardware = derive_hardware(state, now=time.time() if now is None else now, gpu_name=gpu_name, source_status=status)
    write_snapshot(destination, hardware)
    return hardware


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-name')
    parser.add_argument('--interval', type=float, help='Repeat using only existing WAL observations; typically 10 seconds')
    args = parser.parse_args(argv)
    if args.interval is not None and (not math.isfinite(args.interval) or not 1 <= args.interval <= 3600):
        parser.error('--interval must be between 1 and 3600 seconds')
    while True:
        started = time.monotonic()
        try:
            observation = observe_once(scheduler_dir=args.scheduler_dir, output=args.output, gpu_name=args.gpu_name)
            print('Hardware observation exported: ' + observation['status'], flush=True)
        except Exception:
            # Do not expose raw WAL fields, paths, hostnames, commands or errors.
            print('Hardware observation export unavailable', file=sys.stderr, flush=True)
            if args.interval is None:
                return 1
        if args.interval is None:
            return 0
        time.sleep(max(0.1, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)

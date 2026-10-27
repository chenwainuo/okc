"""One-shot scheduler controller; dispatch imports only for explicit GPU actions."""
import argparse
import json
from pathlib import Path
import sys

from .store import QueueStore, QueueError


def _read(path):
    return json.load(sys.stdin) if path == '-' else json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='results/proj-rsi/scheduler')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--actor', default='root', help='local controller identity, not an authentication boundary')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    register = sub.add_parser('register'); register.add_argument('file', help='JSON job descriptor; - for stdin')
    reprice = sub.add_parser('reprice'); reprice.add_argument('id'); reprice.add_argument('file', help='JSON changes: estimate/cost/annotations'); reprice.add_argument('--reason', required=True)
    status = sub.add_parser('status'); status.add_argument('id', nargs='?')
    for name in ('next', 'claim'):
        command = sub.add_parser(name); command.add_argument('resource', choices=('cpu', 'gpu'))
        if name == 'claim': command.add_argument('--owner', default='root')
    finish = sub.add_parser('finish'); finish.add_argument('id'); finish.add_argument('disposition'); finish.add_argument('--evidence', required=True, help='JSON evidence file')
    finish.add_argument('--validate', action='store_true', help='explicit root validation after execution ended')
    for name in ('hold', 'cancel'):
        command = sub.add_parser(name); command.add_argument('id'); command.add_argument('--reason', required=True)
    run = sub.add_parser('run-next'); run.add_argument('--owner', default='root')
    work = sub.add_parser('work', help='bounded drain of approved GPU queue; never validates results')
    work.add_argument('--latest-snapshot', action='store_true',
                      help='strict startup history audit, then verify only the latest complete snapshot per operation; historical corruption detection is deferred until the next strict audit')
    work.add_argument('--owner', default='root')
    work.add_argument('--max-jobs', type=int, default=3)
    work.add_argument('--max-wall-seconds', type=int, default=3600)
    work.add_argument('--poll-seconds', type=float, default=10)
    work.add_argument('--idle-wait-seconds', type=float, default=0,
                      help='wait for late eligible jobs between executions, at most this many seconds per idle gap (default: exit immediately)')
    for name in ('poll', 'collect', 'reconcile'):
        command = sub.add_parser(name); command.add_argument('id')
    args = parser.parse_args(argv)
    try:
        if args.command == 'work' and args.latest_snapshot:
            store = QueueStore(args.directory, repo_root=args.repo_root, latest_snapshot_reads=True)
        else:
            store = QueueStore(args.directory, repo_root=args.repo_root)
        if args.command == 'init':
            result = store.snapshot()
        elif args.command == 'register':
            result = store.register(_read(args.file))
        elif args.command == 'reprice':
            changes = _read(args.file)
            if set(changes) - {'estimate', 'cost', 'annotations'}:
                raise QueueError('reprice accepts estimate/cost/annotations only')
            result = store.update(args.id, changes, actor=args.actor, reason=args.reason)
            if result['state'] == 'HELD' and result.get('eligibility_errors'):
                result = store.transition(args.id, 'BACKLOG', actor=args.actor, reason='reprice: ' + args.reason)
        elif args.command == 'status':
            result = store.get(args.id) if args.id else {**store.snapshot(), 'ranking': {'cpu': store.rank('cpu'), 'gpu': store.rank('gpu')}}
        elif args.command == 'next':
            store.rebalance(args.resource)
            result = next((job for job in store.rank(args.resource) if job['state'] == 'READY'), None)
        elif args.command == 'claim':
            result = store.claim(args.resource, args.owner)
        elif args.command == 'finish':
            evidence = _read(args.evidence)
            if args.validate:
                result = store.transition(args.id, 'DONE', actor=args.actor, reason='explicit controller validation', evidence={**evidence, 'disposition': args.disposition})
            else:
                result = store.finish(args.id, args.disposition, evidence=evidence, actor=args.actor)
        elif args.command in ('hold', 'cancel'):
            result = store.transition(args.id, 'HELD' if args.command == 'hold' else 'CANCELLED', actor=args.actor, reason=args.reason)
        else:
            from .dispatch import GPUDispatcher
            dispatcher = GPUDispatcher(store, args.repo_root)
            if args.command == 'work':
                if args.max_jobs < 1 or args.max_wall_seconds < 1 or not 0 < args.poll_seconds <= 60 or not 0 <= args.idle_wait_seconds <= 86400:
                    raise QueueError('work requires positive job/wall bounds, poll-seconds in (0,60] and idle-wait-seconds in [0,86400]')
                result = dispatcher.work(owner=args.owner, max_jobs=args.max_jobs, max_wall_seconds=args.max_wall_seconds, poll_seconds=args.poll_seconds, idle_wait_seconds=args.idle_wait_seconds)
            elif args.command == 'run-next':
                result = dispatcher.launch_next(owner=args.owner)
            else:
                result = getattr(dispatcher, args.command)(args.id)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (QueueError, OSError, ValueError, KeyError) as error:
        print(json.dumps({'error': str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)

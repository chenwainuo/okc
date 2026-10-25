"""python3 -m rsi --help"""
import argparse
import json
from pathlib import Path
from .ledger import Ledger, ROOT


def main():
    p = argparse.ArgumentParser(description='Agent-directed RSI ledger. No execution, agent spawning, patch application, or publishing.')
    p.add_argument('--ledger', type=Path, default=ROOT / 'results/rsi/ledger.json')
    p.add_argument('--gpu-state', type=Path, default=ROOT / 'results/rsi/gpu-lease.json')
    commands = p.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('init')
    sub.add_argument('--anchor', type=Path, default=ROOT / 'docs/threefold-target.json')
    sub.add_argument('--budget-usd', type=float, required=True)
    sub.add_argument('--hourly-usd', type=float, default=5)
    sub.add_argument('--max-run-minutes', type=float, default=45)
    sub = commands.add_parser('propose')
    sub.add_argument('id'); sub.add_argument('--actor',required=True); sub.add_argument('--plan',type=Path,required=True)
    sub = commands.add_parser('review')
    sub.add_argument('id'); sub.add_argument('--actor',required=True); sub.add_argument('--checklist',type=Path,required=True)
    sub = commands.add_parser('start')
    sub.add_argument('id'); sub.add_argument('--actor',required=True)
    sub = commands.add_parser('finish')
    sub.add_argument('id'); sub.add_argument('--actor',required=True); sub.add_argument('--token',required=True); sub.add_argument('--manifest',type=Path,required=True)
    sub = commands.add_parser('abort')
    sub.add_argument('id'); sub.add_argument('--actor',required=True); sub.add_argument('--token',required=True)
    sub.add_argument('--terminal-proof',type=Path,required=True); sub.add_argument('--reason',required=True)
    for cmd in ('validate','promote','reject'):
        sub = commands.add_parser(cmd)
        sub.add_argument('id'); sub.add_argument('--actor',required=True)
        if cmd == 'reject': sub.add_argument('--reason',required=True)
    sub = commands.add_parser('observe')
    sub.add_argument('id'); sub.add_argument('--actor',required=True); sub.add_argument('--note',required=True); sub.add_argument('artifacts',type=Path,nargs='+')
    commands.add_parser('status')
    args = p.parse_args()
    ledger = Ledger(args.ledger,args.gpu_state)
    def load(path):
        return json.loads(path.read_text())
    try:
        if args.command == 'init':
            result = ledger.initialize(args.anchor,args.budget_usd,args.hourly_usd,args.max_run_minutes)
        elif args.command == 'propose': result = ledger.propose(args.id,args.actor,load(args.plan))
        elif args.command == 'review': result = ledger.review(args.id,args.actor,load(args.checklist))
        elif args.command == 'start': result = ledger.start(args.id,args.actor)
        elif args.command == 'finish': result = ledger.finish(args.id,args.actor,args.token,load(args.manifest))
        elif args.command == 'abort': result = ledger.abort(args.id,args.actor,args.token,load(args.terminal_proof),args.reason)
        elif args.command == 'validate': result = ledger.validate(args.id,args.actor)
        elif args.command == 'promote': result = ledger.promote(args.id,args.actor)
        elif args.command == 'reject': result = ledger.reject(args.id,args.actor,args.reason)
        elif args.command == 'observe': result = ledger.observe(args.id,args.actor,args.artifacts,args.note)
        else: result = ledger.status()
    except (ValueError, KeyError, TypeError, OSError) as error:
        p.exit(2, f'RSI rejected operation: {error}\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()

"""CPU-only agent handoff coordinator; ledger remains the sole gate authority.

Actor labels are unauthenticated coordination records. Root supplies actual
independent agents. Existing outputs are observations, never completion proof.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ROLES = {'supervisor', 'planner', 'implementer', 'critic', 'executor', 'validator', 'method_reviewer'}
TERMINAL_QUEUE = {'rejected', 'measured', 'promoted', 'completed', 'validated',
                  'screen_complete_no_global_winner', 'measured_rejected_variants'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_at(value, root):
    require(isinstance(value, str) and value.strip(), 'Path must be nonempty text')
    require(not any(c in value for c in '*?[]'), 'Ownership and input paths cannot contain globs')
    return str((root / value).resolve())


def artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f'Input is not a regular file: {path}')
    return {'path': str(path), 'sha256': digest(path)}


def overlaps(a, b):
    a, b = Path(a), Path(b)
    if a == b or a in b.parents or b in a.parents:
        return True
    try:
        return a.samefile(b)
    except OSError:
        return False


def text_list(value, label, nonempty=True):
    require(isinstance(value, list) and (value or not nonempty), f'{label} must be a list')
    require(all(isinstance(x, str) and x.strip() for x in value), f'{label} requires nonempty strings')


def validate_contracts(manifest):
    require(manifest.get('schema_version') == 1, 'Unsupported cycle schema')
    require(manifest.get('kind') in ('kernel', 'optimizer_method'), 'Invalid cycle kind')
    require(manifest.get('evidence_scope') in ('development', 'promotion'), 'Explicit evidence scope required')
    require(isinstance(manifest.get('id'), str) and manifest['id'].strip(), 'Cycle ID required')
    tasks = manifest.get('tasks')
    require(isinstance(tasks, list) and tasks, 'Tasks required')
    ids, owners, actors, seen_roles = set(), [], {}, set()
    for task in tasks:
        require(isinstance(task, dict), 'Task must be an object')
        task_id, role, actor = task.get('id'), task.get('role'), task.get('actor')
        require(isinstance(task_id, str) and task_id.strip() and task_id not in ids, 'Unique task IDs required')
        ids.add(task_id)
        require(role in ROLES, f'Invalid task role: {role}')
        require(role not in seen_roles, 'One task per role per cycle; freeze separate cycles for distinct hypotheses')
        seen_roles.add(role)
        for binding in ('experiment_id', 'queue_item_id'):
            require(binding not in task or task[binding] == manifest.get(binding), 'Task binding differs from cycle hypothesis')
        require(isinstance(task.get('handoff_recipient'), str) and task['handoff_recipient'].strip(), 'Explicit handoff_recipient required')
        require(isinstance(actor, str) and actor.strip(), 'Actor ID required')
        actors.setdefault(role, set()).add(actor)
        text_list(task.get('completion_evidence'), 'completion_evidence')
        require(any(isinstance(task.get(k), str) and task[k].strip() for k in ('hypothesis', 'review_question')),
                'Task requires a hypothesis or review_question')
        text_list(task.get('forbidden_actions'), 'forbidden_actions')
        for field in ('owned_files', 'outputs'):
            text_list(task.get(field), field, nonempty=False)
            require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in task[field]), 'Paths must be resolved absolute paths')
        inputs = task.get('inputs')
        require(isinstance(inputs, list) and inputs, 'Task inputs required')
        for item in inputs:
            require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'Invalid input record')
            require(Path(item['path']).is_absolute(), 'Input must be absolute')
            require(isinstance(item['sha256'], str) and len(item['sha256']) == 64, 'Invalid input digest')
        for owned in task['owned_files']:
            require(not any(overlaps(owned, other) for _, other in owners), f'Overlapping write ownership: {owned}')
            owners.append((task_id, owned))
        require(all(any(Path(o) == Path(w) or Path(w) in Path(o).parents for w in task['owned_files']) for o in task['outputs']),
                'Every output must belong to task write ownership')
    require(actors.get('implementer') and actors.get('critic'), 'Implementer and independent critic contracts required')
    require(not actors['implementer'] & actors['critic'], 'Implementer and critic actors must differ')
    require(not actors.get('validator', set()) & (actors['implementer'] | actors.get('executor', set())),
            'Validator must differ from implementer and executor')
    protected = manifest.get('protected_inputs')
    require(isinstance(protected, list) and protected, 'Protected authority inputs required')
    for _, owned in owners:
        require(not any(overlaps(owned, p['path']) for p in protected), 'Task cannot own protected authority files')
        require(not overlaps(owned, manifest['manifest_path']) and not overlaps(owned, manifest['spec']['path']), 'Task cannot own cycle manifest or spec')
        require(not any(overlaps(owned, manifest['sources'][s]['path']) for s in ('ledger', 'gpu_state')),
                'Cycle task cannot own ledger or GPU lease files')


def immutable_write(path, value):
    """Atomic publish without overwrite: a concurrent initializer loses link()."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.cycle-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def initialize(spec_path, manifest_path, ledger_path, queue_path, gpu_state_path, root=None):
    spec_path = Path(spec_path).resolve()
    root = Path(root).resolve() if root is not None else spec_path.parent
    spec, ledger = read(spec_path), read(ledger_path)
    require(isinstance(ledger.get('experiments'), dict), 'Invalid ledger experiments')
    require(isinstance(read(queue_path).get('items'), list), 'Invalid optimization queue')
    protected = []
    for key in ('anchor', 'benchmark_contract', 'measurement'):
        item = ledger.get(key)
        require(isinstance(item, dict), f'Ledger missing protected {key}')
        actual = artifact(item['path'])
        require(actual['sha256'] == item['sha256'], f'Ledger protected {key} changed')
        protected.append(actual)
    protected += [artifact(ROOT / p) for p in ('rsi/protocol.json', 'rsi/ledger.py', 'rsi/cycle.py')]
    result = {k: spec[k] for k in ('id', 'kind', 'evidence_scope')}
    for key in ('experiment_id', 'queue_item_id'):
        if key in spec:
            require(isinstance(spec[key], str) and spec[key].strip(), f'Invalid {key}')
            result[key] = spec[key]
    result.update(schema_version=1, root=str(root), manifest_path=str(Path(manifest_path).resolve()),
                  contract_authority='scheduling only; execution budget/deadline/lease admitted by existing ledger',
                  spec=artifact(spec_path), protected_inputs=protected,
                  sources={'ledger': artifact(ledger_path), 'queue': artifact(queue_path),
                           'gpu_state': {'path': str(Path(gpu_state_path).resolve())}}, tasks=[])
    for task in spec.get('tasks', []):
        item = dict(task)
        for key in ('owned_files', 'outputs'):
            item[key] = [path_at(p, root) for p in task.get(key, [])]
        item['inputs'] = [artifact(path_at(p, root)) for p in task.get('inputs', [])]
        result['tasks'].append(item)
    validate_contracts(result)
    result['manifest_sha256'] = canonical(result)
    immutable_write(manifest_path, result)
    return {'manifest': str(Path(manifest_path).resolve()), 'manifest_sha256': result['manifest_sha256'], 'tasks': len(result['tasks'])}


def check(manifest_path):
    errors, stale = [], []
    try:
        manifest = read(manifest_path)
        expected = manifest.get('manifest_sha256')
        require(expected == canonical({k: v for k, v in manifest.items() if k != 'manifest_sha256'}), 'Manifest self-integrity mismatch')
        validate_contracts(manifest)
        require(str(Path(manifest_path).resolve()) == manifest['manifest_path'], 'Manifest moved from frozen location')
        items = [(None, p) for p in manifest['protected_inputs']] + [(None, manifest['spec'])]
        items += [(t['id'], p) for t in manifest['tasks'] for p in t['inputs']]
        for task_id, item in items:
            try:
                current = artifact(item['path'])
                if current['path'] != item['path'] or current['sha256'] != item['sha256']:
                    stale.append({'task': task_id, 'path': item['path'], 'reason': 'changed or retargeted'})
            except (OSError, ValueError):
                stale.append({'task': task_id, 'path': item['path'], 'reason': 'missing or unreadable'})
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return {'valid': not errors and not stale, 'errors': errors, 'stale_inputs': stale}


def status(manifest_path):
    inspection = check(manifest_path)
    result = {'check': inspection, 'execution_authorized': False, 'authority': 'advisory only; existing ledger owns transitions, leases and promotion',
              'actor_identity': 'labels are not authentication; root must supply actual independent agents'}
    if not inspection['valid']:
        result['next_handoff'] = {'ready': False, 'role': 'supervisor', 'reason': 'invalid_or_stale_contract',
                                  'action': 'Inspect changes and freeze a new reviewed cycle; do not overwrite this manifest'}
        return result
    manifest = read(manifest_path)
    ledger = read(manifest['sources']['ledger']['path'])
    queue = read(manifest['sources']['queue']['path'])
    gpu_path = Path(manifest['sources']['gpu_state']['path'])
    require(gpu_path.is_file(), 'Shared GPU ownership state unavailable; cannot recommend readiness')
    external_lease = read(gpu_path)
    result.update(cycle_id=manifest['id'], evidence_scope=manifest['evidence_scope'],
                  context_changes=[name for name in ('ledger', 'queue') if digest(manifest['sources'][name]['path']) != manifest['sources'][name]['sha256']],
                  budget=ledger.get('budget'), local_incumbent=ledger.get('local_incumbent'),
                  full_serving_goal_achieved=ledger.get('full_serving_goal_achieved', False),
                  experiment_state_counts=dict(sorted(Counter(e.get('state', 'unknown') for e in ledger['experiments'].values()).items())),
                  outputs=[{'task': t['id'], 'path': p, 'present': Path(p).exists(), 'completion_inferred': False}
                           for t in manifest['tasks'] for p in t['outputs']])
    role, action, reason, eligible = 'planner', 'Select a falsifiable hypothesis from the ranked queue', 'queue_selection', True
    leases = [l for l in (external_lease, ledger.get('gpu_lease')) if l is not None]
    if leases:
        role, action, reason, eligible = 'supervisor', 'Inspect actual process/container state and preserve lease until verified terminal; never auto-release expired lease', 'lease_requires_process_inspection', False
        result['leases'] = [{k: v for k, v in lease.items() if k != 'token'} for lease in leases]
    elif manifest['kind'] == 'optimizer_method':
        role, action, reason, eligible = 'method_reviewer', 'Define and implement independently frozen portfolio A/B evaluator before method validation or promotion', 'method_evaluator_unimplemented', False
    elif manifest.get('experiment_id'):
        experiment = ledger['experiments'].get(manifest['experiment_id'])
        require(experiment is not None, 'Bound experiment missing from ledger')
        state = experiment.get('state')
        result['experiment'] = {'id': manifest['experiment_id'], 'state': state, 'revision': experiment.get('revision')}
        mapping = {
            'hypothesis': ('critic', 'Independently review frozen plan/artifacts via existing ledger gates'),
            'reviewed': ('executor', 'Ask supervisor to inspect GPU state and acquire existing ledger lease before execution'),
            'running': ('supervisor', 'Inspect running process and reconcile ownership; do not start another job'),
            'measured': ('validator', 'Use existing ledger validation on full raw evidence; no cycle-local gate'),
            'validated': ('supervisor', 'Inspect current-incumbent and existing promotion gates; this cycle grants no promotion'),
            'rejected': ('planner', 'Analyze failure and rerank queue; freeze a new hypothesis without overwriting evidence'),
            'promoted': ('planner', 'Reprofile incumbent and rerank queue; local promotion is not full serving achievement'),
            'observational': ('planner', 'Use observational findings to rerank hypotheses; cannot promote observational evidence'),
        }
        require(state in mapping, f'Unknown experiment state: {state}')
        role, action = mapping[state]; reason = 'ledger_' + state
        if state in ('running', 'validated'):
            eligible = False
        if manifest['evidence_scope'] == 'development' and state in ('measured', 'validated'):
            role, action, reason, eligible = 'supervisor', 'Keep development evidence observational; independent full-suite holdouts require a new reviewed plan', 'development_not_promotion_eligible', False
    else:
        items = queue.get('active_priority', queue.get('items'))
        require(isinstance(items, list), 'Invalid queue items')
        chosen = None
        if manifest.get('queue_item_id'):
            matches = [i for i in items if i.get('id') == manifest['queue_item_id']]
            require(len(matches) == 1, 'Bound queue item missing or duplicated')
            chosen = matches[0]
        else:
            candidates = [i for i in items if i.get('status') not in TERMINAL_QUEUE and not str(i.get('status', '')).startswith('blocked')]
            for i in candidates:
                rank = i.get('rank')
                require(isinstance(rank, (int, float)) and not isinstance(rank, bool) and math.isfinite(rank), 'Invalid queue rank')
            if candidates:
                chosen = min(candidates, key=lambda i: (i['rank'], i['id']))
        if chosen:
            result['queue_item'] = chosen
            if chosen.get('status') in TERMINAL_QUEUE or str(chosen.get('status', '')).startswith('blocked'):
                role, action, reason, eligible = 'planner', 'Review terminal/blocked queue item and select a new actionable hypothesis', 'queue_item_not_actionable', False
            elif str(chosen.get('status', '')).startswith('reviewed'):
                role, action, eligible = 'executor', 'Supervisor must verify reviewed plan, live budget and lease before any GPU execution', False
            elif chosen.get('status') in ('needs_harness', 'gap_fail_closed', 'design_defined_evaluator_not_implemented') or chosen.get('stage') in ('implement', 'implement/test', 'optimize/reimplement', 'test'):
                role, action = 'implementer', 'Implement scoped candidate/harness and produce required completion evidence; no GPU action inferred'
            else:
                action = chosen.get('next_handoff', chosen.get('next_action', action))
        else:
            reason, action, eligible = 'no_actionable_queue_item', 'Reconcile new evidence and rerank queue', False
    tasks = sorted((t for t in manifest['tasks'] if t['role'] == role), key=lambda t: t['id'])
    selected = tasks[0] if tasks else None
    result['next_handoff'] = {'ready': eligible and selected is not None, 'role': role, 'task': selected['id'] if selected else None,
                              'actor': selected['actor'] if selected else None, 'reason': reason, 'action': action,
                              'handoff_recipient': selected['handoff_recipient'] if selected else None,
                              'missing_contract': selected is None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--spec', type=Path, required=True); init.add_argument('--manifest', type=Path, required=True)
    init.add_argument('--ledger', type=Path, default=ROOT/'results/rsi/ledger.json')
    init.add_argument('--queue', type=Path, default=ROOT/'docs/optimization-queue.json')
    init.add_argument('--gpu-state', type=Path, default=ROOT/'results/rsi/gpu-lease.json')
    init.add_argument('--root', type=Path)
    for command in ('check', 'status'):
        commands.add_parser(command).add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            result = initialize(args.spec, args.manifest, args.ledger, args.queue, args.gpu_state, args.root)
        elif args.command == 'check':
            result = check(args.manifest)
        else:
            result = status(args.manifest)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if args.command == 'check' and not result['valid']:
            raise SystemExit(2)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'Cycle rejected operation: {error}\n')


if __name__ == '__main__':
    main()

"""Persistent, fail-closed experiment ledger. No shell commands or LLM calls."""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT = ROOT / 'benchmarks/measurement.py'
CONTRACT = ROOT / 'serving/benchmark-contract.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'Artifact must be a regular file')
    return {'path': str(path), 'sha256': digest(path)}


def verify(item):
    require(digest(item['path']) == item['sha256'], f"Artifact changed: {item['path']}")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value):
    return finite(value) and value > 0


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.ledger-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(data, out, indent=2, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Ledger:
    def __init__(self, path, gpu_path=None):
        self.path = Path(path).resolve()
        self.gpu_path = Path(gpu_path or ROOT / 'results/rsi/gpu-lease.json').resolve()

    def gpu_claim(self, lease):
        self.gpu_path.parent.mkdir(parents=True, exist_ok=True)
        with self.gpu_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(self.gpu_path.read_text()) if self.gpu_path.exists() else None
            require(current is None, 'GPU leased by another ledger; verify terminal process before recovery')
            atomic(self.gpu_path, {**lease, 'ledger': str(self.path)})

    def gpu_release(self, token):
        with self.gpu_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(self.gpu_path.read_text())
            require(current and current['ledger'] == str(self.path) and
                    secrets.compare_digest(current['token'], token), 'Global GPU lease mismatch')
            atomic(self.gpu_path, None)

    @contextmanager
    def transaction(self, initialize=False, check_frozen=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if initialize:
                require(not self.path.exists(), 'Ledger already exists; immutable objective cannot be replaced')
                state = {}
            else:
                state = json.loads(self.path.read_text())
                if check_frozen:
                    verify(state['anchor'])
                    require(canonical(state['objective']) == state['objective_sha256'], 'Objective snapshot changed')
                    verify(state['measurement'])
                    if state['schema_version'] >= 2:
                        verify(state['benchmark_contract'])
                        pinned_image = json.loads(Path(state['benchmark_contract']['path']).read_text())['base']['image']
                        require(state['expected_container_image'] == pinned_image and
                                state['expected_container_digest'] == pinned_image.split('@', 1)[1],
                                'Stored container pin differs from frozen benchmark contract')
            yield state
            atomic(self.path, state)

    def initialize(self, anchor, budget_usd=25, hourly_usd=5, max_run_minutes=45):
        require(all(positive(x) for x in (budget_usd, hourly_usd, max_run_minutes)), 'Budgets must be finite and positive')
        with self.transaction(initialize=True) as state:
            source = artifact(anchor)
            objective = json.loads(Path(source['path']).read_text())
            require(objective.get('target_speedup') == 3, 'Expected the frozen threefold target')
            contract = artifact(CONTRACT)
            image = json.loads(Path(contract['path']).read_text())['base']['image']
            require(isinstance(image, str) and '@sha256:' in image, 'Pinned image digest required in benchmark contract')
            image_digest = image.split('@', 1)[1]
            require(len(image_digest) == 71 and all(c in '0123456789abcdef' for c in image_digest[7:]), 'Invalid image digest')
            state.update(schema_version=2, anchor=source, objective=objective,
                         benchmark_contract=contract, expected_container_image=image,
                         expected_container_digest=image_digest,
                         objective_sha256=canonical(objective), measurement=artifact(MEASUREMENT),
                         budget={'usd': budget_usd, 'hourly_usd': hourly_usd,
                                 'max_run_minutes': max_run_minutes, 'spent_usd': 0},
                         experiments={}, gpu_lease=None, consumed_holdouts=[],
                         local_incumbent=None, full_serving_goal_achieved=False, events=[])
            self.event(state, 'initialized', 'supervisor', {})
        return self.status()

    @staticmethod
    def event(state, action, actor, data):
        require(isinstance(actor, str) and actor.strip(), 'Actor identity required')
        state['events'].append({'sequence': len(state['events']), 'time_unix': time.time(),
                                'action': action, 'actor': actor, 'data': data})

    @staticmethod
    def unchanged(exp):
        require(canonical(exp['plan']) == exp['plan_sha256'], 'Preregistered plan changed')
        for item in exp['artifacts']:
            verify(item)

    def propose(self, experiment_id, author, proposal):
        with self.transaction() as state:
            require(experiment_id and experiment_id not in state['experiments'], 'Unique experiment ID required')
            plan = copy.deepcopy(proposal)
            require(plan['kind'] in ('kernel', 'optimizer_method'), 'Unknown loop kind')
            require(isinstance(plan['hypothesis'], str) and plan['hypothesis'].strip(), 'Falsifiable hypothesis required')
            shapes = state['objective']['local_primary_batches'] + state['objective']['regression_batches']
            require(sorted(plan['suite']) == sorted(set(shapes)), 'Exact frozen primary+regression suite required')
            seeds = plan['holdout_seeds']
            require(len(seeds) >= 2 and len(set(seeds)) == len(seeds) and
                    all(isinstance(x, int) and not isinstance(x, bool) for x in seeds), 'At least two distinct integer holdout seeds required')
            require(not set(seeds) & set(plan['screen_seeds']), 'Screen and holdout seeds overlap')
            require(not set(seeds) & set(state['consumed_holdouts']), 'Holdout was already consumed; preregister fresh seeds')
            require(isinstance(plan['min_pairs'], int) and plan['min_pairs'] >= 20 and plan['min_pairs'] % 2 == 0,
                    'Pairs must be even and >=20')
            require(positive(plan['max_gpu_minutes']) and plan['max_gpu_minutes'] <= state['budget']['max_run_minutes'], 'Run budget exceeds policy')
            roi = plan['roi_prior']
            require(roi['basis'] == 'engineering_prior' and len(roi['speedup_range']) == 2 and
                    all(positive(x) for x in roi['speedup_range']) and roi['speedup_range'][0] <= roi['speedup_range'][1], 'ROI must be labeled finite engineering prior')
            require(plan.get('baseline_isolation') in ('immutable_tree', 'isolated_process'), 'Baseline isolation strategy required')
            require(digest(plan['baseline_anchor_source']) == state['objective']['anchor_source_sha256'], 'Baseline source does not match frozen goal anchor')
            require(str(Path(plan['baseline_anchor_source']).resolve()) in
                    [str(Path(p).resolve()) for p in plan['dependencies']], 'Baseline anchor source must be in dependency bundle')
            paths = [plan['candidate'], plan['baseline'], *plan['dependencies']]
            require(len(paths) >= 3, 'Transitive source/fixture/harness dependency manifest required')
            snapshots = [artifact(p) for p in paths]
            require(snapshots[0]['path'] != snapshots[1]['path'], 'Baseline and candidate must be distinct artifacts')
            if plan['kind'] == 'optimizer_method':
                require(plan.get('method_trial_plan'), 'Optimizer changes require a frozen method-trial plan')
                snapshots.append(artifact(plan['method_trial_plan']))
            exp = {'id': experiment_id, 'author': author, 'state': 'hypothesis', 'revision': 1,
                   'plan': plan, 'plan_sha256': canonical(plan), 'artifacts': snapshots, 'evidence': []}
            state['experiments'][experiment_id] = exp
            self.event(state, 'proposed', author, {'id': experiment_id, 'plan_sha256': exp['plan_sha256']})
        return exp

    def review(self, experiment_id, reviewer, checklist):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] == 'hypothesis', 'Only a hypothesis can enter preflight review')
            require(reviewer != exp['author'], 'Author cannot independently review own hypothesis')
            self.unchanged(exp)
            required = ('baseline_bundle_complete', 'candidate_bundle_complete', 'baseline_isolated',
                        'hypothesis_falsifiable', 'suite_and_holdouts_frozen', 'precision_preserved')
            require(all(checklist.get(k) is True for k in required), 'Preflight checklist incomplete')
            exp.update(state='reviewed', preflight_reviewer=reviewer, preflight_checklist=checklist)
            self.event(state, 'reviewed', reviewer, {'id': experiment_id})
        return exp

    def start(self, experiment_id, executor):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(state['schema_version'] >= 2, 'Legacy v1 is read-only for execution: initialize a v2 ledger and re-review before running')
            require(exp['state'] == 'reviewed', 'Run requires reviewed hypothesis')
            self.unchanged(exp)
            require(state['gpu_lease'] is None, 'GPU already leased; never steal an expired-looking lease')
            reserve = exp['plan']['max_gpu_minutes'] / 60 * state['budget']['hourly_usd']
            require(state['budget']['spent_usd'] + reserve <= state['budget']['usd'], 'Insufficient remaining run budget')
            token = secrets.token_hex(16)
            lease = {'experiment': experiment_id, 'executor': executor, 'token': token,
                     'started_at': time.time(), 'reserved_usd': reserve,
                     'deadline_unix': time.time() + exp['plan']['max_gpu_minutes'] * 60}
            require(isinstance(executor, str) and executor.strip(), 'Executor identity required')
            self.gpu_claim(lease)
            state['gpu_lease'] = lease
            exp.update(state='running', executor=executor)
            state['consumed_holdouts'].extend(exp['plan']['holdout_seeds'])
            self.event(state, 'started', executor, {'id': experiment_id, 'deadline_unix': lease['deadline_unix']})
        return lease

    def release(self, state, experiment_id, executor, token):
        lease = state['gpu_lease']
        require(lease is not None and lease['experiment'] == experiment_id and
                lease['executor'] == executor and secrets.compare_digest(lease['token'], token), 'GPU lease ownership mismatch')
        elapsed = max(0, time.time() - lease['started_at'])
        spent = elapsed / 3600 * state['budget']['hourly_usd']
        state['budget']['spent_usd'] += spent
        self.gpu_release(token)
        state['gpu_lease'] = None
        return {'elapsed_seconds': elapsed, 'accounted_gpu_usd': spent,
                'deadline_exceeded': time.time() > lease['deadline_unix']}

    def finish(self, experiment_id, executor, token, manifest):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] == 'running', 'Only running experiments can record measurements')
            self.unchanged(exp)
            # Terminal process proof is an executor attestation, never a stale statefile.
            require(manifest.get('process_terminal') is True and manifest.get('process_handle'), 'Terminal process handle/proof required')
            evidence = []
            for run in manifest['runs']:
                evidence.append({'report': artifact(run['report']), 'correctness': artifact(run['correctness'])})
            require(evidence, 'At least one run artifact required, including failed runs')
            accounting = self.release(state, experiment_id, executor, token)
            exp.update(state='measured', evidence=evidence, terminal_proof=manifest['process_handle'], accounting=accounting)
            self.event(state, 'measured', executor, {'id': experiment_id, 'evidence': evidence})
        return exp

    def abort(self, experiment_id, executor, token, terminal_proof, reason):
        # Recovery must remain available after an artifact/anchor mutation. It can
        # only release owned resources and reject; it cannot validate or promote.
        with self.transaction(check_frozen=False) as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] == 'running', 'Abort requires a running experiment')
            require(terminal_proof.get('process_terminal') is True and
                    terminal_proof.get('process_handle') and reason.strip(), 'Abort requires verified terminal handle and reason')
            accounting = self.release(state, experiment_id, executor, token)
            exp.update(state='rejected', rejection_reason=reason, terminal_proof=terminal_proof,
                       accounting=accounting, aborted=True)
            self.event(state, 'aborted', executor, {'id':experiment_id, 'reason':reason, 'terminal_proof':terminal_proof})
        return exp

    def reject(self, experiment_id, reviewer, reason):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] in ('hypothesis', 'reviewed', 'measured', 'validated'), 'Running job requires finish with terminal proof before rejection')
            require(reviewer != exp['author'] and reason.strip(), 'Independent rejection reason required')
            exp.update(state='rejected', rejection_reason=reason)
            self.event(state, 'rejected', reviewer, {'id': experiment_id, 'reason': reason})
        return exp

    def validate(self, experiment_id, reviewer):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] == 'measured', 'Validation requires completed measurements')
            require(reviewer not in (exp['author'], exp['executor']), 'Validation reviewer must be independent of author and executor')
            self.unchanged(exp)
            scores = self.validate_evidence(state, exp)
            exp.update(state='validated', validation_reviewer=reviewer, recomputed_results=scores)
            self.event(state, 'validated', reviewer, {'id': experiment_id, 'scope': 'local_component_only'})
        return exp

    @staticmethod
    def validate_evidence(state, exp):
        require(state['schema_version'] >= 2, 'Legacy v1 evidence has no frozen image pin: initialize a v2 ledger; no retroactive promotion')
        spec = importlib.util.spec_from_file_location('rsi_measurement', state['measurement']['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = exp['plan']
        seen_seeds, sessions, results = set(), set(), []
        reference_environment = None
        for evidence in exp['evidence']:
            verify(evidence['report']); verify(evidence['correctness'])
            report = json.loads(Path(evidence['report']['path']).read_text())
            correctness = json.loads(Path(evidence['correctness']['path']).read_text())
            require(report.get('status') == 'complete', 'Incomplete run')
            require(report.get('candidate_sha256') == exp['artifacts'][0]['sha256'] and
                    report.get('baseline_sha256') == exp['artifacts'][1]['sha256'], 'Run provenance mismatch')
            require(report.get('rsi_plan_sha256') == exp['plan_sha256'], 'Report not bound to preregistered plan')
            require(report.get('dependency_hashes') == {a['path']: a['sha256'] for a in exp['artifacts'][2:]}, 'Transitive dependency provenance mismatch')
            require(correctness.get('report_sha256') == evidence['report']['sha256'], 'Correctness record not bound to report')
            require(all(correctness.get(k) is True for k in ('reference_pass', 'empty_local_eager_pass', 'empty_local_graph_pass', 'precision_preserved')), 'Correctness gate missing or failed')
            require(type(correctness.get('reference_cases')) is int and correctness['reference_cases'] > 0, 'Correctness case count missing')
            seed = report.get('seed')
            require(type(seed) is int, 'Integer report seed required')
            require(seed in plan['holdout_seeds'] and seed not in seen_seeds, 'Unexpected or repeated holdout seed')
            seen_seeds.add(seed)
            session = report.get('session_id')
            require(isinstance(session, str) and session and session not in sessions, 'Independent session ID required')
            sessions.add(session)
            env = report.get('environment', {})
            fingerprint_keys = ('gpu_uuid','driver','torch','cuda','container_digest')
            require(all(isinstance(env.get(k), str) and env[k].strip() for k in fingerprint_keys), 'Environment fingerprint incomplete')
            require(env['container_digest'] == state['expected_container_digest'], 'Container digest differs from frozen benchmark contract')
            fingerprint = {k: env[k] for k in fingerprint_keys}
            if reference_environment is None:
                reference_environment = fingerprint
            else:
                require(fingerprint == reference_environment, 'GPU/device/runtime fingerprint changed across holdout sessions')
            require(env.get('exclusive_gpu') is True and report.get('profiling') is False, 'Clean exclusive-GPU timing required')
            rows = report.get('results', [])
            require(len(rows) == len(plan['suite']) and sorted(r['bs'] for r in rows) == sorted(plan['suite']), 'Missing/duplicate/unexpected suite rows')
            clean = []
            for row in rows:
                baseline, candidate = row.get('baseline_samples_us', []), row.get('candidate_samples_us', [])
                require(len(baseline) == len(candidate) and len(baseline) >= plan['min_pairs'] and len(baseline) % 2 == 0, 'Insufficient or unmatched raw pairs')
                require(all(positive(x) for x in baseline + candidate), 'Nonfinite/nonpositive raw sample')
                summary = module.summarize(baseline, candidate)
                require(summary['bootstrap_95'][0] >= 1 / 1.02, f"Noninferiority failed at M={row['bs']}")
                clean.append({'bs': row['bs'], **summary})
            primary = [r for r in clean if r['bs'] in state['objective']['local_primary_batches']]
            require(any(r['bootstrap_95'][0] > 1.02 for r in primary), 'No established primary-shape gain')
            primary_geomean = math.exp(sum(math.log(r['balanced_speedup']) for r in primary)/len(primary))
            lower_geomean = math.exp(sum(math.log(r['bootstrap_95'][0]) for r in primary)/len(primary))
            require(primary_geomean > 1.02 and lower_geomean > 1.0, 'Primary aggregate gain not established')
            results.append({'seed':seed,'session':session,'rows':clean,
                            'primary_geomean':primary_geomean,
                            'primary_lower_geomean_screening_bound':lower_geomean,
                            'bound_note':'Geomean of marginal lower bounds; screening statistic, not a joint 95% confidence interval'})
        require(seen_seeds == set(plan['holdout_seeds']), 'All preregistered holdout sessions required')
        if plan['kind'] == 'optimizer_method':
            raise ValueError('Optimizer-method promotion requires a separate frozen portfolio trial; local kernel evidence is insufficient')
        return results

    def promote(self, experiment_id, reviewer):
        with self.transaction() as state:
            exp = state['experiments'][experiment_id]
            require(exp['state'] == 'validated' and reviewer == exp['validation_reviewer'], 'Independent validated review required')
            self.unchanged(exp)
            self.validate_evidence(state, exp)
            exp['state'] = 'promoted'
            state['local_incumbent'] = experiment_id
            self.event(state, 'promoted', reviewer, {'id':experiment_id, 'scope':'local_component_only', 'does_not_apply_patch':True})
        return exp

    def observe(self, experiment_id, actor, paths, note):
        with self.transaction() as state:
            require(experiment_id not in state['experiments'], 'Unique experiment ID required')
            require(paths and note.strip(), 'Observation artifacts and scope note required')
            exp = {'id': experiment_id, 'state': 'observational', 'actor': actor,
                   'artifacts': [artifact(p) for p in paths], 'note': note,
                   'promotion_eligible': False}
            state['experiments'][experiment_id] = exp
            self.event(state, 'legacy_observation', actor, {'id':experiment_id})
        return exp

    def status(self):
        with self.transaction() as state:
            result = copy.deepcopy(state)
            if result['gpu_lease']:
                result['gpu_lease'].pop('token', None)
        return result

"""Pure CPU scheduling validation and conditional ROI, never performance admission."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

STATES = {'BACKLOG', 'READY', 'RUNNING', 'VALIDATING', 'DONE', 'HELD', 'CANCELLED'}
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')

DEFAULT_GPU_IMAGE = 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel'
SGLANG_GPU_IMAGE = 'lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a'
GPU_IMAGE_TOOLS = {
    DEFAULT_GPU_IMAGE: ('proj_rsi/tools/h100_trial.sh', 'proj_rsi/tools/h100_container.sh', '/root/okc_run.sh'),
    SGLANG_GPU_IMAGE: ('proj_rsi/tools/h100_sglang_trial.sh', 'proj_rsi/tools/h100_sglang_container.sh', '/root/okc_run_sglang.sh'),
}

def launch_tools(launch):
    image = launch.get('image', DEFAULT_GPU_IMAGE)
    if not isinstance(image, str) or image not in GPU_IMAGE_TOOLS:
        raise ValueError('launch image must be one of the two explicitly allowed GPU images')
    return GPU_IMAGE_TOOLS[image]

def load_policy(path=None):
    policy = json.loads(Path(path or Path(__file__).with_name('policy.json')).read_text())
    return _validate_policy(policy)

def _validate_policy(policy):
    if not isinstance(policy, dict): raise ValueError('policy must be an object')
    if not _positive(policy.get('gpu_usd_per_hour')) or policy['gpu_usd_per_hour'] != 5:
        raise ValueError('policy gpu_usd_per_hour must equal authorized USD5 rate')
    for key in ('ready_capacity', 'active_capacity'):
        if not isinstance(policy.get(key),dict): raise ValueError(f'{key} must be an object')
        for resource in ('gpu', 'cpu'):
            value = policy.get(key, {}).get(resource)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key}.{resource} must be positive integer')
            ceiling = {'ready_capacity': {'gpu':10,'cpu':20}, 'active_capacity': {'gpu':1,'cpu':10}}[key][resource]
            if value > ceiling: raise ValueError(f'{key}.{resource} exceeds authorized cap')
    age = policy.get('aging', {})
    if not isinstance(age,dict): raise ValueError('aging must be an object')
    if not _number(age.get('max_bonus_fraction')) or not 0 <= age['max_bonus_fraction'] <= .1:
        raise ValueError('age bonus must be between zero and ten percent')
    if not _positive(age.get('full_bonus_seconds')):
        raise ValueError('full_bonus_seconds must be finite positive')
    if not _positive(policy.get('incumbent_score_us',3200.6150084373335)):
        raise ValueError('incumbent score must be finite positive')
    limits=policy.get('launch_limits', {'max_inner_seconds':3600,'max_outer_seconds':3720})
    if not isinstance(limits,dict) or any(not isinstance(limits.get(k),int) or isinstance(limits.get(k),bool) or limits[k] <=0 for k in ('max_inner_seconds','max_outer_seconds')):
        raise ValueError('launch limits must be positive integer seconds')
    if limits['max_outer_seconds'] <= limits['max_inner_seconds'] + 20:
        raise ValueError('outer launch limit must exceed inner launch limit +20')
    policy['launch_limits']=limits
    if not isinstance(policy.get('assumptions',[]),list) or not all(isinstance(v,str) for v in policy.get('assumptions',[])):
        raise ValueError('assumptions must be a list of strings')
    required = ['All probabilities are subjective conditional priors, not measured speedups.', 'No H100-to-H200 performance transfer is inferred.']
    policy['assumptions'] = list(dict.fromkeys(policy.get('assumptions',[]) + required))
    return policy

def _number(value):
    if not isinstance(value, (float, int)) or isinstance(value, bool): return False
    try: return math.isfinite(value)
    except (OverflowError, TypeError, ValueError): return False

def _positive(value):
    return _number(value) and value > 0

def _relative(path):
    return isinstance(path, str) and bool(path) and not any(ord(c)<32 for c in path) and '\\' not in path and '\x00' not in path and not PurePosixPath(path).is_absolute() and all(p not in ('..', '.') for p in path.split('/')) and '' not in path.split('/')

def _text(value):
    return isinstance(value, str) and bool(value.strip()) and '\x00' not in value

def ncu_target(argv):
    """Return actual Python target index for the reviewed NCU option grammar.

    Unknown flags/positionals fail closed; NCU never receives a shell target.
    Clock control must be explicitly disabled, once, before the target.
    """
    if not isinstance(argv,list) or not argv or argv[0] not in ('ncu','/usr/local/cuda/bin/ncu'):
        return None
    if not all(isinstance(v,str) and v and not any(ord(c)<32 for c in v) for v in argv):
        return None
    value_options={'--target-processes','--kill','--replay-mode','--graph-profiling',
        '--cache-control','--clock-control','--nvtx-include','--launch-count',
        '--section','--export','--log-file','--set','--metrics','--launch-skip',
        '--profile-from-start','--kernel-name','--kernel-name-base'}
    flag_options={'--nvtx','-f','--force-overwrite'}
    index=1
    clock_count=0
    while index < len(argv):
        token=argv[index]
        if token in flag_options:
            index+=1
            continue
        option,separator,value=token.partition('=')
        if option in value_options:
            if not separator:
                if index+1 >= len(argv): return None
                value=argv[index+1]
                index+=2
            else:
                index+=1
            if not value or value.startswith('-'): return None
            if option=='--clock-control':
                if value!='none':return None
                clock_count+=1
                if clock_count>1:return None
            continue
        # First positional is the actual profiled executable, never search later.
        if token not in ('python3','python','/usr/bin/python3') or clock_count!=1:
            return None
        if index+1 >= len(argv):return None
        driver=argv[index+1]
        if not driver.startswith('/okc/proj_rsi/') or not _relative(driver[len('/okc/'):]):
            return None
        return index
    return None

def validate_job(job, *, for_admission=False):
    return _validate_job(job, for_admission=for_admission, policy=load_policy())

def _validate_job(job, *, for_admission=False, policy):
    errors = []
    if not isinstance(job, dict):
        return ['job must be an object']
    for field in ('id', 'author'):
        if not isinstance(job.get(field), str) or not _ID.fullmatch(job[field]):
            errors.append(f'{field} must be a safe nonempty identifier')
    resource = job.get('resource')
    if resource not in ('gpu', 'cpu'): errors.append('resource must be gpu or cpu')
    if job.get('kind') not in ('optimization', 'diagnostic', 'implementation', 'review', 'analysis', 'profile', 'validation', 'import'):
        errors.append('kind must describe optimization/diagnostic/implementation/review/analysis/profile/validation/import')
    if not isinstance(job.get('state'),str) or job.get('state') not in STATES: errors.append('invalid state')
    if not _number(job.get('created_at')) or job.get('created_at', -1) < 0: errors.append('created_at must be finite nonnegative epoch seconds')
    for field in ('depends_on', 'write_paths'):
        if not isinstance(job.get(field), list): errors.append(f'{field} must be a list')
    for path in job.get('write_paths', []) if isinstance(job.get('write_paths'), list) else []:
        if not _relative(path): errors.append('write_paths must be repo-relative without traversal')
    for dep in job.get('depends_on', []) if isinstance(job.get('depends_on'), list) else []:
        depid = dep.get('id') if isinstance(dep, dict) else dep
        if not isinstance(depid, str) or not _ID.fullmatch(depid) or depid == job.get('id'):
            errors.append('dependency must name another safe job id')
    if job.get('state') in ('DONE', 'CANCELLED') and not for_admission and job.get('kind') == 'import':
        return errors
    estimate = job.get('estimate')
    if not isinstance(estimate, dict): estimate = {}; errors.append('estimate must be an object')
    for field in ('basis', 'confidence'):
        if not _text(estimate.get(field)): errors.append(f'estimate.{field} required')
    evidence = estimate.get('evidence')
    if not isinstance(evidence, list) or not evidence or not all(_relative(p) for p in evidence): errors.append('estimate.evidence requires repo-relative evidence paths')
    scenario_ceiling=policy.get('incumbent_score_us',3200.6150084373335)
    for branch, probability_keys in [('direct', ('p_gain', 'p_valid')), ('information', ('p_resolve', 'p_followup_gain'))]:
        values = estimate.get(branch)
        if not isinstance(values, dict): values = {}; errors.append(f'estimate.{branch} required')
        scenario = values.get('scenario_score_us')
        if not _number(scenario) or scenario < 0 or scenario > scenario_ceiling: errors.append(f'{branch}.scenario_score_us unknown or outside finite [0, official score ceiling]')
        for key in probability_keys:
            value = values.get(key)
            if not _number(value) or not 0 <= value <= 1: errors.append(f'{branch}.{key} unknown or outside [0,1]')
        if branch == 'information' and _number(scenario) and scenario > 0:
            if not _text(values.get('decision')): errors.append('information requires a concrete decision')
            unblocks = values.get('unblocks')
            if not isinstance(unblocks, list) or not unblocks or not all(isinstance(v, str) and _ID.fullmatch(v) and v != job.get('id') for v in unblocks): errors.append('information requires concrete unblocks job ids')
    if job.get('kind') in ('profile','diagnostic') and isinstance(estimate.get('direct'),dict) and estimate['direct'].get('scenario_score_us') != 0:
        errors.append('diagnostic/profile benefit must be information-only: direct scenario must equal zero')
    cost = job.get('cost')
    if not isinstance(cost, dict): cost = {}; errors.append('cost must be an object')
    for key in ('gpu_seconds', 'followup_gpu_seconds', 'cpu_minutes'):
        value = cost.get(key)
        if not _number(value) or value < 0: errors.append(f'cost.{key} unknown or not finite nonnegative')
    if resource == 'gpu' and not _positive(cost.get('gpu_seconds')): errors.append('gpu_seconds must be positive for GPU job')
    if resource == 'cpu' and not _positive(cost.get('cpu_minutes')): errors.append('cpu_minutes must be positive for CPU job')
    info = estimate.get('information', {})
    if resource == 'gpu' and isinstance(info, dict) and _number(info.get('scenario_score_us')) and info['scenario_score_us'] > 0 and not _positive(cost.get('followup_gpu_seconds')):
        errors.append('GPU information value requires positive followup_gpu_seconds chain cost')
    files = job.get('files', {})
    if not isinstance(files, dict) or any(not _relative(k) or not isinstance(v, str) or not _HASH.fullmatch(v) for k,v in files.items()): errors.append('files must map safe repo-relative paths to sha256')
    review = job.get('review')
    if review is not None:
        if not isinstance(review, dict): errors.append('review must be an object')
        else:
            if not _text(review.get('actor')) or review.get('actor') == job.get('author'): errors.append('review actor must be independent of author')
            if not _relative(review.get('path')) or not isinstance(review.get('sha256'), str) or not _HASH.fullmatch(review['sha256']): errors.append('review requires safe path and sha256')
            if review.get('decision') not in ('GO', 'HOLD', 'NO_GO'): errors.append('invalid review decision')
    launch = job.get('launch')
    if launch is not None:
        if not isinstance(launch, dict): errors.append('launch must be an object')
        else:
            allowed = {'host','remote_repo','container','inner_seconds','outer_seconds','profile','env','argv','artifacts','image'}
            if set(launch)-allowed: errors.append('unknown launch keys')
            try: launch_tools(launch)
            except ValueError as error: errors.append(str(error))
            if launch.get('host') != 'root@107.170.6.60': errors.append('launch host must be approved H100 host')
            if launch.get('remote_repo') != '/root/okc': errors.append('remote_repo must be /root/okc')
            if not isinstance(launch.get('container'), str) or not re.fullmatch(r'okc-proj-[a-z0-9-]{1,101}', launch['container']): errors.append('container must be safe unique okc-proj name')
            inner, outer = launch.get('inner_seconds'), launch.get('outer_seconds')
            limits=policy['launch_limits']
            if (not isinstance(inner,int) or isinstance(inner,bool) or not isinstance(outer,int) or isinstance(outer,bool) or not _positive(inner) or not _positive(outer) or inner > limits['max_inner_seconds'] or outer > limits['max_outer_seconds'] or outer <= inner + 20): errors.append('deadlines must be positive integer seconds within policy launch_limits and outer > inner + 20')
            if not isinstance(launch.get('profile'), bool): errors.append('profile must be boolean')
            if launch.get('profile') is True and job.get('kind') not in ('profile','diagnostic'):
                errors.append('profile=true requires profile/diagnostic kind')
            env = launch.get('env')
            if not isinstance(env, dict) or not all(isinstance(k,str) and re.fullmatch('[A-Z_][A-Z0-9_]*',k) and isinstance(v,str) and '\x00' not in v for k,v in env.items()): errors.append('launch env must be string environment map')
            elif any(k in env for k in ('LD_PRELOAD','BASH_ENV','ENV','PROJ_GPU_CONTAINER')): errors.append('reserved unsafe environment override')
            elif 'PROJ_GPU_PROFILE' in env and (not launch.get('profile') or env['PROJ_GPU_PROFILE'] != '1'): errors.append('PROJ_GPU_PROFILE=1 only with profile=true')
            argv = launch.get('argv')
            if not isinstance(argv,list) or not argv or not all(isinstance(v,str) and v and not any(ord(c)<32 for c in v) for v in argv): errors.append('argv must be nonempty literal argument list')
            elif argv[0] in ('ncu','/usr/local/cuda/bin/ncu'):
                if launch.get('profile') is not True or job.get('kind') not in ('profile','diagnostic'):
                    errors.append('ncu executable requires profile=true and profile/diagnostic kind')
                target=ncu_target(argv)
                if target is None:
                    errors.append('ncu requires supported literal options, explicit clock-control none, then actual Python target and driver under /okc/proj_rsi')
                elif not isinstance(files,dict) or argv[target+1][len('/okc/'):] not in files:
                    errors.append('ncu driver must be pinned in reviewed files')
            elif argv[0] not in ('python3','python','/usr/bin/python3'):
                errors.append('argv executable must be Python or explicitly reviewed ncu profile; no shell launcher')
            artifacts=launch.get('artifacts')
            if not isinstance(artifacts,list) or not artifacts or not all(isinstance(v,str) and v.startswith('/root/okc/') and '..' not in PurePosixPath(v).parts and '\x00' not in v for v in artifacts): errors.append('artifacts must be absolute paths under remote repository')
    if for_admission and resource == 'gpu':
        if not files: errors.append('GPU admission requires pinned files')
        if isinstance(files,dict):
            try: selected_tools = launch_tools(launch)[:2] if isinstance(launch,dict) else ()
            except ValueError: selected_tools = ()
            for required in selected_tools:
                if required not in files: errors.append(f'GPU admission requires pinned tool: {required}')
        if not isinstance(review,dict) or review.get('decision') != 'GO': errors.append('GPU admission requires independent GO review')
        if launch is None: errors.append('GPU admission requires launch descriptor')
        if job.get('kind') == 'import': errors.append('historical imports must never execute')
    return errors

def score(job, resource=None, now=None, policy=None):
    policy = _validate_policy(copy.deepcopy(policy)) if policy is not None else load_policy()
    errors = _validate_job(job, policy=policy)
    if not isinstance(job,dict):
        return {'priority':None,'direct_expected_us':None,'information_expected_us':None,'effective_value_us':None,'cost_usd':None,'cpu_minutes':None,'age_multiplier':1.0,'assumptions':list(policy.get('assumptions',[])),'errors':errors,'rankable':False}
    if job.get('kind') == 'import': errors.append('historical import is unrankable')
    chosen = resource or job.get('resource')
    if chosen != job.get('resource'): errors.append('resource override does not match job resource')
    output = {'priority':None,'direct_expected_us':None,'information_expected_us':None,'effective_value_us':None,'cost_usd':None,'cpu_minutes':None,'age_multiplier':1.0,'assumptions':list(policy.get('assumptions',[])), 'errors':errors,'rankable':False}
    if errors: return output
    direct=job['estimate']['direct']; info=job['estimate']['information']; cost=job['cost']
    d=direct['scenario_score_us']*direct['p_gain']*direct['p_valid']
    i=info['scenario_score_us']*info['p_resolve']*info['p_followup_gain']
    effective=max(d,i)
    usd=(cost['gpu_seconds']+cost['followup_gpu_seconds'])*policy['gpu_usd_per_hour']/3600
    denominator=usd if chosen=='gpu' else cost['cpu_minutes']
    if not _positive(denominator):
        output['errors'].append('ROI denominator underflow/nonfinite'); return output
    timestamp=time.time() if now is None else now
    if not _number(timestamp): output['errors'].append('now must be finite'); return output
    age=max(0,timestamp-job['created_at']); aging=policy['aging']
    multiplier=1+min(aging['max_bonus_fraction'],aging['max_bonus_fraction']*age/aging['full_bonus_seconds'])
    priority=effective/denominator*multiplier
    if not all(math.isfinite(v) for v in (d,i,effective,usd,priority)):
        output['errors'].append('ROI arithmetic overflow/nonfinite'); return output
    output.update(priority=priority,direct_expected_us=d,information_expected_us=i,effective_value_us=effective,cost_usd=usd,cpu_minutes=cost['cpu_minutes'],age_multiplier=multiplier,rankable=True)
    output['assumptions'].append(job['estimate']['basis'])
    return output

"""Read-only, stdlib lesson validation/query. No GPU, network, queue or API actions."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path(__file__).with_name('learnings.json')
class ValidationError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ValidationError(message)

def keys(value, expected, where):
    require(isinstance(value, dict) and set(value) == set(expected.split()), f'{where}: exact keys required: {expected}')

def nonempty(value, where):
    require(isinstance(value, str) and bool(value.strip()), f'{where}: nonempty text required')

def digest(record):
    body = {k:v for k,v in record.items() if k != 'record_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def checked_path(root, relative):
    nonempty(relative, 'path')
    p = (root / relative).resolve()
    require(not Path(relative).is_absolute() and p.is_relative_to(root.resolve()), 'evidence/source path escapes root')
    require(p.is_file(), f'missing evidence/source: {relative}')
    return p

def pointer(value, address):
    require(isinstance(address, str) and (address == '' or address.startswith('/')), 'invalid JSON pointer')
    for part in address.split('/')[1:]:
        require(not re.search(r'~(?![01])',part), 'invalid JSON pointer escape')
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value,list): require(re.fullmatch(r'0|[1-9][0-9]*',part) is not None, 'invalid JSON pointer array index')
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValidationError(f'unresolved JSON pointer: {address}') from error
    return value

def exact_json_equal(actual, expected):
    if type(actual) is not type(expected): return False
    if isinstance(actual,dict): return set(actual)==set(expected) and all(exact_json_equal(actual[k],expected[k]) for k in actual)
    if isinstance(actual,list): return len(actual)==len(expected) and all(exact_json_equal(a,b) for a,b in zip(actual,expected))
    return actual==expected

def estimate(item):
    keys(item, 'kind value unit conditions', 'conditional benefit')
    require(item['kind'] == 'conditional_scenario_not_measured', 'predicted benefit cannot be a measured/counterfactual gain')
    require(item['value'] is None or (type(item['value']) in (int,float) and math.isfinite(item['value']) and item['value'] >= 0), 'invalid conditional value')
    require(item['unit'] in ('us', 'bytes', 'decision'), 'invalid benefit unit')
    require(isinstance(item['conditions'], list) and item['conditions'], 'conditional benefit requires conditions')
    for value in item['conditions']: nonempty(value, 'benefit condition')

def cost(item):
    keys(item, 'basis cpu_minutes gpu_seconds', 'cost')
    nonempty(item['basis'], 'cost basis')
    for field in ('cpu_minutes', 'gpu_seconds'):
        require(item[field] is None or (type(item[field]) in (int,float) and math.isfinite(item[field]) and item[field] >= 0), 'invalid cost; unknown must be null')

def chain(records, where):
    previous = {}
    for record in records:
        nonempty(record['id'], where+' id')
        require(type(record['version']) is int and record['version'] >= 1, 'invalid record version')
        require(record['record_sha256'] == digest(record), f'{record["id"]}: immutable record digest mismatch')
        old = previous.get(record['id'])
        require(record['version'] == (old['version']+1 if old else 1), 'versions must append sequentially')
        require(record['supersedes'] == (old['record_sha256'] if old else None), 'broken supersedes chain')
        previous[record['id']] = record
    return previous

def validate(data, root=ROOT):
    keys(data, 'schema_version registry_version lessons ideas', 'registry')
    require(type(data['schema_version']) is int and data['schema_version'] == 1 and type(data['registry_version']) is int and data['registry_version'] >= 1, 'unsupported registry version')
    require(isinstance(data['lessons'],list) and data['lessons'], 'empty lesson registry')
    require(isinstance(data['ideas'],list), 'ideas must be a list')
    lesson_keys = 'id version supersedes record_sha256 title tags context evidence observations hypotheses scope transfer_limits confidence falsification next_test related_queue_ids conditional_benefit cost'
    references = set()
    for lesson in data['lessons']:
        keys(lesson, lesson_keys, 'lesson')
        for field in ('title','scope','falsification'): nonempty(lesson[field], field)
        for field in ('tags','transfer_limits','related_queue_ids'):
            require(isinstance(lesson[field],list) and lesson[field], f'empty {field}')
            for value in lesson[field]: nonempty(value, field)
        c = lesson['context']; keys(c, 'experiment_id family parents comparators hardware validity disposition sources', 'context')
        for field in ('experiment_id','family'): nonempty(c[field], field)
        require(c['validity'] in ('VALID_LOCAL','VALID_OFFICIAL_CHECK','VALIDATED_DIAGNOSIS','CPU_VALIDATED'), 'unvalidated experiment cannot supply a lesson')
        require(c['disposition'] in ('VALID_GAIN','VALID_REGRESSION','VALID_MIXED','INVALID_SETUP','CPU_ACCOUNTING'), 'invalid outcome disposition')
        keys(c['hardware'], 'kind model', 'hardware')
        require(c['hardware']['kind'] in ('CPU','GPU'), 'hardware kind')
        nonempty(c['hardware']['model'], 'hardware model')
        if c['validity']=='CPU_VALIDATED':
            require(c['hardware']['kind']=='CPU' and c['hardware']['model'].startswith('CPU'), 'CPU evidence cannot be promoted to a GPU observation')
        for field in ('parents','comparators'):
            require(isinstance(c[field],list) and c[field], f'missing {field}')
            for value in c[field]: nonempty(value, field)
        require(isinstance(c['sources'],list) and c['sources'], 'missing source parents')
        for source in c['sources']:
            keys(source,'path sha256 role','source'); nonempty(source['role'],'source role')
            require(hashlib.sha256(checked_path(root,source['path']).read_bytes()).hexdigest()==source['sha256'], 'stale source hash: '+source['path'])
        require(isinstance(lesson['evidence'],list) and lesson['evidence'], 'missing evidence')
        evidence = {}
        for entry in lesson['evidence']:
            keys(entry,'id path sha256','evidence'); nonempty(entry['id'],'evidence id')
            require(entry['id'] not in evidence, 'duplicate evidence id')
            raw=checked_path(root,entry['path']).read_bytes()
            require(hashlib.sha256(raw).hexdigest()==entry['sha256'], 'stale evidence hash: '+entry['path'])
            evidence[entry['id']] = raw
        require(isinstance(lesson['observations'],list) and lesson['observations'], 'empty outcomes are prohibited')
        facts = set()
        for observation in lesson['observations']:
            keys(observation,'id statement outcome evidence selector','observation')
            nonempty(observation['id'],'observation id'); nonempty(observation['statement'],'observation statement')
            require(observation['id'] not in facts,'duplicate observation id'); facts.add(observation['id'])
            require(observation['outcome'] in ('positive','negative','mixed','neutral','invalid'), 'negative/invalid evidence must remain explicit')
            require(observation['evidence'] in evidence,'unbound observation')
            sel=observation['selector']; raw=evidence[observation['evidence']]
            require(isinstance(sel,dict), 'observation selector must be an object')
            if set(sel)=={'json_pointer','equals'}:
                try: actual=pointer(json.loads(raw),sel['json_pointer'])
                except (json.JSONDecodeError,UnicodeDecodeError) as e: raise ValidationError('JSON selector on non-JSON evidence') from e
                require(exact_json_equal(actual,sel['equals']), 'observation does not match evidence')
            elif set(sel)=={'contains'}:
                nonempty(sel['contains'],'text anchor'); require(sel['contains'] in raw.decode(), 'observation text anchor missing')
            else: raise ValidationError('observation requires exact JSON value or text anchor')
        outcomes={o['outcome'] for o in lesson['observations']}
        if c['disposition'] in ('VALID_REGRESSION','VALID_MIXED'): require('negative' in outcomes, 'negative evidence cannot be erased from a regression/mixed lesson')
        if c['disposition']=='VALID_GAIN': require('positive' in outcomes, 'VALID_GAIN requires positive evidence')
        if c['disposition']=='VALID_MIXED': require(bool(outcomes & {'positive','mixed'}), 'mixed result requires positive or mixed evidence as well as negative')
        if c['disposition']=='INVALID_SETUP': require('invalid' in outcomes, 'invalid setup evidence cannot be relabeled valid')
        require(isinstance(lesson['hypotheses'],list) and lesson['hypotheses'],'missing causal hypothesis')
        for h in lesson['hypotheses']:
            keys(h,'statement status observation_ids','hypothesis'); nonempty(h['statement'],'hypothesis statement')
            require(h['status'] in ('unproven','refuted','supported_in_scope'), 'hypothesis is not a measured causal conclusion')
            require(isinstance(h['observation_ids'],list) and h['observation_ids'], 'hypothesis requires bound facts')
            for value in h['observation_ids']: nonempty(value, 'hypothesis fact id')
            require(set(h['observation_ids']) <= facts, 'hypothesis requires bound facts')
        keys(lesson['confidence'],'observation causality transfer','confidence')
        require(all(v in ('low','medium','high') for v in lesson['confidence'].values()),'invalid confidence')
        require(lesson['confidence']['transfer']=='low', 'unsupported transfer confidence; record a separate validated target experiment')
        # This registry never infers transfer from a different GPU, CPU byte ledger,
        # source commutativity, or an official prepare bake. A validated transfer
        # requires its own new in-scope lesson, not relabeling an old observation.
        for limit in lesson['transfer_limits']:
            nonempty(limit,'transfer limit')
        test=lesson['next_test']; keys(test,'question controls discriminator stop_rule required_gates','next test')
        for field in ('question','discriminator','stop_rule'): nonempty(test[field],field)
        for field in ('controls','required_gates'):
            require(isinstance(test[field],list) and test[field],f'empty next-test {field}')
            for value in test[field]: nonempty(value,field)
        estimate(lesson['conditional_benefit']); cost(lesson['cost'])
        references.add(lesson['id']+'@'+str(lesson['version']))
    latest = chain(data['lessons'],'lesson')
    for idea in data['ideas']:
        keys(idea,'id version supersedes record_sha256 title tags lesson_refs existing_roi_ids status mechanism discriminator conditional_benefit cost transfer_status','idea')
        require(idea['status']=='proposed' and idea['transfer_status']=='unproven', 'unsupported transfer/counterfactual promotion')
        for field in ('title','mechanism','discriminator'): nonempty(idea[field],field)
        for field in ('tags','lesson_refs','existing_roi_ids'):
            require(isinstance(idea[field],list) and idea[field], f'empty idea {field}')
            for value in idea[field]: nonempty(value, 'idea '+field)
        require(set(idea['lesson_refs']) <= references,'idea refers to absent lesson version')
        estimate(idea['conditional_benefit']); cost(idea['cost'])
    ideas = chain(data['ideas'],'idea')
    return {'status':'VALID','schema_version':1,'registry_version':data['registry_version'],'lesson_versions':len(data['lessons']),'latest_lessons':len(latest),'idea_versions':len(data['ideas']),'GPU_calls':0}

def latest(records):
    return list({record['id']:record for record in records}.values())

def query(data, text, hardware=None):
    nonempty(text,'query')
    words=set(re.findall(r'[a-z0-9]+',text.lower())); require(bool(words),'query requires terms')
    results=[]
    for kind in ('lessons','ideas'):
        for record in latest(data[kind]):
            hay=json.dumps(record,sort_keys=True).lower(); score=sum(len(re.findall(r'\b'+re.escape(w)+r'\b',hay)) for w in words)
            if score and (not hardware or kind=='ideas' or hardware.lower() in record['context']['hardware']['model'].lower()):
                results.append({'kind':kind,'id':record['id'],'version':record['version'],'title':record['title'],'score':score,'transfer':'unproven; inspect linked scope','scope':record.get('scope'),'hardware':record.get('context',{}).get('hardware'),'lesson_refs':record.get('lesson_refs',[]),'negative_evidence':[o['statement'] for o in record.get('observations',[]) if o['outcome'] in ('negative','invalid')],'next_test':record.get('next_test',record.get('discriminator'))})
    return sorted(results,key=lambda r:(-r['score'],r['id']))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registry',type=Path,default=DEFAULT)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('validate');sub.add_parser('summarize')
    q=sub.add_parser('query');q.add_argument('text');q.add_argument('--hardware')
    args=parser.parse_args(argv)
    try:
        data=json.loads(args.registry.read_text());result=validate(data)
        if args.command=='summarize':result={**result,'lessons':[{'id':x['id'],'version':x['version'],'title':x['title'],'outcomes':[o['outcome'] for o in x['observations']],'scope':x['scope']} for x in latest(data['lessons'])],'proposed_ideas':[x['id'] for x in latest(data['ideas'])]}
        elif args.command=='query':result=query(data,args.text,args.hardware)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValidationError,OSError,json.JSONDecodeError) as error:
        print(json.dumps({'status':'INVALID','error':str(error)}),file=__import__('sys').stderr);return 1
if __name__=='__main__':raise SystemExit(main())

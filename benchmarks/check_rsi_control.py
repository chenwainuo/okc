#!/usr/bin/env python3
"""Preregistered practical-equivalence check for an unchanged RSI control."""
import argparse
import json
import math
from pathlib import Path
from measurement import summarize


def check(report,plan):
    if report.get('status')!='complete' or report.get('run_kind')!='control' or report.get('profiling') is not False:
        raise ValueError('Completed clean control required')
    if report.get('candidate_sha256')!=report.get('baseline_sha256'):
        raise ValueError('Control source bundles differ')
    if report.get('seed') not in plan['screen_seeds']:
        raise ValueError('Control exposed a holdout/unregistered seed')
    if not all(report.get('correctness',{}).get(k) is True for k in
               ('reference_pass','empty_local_eager_pass','empty_local_graph_pass')):
        raise ValueError('Control correctness failed')
    if report.get('process_groups_terminated') is not True:
        raise ValueError('Worker termination unverified')
    rows=report.get('results',[])
    if len(rows)!=len(plan['suite']) or sorted(r['bs'] for r in rows)!=sorted(plan['suite']):
        raise ValueError('Control suite incomplete or duplicated')
    pairs=plan['control_policy']['pairs'];lower=1/1.02;upper=1.02
    result=[]
    for row in rows:
        a,b=row['baseline_samples_us'],row['candidate_samples_us']
        if len(a)!=pairs or len(b)!=pairs:raise ValueError('Unregistered control sample count')
        summary=summarize(a,b)
        result.append({'bs':row['bs'],**summary,'equivalent':summary['bootstrap_95'][0]>=lower and summary['bootstrap_95'][1]<=upper})
    primary=plan['control_policy']['primary_batches']
    geo=math.exp(sum(math.log(r['balanced_speedup']) for r in result if r['bs'] in primary)/len(primary))
    return {'passed':all(r['equivalent'] for r in result) and lower<=geo<=upper,
            'primary_geomean_point_estimate':geo,'equivalence_band':[lower,upper],
            'rows':result,'scope':'practical equivalence; marginal bootstrap intervals, not a simultaneous or aggregate CI'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=check(json.loads(a.report.read_text()),json.loads(a.plan.read_text()))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed']}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()

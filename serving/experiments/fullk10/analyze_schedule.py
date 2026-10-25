#!/usr/bin/env python3
"""CPU model of the existing stripe arithmetic, not a GPU performance test."""
import json
import math
from pathlib import Path
import statistics

K_TILES = 64

def cdiv(a,b): return (a+b-1)//b

def rounded_iters(output_tiles,grid):
    old=cdiv(output_tiles*K_TILES,grid)
    new=cdiv(old,K_TILES)*K_TILES if old else 0
    use=old>0 and output_tiles>=2*grid and new<=old*11//10 and new<=2147483647
    return old,new if use else old,use and new!=old


def verify_partition(output_tiles,grid,iters):
    total=output_tiles*K_TILES
    last=0
    splits=0
    for cta in range(grid):
        start=iters*cta
        stop=min(iters*(cta+1),total)
        if start>=total:continue
        assert start==last
        last=stop
        row=start%K_TILES
        col=start//K_TILES
        while col*K_TILES+row<stop:
            slice_iters=min(stop-(col*K_TILES+row),K_TILES-row)
            assert slice_iters>0
            col_first=iters*cdiv(K_TILES*col,iters)
            count=1
            if col_first<=K_TILES*(col+1):
                off=col_first-K_TILES*col
                count=cdiv(K_TILES-off,iters)+(off>0)
            if count>1:splits+=1
            if iters%K_TILES==0:
                assert row==0 and slice_iters==K_TILES and count==1
            col+=1;row=0
    assert last==total
    return splits


def main():
    checks=0;changed=0
    for grid in (1,132,264,396,528):
        for output_tiles in list(range(0,1025))+[2048,4096,8192,16384]:
            old,new,use=rounded_iters(output_tiles,grid)
            verify_partition(output_tiles,grid,old)
            verify_partition(output_tiles,grid,new)
            if use:
                assert new*10<=old*11
                assert verify_partition(output_tiles,grid,new)==0
                changed+=1
            checks+=1
    source=Path('results/selected-validation.json')
    report=json.loads(source.read_text())
    shape_rows=[]
    for row in report['results']:
        m=row['bs']
        if m not in (1,8,16,32,64,128,256):continue
        bm=8 if m<=16 else 16
        samples=[]
        for route in row['routing_samples']:
            valid=sum(cdiv(n,bm) for n in route['routes_per_local_expert'])
            invalid=cdiv(route['route_slots']-route['local_routes'],bm)
            one={'valid_blocks':valid,'invalid_prefix_blocks':invalid,'total_blocks':valid+invalid,
                 'count_scan_CTA_iterations_per_GEMM':396*(valid+invalid)}
            for name,nt in [('up',64),('down',32)]:
                outputs=valid*nt
                old,new,use=rounded_iters(outputs,396)
                boundaries=[cta*old for cta in range(1,396) if cta*old<outputs*64 and (cta*old)%64]
                split_tiles=len({b//64 for b in boundaries})
                one[name]={'output_tiles':outputs,'stripe_k_tiles':old,'split_output_tile_fraction':split_tiles/outputs if outputs else 0,
                           'cross_CTA_K_boundaries':len(boundaries),
                           'fp32_scratch_read_plus_write_upper_bytes':len(boundaries)*2*bm*128*4,
                           'unconditional_rounding_max_stripe_ratio':cdiv(old,64)*64/old if old else 1,
                           'bounded_guard_changes_schedule':bool(use and 128<=m<=256),
                           'bounded_stripe_k_tiles':new if 128<=m<=256 else old}
            samples.append(one)
        shape_rows.append({'M':m,'M_tile':bm,'samples':len(samples),
            'median_valid_blocks':statistics.median(r['valid_blocks'] for r in samples),
            'median_invalid_prefix_blocks':statistics.median(r['invalid_prefix_blocks'] for r in samples),
            'median_count_scan_CTA_iterations_per_GEMM':statistics.median(r['count_scan_CTA_iterations_per_GEMM'] for r in samples),
            'stages':{name:{'median_output_tiles':statistics.median(r[name]['output_tiles'] for r in samples),
                'median_split_output_tile_fraction':statistics.median(r[name]['split_output_tile_fraction'] for r in samples),
                'median_cross_CTA_K_boundaries':statistics.median(r[name]['cross_CTA_K_boundaries'] for r in samples),
                'median_fp32_scratch_read_plus_write_upper_bytes':statistics.median(r[name]['fp32_scratch_read_plus_write_upper_bytes'] for r in samples),
                'median_unconditional_rounding_max_stripe_ratio':statistics.median(r[name]['unconditional_rounding_max_stripe_ratio'] for r in samples),
                'fraction_samples_changed_by_candidate':sum(r[name]['bounded_guard_changes_schedule'] for r in samples)/len(samples)} for name in ('up','down')}})
    result={'status':'cpu_arithmetic_model_pass','partition_cases_checked':checks,'cases_with_bounded_schedule_change':changed,
            'source_report':str(source),'assumptions':['N tile128,K tile64; grid396 CTAs from incumbent Nsight M8/M128; confirm runtime grid before extrapolating.','Histograms reconstruct alignment padding; invalid bucket is ordered first by the +1 bucket convention.','Source-level CTA loop and scratch upper-bound counts, not executed instructions, DRAM traffic, or measured time.','FP32 scratch upper bound ignores partial-row masks; real traffic may be smaller.'],'shapes':shape_rows}
    Path(__file__).with_name('schedule-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

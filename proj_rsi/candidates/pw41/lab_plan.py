"""CPU-validated narrow discriminator plan; deliberately not a GPU executor.

The executable GPU wrapper is a separate review/registration step. Do not turn
this plan into an unreviewed kernel launch, reset prepared counters, or force a
serving choice. This module imports no GPU libraries and launches no processes.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_patch import HERE,PARENT,TARGETS,LATE,catalog
from protocol_proof import admit

def proof():
    old=catalog(PARENT.read_text());new=catalog((HERE/'submission.py').read_text())
    cells=[('router',8,256,8192,0),('router',16,256,8192,1),('router',32,256,8192,2),('unpooler',1,1024,4096,3),('unpooler',8,1024,4096,3)]
    cases=[]
    for layer,m,n,k,which in cells:
        args=TARGETS[which];late=args[:-1]+(args[-1]|LATE,)
        assert args in old['GVI'] and late in new['GVI'];fmt,mt,st,wr,wk,ku,s,rt,xs,v=args
        assert k==s*wk*ku*64 and m<=8*mt and n%(16*wr*rt)==0
        tiles=((m+8*mt-1)//(8*mt))*(n//(16*wr*rt));admit(73*s,73,s)
        cases.append(dict(layer=layer,m=m,n=n,k=k,early_args=args,late_args=late,early_index=old['GVI'].index(args),late_index=new['GVI'].index(late),tiles=tiles,S=s,grid_CTAs=tiles*(s+int(bool(v&(1<<29)))),counter_delta_per_completed_launch={'even_per_tile':s,'odd_per_tile':1,'all_other_storage':'UNCHANGED'},scope='AFFECTED_SCORED_P1_CASE; local actual selection must be recorded, not assumed'))
    return dict(status='CPU_VALIDATED_NARROW_TEST_PLAN_NOT_EXECUTABLE_GPU_WRAPPER_PENDING_REVIEW',source_sha256=hashlib.sha256((HERE/'submission.py').read_bytes()).hexdigest(),cases=cases,
      arms=['Actual full prepared PW38 kernel_fn policy','Fixed geometry early-ticket original kernel','Same fixed geometry late-ticket kernel'],
      allocation_control='For the fixed early/late causal comparison use the SAME source-bound counter storage/output/workspace roles, one ordered stream and no counter reset; swap/match pointer roles in a separately frozen harness if needed. Actual selected policy remains a distinct pipeline control.',
      semantics='Fixed-arm progress is mechanism evidence only. If local prepare selects a different path, retain that fact; never relabel fixed-arm improvement as actual-incumbent or cumulative-policy gain.',
      admission=['Independent source + exact executable wrapper review and complete immutable closure before registration',
       'Before BOTH parent and candidate prepare, not merely after selection: actual SM90 H100/H200 identity and at least128SMs; exact-symbol resources/local0 and actual gvocc full-grid eligibility before any P1 launch. Register/thread capacity calculations are upper bounds, never guaranteed occupancy.',
       'Retain every actual loaded SO/hash/PC stream and all95old controls/4new canonical target streams; do not accept missing modules',
       'AllKbasis and random FP64/task checks for all four fixed types, normal/outlier/probe and odd-m liveness; preserve FP32 router hi+lo precision',
       'Exact per-call full-storage counter/offset/stride/dtype/physical alias deltas, nonzero dormant baselines and all W/packed/CSR/input immutability; no harness reset',
       'Fresh original-X retained graphs, output/float scratch NaN overwrite, zero/outlier/all-row probe mutations; classify unknown persistent workspace HOLD',
       'Balanced same-fixture F1 and paired complete-pipeline timing, every signed raw call retained, own-fixture actual last outputs checked BEFORE any replay/input mutation at both boundaries',
       'One owner/ordered stream for shared P1 state; reject concurrent-state streams and int32 wrap; physical cleanup and independent terminal analysis'],
      falsifiers=['Actual SASS hoists ticket back before useful loads/math','Late atomic stalls final acquire or increases burst contention enough to lose','Any epoch/publication/alias/resource/numeric/graph/current-input gate fails','A fixed-arm gain disappears under matched pointers or does not improve actual PW38 selected pipeline'],
      forecast={'attainable_gain_us':None,'minimum_gain_cutoff':None,'family_rejection_after_one_failure':False},
      cost={'expected_CPU_implementation_minutes':45,'expected_CPU_independent_review_minutes':15,'expected_CPU_analysis_minutes':15,'total_expected_CPU_minutes':75,
       'provisional_narrow_GPU_full_reservation_seconds':1020,'conditional_cumulative_GPU_followup_seconds':3420,'total_local_GPU_chain_seconds_if_both_stages':4440,'official_H200_cost':None,
       'common_budget_scenario':'The ROI scan90CPUmin/300narrow/3420cumulative caps are comparison scenarios. This provisional narrow quote exceeds300s; rerank or redesign before runtime admission. No extra authorization follows from either number.'},
      rationale='A correct reusable P1 GPU harness needs source-bound monotone/dormant alias and matched-allocation controls. This package does not pretend an unreviewed ad-hoc runner meets those gates; the full fixed-arm/current-policy executable wrapper is intentionally a separate bounded artifact.')
if __name__=='__main__':print(json.dumps(proof(),indent=2))

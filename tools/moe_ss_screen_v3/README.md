# v3 immutable screen harness repair

v2 failed on the actual H100 before any production launch: cudaFuncGetAttributes static shared was0 although the exact matching cuobjdump table reports1024. The complete original result and raw artifacts remain pinned in failure-evidence.json. This is a failed harness admission, not an arithmetic, performance, or family result.

v3 changes only host resource admission and evidence retention. The frozen v2 kernel, production source, appended CUDA diagnostic, actual exact eight production SASS checks, exact binary REG/STACK/SHARED/LOCAL table, numerical gates, lifetime checks and timings remain unchanged. Runtime static shared is constrained to0 or canonical1024; register/local/maxthreads/maxdynamic/occupancy checks remain. Dynamic extern alignment plausibly accounts for the difference but driver causality is unproved; this explicit bounded reconciliation requires independent review. All eight runtime rows persist before resource assertions.

CPU checks: `python3 -m unittest discover -s tools/moe_ss_screen_v3 -p test_screen.py -v`; `python3 tools/moe_ss_pipeline_screen_v3.py --cpu-check`; `python3 tools/moe_ss_screen_v3/prove.py`. No Torch import, CUDA API, context, SSH, queue or GPU use is authorized by these commands.

next_job_proposal.json is a frozen bounded proposal for root admission, not a queued job. It is intentionally HELD with independent review and utility unresolved. Review must bind the manifest and exact launch, then replace the review path/SHA sentinels in an admitted descriptor outside this immutable proposal. Reserved cost720GPUseconds,600inner; retain failures; no retry. Stage-only evidence cannot establish the original861333f full-MoE3x goal.

# Current leaderboard stack: pw38

Board-verified scored job `a3f0ebf25a754c01`: **3198.6175079817267us**,9/9 and30 timed shapes. Source `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`. PW36 + selectable EH, no cursor. The preceding CHECK is a distinct job.

The old [rs32 source audit](audits/current-leaderboard-stack-source.md) and [official history](audits/current-leaderboard-gains-official.md) are HISTORICAL snapshots retained unchanged. Current bindings below are regenerated from frozen PW38 release bytes and this scored job; they are not copied CHECK selections.

|Component|Source/selection evidence|
|---|---|
|Lossless 12-bit BF16 codec with escapes|SOURCE_PRESENT; actual applicability is in scored routes; 5 current AST/source bindings; no isolated causal delta.|
|Prepacked MMA fragment layouts|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Register-direct GV stream and fast decode|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|GV local K reduction and retained fallback reductions|SOURCE_PRESENT; actual applicability is in scored routes; 1 current AST/source bindings; no isolated causal delta.|
|PW33 short per-warp QKV chains|SOURCE_PRESENT; actual applicability is in scored routes; 1 current AST/source bindings; no isolated causal delta.|
|P1 split-K with in-kernel zeroing|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Direct FP32 router GEMV|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|Router BF16 hi+lo decomposition and FP32 accumulation|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Old RW cp.async/WGMMA router|SOURCE_PRESENT; actual applicability is in scored routes; 6 current AST/source bindings; no isolated causal delta.|
|RS128/RQ swap-AB producer/math pipeline|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Staged register escape cursor|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|RQ fused zeroing, STMatrix and TMA reduction|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|RS32 small-token specialization|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Early first loads and L2 hints|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|64-row HALF/fullK plain-store alternatives|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|cuBLASLt tuning and valid fallback paths|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Shape-exact routing, dimensions and safe fallback|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Parallel build, exact validation, bounded F1 bake|SOURCE_PRESENT; actual applicability is in scored routes; 6 current AST/source bindings; no isolated causal delta.|
|PW35 one-node TMA RTK router128/256|SELECTED_IN_SCORED_PROFILE; 5 current AST/source bindings; no isolated causal delta.|
|Optional EH early-half-slab epilogue bit65536|SELECTED_IN_SCORED_PROFILE; 4 current AST/source bindings; no isolated causal delta.|

|Load / layer / M|Actual scored family / prepare pick|EH / HALF / EARLY flags|
|---|---|---|
|decode 1 / qkv_gate / 1|gv / See actual kernel profile|False / False / False|
|decode 1 / o / 1|gv / See actual kernel profile|False / False / False|
|decode 1 / pooler / 1|gv / See actual kernel profile|False / False / False|
|decode 1 / unpooler / 1|gv / uk-s4x|False / False / False|
|decode 1 / router / 1|rr / See actual kernel profile|False / False / False|
|decode 16 / qkv_gate / 16|rq / rq:t16s5hze-k7|False / False / True|
|decode 16 / o / 16|gv / r6|False / False / False|
|decode 16 / pooler / 16|gv / r6|False / False / False|
|decode 16 / unpooler / 16|gv / r6|False / False / False|
|decode 16 / router / 16|gv / rk-s16w8z|False / False / False|
|decode 32 / qkv_gate / 32|rq / rq:t32s5hze-k7|False / False / True|
|decode 32 / o / 32|rq / rq:t32s6hze-k2|False / False / True|
|decode 32 / pooler / 32|rq / rq:t32s6hze-k4|False / False / True|
|decode 32 / unpooler / 32|rq / rq:t32s6hze-k2|False / False / True|
|decode 32 / router / 32|gv / rk-s8z|False / False / False|
|verify 1 / qkv_gate / 8|gv / See actual kernel profile|False / False / False|
|verify 1 / o / 8|gv / See actual kernel profile|False / False / False|
|verify 1 / pooler / 8|gv / See actual kernel profile|False / False / False|
|verify 1 / unpooler / 8|gv / uk-s4x|False / False / False|
|verify 1 / router / 8|gv / rk-s16w8x|False / False / False|
|verify 16 / qkv_gate / 128|rq / rq:s4s5hzEH|True / False / False|
|verify 16 / o / 128|Lt/vendor / r6|False / False / False|
|verify 16 / pooler / 128|rq / rq:s4s5zEH|True / False / False|
|verify 16 / unpooler / 128|Lt/vendor / r6|False / False / False|
|verify 16 / router / 128|rtk / rt64k32b3x2z|False / False / False|
|verify 32 / qkv_gate / 256|Lt/vendor / r6|False / False / False|
|verify 32 / o / 256|Lt/vendor / r6|False / False / False|
|verify 32 / pooler / 256|Lt/vendor / r6|False / False / False|
|verify 32 / unpooler / 256|Lt/vendor / r6|False / False / False|
|verify 32 / router / 256|rtk / rt64k32b3x2z|False / False / False|

Full per-route kernel reports, exact current source-node SHA256/line bindings and historical equivalence checks are in [machine view](current_leaderboard_stack.json). Source presence does not imply measured benefit on every route. Server payload hash remains opaque; no component-wise or cross-run additive gain claim is made.

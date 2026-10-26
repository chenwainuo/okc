# Native allocation: valid no-gain result; traffic diagnosis next

`run=hw-native01; terminal=2026-10-26T23:07:54Z; exit=0; cleanup=CHECKED_CLOSE_PASS`

Same-run F1(us): actual incumbent RQ24.0640007, ordinary Lt26.5280008,
VMM NONE Lt26.5599992, GENERIC Lt26.6239997. GENERIC paired median gain versus
ordinary/NONE=+0.096001us each, versus incumbent=-2.623999us. Its F1 differences
are negative against all three controls. No promotion and no demonstrated gain.

All4096basis columns, normal/outlier/FP64/task oracles,20retained-graph checks,
4fresh-output checks, fixed shader/config identity, byte invariance and counters
pass. Both VMM owners reachRegion0 after graphs reset and aliases drop; no
retained-resource warning. Native Lt config[66,18,1,0,0,4,35,0,2], active
unpooler_t64MiBBF16, D2D population unchanged.

GENERIC/NONE were granted/query-confirmed1/0; effective compression remains
unmeasured. A flat latency result cannot decide whether compression occurred.
Next iteration compares physical DRAM traffic for the same fixed Lt, before
changing population or arithmetic. Original packed trial remains invalid for
timing because its cleanup failed; the independent ownership correction was
validated separately and does not repair that historical measurement.

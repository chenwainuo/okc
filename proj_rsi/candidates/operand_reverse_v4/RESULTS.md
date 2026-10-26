# V4: valid address-policy improvement; family best remains V1

`run=operand-reverse-v401; terminal=2026-10-26T23:48:38Z; exit=0`

Same-run conservative F1(us): incumbent24.192000, V146.944000,
V354.784000, V452.096002. V4 paired median gain(us): incumbent−28.416000,
V1−4.960001, V3+4.608002; wins0/21,0/21,21/21 respectively. Full F1 gain
against V3=+2.687998us. No integration or official speedup claim.

All four basis/FP64/normal/outlier suites,20actual retained-graph poison/mutation
checks, W/packed/escape immutability and counter resets pass. V4 actual resources
150regs/0local/103600Bdynamicshared/occupancy1; profile is the exact rq_v4 kernel.
All four graphs reset; PID373129/container gone and GPU0MiB after termination.

The complete fixed-stride/address-hoist version improved V3. Arithmetic removal,
register change and compiler rescheduling are not separately attributed. It still
loses against the family's faster V1 and the actual incumbent. Sources/data remain.

Next branch freshly packs X into a native-A fragment workspace every call, then
uses four coalesced128bit loads without consumer shuffles. Its timed graph must
include BOTH pack and main, with poisoned workspace/output and fresh-X checks.
The initial identity rawQ-to-A copy was caught merging physical registers before
GPU admission; preserve that failure and revise to explicitly retired A4 slots.
Actual source/SASS/lifetime review is required before measuring the new branch.

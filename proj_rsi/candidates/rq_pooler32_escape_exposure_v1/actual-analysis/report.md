# Exact pooler32 escape exposure — DONE / VALID_NOT_EXPOSED

Actual seed5151 original CUDA weights and RQ packing give **cap32,1024 lists,4559 records total, maximum11 per list**. All lists remain inside the initial bank: **zero subsequent-window loads**. The weight hash matches the known actual fixture.

The proposed lookahead targets only subsequent32-entry banks. Its four native refill LDG→SHFL sites cannot execute on this fixture; it does not shorten the initial load0x570→firstSHFL0x15f0 path. Thus aggregate54.07% long scoreboard from the earlier profile cannot be attributed to these dormant refill sites.

**Park this source/version and its frozen full trial for the intended fixture.** No~43sJIT/timing is needed to test a nonexistent path. Preserve CPU correctness/native-overlap evidence and reopen only on real intended workload exposure or a separate active-path hypothesis. This is an applicability result, not a measured runtime loss or rejection of the prefetch family. No synthetic tail benchmark is proposed.

Stats runtime3.102s; no JIT/nativeRQ/timing. Allocation/reservation0 and external exit0/GPU0MiB. All1024 lengths and hashes remain in the immutable raw receipt/analysis.

Loop: profile → measure → diagnose → optimize/test → validate → learn → reprioritize → repeat. [Contract](../../../ITERATION.md).

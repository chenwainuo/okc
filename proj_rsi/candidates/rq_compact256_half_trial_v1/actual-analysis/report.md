# Compact256 HALF: valid resource change, no measured gain

Actual compact query: **128 registers, 0 local bytes, 256 threads, 62,624 shared bytes, occupancy2**. Capacity264 covers grid256. Full incumbent is168 registers/384 threads/occupancy1, capacity132 covers grid128. Old384-thread HALF was safely skipped: capacity132<256; its performance is unavailable.

Compact versus full ST6: paired median **−0.016000µs** saving,5 wins/6 losses/1 tie,12 matched pairs with6/6 order. Fixed F1 average saving **−0.064000µs**; endpoints **+0.095999µs / -0.224000µs**, preserving drift and opposite signs. No incumbent speed gain or official claim.

Source/job/CUDA/SO pins and all51 raw journal rows match. Eight validation, five own-terminal and six mutation records pass. Exact actual compact native stream matches offline2232 instructions, inc216/dec40, no local-memory instructions. Graphs drained/reset; external exit0, GPU0MiB.

The intended resource point was reached; latency benefit was not. Two compact CTAs still provide only two math warpgroups while doubling X TMA bytes and producer groups. Those structural costs are proven; which cost or scheduling/context effect explains this small loss remains UNKNOWN. Numerical validity and native identity do not establish a unique performance cause.

Disposition: **DONE / VALID_NO_GAIN** for this configuration. Preserve the valid implementation and historical HALF counterevidence. Park unchanged configuration by ROI; reopen with direct full-kernel stall evidence or a concrete way to remove duplicated work. No broad rewrite/full30 retry proposed.

Runtime46.159s, compile42.866s. No author GPU/API/SSH/queue/git action.

Loop: profile → measure → hypothesis → optimize/test → validate → stack/submit → learn → reprioritize → repeat. [Contract](../../../ITERATION.md).

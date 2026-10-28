# Projection ROI queue

ONLY `muse_bf16_proj`, official target <3000µs. Best9916 remains **3157.038404µs**, gap157.038404µs. Bounded root runtime observation **1935: QKV spill tradeoff RUNNING**; view generated2026-10-28 00:06:21UTC, no fresh runtime poll. Runtime READY is distinct from admitted/running; physical utilization unknown.

| Priority | Active experiment | Pricing and uncertainty | Runtime at root cut |
|---|---|---|---|
| 1 | QKV M1/M8 full/half X prefetch versus parent | Root hypothetical5%both-case cut:13.126129µs score sensitivity; priors p_gain=.08,p_valid=.85,120s bound. Attainable/expected score gain unknown. | Admitted; RUNNING1935 |

CPU full-X has32Bstack/48Bstore+48Bload spills; half-X32Bstack/56Bstore+100Bload. Preserve both sources/native/operand proofs. These are resource facts, not a correctness failure or zero-spill admission gate; the bounded trial measures net overlap versus spill cost. Own runtime outcome is outside this completed cut.

Scored **3ef736760fda465b**, source2dda1b25, has an accepted receipt and frozen threeway-M8 release; **pending at this cut, no score claim**. Completed official count**74=20scored+54CHECK**;75 distinct accepted including this pending job.

Completed CHECK**38821b954fb14978**, source48af8c, passed9/9 at**3225.130858482905µs**, REFINE. Actual unpoolerM16 Xprefetch selected; same-prepare tuning+.095999800µs, M1parent/M8earlyload selected. Tuning/local gains and whole transfer are distinct; best unchanged. W-only fixed M8 paired+.096001µs versus parent but−.031999µs versus WX; mixed orders/all signed rounds/F1 brackets retained. Escape exposure:0subsequent32-window refills,max11/cap32; current lookahead version parked **unrun**, not a measured kernel loss or codec-family rejection. InitialLDG is another path.

Root CLI22:14 rank64 remains the latest recorded board observation; rank62/1173 is an archived21:25 capture. No new leaderboard snapshot or quota assumption. [Machine view](roi_queue.json) · [All signed history](HISTORY.md) · [Current stack](CURRENT_STACK.md). Inventory covers an explicit source cut; discover new proposals with `python3 -m proj_rsi.history_catalog --discover-candidates`, review/append roots/revision/timestamp, regenerate/check. Pending/running source outcomes remain separate.

Loop: profile → measure → hypothesis → optimize/reimplement → test → validate → stack → submit measured gains → learn → reprioritize → repeat; idle≤10s. [Contract](ITERATION.md).

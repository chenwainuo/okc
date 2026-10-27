# Projection ROI queue

ONLY `muse_bf16_proj`, official target <3000µs. Best scored9916 remains **3157.038404µs**, gap157.038404µs. This is the bounded completed cut at **2026-10-27 23:40:45 UTC**, root sequence1897: queueempty; physical GPU utilization unknown. Agent preparation/review is separate from runtime READY (zero at this cut).

| Priority | Active CPU follow-up | Evidence / uncertainty | Runtime |
|---|---|---|---|
| 1 | Optional unpoolerM16 Xprefetch integration | Fixed paired+.576001µs,21W3L; F1−.128/+.544µs. Policy/official transfer unknown. | Preparing/reviewing; not READY/admitted |
| 2 | GV53 M8 W-only earlier issue | New falsifier after optional earlyload selected parent on40d. No measurement yet. | CPU preparation |
| 3 | RQ poolerM32 escape lookahead | Exact current counter diagnostic; static dependency change is not a measured speed gain. | CPU preparation/review |

Ordering is the root's active work priority, not a numeric forecast. Attainable score gains and new cost/prior estimates are unknown; previous experiment prices do not automatically price these new versions. No MoE experiment is in the active queue.

Scored **40d856e085274f51**, source17d44d, passed9/9 at**3175.899650724306µs**, root REFINE. Actual M1keyed selected; M8parent selected despite earlyload source inclusion. Its same-prepare choice points+.016/−.016µs are tuning inputs, not paired official case gains. Crossjob loss18.861247µs versus best is not a causal feature increment. Fixed production-body M1−.048µs/F1−.960µs and M8+.032µs/F1+.176µs remain mixed (M8 mean/order counterevidence retained). RQ exactSO12metrics/4passes is diagnostic only. No achieved combined Xprefetch stack is claimed.

Latest leaderboard observation remains root CLI22:14 rank64, leader2951.538µs; rank62/1173 is an archived21:25 capture, not current rank. [Machine view](roi_queue.json) · [All signed history](HISTORY.md) · [Current stack](CURRENT_STACK.md). Source inventory coverage is an explicit cut; discover newer sources using `python3 -m proj_rsi.history_catalog --discover-candidates`, then review/append roots/revision/timestamp and regenerate. New proposals remain visible through ROI preparation.

Loop: profile → measure → hypothesis → optimize/reimplement → test → validate → stack → submit measured gains → learn → reprioritize → repeat; deliberate idle≤10s. [Contract](ITERATION.md).

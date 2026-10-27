# Read-only official choice analyzer

Run after root archives the official result and its local stdout log:

```
python3 proj_rsi/audits/gv53-keyed-choice-analyzer-v1/analyze.py \
  --job results/proj-rsi/jobs/fe5c45197ca041bb.json \
  --log results/proj-rsi/jobs/fe5c45197ca041bb.log \
  --control results/proj-rsi/jobs/9916d34357b745cf.json \
  --output proj_rsi/audits/gv53-keyed-choice-result-v1
```

The output directory must not exist. This tool only reads local inputs and writes a fresh analysis directory. It makes no service calls. Parser checks all completed four-block AB/BA receipts, recomputes signed median(parent-keyed), independently verifies strict-positive/ties-parent choice and raw sample consistency. It retains every receipt and incomplete raw sample; budget, RQ precedence, unselected geometry, shape skip and bad timing remain reasons, not measured gains. Missing receipts remain unknown. Multiple preparations remain separate; no unverified final-record association is made.

Per-target profiler identity is reported separately from source inclusion, receipt eligibility and accepted plan. Matched prepare effects use a tuning input and actual W; official raw/calibrated times are a different workload observation. All30 optional control comparisons are explicitly cross-job noncausal, retain seeds/routes, and separate calibrated/raw/overhead/reference ratios. No aggregate speed claim is created from the tuning effects.

Focused tests parse22 receipts from the actual helper's CPU transport log, exercise wrong-choice rejection/zero/tiny-positive/partialNaN cases, and parse30 real archived official rows including both keyed entries. This tests parser arithmetic and interpretation, not GPU performance.

Loop: profile -> measure -> hypothesize -> implement -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](../../ITERATION.md).

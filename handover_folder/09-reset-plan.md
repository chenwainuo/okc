# 09 — Plan for the 00:00 UTC reset (2026-10-27) and beyond

At the reset: 20 checks + 10 scored. Scored slots left today (expire at reset): 4 → pw35 is using one now (launched 19:08 UTC);
the other 3 have no candidate. The old `wait_rs128.sh` reset waiter was killed (rs128-only check is redundant: pw34 carries
those paths; its file is mid-edit). No waiter is armed at handover — arm one when a candidate is ready:
```
# detached (survives the turn): write a script, then launch it
S=<scratchpad or results/proj-rsi>; cat > $S/submit_X.sh <<'EOS'
#!/bin/bash
cd /home/ubuntu/okc; out=results/proj-rsi/jobs/X.report.txt
for i in $(seq 1 1500); do left=$(python3 -m proj_rsi quota | python3 -c "import json,sys;print(json.load(sys.stdin)['checks']['left'])"); [ "${left:-0}" -gt 0 ] && break; sleep 60; done
python3 -m proj_rsi check proj_rsi/candidates/X --experiment X --against 846e25d9df09433f > "$out.tmp" 2>&1; mv "$out.tmp" "$out"
EOS
chmod +x $S/submit_X.sh; setsid nohup $S/submit_X.sh >/dev/null 2>&1 < /dev/null &
```
(add `--scored` for a scored run; `--against <job>` sets the comparison incumbent in the summary.)

## Order of candidates (check first, scored only after a check confirms)
1. **pw35** — already scored (19:08 UTC). If its report shows the router build "skipped-budget", fix the ordering and re-check.
2. **rs32** (05) when its box table shows qkv m=32 ≥ 2 µs under pw34's pick → check standalone rs32 first (it is a full
   submission), then build pw36 = pw35 + rs32 section.
3. **P4's escape patch table** (07) and the author's stream-K / multicast (04) if they beat shipped rq by ≥ 0.3 in-prepare →
   check as rs128 (standalone), then port into pw36.
4. **latency school** (08) if its loader reaches the targets → it is a separate full submission; check it standalone, then port
   its small-shape kernels into pw36 behind the bake.
5. Clean-room plan mechanisms (audits/cleanroom-plan.md) as they are measured.
Every check: grep the judge `.log` for `skipped`, `BAKE`, `PICK`; compare the within-run bake; record in LOOP.md + ledger;
only then spend a scored slot. Never let two scored runs of the same file go out (drift ±3% is not signal).

## Realistic path to the top 10 (≈ 3120) and to 3000
pw35 (−10…−17) + rs32 (−45) + rq fixed-term cuts (−30…−55) + router m ≤ 32 / small-shape latency (−30…−60) ≈ 3150–3180,
inside striking distance of the top 10; 3000 additionally needs m=256 beaten (≈ −85) or the small loads at the latency
school's projections (≈ −90). Keep the H100 queue full; one agent per track; batch locked runs.

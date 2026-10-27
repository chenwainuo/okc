# PW44 private EH composition screen on current parent

CPU-built, GPU unrun. Root owns GPU, queue, official API and git. This directory preserves old parents and creates two composition alternatives. Use `--parent pw45` while 835acb is current; choose `--parent rq32_predxor` if cb7df is accepted/current. Never mix their prepared states.

The exact retained PW44 `delta.json` applies unchanged, with a byte-for-byte inverse back to each parent:

| Parent | Composed source SHA256 |
|---|---|
| PW45 835acb90b47b… | 196da9966ca2452a94ad370cf025a095e6abf853c35eb9d245d06953e1e15595 |
| RQ32 cb7df8085391… | 8140168cb95bb196da60f1ea3adc5998288ff9f98af96b974700bc493d246d64 |

The touched serving kernels are M128/ST5/NT128 QKV KS7 (126 CTAs) and pooler KS4 (128 CTAs), not the o32 HALF52 or RQ32 predxor kernels. The promising private-free flags are851970 (`ehpriv_z_free`) and851971 (`ehpriv_hz_free`): EH bit65536 + private slabs262144 + remove alias-protecting pre-store barrier524288 + original flags2/3. Original retirement/output-publication/read-drain barriers remain. All four retained PW44 private variants are composed to reuse the existing exact compiled proof; only the two free variants are extra fixed timing arms.

Why private was unselected in PW44 retry02: all560 prepare host calls succeeded; no decline, query error or budget skip. Actual picks were old EH65539 for QKV128 and old nonEH3 for pooler128. `_rqbake` sorts the complete old+new candidate list and considers its fastest point against r6; private variants were not that selected minimum. `MARGIN=.99` compares the fastest RQ contender against r6. It is **not** a separate 1% private-versus-old-RQ hurdle. The retained result lacks full bake timing rows, so exact private rank/timing gaps cannot be reconstructed. Narrow fixed-EH wins cannot establish wins versus actual nonEH3; this missing comparator is the reason to run this screen.

`screen.py` uses two actual complete prepares, current parent and composition, keeping original prepare clocks/options. It saves selected plans and stdout supplies actual bake rows. It measures only QKV128 and pooler128 with four arms: actual current policy, actual composed policy, fixed private-free z, fixed private-free hz. Actual policies use their own untouched prepared selections; no selection is forced. Fixed arms have explicit independent packing/output/counters and are labeled separately. Both estimators and all raw signs remain; there is no minimum-gain or estimator-agreement gate. A kernel win can justify the next selection refinement even if unselected; a losing screen avoids a new all30 run.

Existing PW44 resource/SASS/physical helpers are reused at the actual private module's load return, before private prepare math. They reconcile runtime user-static0 + queried driver-reserved1024 with actual dump1024 and check all six flags/occupancy. Source delta/inverse check is the new CPU proof, not a rerun of the old proof suite. Two touched shapes use inherited full-K source validators and current-output task checks; paired arm outputs are checked immediately, with source-aware counter checks and separate poisoned zero/outlier replay checks. This is a scoped local screen, not official validity or all30 validation. External root guard supplies process/GPU cleanup and a wall cap; script drains graphs in finally. Historical timings are not a forecast of this new adapter's runtime.

CPU check (completed):

```bash
python3 proj_rsi/candidates/pw45_private_eh_screen_v1/screen.py --cpu-check
```

Runnable root command after reviewing the narrow adapter delta and choosing the actual current parent:

```bash
python3 proj_rsi/candidates/pw45_private_eh_screen_v1/screen.py --parent pw45 --output /root/okc/logs/scheduler/pw45-private-eh-screen-01
```

No new all30 harness is assembled. If the screen supplies useful positive actual-incumbent points, retain them regardless of size, diagnose any prepare selection mismatch, and advance one combined full-policy verification. Do not count the two source alternatives as two measured gains.

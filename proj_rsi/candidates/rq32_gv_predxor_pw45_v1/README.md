CPU source composition only; compiled resources, runtime numerics, selection and combined performance are UNRUN. This new candidate applies the exact GV predxor v2 change onto exact RQ32 predxor cb7df808. It does not modify either frozen parent or their harnesses.

Combined submission SHA256: `137e0dbce0f0126ef9c72d4b6e9b48e0448b02d0c397d9cc0ba927bac2d2a4a6`.

| Source | SHA256 | Difference from PW45 |
|---|---|---|
| PW45 base | `835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3` | None |
| Exact RQ parent | `cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210` | RQCU only |
| Exact GV parent | `501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9` | GVCU and _gvext only |
| Combined | `137e0dbce0f0126ef9c72d4b6e9b48e0448b02d0c397d9cc0ba927bac2d2a4a6` | Exact union of those three nodes |

The RQ edit adds its separate `rq_fix_pred` helper and uses it only for NT32/ST5-or6/GS1/PRB0/WM0/INS0/FL35. The GV edit adds its separate `gv_fix_pred` helper and seven copied template instantiations, redirects only its original shape-whitelisted `gvl` pointer, and retains its exact v2 JIT prefix. The seventh GV target is poolerM8 WR1/ST2/S1, not split S2. There is no overlapping dispatch edit: RQ's type condition is inside RQCU; GV's pointer change is inside GVCU. Python selectors, template inventories, prepare budgets and all other top-level nodes remain exact PW45.

`verify.py` uses only stdlib AST and bytes. Both patch orders produce this same entire file; removing GV gives byte-exact RQ cb7df808, removing RQ gives byte-exact GV501d901d, and removing both gives byte-exact PW45835acb90. All176 top-level nodes retain ordering; only indices10 GVCU,57 _gvext and149 RQCU differ from PW45. This proves source composition, not generated-machine-code identity or numerical/performance composition.

Run `python3 proj_rsi/candidates/rq32_gv_predxor_pw45_v1/verify.py` for the read-only proof and all local/parent pins. No new compilation, exhaustive proof suite, GPU harness, queue descriptor or performance forecast was created. RQ official check CHECK87044866f4494726 was reported submitted by root; no result is inferred here. Root may later admit a minimal combined screen and then full30 controls if warranted by both marginal observations. No stacked or additive gain is claimed before actual combined measurement. Promotion still requires final correctness and comparison against the current incumbent.

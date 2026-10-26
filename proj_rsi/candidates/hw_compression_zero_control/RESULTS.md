# Known-zero control: allocation/consumer/observer responds

`run=hw-zero-counters01; terminal=2026-10-26T23:49:24Z; exit=0; passes=3/node`

Observed DRAM-unit reads: GENERIC17842944B, NONE68174336B; signed reduction
50331392B (73.82747666%). Both use the same known-zero64MiBBF16 Wt, unchanged
D2D population, fixed native Lt config/shader/grid/block/register/shared resources.
This establishes responsiveness under this control. It is not ordinary-weight
compression performance, a pure-W wire factor, an H200 result or a latency win.

All zero-byte/query flags,4pre+4postactualgraph/freshnonzeroX mutation checks,
2fresh outputs,2immediate selected-output poison/oracles, exact byte/input checks
and both checked Region0 closures pass. No trivial full-basis claim is made.
PID373969/container gone and GPU0MiB after completion; raw report/base-unit CSV
and analysis are in ../../diagnostics/hw-zero-counters/.

ILC evaluated-compressible count remains0 in both despite the read response;
those fields are not a sufficient witness for this read benefit. Selected metrics
do not directly count every TMA/load instruction; fixed source/config/resource
identity constrains bypass without proving dynamic instruction-count equality.

The normal-weight pair had nearly flat reads. The next causal question is data
pattern/layout and inverse-decode cost. D2D-to-SM rewrite is not the next step:
this unchanged D2D path has now shown a response. Bit-exact reversible layouts
need CPU proofs, actual local patterns and cost assessment before another GPU test.

# PW37 v1 remains an experimental stack; no cursor improvement established

The frozen source SHA7913564fe7912cc69eace3e60b595ef3865440d60acc25e7fe0af7fda67a0330
and manifest94afa2a347bbfecf56eed1bdc2c9061a8e1413b5965e1e90927dc685a6604c08
remain immutable. This results annotation is deliberately outside that manifest.

The standardized all-five cursor v3 comparison completed validly with guard398864,
exit0 and verified cleanup. It compares freshly prepared RS32/PW36/cursor policies;
it does not measure this cumulative PW37 file. The result is a valid regression
against RS32 overall and establishes no isolated cursor improvement.

The clearest matching-path case is pooler32, t32s5hze KS4: cursor loses0.144us
paired and0.096us F1 against RS32. Unpooler loses0.352us paired with different
ST picks; o32 loses0.112us with different GV picks. qkv16's+0.032us occurs on the
same unchanged GV kernel and cannot establish a cursor effect. Retain all small
wins/ties/regressions in the raw report, without a fabricated feature gain.

Evidence: `proj_rsi/audits/smallm-v3-01/report.md` and `analysis.json`.
Do not submit/score PW37 as a forced global cursor improvement. The next
cumulative candidate PW38 starts from PW36 without the cursor and adds optional
EH variants while retaining controls. This is a version-level result, not
retirement of every cursor implementation or a prohibition on future evidence.

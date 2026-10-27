# EH+O32 versus local EH: incremental full30

**Valid positive incremental package point:** paired1.0006856973480098×; F1 endpoint-average1.0007206029844913×. Control is measured localEH aaf40d39, candidateEH+O32c425148c. This is **not** a comparison with currentGV501d and cannot be added to prior ratios.

O32 actual dispatch changes originalGV47→originalRQ`t32s5hze`,flags35/KS2,168registers/0local/83080shared. It gains+0.032000244us paired but loses−0.032000244us F1. Candidate same-prepare best9.82us versus incumbent9.89us is rejected by old0.99threshold9.7911 but admitted by1.0; rounding leaves a positive0.06–0.08us gap. ControlEH separately kept r6 at9.95 despite best9.86. Exact complete target receipts are in `analysis.json`; no kernel arithmetic or RQpredxor is changed by this incremental source.

Both arms select private-free hz851971 for QKV128 and pooler128. QKV is+0.064000487paired/−0.144001096F1; pooler is−0.016000122paired/+0.607999973F1. These are same-template fresh-state/context contrasts, not newly isolated EH gains. Fourteen selected-plan descriptions differ across fresh prepares.

All148 task cases,2910 finite raw timings equal the journal;120 own-output terminal receipts and240 mutations pass. All23 artifact hashes verify, both O32basis checks pass, noJIT/graph errors, exit0 and clean root process/container/GPU cleanup. Original prepare clocks remain charged (58.511767/41.061868s walls). Existing checks are reused; no new proof or repeat is required.

|Shape|Paired localEH minus EH+O32 us|F1 us|
|---|---:|---:|
|decode1/qkv_gate|-0.159999821|+0.063999556|
|decode1/o|-0.016000122|-0.080000609|
|decode1/pooler|-0.048000365|-0.191999599|
|decode1/unpooler|-0.128000043|+0.223999843|
|decode1/router|+0.016000122|+0.015999889|
|decode16/qkv_gate|+0.239999965|+0.208000187|
|decode16/o|-0.015999656|-0.272000208|
|decode16/pooler|+0.016000122|+0.032000244|
|decode16/unpooler|-0.063999556|-0.143999234|
|decode16/router|+0.064000487|+0.031999778|
|decode32/qkv_gate|-0.128000043|-0.111999922|
|decode32/o|+0.032000244|-0.032000244|
|decode32/pooler|-0.032000244|-0.127999112|
|decode32/unpooler|-0.096000731|-0.112000853|
|decode32/router|+0.064000022|-0.080000143|
|verify1/qkv_gate|-0.128000043|-0.064000487|
|verify1/o|-0.064000022|-0.047999900|
|verify1/pooler|+0.032000244|+0.128000975|
|verify1/unpooler|-0.064000022|-0.207999954|
|verify1/router|+0.063999789|+0.032000244|
|verify16/qkv_gate|+0.064000487|-0.144001096|
|verify16/o|+0.032000244|-0.128000043|
|verify16/pooler|-0.016000122|+0.607999973|
|verify16/unpooler|+0.128000043|+0.687999651|
|verify16/router|+0.047999900|+0.064000022|
|verify32/qkv_gate|+0.384000130|+0.608000904|
|verify32/o|+0.448000152|+0.064000022|
|verify32/pooler|-0.560000539|-0.544000417|
|verify32/unpooler|-0.032000244|+0.079999678|
|verify32/router|-0.080000143|+0.000000000|

Paired14positive/0ties/16negative; F114positive/1tie/15negative. Every signed shape and32paired differences remain in `analysis.json`. Root may use this incremental evidence while measuring the completeGV+RQ+EH+O32 package directly against currentGV. No official/additive gain is claimed. No GPU/API/queue/git action occurred in analysis.

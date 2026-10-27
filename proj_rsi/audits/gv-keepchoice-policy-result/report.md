# Keepchoice versus current GV

Valid full30 local package improvement: **paired1.0034301577777616×**, **F1 1.0077656313114494×**. CurrentGV501d901d compared directly to keepchoice244e5b7d; positive points support root’s authorized scored decision without another significance/proof gate.

148 task checks,2910 finite raw rows matching journal,120 terminal receipts and240 mutations pass. Existing source/state/counter checks complete, exit0 and cleanup clean. No profiler/SASS campaign repeated.

| Shape | Paired GV−keepchoice µs | F1 µs |
|---|---:|---:|---:|
| decode1/qkv_gate | +0.159999356 | -0.016000122 |
| decode1/o | +0.047999900 | +0.064000022 |
| decode1/pooler | -0.080000609 | -0.047999434 |
| decode1/unpooler | +0.287999865 | +0.352000119 |
| decode1/router | +0.032000244 | +0.032000011 |
| decode16/qkv_gate | +0.208000187 | +0.192000065 |
| decode16/o | -0.080000143 | -0.031999778 |
| decode16/pooler | +0.079999678 | +0.048000365 |
| decode16/unpooler | +0.015999191 | +0.080000609 |
| decode16/router | -0.047999900 | -0.047999900 |
| decode32/qkv_gate | +0.031999778 | -0.192000065 |
| decode32/o | +0.032000244 | -0.064000487 |
| decode32/pooler | -0.335999765 | -0.160000287 |
| decode32/unpooler | +0.032000244 | +0.464000739 |
| decode32/router | -0.047999900 | +0.191999134 |
| verify1/qkv_gate | +0.223999843 | +0.223999843 |
| verify1/o | -0.063999556 | +0.095999800 |
| verify1/pooler | -0.159999356 | +0.032000244 |
| verify1/unpooler | +0.320000108 | +0.224000076 |
| verify1/router | +0.015999889 | +0.128000276 |
| verify16/qkv_gate | +0.207999721 | +0.256000087 |
| verify16/o | +0.000000000 | +0.047999900 |
| verify16/pooler | +0.096000731 | +0.448000617 |
| verify16/unpooler | -0.256000087 | +0.096000731 |
| verify16/router | +0.032000244 | +0.159999821 |
| verify32/qkv_gate | +0.624000095 | +0.576001592 |
| verify32/o | +0.063999556 | +0.319999643 |
| verify32/pooler | +0.512000173 | +0.159999356 |
| verify32/unpooler | -0.191999599 | -0.143999234 |
| verify32/router | +0.032000244 | +0.143999700 |

Paired20wins/1tie/9losses; F1 22wins/0ties/8losses. Actual poolerM1 loses both; poolerM8 paired loses/F1 gains. PoolerM16 improves both. UnpoolerM16 improves slightly while selecting RQ rather than changed GV. Plan markers/native API receipts preserve implementation identity separately from variant ID. Exact picks/load sums/hashes remain in analysis.json. One prepare perpolicy includes selector effects and does not establish an official gain or isolated cycle attribution.

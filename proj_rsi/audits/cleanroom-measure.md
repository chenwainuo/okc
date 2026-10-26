# Cleanroom verifier: bounded H100 comparison against frozen pw36

Snapshot: 2026-10-26T20:00:44.927961+00:00. Independent readonly harvest by `/root/proj_evidence`; root ran the GPU job. No kernels edited, GPU launches, remote writes/kills, new variants or commits by reviewer.

## Decision

**REJECT all four unchanged full-school submissions as replacements; no measured mechanism qualifies for a port or a paired follow-up.** Every school loses every six-load total to both pw36 brackets. Zero school shapes beat both brackets by >=0.3us. Preserve every school and its evidence as a closed measured hypothesis, not a performance candidate.

The only >=0.3us apparent win against the *opening* baseline is contrarian o32:13.264us vs13.632us (-0.368). Closing pw36 is13.152us, making contrarian **+0.112us slower**. Contrarian's own bake chose **torch fallback** for this shape: best custom16.90us versus torch13.15us. This does not justify a new kernel variant or a further GPU run.

Tensorcore **MODE0 current v25 completes65/65** under the default serving policy; the inherited MODE3 router diagnostic deadlock remains blocked, as explained in `cleanroom-takeover-plan.md`. Completion of MODE0 does not repair the diagnostic path.

## Run identity and admission counters

- Host `root@107.170.6.60`; named container `okc-proj-cleanroom-takeover-v1`; root launcher PID345107.
- Window2026-10-26T19:57:55Z..20:00:20Z,145s. Host log ends `VERIFIER_DONE`; all six harness processes exit0. At20:00:44 reviewer observed empty `docker ps` and empty compute-app list. Earlier spot checks showed one CUDA process in the named container; root owns the continuous exclusivity/lock evidence.
- NVIDIA H10080GB HBM3, torch2.8.0+cu128, CUDA12.8. Corrected in-container flock; no inherited soak or second GPU process observed. This is a coarse exclusive-session comparison, not balanced same-process AB/BA.
- Harness: `proj_rsi/tools/h100_bench.py --official`, default20 replays, seed1234, full30 load/layer shapes. Separate fresh Python processes, in order pw36/BW/contrarian/latency/TC/pw36_after.240s process deadlines; no deadline fired.
- Reviewer independently rejects any CHECK failure even if exit0. Verified complete expected5layers x13m values,65 unique CHECK keys, no failures,30 unique timing rows, all positive finite, terminal JSON, exit0, and no traceback/CUDA fault in every run.

|run|CHECK pass/expected|unique checks|finite timing rows/expected|exit|admit coarse table|
|---|---:|---:|---:|---:|---|
|pw36|65/65|65|30/30|0|PASS|
|pw36_after|65/65|65|30/30|0|PASS|
|bandwidth|65/65|65|30/30|0|PASS|
|contrarian|65/65|65|30/30|0|PASS|
|latency|65/65|65|30/30|0|PASS|
|tensorcore|65/65|65|30/30|0|PASS|

`--official` here expands normal-input m values to1,5,7,8,9,16,32,33,128,130,256,512,1000 for every layer, with generated probe rows. It does **not** exercise the official outlier distributions. Therefore this table is useful for screening, not full official-check certification. No port was admitted, so no additional exact-nine run is justified for these losing schools.

Latency additionally ran `validate_replay.py` before its harness: **45/45PASS**,15(layer,m) at m1/8/32 x original/zero/fresh-outliers. Captured output retained; fixed X allocation mutated; input unchanged by kernel; zero output exact; fresh outliers checked by task oracle. Weights seed5151 and fresh input seeds differ from harness1234. This establishes graph-replay correctness for this tested small-m matrix, not performance or all possible odd-m paths.

## Preparation and measurement context

|run|prepare s|tiny graph replay us|read6MB us|
|---|---:|---:|---:|
|pw36|46.6|11.344|49.984|
|pw36_after|45.6|12.864|49.984|
|bandwidth|0.3|12.144|50.288|
|contrarian|4.2|11.008|50.304|
|latency|0.2|12.336|49.904|
|tensorcore|0.2|12.864|50.240|

All preparation is outside the timed shape replays and inside root's GPU ownership window. Warm extension cache explains the school preparation times; these are not cold judge compile/prepare measurements.

The pw36 load-sum geometric mean changes **-0.403%** between brackets, but shape drift reaches0.608us and actual prepare choices differ. Examples: qkv256 opening rq:n256s4hz -> closing r6; o128 opening rq:h128s8e-k1 -> closing r6; router128 opening rt64k32b3x2z -> closing rt64k32b4z; router256 opening rt64k16b6z -> closing rt64k32b3x2z. The comparator is the same source with fresh bakes, not identical frozen dispatch state.

Replay probes11.008..12.864us remain high even with exclusive CUDA ownership. Do not subtract them to derive H200 kernel times or infer that the GPU is still contended solely from that floor. Read6MB probes are49.904..50.304us. Raw event-timed graph medians are retained below; no profiler-derived timing is mixed in.

## Full30-shape table

Units us; lower is better. `before`/`after` are frozen-pw36 source prepared independently at the session boundaries. BW=bandwidth, CR=contrarian, LAT=latency, TC=tensorcore.

|load|layer|m|pw36 before|pw36 after|BW|CR|LAT|TC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|decode1|qkv_gate|1|16.880|16.944|23.936|21.728|32.624|23.088|
|decode1|o|1|12.544|12.400|16.080|14.128|16.208|13.808|
|decode1|pooler|1|24.128|23.984|30.496|28.256|35.584|32.624|
|decode1|unpooler|1|12.128|11.520|16.320|12.800|16.736|13.968|
|decode1|router|1|12.032|12.000|22.016|17.808|31.600|19.712|
|decode16|qkv_gate|16|17.456|17.424|23.872|22.688|34.192|23.072|
|decode16|o|16|12.560|12.464|16.416|13.424|16.480|13.648|
|decode16|pooler|16|24.096|24.304|31.248|28.656|36.976|32.992|
|decode16|unpooler|16|24.624|24.384|29.408|28.208|28.992|30.432|
|decode16|router|16|12.096|11.744|21.360|18.272|33.392|19.504|
|decode32|qkv_gate|32|17.712|17.680|26.688|22.832|41.600|23.504|
|decode32|o|32|13.632|13.152|18.496|13.264|20.480|13.888|
|decode32|pooler|32|24.672|24.800|33.120|32.496|44.896|34.976|
|decode32|unpooler|32|24.880|24.768|32.400|30.000|37.520|30.560|
|decode32|router|32|12.112|11.664|21.504|19.632|35.232|20.592|
|verify1|qkv_gate|8|17.184|17.088|23.872|21.488|32.544|23.136|
|verify1|o|8|12.608|12.496|16.144|13.280|16.240|13.776|
|verify1|pooler|8|24.160|24.144|30.480|28.096|35.200|32.448|
|verify1|unpooler|8|11.936|11.872|16.528|13.152|17.088|13.776|
|verify1|router|8|11.632|11.552|22.032|18.128|30.944|19.040|
|verify16|qkv_gate|128|19.424|19.680|23.520|23.040|23.824|23.728|
|verify16|o|128|13.472|13.296|14.272|13.696|14.112|14.192|
|verify16|pooler|128|26.208|26.496|31.920|31.152|32.576|31.616|
|verify16|unpooler|128|26.768|27.200|31.136|30.928|31.424|32.096|
|verify16|router|128|13.088|13.344|23.712|22.976|23.488|24.992|
|verify32|qkv_gate|256|23.696|23.888|26.352|26.048|26.064|26.496|
|verify32|o|256|14.464|14.496|15.248|14.880|15.104|15.856|
|verify32|pooler|256|32.192|31.920|38.608|35.520|39.120|36.960|
|verify32|unpooler|256|30.816|30.304|33.808|32.896|33.152|33.264|
|verify32|router|256|15.376|15.584|29.808|28.800|29.680|31.440|

## Six-load sums and aggregate screening

Each cell sums the five layer graph times; no layer-count weighting or per-shape ratio geomean.

|load|pw36 before|pw36 after|BW|CR|LAT|TC|
|---|---:|---:|---:|---:|---:|---:|
|decode1|77.712|76.848|108.848|94.720|132.752|103.200|
|decode16|90.832|90.320|122.304|111.248|150.032|119.648|
|decode32|93.008|92.064|132.208|118.224|179.728|123.520|
|verify1|77.520|77.152|109.056|94.144|132.016|102.176|
|verify16|98.960|100.016|124.560|121.792|125.424|126.624|
|verify32|116.544|116.192|143.824|138.144|143.120|144.016|

|school|geomean(load sum / pw36 before)|geomean(load sum / pw36 after)|>=0.3us wins vs both brackets|
|---|---:|---:|---:|
|bandwidth|1.34265|1.34807|0/30|
|contrarian|1.22395|1.22889|0/30|
|latency|1.56071|1.56702|0/30|
|tensorcore|1.30066|1.30592|0/30|

These are **H100 full-policy latency ratios**, not official speedup scores, H200 projections, or a pure custom-kernel comparison. Schools often dispatch to torch. The official score additionally uses per-load SGLang baselines and H200 measurements. No new board-score claim follows.

## Preserved track dispositions

|track|tested policy / observed result|disposition|
|---|---|---|
|bandwidth|BW_BIG=0, packed small paths; torch above32. All65 checks pass; no>=0.3us win against both baselines.|Close unchanged file. Keep12-bit layout and epilogue evidence; large-m mma.sync stays closed. No batch5 sweep restart.|
|contrarian|Default bake+torch competition. Cooperative split-K candidate configs run alone; large paths usually choose torch.65/65pass; apparent o32 win is fallback plus bracket drift.|Close unchanged file. Preserve issue/decode/epilogue findings; do not reopen wgmma solely from raw mainloop predictions.|
|latency|LOADER3, DBG512, empty PLAN, SMAX16.45/45 replay mutations plus65/65normal-m checks pass. Full policy1.56x baseline latency.|Retain correctness and fixedcost A/B evidence; default-change mechanism can be valid internally while remaining behind pw36. No automatic port or loader retuning from these results.|
|tensorcore|MODE0, default BF16=torch/router=packed<=256, TC overrides unset.65/65pass. Router m12824.992 versus pw3613.088/13.344; m25631.440 versus15.376/15.584.|Close full policy; MODE0 no longer hang-blocked for tested matrix. MODE3/other diagnostic converter deadlock remains blocked. No unchanged ladder/cmp8 rerun.|
|pw36|Both brackets65/65pass, all prepare phases complete; source unchanged.|Keep frozen candidate for root's official check at reset. This H100 table does not replace that check.|

No winner enters the integration queue. The bounded verifier stage now has its missing measurement artifact. Synthesis/reviews can consume this evidence without new GPU work; potential future hypotheses need distinct source mechanisms and measured ROI evidence, not rerunning these losing files. Stream-K and P4 track decisions remain independently recorded by their owners; this table does not change or reopen them.

## Provenance and raw artifact hashes

Primary remote directory: `/root/okc/logs/takeover-cleanroom/`. Host log: `/root/okc/logs/takeover-cleanroom-host.log`. Root requested to archive these into local `results/proj-rsi/takeover/20261026/`; remote paths below identify the original files regardless of archive naming.

|source|bytes|SHA256|
|---|---:|---|
|pw36/submission.py|272704|`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`|
|bandwidth/submission.py|54770|`9ac20ff1a8711db694a7ab1cc70066f33dee402bc2328da520221e119f641fe6`|
|contrarian/submission.py|51987|`47027d0dbeff674d95f4ee80a37f0e6f1e8c8919f112e7528a26ce10d031c094`|
|latency/submission.py|59309|`e4c68752a7734d3a4644a21bef3df649435ece3b8f126959a755bf3f9678cb1b`|
|tensorcore/submission.py|42244|`b92f012bf96a168fda296073e2e1fb3030289caa258a54a09a0ff6b07295b522`|

Post-run hashes agree with preflight frozen source hashes; root's serial script and exclusive ownership supply execution lineage. Harness does not emit an uploaded-source digest or signed execution attestation, so a post-run file hash alone is not mathematical proof of runtime source identity.

|run|log SHA256|JSON SHA256|
|---|---|---|
|pw36|`cd6b4f9e47c1436d56d8ccf8434b064ef37f9d1f6f0de9fd2af13735ac6e030f`|`a53c5849ac4519f6e78dbbffded5c92569baa0343df5bc7cb6ea272a82dc08b6`|
|pw36_after|`663bc3f9fca3aa407be33c4231681b7bd945b67bf03e3c0be243ccb39c34e9d3`|`c25683d9abd957fcd77f4a1264b7795c156c90514d72e275f9cb16e0b741cb08`|
|bandwidth|`fcaaf12cf5e0f349103f418999e4ffcca3334f796ca97747d56bfcc83e117837`|`e33b105197115dd51ee04221257f20c1a7f40a269debfa7a105d6ccfe9d1d050`|
|contrarian|`9d8c3779fefe9a5d4d83c774a8c4a5b8b6d85ce36586460a7808312dede638a9`|`51b5a256b764c4ef947038875bbb5e7c4b47821af394fd6c98f22544b69b5122`|
|latency|`9337838cdfd206b6e6a85c024cffb5c47832374ee7bc526e8501edaeb65cf0c1`|`6461534bbfad431d55765d7d5c3f9bae3751bbdfaf4f682c0377868a77e575c8`|
|tensorcore|`2466632db414cbd081a3f255d1f432615392437e678843e8e8781e021c892a2f`|`68a14f426fb5bbf5da080ebf11f03a434c568e4e50d74d38d88daed075370319`|

- Host log SHA256: `d50e24c9df2bf90fe30aad28c55bc677d88766d47cd1a1af52a254487b26800a`.
- Latency replay log SHA256: `f76a15ea6baec622c41f6126f803cc17707a3f90596ecdb15cbe7953d46ca2f9`.
- Root run script `results/proj-rsi/takeover/20261026/run-cleanroom-verifier.sh` SHA256: `2bd0d7972e1c4ee96db6e5d00da0a2a39dfb12b02aef1b52a923a8cd4840907b`.
- Harness SHA256: `3a5b5a2af4dae0c81fed93797959c674297fe71eb276fd1c0553d00703b9a7e4`.
- Replay validator SHA256: `27aa3d7e5dbecc52068626554b1d128e18aa7d74f422c8726fd6bf8db93e95ea`.

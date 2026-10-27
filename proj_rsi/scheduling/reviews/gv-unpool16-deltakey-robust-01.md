# GO — exact-SO projection robustness discriminator

Independent gv_codec_frontier focused changed-driver review. Driver
fa2512109c267af3e1c1c555fc10ab4ff17ef2929fbed5e534818d5107b12d70;
manifest6dd924b14c437c79b81f18c4f3d5246c12cddc6e289aca692f10a47615952f0d;
proposal978c47ae2e19379e503226912cd9c11afafa802e777110afc2cd7aa5dd96da36.
GO for one60s-inner/120s-outer muse_bf16_proj unpooler M16 robustness run.

The native code is unchanged: exact collected SO
d1c06fd80d1f5391986c5257765851e95eb749ded49129601a5183b0871f3c49,
original PyInit name gv_unpool16_deltakey_e418b7d8bc4d29788ff7. Prior result/SO/
candidate-source hashes are checked before loading; the exact SO is copied into
new outputs. No JIT, new kernel, profiler or native-proof campaign. Existing
source/math/codec/transport/resource proof and prior actual native module are
reused. The prior one-weight paired loss/F1 positives remain evidence, not a
family rejection or a claimed measured paired gain.

Three weight seeds6203/6257/6287 are fixed prospectively. Both arms share each
weight/pack/CSR. Each consumes the same stored eight X fixtures within a weight;
byte hashes bind identical X values across regenerated banks for all weights.
Four normal/four outlier inputs are shared across both timing styles and labels.
There is no outcome-dependent seed or arm selector.

Per weight: A-B-A and B-A-B blocks both consume all eight fixtures; interleaved
AB and BA sweeps consume every fixture in both orders. The middle weight flips
the bracket orientation.40 clean observations per arm/weight gives240 clean
calls+30warmups. Thirty own retained timed-output terminal oracles occur before
later replay/mutation of those private banks; no claim that every timed output
is individually checked. Private X/output, current-X/task/FP64 checks, guards,
complete S1 counter/scratch and pack/weight integrity, retained-graph mutations
and all signed raw observations remain. Per-weight/fixture/condition/style
effects survive; estimator nonlinearity and time/order are still limitations.

The unchanged known BLAS clear runs only after graph drain/reset and run-scope
alias release, records cache/allocator before/after and requires empty map and
allocated0. Manifest/12 source pins and CPU schedule checks pass. Reviewed only
the changed SO load, prospective weight loop, matched-bank scheduling and own
terminal placement; no added generic gates or repeated native proof. Existing
prior run directory is an explicit hashed input, and a missing/mismatched SO
must fail setup rather than rebuild or substitute code.

No GPU/SSH/queue/git/compiler action or frozen-author mutation by reviewer.
Root owns admission, result validation and any separate official decision.
This fixed implementation robustness run makes no H200/policy/score claim.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../ITERATION.md)

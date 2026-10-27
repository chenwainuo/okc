# GO — weight-only exponent8/top8 capacity diagnostic

Independent focused measurement_contract_audit CPU review. ONLY muse_bf16_proj.
Driver `ff04b84167fb412f5339c4f2ca02f3c117bed36abecf6863c437ba23401cc096`,
manifest `67e2e9378b4570cf0ebfdc593db4e24fc7adf9126cb64d232f2bfc011c398213`,
proposed `70104d4ff48ad84187890f47407d41a8a508fd1b116c3ac77b2a5a0c1297ada2`.
All7 launch pins verified. Reuse reviewed reference AST generation, original
production _pack and group_counts proof; no new codec/native campaign.

Two complete original CUDA weight generations use5151 and60631, preserving
original layer order and reference implementation. Only pooler/unpooler W are
copied to CPU and packed/analyzed. No activations, task outputs, model prepare,
custom JIT/native codec, profiler or timing policy. Runtime records Torch/CUDA,
device, generation order and raw/fragment hashes. Known5151 hash equality is
recorded rather than imposed, so environment/RNG differences remain visible.

Full exponent extraction is (BF16bits>>7)&255, excluding sign, not old high7.
Codebooks are global per matrix: deterministic top8 arbitrary exponents,
unrestricted contiguous8 and original even-aligned width8 (four old high7 bins).
All valid starts including248 are considered. Same actual fragment tensor is
used for each candidate and every result is retained even if capacity fails.
No per-group dictionary, truncation, chosen decoder or losslessness claim follows.

Actual selected geometries are pooler(2,16,8,1,1,0) and
unpooler(1,4,16,1,1,2). Original group_counts folds RT/KU and orders groups as
original _gvesc:4096 pooler and2048 unpooler groups over complete matrices.
Histograms count all coefficients; repair sums and >32/>64 tails concern those
real groups, not independent flat chunks. Existing32-record threshold is exact
capacity accounting; >64 is explicitly optimistic without metadata. Prospective
11-bit payload+128B/group+8B table is static conditional storage only, not an
implemented packet, physical traffic or measured gain.

Independent CPU checks passed7pins/proposed hash, a noncontiguous-codebook
discriminator, both actual group geometries/mass and a small real original-pack
analysis with histogram/repair mass agreement. CUDA stayed uninitialized and no
files were written by tests. Unchanged full generation/packing proofs reused.

Analysis locals contain CPU tensors/statistics only. GPU weights release per
seed and in finally; synchronize/GC/empty_cache and allocated0 cleanup are
recorded, with errors preserving traceback and invalid disposition. External
process/device cleanup remains root receipt validation. Internal75s checks are
between bounded stages, not interruption of a running Torch operation;90inner/
120outer watchdog bounds cover the whole task. Review hash is bound by CLI and
file pin. Existing exclusive worker/normal helper only.

GO for this bounded diagnostic; root alone prices/adopts/launches. No speed,
native decoder, output correctness or production-selection claim. No GPU/SSH/
API/queue/git action by reviewer. New review/root-ready only; delivered hashes
freeze and future evidence belongs in supplements.

Loop: profile -> measure -> hypothesis -> optimize/test -> validate -> stack ->
submit measured gains -> learn -> reprioritize -> repeat; idle <=10s.
[Contract](../../ITERATION.md)

# Native-X TMA plus dual producer STMatrix, v1

CPU feasibility/proofs pass; GPU UNRUN. This is a causal composition on frozen
`reverse_tma_x`: only the decoded-W producer's sixteen scalar u32 stores become
four full-warp `stmatrix.sync.aligned.m8n8.x4.shared.b16` operations, copied exactly
from frozen `operand_reverse_dual_stmatrix`. Kernel and extension cache identities
are unique. Undoing the exact store block and kernel name reproduces the entire
native parent's generated extension source byte for byte.

All native-X two-slot SW128 shared ring, math barrier5, elected refill, four A
register slots, W6/B8 rings, alternating ownership, CSR repair, output, publisher,
proxy and barriers3/4 remain unchanged. CPU proofs bind all8192 stage bytes,
full-warp provider16B alignment, all65536 BF16 patterns, complementary owner
barriers, proxy-after-store order, native LDSM/TMA/bar5 and physical A4 retirement.

Local CUDA12.8 cubin: 122 registers, zero stack/spill stores/spill loads, 512threads,
137232 dynamic shared bytes and 65536 rounded register pool. No C75 warning.
Neither import nor preflight loads an extension or creates a CUDA context.

Run CPU-only validation: `python3 preflight.py` from this directory.
`actual_binary_gate.py --sass <loaded-extension-sass>` is a CPU-only actual-JIT
checker: it requires exact main PC/predicate/instruction equality to this frozen
CPU artifact, then reruns physical lifetime, native-X and producer store gates.
It is a gate component, not a GPU execution driver.

GPU admission is pending independent review and a separately frozen standalone
lab outside this immutable kernel directory. That lab must compare the actual
selected incumbent, frozen native parent and this composition; verify actual JIT
before first math, resources/optin occupancy, correctness/graphs/mutations,
immutability and cleanup. Native raw TMA/LDSM all-bit hardware roundtrip remains
unprobed, as in the parent. No performance claim, benchmark-ready promotion,
official submission or global ledger change is made here.

Checker history retains the first preflight failure. Kernel compilation passed;
the checker initially reused the parent's fixed math-dispatch PC0xaf0/target2cf0.
STMatrix code changes shifted dispatch to PC0xad0/target2cd0. Only the copied
checker constants were corrected; candidate CUDA source never changed.

# GO — M8 quarter-stream native/escape-packet prototype

Independent focused measurement_contract_audit CPU review. ONLY muse_bf16_proj.
Pilot `b7c8fccd8a92c57387872f41e164aefb86d70f5756ebef6ba5ae6b8a2bc7a7fb`,
kernel `557c13e62b54dc595f7b62bddd66ff2249bdaa2577cf2b7d17b84928bda1488b`,
transport `965d21022c14fb0ec7a4b68b1e54f6109443184ca0119dbe13b0adc75f6d3ab9`,
manifest `4c208310a89a1280c3c35cf05b86f7937bde36473612d50879e48fcca63ae63a`.
All18 manifest pins verified. This GO covers native/packet source only, not a
complete production integration, runtime harness or GPU admission.

Independent in-memory reconstruction reproduces the exact parent501d native
control and new clone/helper/step/derivation. Target static assertion and device
shape guard restrict new kernel to poolerM8/N4096/K8192 and the exact original
tuple1,1,1,2,16,8,1,1,0,3. Other kernels/pickers are not changed by this isolated
prototype. Any later host wrapper must enforce this supported shape rather than
treating the device early-return on unsupported shape as successful computation.

Packet remains32 uint32 words/128B per original warpgroup:8 unit headers,
16 packed-record words and8 pad words. Each header has five6-bit absolute offsets
0..32, fitting30 bits. Each16-bit record preserves XOR7, highHalf, hm, local
register and lane; unit/quarter move into bucket position. Stable grouping keeps
duplicates and order within a bucket. hm1 is consumed and ignored exactly as
original FMT1/NA1. Empty through32-record groups are supported; original None
overflow/fallback propagates and33 records are rejected, never truncated.
Consumer shuffles unit header from laneu, records from lane8+(index>>1), uses
correct16-bit half and quarter-local register. Maximum record lane23 is valid;
unused/pad records are never consumed. Unsigned packet words must preserve bits
when a future runtime adapter stores them in signed Torch int32 backing.

New four decode4/repair4/HMMA phases retain original dec1 operands, B fragment
mapping and each accumulator's u/q order. Current W and X payload snapshots
precede ld() overwriting the slot. Quarter repairs touch only that phase's four
registers; operations on other quarters can be rebucketed without changing the
current MMA. Helper copies register/mask operands before modifying outputs and
consumes lane/target/hm predicates before the first XOR, retaining alias safety.
Original reduction/output/lifetimes remain unchanged for this supported tuple.

Reran existing1728 changed-transport groups,4 malformed/overflow rejections and
256 payload mapping cases with every write intercepted: PASS. Source field
anchors, None fallback and packet-bank cases pass. These CPU tests establish
transport/source semantics; existing unchanged codec proofs are reused, not
replaced by a new full-matrix campaign. GPU outputs remain untested.

Reuse target-only offline compile: both kernels64regs,0 stack/spills,17408B
dynamic shared. Read-only code inspection confirms first HMMA follows4PRMT
versus16 for parent, establishing removal of the full-fragment decode dependency
before first MMA. Later quarter decodes are partly hoisted; four source phases
do not imply strictly sequential native decode/MMA blocks. New472 slots/462
nonNOP versus parent360/348 means+114 nonNOP, more header/control/shuffle work,
and no register reduction. PC distance/instruction counts are not latency.
Measured speed and official transfer remain UNKNOWN; prior keyed/713c/C11 losses
and mixed signs remain counterevidence.

Root may bind a separate reused one-cell matched-W/X/current-output/fresh-zero/
cleanup harness after its focused integration review. No GPU/SSH/API/queue/git or
compile by reviewer. Only new review/binding paths written; delivered hashes freeze.

Loop: profile -> measure -> hypothesis -> optimize/test -> validate -> stack ->
submit measured gains -> learn -> reprioritize -> repeat; idle <=10s.
[Contract](../../ITERATION.md)

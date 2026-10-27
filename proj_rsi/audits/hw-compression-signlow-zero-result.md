# Six-node copy calibration: responsive zero, flat normal layouts

`state=VALID_SIX_NODE_COUNTER_RESULT; representations=PARKED_BELOW_PRODUCER_ROI; family=NOT_BANNED; matmul_port=NONE; speed_claim=NONE`

Actual [driver](../../results/proj-rsi/takeover/20261027/hw-signlow-zero/hw-signlow-zero01.json), [CSV](../../results/proj-rsi/takeover/20261027/hw-signlow-zero/hw-signlow-zero01.raw.csv), [analysis](../../results/proj-rsi/takeover/20261027/hw-signlow-zero/hw-signlow-zero01.analysis.json), [host log](../../results/proj-rsi/takeover/20261027/hw-signlow-zero/hw-signlow-zero01.host.log). Root ran frozen2d51/driver13d27 2026-10-27 00:37:50–00:38:08UTC:6nodes,3passes each, exit0, no retained-resource warning; root confirmed process gone/node0MiB. Prior four-node sign-low11fff remains unrun/superseded, preserved along with earlier native/packed/byte-plane versions.

|source, same frozen volatile copy reader|GENERIC observed DRAM-read B|NONE B|NONE−GENERIC B|observed read reduction|
|---|---:|---:|---:|---:|
|actual original native Wt|67,114,240|67,114,240|0|0%|
|exact sign-low XOR16|67,114,496|67,113,984|−512|−0.0007629%|
|known-zero64MiB bytes|16,782,848|67,113,472|50,330,624|74.9933247%|

Raw allocation ratio1, sign-low1.0000076288, zero0.2500667526. This supplies the previously missing **same-reader known-zero responsiveness**. Flat native/byteplanes64/sign-low observations are consequently stronger evidence that these exact representations do not yield a useful response under the measured population/reader/policy. It does not prove universal incompressibility or make the zero factor transferable to normal weights/WGMMA.

Independent raw parsing/driver schema checks pass: exactly6unique range-bound nodes, all21selected values finite, retained base units, read/write bytes=32×sectors, same shader/grid128×1×1/block256×1×1/32regs/1024B NCU total shared. Actual JIT hash150ef389 matches the prior byte-plane reader, CUDA d5b5/bindings e391 unchanged; attributes0local/shared, max256threads, occupancy8, SASS resource0stack/spills. Fixed volatile copy addresses cover all64MiB; every selected output is full-byte checked immediately with no later repair replay.

Six JSON preprofile rows each include eager+actual graph, plus6fresh and6immediate selected outputs:24successful copy executions,18grouped oracle records. Full actual GPU sign-low inverse recovers all original64MiB bytes before cloning; CPU/GPU encoding agrees. Known-zero source matches SHA3b6a07d0…421351/allbytes0 and complement255. Parser reconstructs all18disjoint source/main/fresh spans from recorded pointers. Every source grant/query matches1/0; ordinary outputs/fresh outputs remain UNKNOWN if query unsupported. Source bytes remain immutable. All graphs reset before alias drop; all6owners close idempotently to Region0/undefined Tensor. No author GPU work.

Read-side ILC evaluated/success fields remain0 **even for the responsive zero pair**; input/output counts remain~2,097,165sectors for every case. This further demonstrates they are not sufficient read-benefit observers. L2 device fills zeroG/N524434/2097308 and DRAM read reduction agree qualitatively. Aggregate DRAM write bytes remain~51MB and output flags UNKNOWN; do not derive W-only/on-wire compression from them. Unmodified clocks/ordered cases/one capture remain qualifications;3passes are metric replay passes, not3performance observations or a CI.

Disposition: preserve and park these normal layouts below the fresh producer-kernel work. Native uses16logical bits versus packed12's12, so a future representation must overcome the25%payload saving already shipped, uncompressed TMA/LDS/storage, and its exact decode cost. Sign-low's estimated24ops/8weights is worse than packed12~9/8 despite more leading-zero bits; byte planes' cheaper4PRMT/8 were not enough to establish traffic benefit. Reopening needs a materially new representation with credible **full-path** cost and observed read response against its controls, then the actual12bit incumbent comparison. No extra format GPU grid, matmul port, timing inference, SM rewrite or whole-family ban follows from these versions.

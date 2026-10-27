# Conditional inverse-only proof

Run `python3 proj_rsi/candidates/hw_compression_byteplanes_inverse/proof.py`.

CPU-only all65536bit/lane-layout proof plus local SM90a compilation. The device microkernel only loads fixed native-fragment byte planes, reconstructs32BF16 values with16PRMT and stores every output. It has no host launcher, GEMM/WGMMA or GPU/API execution. Inverse-only register count does not establish full-kernel resource fit. Frozen four-node counter package is unchanged.

See [conditional design/accounting](../../audits/hw-compression-byteplanes-consumer-design.md). A positive root-owned four-case counter result and subsequent authorization are required before a matmul implementation.

# Ownership iteration: exact-runtime lifetime PASS

Original v1 trial remains INVALID; no timing was repaired or relabeled. Exact
Python3.11/Torch2.8 CPU probe distinguishes harmless cached wrappers from the
actual locals() snapshot alias: last cp dictionary persists after del; unconditional
cp=None releases it. Positive/control artifact hw-owner-cpu28-locals.json.

Non-timing GPU lifetime gate hw-owner-lifetime01 terminated2026-10-26T23:01:55Z,
exit0.22counttraces,8outputoracles and originalRQprofiles pass. GenuineTensor,
dtypeview andStoragealiases reject release and remain valid. After graphreset,
state/output/cpdrop andgc, both Regionowners reach0 andownTensorsareundefined.
No cleanup-retention warning orTraceback. Independentreviewconfirmsno live
process/container/CUDAallocation. Sourcehelper7f2 anddriver5e0 remainfrozen.

This validates allocation ownership and cleanup; it is not a performance result.
NativeBF16existingLt allocation test is a separate next experiment, with its own
correctness/control/actualgraph/performance gates. Sources and previous versions
are retained so later versions can be compared and diagnosed.

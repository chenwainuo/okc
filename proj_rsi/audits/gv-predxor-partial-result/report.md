# GV predxor partial evidence

**Keep the partial direct measurements and numerical checks.** The optional profiler failure occurred after148 actual-policy correctness cases, both actual O32 all-basis checks, and five fixed cells with780 direct rows (705 timed excluding warmups),30 own-terminal receipts and75 mutations. Journals match saved rows and each terminal matches its own last fixture. Four cells also completed profiles; the fifth completed timing/terminals/mutations before profiling failed.

| Fixed pooler geometry / m | Parent minus predxor paired µs | F1 µs |
|---|---:|---:|
| pooler/1/1.1.1.2.16.8.1.1.0.3 | +0.095998868 | +0.064000487 |
| pooler/8/1.1.1.2.16.8.1.1.0.3 | +0.128000043 | +0.000000000 |
| pooler/1/1.1.2.2.8.16.1.1.0.0 | -0.031999312 | +0.064000487 |
| pooler/8/1.1.2.2.8.16.1.1.0.0 | +0.256000087 | +0.176001340 |
| pooler/1/1.1.2.2.8.16.1.1.0.1 | +0.143999234 | +0.064000487 |

The failure is an empty filtered Kineto CUDA-event list for the fifth cell’s parent posttiming profile. This path does not parse Chrome trace categories. Earlier profiles succeed; missing raw collector events/trace prevent a more precise underlying diagnosis. It does not invalidate preceding direct event timings or outputs.

Full30 integration was never reached:0 actual-policy timing shapes/rows. Retain fixed gains, losses and F1 disagreement as mechanism evidence, with no fullpolicy claim. Proceed to the minimal selected-policy integration without mandatory profiler; reuse the existing checks rather than repeating them. Graph cleanup was clean. Full contrasts including wrapper controls are in report.json.

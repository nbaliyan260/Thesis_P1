| Detector endpoint | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| Actually changed prefix cases localized | 96/96 | 96/96 |
| No-op controls with an alarm | 0/12 | 0/12 |
| Comparison time, all cases: median [IQR] | 0.930 ms [0.852, 0.999] | 0.727 ms [0.678, 0.801] |
| Logical comparison reads per window | 2.81-5.62 MiB | 1.50-3.00 MiB |

The comparison localized 192/192 actually changed prefix cases and raised 0 alarms on 24 no-op controls. There were 0 intended perturbations with no actual rounded change. These are controlled-case results, not sensitivity/specificity estimates for hardware faults. All 1,080 recorded detection timings are separate from recovery timings; no new normal-journal overhead experiment was run.

The supplemental repair policy used the detector's unique layer result, not the injected layer. Ground truth was checked only to evaluate the controlled experiment. Exact comparison consumes the already-required trusted snapshot and at least two copies of its logical prefix payload in reads; the table does not count all temporary-flag traffic or protected-storage cost. No claim of low-overhead deployment follows from these short-context measurements.

| Method | Alias-full / method: SmolLM2 | Alias-full / method: Qwen |
| --- | --- | --- |
| Conservative full replay | 1.00x [0.99, 1.00] | 0.99x [0.99, 1.00] |
| View / one-layer restore | 1.00x [0.99, 1.00] | 1.00x [0.99, 1.00] |
| Checkpoint-alias full replay | 1.00x [1.00, 1.00] | 1.00x [1.00, 1.00] |
| Sparse RECUT | 1.29x [1.01, 3.45] | 1.30x [1.00, 3.43] |
| All-layer journal | 1.83x [1.01, 17.93] | 1.76x [1.00, 11.62] |

Ratios are paired per-case medians with interquartile ranges. Above 1 means the listed method is faster than checkpoint-alias full replay; below 1 means slower. All 96 perturbed cases per model are included.

Every method recovered all 192 reused cases, with exact state/output/continuation checks and immutable-source guards in all 4,800 measured recovery trials. A separate evidence audit passed 98,997 consistency checks. It does not independently reload supplemental tensors: none were saved. The final expanded unit suite contains 76 passing tests, distinct from the original 58-test primary gate.

Median checkpoint-alias replay times were SmolLM2-135M: 239.42 ms; Qwen2.5-0.5B: 199.71 ms.

At layer zero, conservative full and sparse invoke the same replay algorithm. Their same-algorithm control ratios were SmolLM2-135M: 1.00x; Qwen2.5-0.5B: 1.00x. The alias-full/sparse ratios at that layer were SmolLM2-135M: 1.00x; Qwen2.5-0.5B: 0.99x, reflecting restoration differences as well as timing variability. These cases save no decoder-layer work. Per-layer/window strata remain in the summary; a pooled gain is not a gain at every layer.

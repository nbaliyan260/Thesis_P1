# RECUT post-hoc exploratory trade-offs

This is descriptive analysis of the complete frozen run, not a preregistered test or evidence of a timing Pareto frontier. All cases remain in the JSON outputs; noops are summarized separately. Quantiles describe case medians, not independent trials.

## Injected-fault cases, by model

Recovery ratios below are full replay / journal recovery (larger is faster). Nonnegative-cost sensitivity is a hypothetical ratio, not a measured fault probability.

| Model | Cases | Policy | Median recovery ratio | Median journal KiB | Mean nonnegative normal cost (ms) | Mean signed recovery saving (ms) | Clipped sensitivity | Original primary ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-135M | 96 | sparse | 1.297 | 27.0 | 1.454 | 75.498 | 0.0193 | 0.0108 |
| HuggingFaceTB/SmolLM2-135M | 96 | all | 1.822 | 261.0 | 4.477 | 112.154 | 0.0399 | 0.0375 |
| Qwen/Qwen2.5-0.5B | 96 | sparse | 1.308 | 42.0 | 1.318 | 63.931 | 0.0206 | 0.0108 |
| Qwen/Qwen2.5-0.5B | 96 | all | 1.774 | 322.0 | 4.246 | 90.460 | 0.0469 | 0.0451 |

## Sparse versus all-layer journals

Positive timing differences mean sparse is slower. Observed repeat-range overlap and sign counts are noise diagnostics only; neither supplies a confidence interval.

- HuggingFaceTB/SmolLM2-135M: sparse-minus-all median recovery difference 27.193 ms; median payload ratio 0.103; sparse repair medians lower/equal/higher in 14/0/82 cases; observed repair ranges overlap in 32/96 cases.
- Qwen/Qwen2.5-0.5B: sparse-minus-all median recovery difference 20.393 ms; median payload ratio 0.130; sparse repair medians lower/equal/higher in 15/0/81 cases; observed repair ranges overlap in 31/96 cases.

## Interpretation limits

- Post-hoc exploratory groupings were selected after the debug pilot, before primary completion; they do not change the frozen experiment or primary analyzer.
- Constructed prompts, seeds, finite injections, and repeated policies/timings are not IID deployment trials; no field fault or divergence rate is inferred.
- Nonnegative-cost sensitivity is a noise-sensitive cost/saving diagnostic for this artificial case mix. Per-case positive clipping prevents negative observed overhead from funding the ratio, but is not an estimator of true cost, is not a statistical upper bound, and can bias the numerator upward. Ratios above one give no feasible p in the toy interval [0,1].
- Conditional divergence groups describe an observed outcome, not a prospective policy or causal comparison. Missing factorial cells are absent, not imputed as zero.
- Timing repetitions are too few to establish a Pareto frontier. Median orderings, range overlap and paired signs are descriptive only; no categorical timing dominance is claimed.
- Oracle fault detection/localization, trusted checkpoint availability, commit delay, full reference verification, and production protection costs are not all charged. No system break-even or deployment recommendation follows from this ratio.
- Journal bytes are logical tensor payload, not protected total storage or allocator peaks. Normal journal capture retains every configured cut even when selected repair cut is zero.
- All-layer journals use every nonzero decoder-layer input; sparse journals use the fixed configured cuts. No placement search or optimized serving-kernel baseline is evaluated.
- This analysis covers only the frozen primary/debug runner schema; any supplemental optimized restoration/cropping baseline is a separate experiment and is not combined here.
- Equality counts reuse recorded verification flags. Consult the independent raw-results audit and its snapshot limits; this script does not reload models or replay inference.

Per-model/layer/window/fault-shape/divergence tables are in `frontier_exploratory.json`; every case, raw timing sample, negative saving, and noop control is retained in `frontier_cases.json`.

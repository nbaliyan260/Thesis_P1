# RECUT post-hoc exploratory trade-off analysis

Scope: a separate, descriptive analysis of a **complete** RECUT run, authored after examining the debug pilot while the frozen primary experiment was still running. This is not a preregistered hypothesis test, novelty clearance, a replacement for the primary analyzer, or an independent implementation of RECUT. No experiment settings, raw rows, runner, engine, primary analyzer, or auditor are changed.

## Questions and complete-case policy

The analysis asks where the measured trade-off between the fixed sparse and all-layer journals changes with the model, faulty layer, replay window, injected fault shape, and observed first token divergence. It retains every configured model-by-case row. Noops are a separate population; nominal injections with zero changed elements remain visible. Negative observed recovery savings and negative observed overhead are not removed. Incomplete, duplicate, malformed, or verification-failing runs are rejected as a whole rather than reduced to a favorable subset.

The independent script `experiments/frontier_analysis.py` reads `metadata.json` and `episodes.jsonl`, checks completion, raw SHA, complete unique model-by-case coverage, per-row configuration/revision agreement, and verified timing repetition/order coverage. It records any existing independent audit's status and whether its episode and metadata hashes match. This script does not substitute for that audit, reload models, or independently establish equality.

All summaries are explicitly separated by model and control/fault population. Reported groups are: model; layer; window; fault shape; K/V kind plus fault shape; observed divergence class; layer plus window; and layer plus window plus shape plus divergence. The last grouping exposes intersections rather than hiding selection. Only observed cells exist; missing cells are not zero observations. Every cell lists its case IDs and size. Debug cases are too sparse and confounded to estimate independent effects of these factors.

## Measurements and direction conventions

For each case, use the median of the recorded recovery repetitions for full, sparse, and all-layer policies, and the median of normal decoding repetitions for none, sparse, and all. Preserve every raw repetition in the case output. The primary run has three recovery repetitions and five normal repetitions; debug has one and two respectively. A repetition, policy, shared prompt, or artificial fault variant is not an independent deployment trial.

- Full-recovery time / journal-recovery time: values above one mean a lower observed journal-recovery median.
- Sparse-recovery time / all-layer-recovery time: values above one mean a higher observed sparse-recovery median.
- Sparse minus all-layer time: positive means sparse was slower in that measurement.
- Sparse / all-layer payload: logical tensor bytes only; it omits protection, metadata, allocation and serving integration costs.
- Executed layer-token fraction: recorded layer steps divided by decoder layers times the replay window. This is a work count, not a latency model.

Show case-median quantiles, recovery savings, normal overhead, payload, work fractions, and exact-recovery flag counts. Sparse versus all-layer comparisons also show same-repetition difference signs and observed min–max range overlap. These are timing-noise diagnostics only. Even disjoint observed ranges do not establish population timing dominance; overlap does not establish equivalence. No confidence intervals, significance tests, bootstrap over non-independent rows, or categorical Pareto claims are produced. The script reports descriptive trade-offs, not a demonstrated frontier.

## Nonnegative-cost sensitivity analysis

For case i and journal policy m, define:

`H_i = max(median(normal_m_i) - median(normal_none_i), 0)`

`S_i = median(repair_full_i) - median(repair_m_i)`

For every explicitly enumerated cell, report `mean(H_i) / mean(S_i)` only when `mean(S_i) > 0`. Both means use **all** cases in that cell, including negative or zero recovery savings. Also expose the signed mean normal cost, negative-cost count, and nonpositive-saving count. The original primary ratio is shown alongside: signed mean normal cost divided by mean saving, defined only for nonnegative mean cost and strictly positive mean saving. An unrestricted signed ratio is retained separately so a negative observed mean cost is not hidden. Do not compute a ratio on only beneficial cases or pool ratios with different denominators. Per-case diagnostics remain available to expose heterogeneity.

The accounting assumption is one normal window plus probability p of one recovery event. A ratio above one has no feasible break-even p in the toy interval [0,1], under the assumed maximum of one recovery event per window. A zero ratio caused by nonpositive measured overhead is not evidence that journals are free. Per-case clipping is intentionally stricter than allowing negative measured overhead to cancel positive costs, but it is noise-sensitive and upward-biased, not an estimator of true cost or a statistical upper confidence bound. Call it **nonnegative-cost sensitivity**, not a statistically conservative bound. Do not infer real fault frequencies, incident independence, or a deployment recommendation from this ratio.

Important omitted or oracle-provided costs include practical detection/localization, protected journal/checkpoint storage, detector latency and false positives, commit holdback, and production integration. Full-reference correctness checks and fixtures are experimental, not a deployed verifier. Checkpoint restoration is inside measured recovery, whereas common working-cache fixture clones are excluded. The normal capture measurements retain all configured journal cuts, including cases whose selected recovery cut is zero. The ratio cannot establish an end-to-end operational break-even point.

## Interpretation and artifact contract

Run locally on debug for validation:

```text
research-next/.venv/bin/python research-next/experiments/frontier_analysis.py research-next/runs/debug-v1
```

Run the same unchanged script on the completed primary folder after the main analysis/audit. It creates only `frontier_exploratory.json`, `frontier_cases.json`, and `frontier_exploratory.md` in the selected run. They contain source/input hashes and an explicit post-hoc label. Running it again updates only these derived files. Supplemental prefix-view/known-layer-only restoration baseline experiments are outside this analysis and must be reported separately, not silently combined with the frozen primary timings.

Potentially useful follow-up evidence is a reproducible region where sparse journals use less protected storage, retain acceptable recovery latency, and have a favorable normal-overhead/recovery-saving balance against strong activation-cache/recompute baselines. This pilot alone cannot establish such a production region, optimal placement, generalization across models or prompts, a new fault-propagation phenomenon, or a publishable novelty claim. Outcome-conditioned divergence groups are descriptive and cannot choose a journal policy prospectively.

## Debug validation performed

The script completed on all eight debug rows (six nominal injections and two noops; four cases per model), with matching raw-result and independent-audit hashes. Twenty-seven ad hoc, in-memory assertions checked complete coverage across each grouping, agreement of the original-primary ratios with the existing primary analyzer, inclusion of negative-saving cases in the sensitivity denominator, rejection of incomplete/duplicate/unverified runs, and unchanged raw-result SHA. These are software validation checks, not additional scientific trials. Debug timings are too sparse to support substantive frontier conclusions. No primary data or supplemental-baseline data were analyzed during this validation.

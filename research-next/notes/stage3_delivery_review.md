# Stage-3 delivery arithmetic and claim review

Reviewed 2026-09-16 while the main experiment was still running. No final main-result numbers were available to this review. Only the non-frozen delivery-table generator and its new regression tests were edited; no experiment/controller/auditor source or PDF-template source was changed.

## Delivery generator: defects fixed

The review found a real completion-path crash: the original sha helper called .open() on its argument, while main called sha(__file__) with a string. It now normalizes paths before opening.

Provenance gates were strengthened without changing measured data or the statistical hierarchy:

- Validation uses explicit ValueError checks rather than assert, so running Python with optimization cannot disable the gates.
- Duplicate model entries cannot silently overwrite earlier entries; duplicate case-median entries are rejected.
- Both HPC and local audits must bind all five current raw input files, configuration/manifest hashes and the current frozen auditor source, not just metadata.json.
- Correction sidecars must bind their generator, original/raw inputs and corrected summary. The independent recomputation must report agreement specifically against that same corrected summary and sidecar.
- Complete local tensor-reload coverage is required, and local audit counts are cross-checked against the independently reconstructed main-session counts.
- Nonstandard JSON values such as NaN, missing displayed strata and nonfinite metric values are refused.
- Output still refuses overwriting. Input-hash ordering is deterministic.

The generator now requires analyzer_comparison.status = agree with summary_name = summary_corrected.json. An older comparison against original summary.json, or an unresolved refusal-counter discrepancy, is insufficient for final delivery.

These are hash/provenance relationships, not digital signatures or proof that arbitrary untrusted JSON was produced by a particular program. The underlying independent audit, source freeze, raw recomputation and restricted reporting correction remain necessary evidence layers.

## Arithmetic and aggregation checked

| Quantity | Generator behavior and interpretation |
| --- | --- |
| Clean milliseconds | Median across the three prompt-level case medians, separately for each model and initial prefix/window shape. Each case's session value already includes its separately timed constructor. |
| Protection cost ratios | Median of matched case-level protected/unprotected ratios. This is intentionally not a ratio obtained by dividing the displayed cross-prompt medians. |
| Fault-window ratios | All eight configured strata have their own column; only affected windows and the full-batched/sparse-batched pair enter these columns. Timing repetitions were collapsed before forming those matched ratios. |
| Fault-session ratios | Constructor-inclusive matched case-session ratios, pooled descriptively across the eight supported fault scenarios for that model/shape. Journal/checkpoint guard challenges are excluded. This is not an incidence-weighted deployment speedup. |
| Sealing ablation | Constructor-inclusive legacy/batched ratios are separated by full versus sparse and clean versus late-value fault. |
| GPU memory | Maximum recorded allocation over every measured variant/scenario at that model/shape, not the median and not a production minimum. Logical journal KiB are separate clean payload medians. |
| Availability | Median of per-case-window first-token availability. The hold column is the median of per-case-window mean token-hold durations. It is not pooled-token mean hold or end-to-end/network TTFT. |
| Break-even | The median p* uses only defined positive-H/positive-R cases, but Defined/all retains every matched affected case-window in its denominator. Nonpositive-H/positive-R cases and all other sign combinations remain counted separately. |
| Audit checks/tensors | Totals use the local audit once. The HPC audit of the same evidence is checked but not added as another population of observations. |
| Largest seal batch | Combined-model summary takes a maximum, not a sum. |

The Defined/all denominator is 60 per model/shape for the frozen full/sparse matrix: three prompts times two seeds times ten affected windows across scenarios. Seven fault scenarios contribute one affected window each; repeated-prefix contributes three. Its additional windows therefore have explicit extra weight in the pooled break-even display. Neither the denominator nor the repeated timing executions represent independent physical fault samples.

For context labels, 128 and 2,048 are the **initial** prefix lengths. Later windows start at longer prefixes. Non-repeated ordinary faults occur in window two; repeated-prefix spans all three. Pairing is still correct because clean and fault comparisons match the same window index. Calling table rows “Model / initial prefix” or explaining this in the caption would avoid ambiguity.

Likewise, partial-to-full counts are measured window executions, including policy variants and timing repetitions; they should not be described as that many distinct naturally occurring faults.

## Regression evidence

The added reporting/test_report_tables.py contains 14 synthetic tests. Together with the 13 existing reporting tests, all 27 passed locally. No models, GPU, HPC execution or real study results are used by these tests.

The synthetic cases explicitly distinguish:

- case-first median ratios from ratios of cross-prompt medians;
- short and long context shapes;
- all eight fault columns;
- fair fault-session comparisons from an artificially huge policy-specific guard ratio;
- maximum GPU allocation from a much smaller median allocation;
- defined break-even thresholds from the full affected-window denominator;
- the local audit total from deliberately much larger HPC audit counts;
- a maximum seal batch from a sum;
- stale raw, corrected-summary and sidecar bindings;
- duplicate cases/models, missing strata and provenance gates under python -O.

The fixture exercises delivery arithmetic and validation logic. It is not a complete experimental record and does not replace the independent run auditor.

## PDF-template claim review

Inspected artifacts/RECUT_04_Complete_Prototype_and_Validation.template.md for substantive contract wording, not visual layout or final rendering.

No blocking core-contract overclaim was found. It correctly distinguishes:

- a completed bounded study from a completed thesis or A/A*-ready paper;
- numerical persistent old-prefix detection from byte-sensitive seals;
- trusted initial/model/guard execution from demonstrated properties;
- every potentially corrupting layer retaining evidence, including the reverted-low-layer/high-layer-alarm counterexample;
- irreversible history fallback from reusing activations after apparent token reconvergence;
- checkpoint refusal from checkpoint repair;
- clean model-state equality from truthful or safe answers;
- native/bare cost baselines from resilience-equivalent methods;
- raw timing from injection-subtracted diagnostics;
- common prefill from post-prefill parity;
- internal saved-state audit from re-execution or external replication;
- a local single-owner method boundary from concurrent/durable/network transactions.

Two optional precision edits for the final human-facing prose:

1. “Ordinary recovery repeats a whole uncommitted window” is better phrased as “our protected full-replay baseline repeats the whole uncommitted window.” The existing statement should not be read as a universal description of published recovery systems.
2. “Stronger replay baseline” for all-cuts is defensible as a denser reuse comparator, but “denser-journal ablation” is more neutral. It stores more cuts; it is not necessarily faster or equivalent to a published optimized restoration system.

All final numeric placeholders, natural-divergence claims, snapshot outcomes and hardware statements must be populated only after the completed main artifacts and local audits pass their gates. This review does not authorize claiming favorable results that are still pending.

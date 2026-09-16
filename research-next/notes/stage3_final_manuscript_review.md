# Final filled Stage 3 manuscript review

Reviewed 16 September 2026. This was a read-only numerical and claim-scope review; no frozen source, raw experiment record, report tool, manuscript or table was changed. PDF rendering and visual QA are a separate check.

## Exact reviewed artifact

- Manuscript: `artifacts/RECUT_04_Complete_Prototype_and_Validation.md`
- SHA256: `b88e8d8e37c897d15053b237cf5305ea58e128c746af8a996f0f9acca55959c7`
- `stage3/RESULTS.md` is exactly the same content with page-break comments removed.
- Independent raw recomputation: `notes/stage3_main_recomputed.json`, SHA256 `ff14bea7a237b171e3c7db8a2b224e459378e983ef5186188abbc94b1dfe3121`.

The current manuscript, template, table data, interpretation, independent recomputation, composer, selected diagnostic and diagnostic amendment hashes agree with manuscript provenance. The report table input bindings agree with the current main raw records, original/corrected summaries, correction sidecars and audits. All direct Markdown tables and the transposed two-shape fault table match the table artifact. The aggregate table totals match the per-model independent recomputation.

## Headline numbers

No blocking numerical discrepancy was found.

- Clean initialization-inclusive sparse/bare overhead is approximately 5.24–9.17% at initial prefix 128 and 14.06–48.55% at 2,048. The executive's approximate 5–9% and 14–49% ranges are supported.
- The twelve late-K/V, model/shape affected-window median ratios range from 1.3753 to 1.5320 full/sparse. The approximate 1.38–1.53 range is supported; it is not a speedup over unprotected inference.
- Long-shape Qwen-7B sparse/native initialization-inclusive matched cost is approximately 1.4955. The roughly 50% overhead and 2.38-second versus 1.59-second cross-prompt session summaries are correctly distinguished.
- Main totals are 2,358 measured sessions, 7,020 attempted windows, 6,966 exact-checked commits and 54 specified refusals. The 2,358 recovered-window count includes guard recovery and policy/repetition executions, not independent fault incidents.
- Main state evidence is 21 archives and 2,338 actual/reference tensor pairs, with all pairs passing numeric and raw-byte checks. Local audits total 1,018,993 checks; independent recomputation agrees on 81,309 compared quantities.

## Denominators and qualifications

The report correctly states the case-first hierarchy: reduce timing repetitions before forming matched ratios. Three prompt-level cases underlie clean session summaries; ordinary fault strata use six affected case-windows and the repeated stratum uses eighteen. The equal-case session mixture contains 48 cases per model/shape, not a deployment incidence distribution. Sensitivity calculations retain all 60 affected case-windows, including undefined/sign-changing cases, and report threshold medians only over the positive-H/positive-R subset.

Memory increases are distinguished from total GPU allocation, logical checkpoint/journal payloads and unmeasured host peaks. Native/bare step-completion availability is a local proxy, not network TTFT. Common prefill/copy/reference work is excluded, session initialization is explicitly added where claimed, and guard treatments are excluded from fair timing comparisons.

The 12 main divergent replay executions and four partial-to-full executions are confined to Smol; main Qwen runs had none. The report explains that the four partial-to-full executions arise from one configured case, two policies and two timing repetitions. Debug and forced-test evidence are not pooled into that main count.

The selected post-hoc fallback replay is clearly separated: 13 sessions, 33 commits, three expected refusals, three additional archives and 366 tensor pairs. None enters primary sample/performance totals. The one-based `first_divergence=7` explanation correctly locates the first layer-zero replay at the eighth step. The original main archive gap is disclosed rather than retroactively described as prespecified coverage.

## Claim-scope assessment

The narrow persistent, finite, numerically unequal old-prefix fault contract and trusted phases are explicit. Signed-zero, reverted, suffix-only, computation-origin, between-window and physical GPU faults are not silently claimed. The report does not infer production effectiveness, physical incident rates, optimal cut placement, semantic AI safety, first-method novelty or A/A* acceptance from these experiments.

The prior-work table and publication judgment distinguish internal same-contract comparators from reproductions of published serving systems. The large clean cost and output holdback are prominent adverse findings. The internal AI-assisted audit is not presented as human peer review or independent external replication.

Two nonblocking language nuances remain worth remembering in later paper writing: “necessary upper-layer work” is an informal mechanism explanation, not a minimality result; and evidence for divergence fallback across the model set includes debug/regression evidence, not observed main-run Qwen divergence. The current manuscript already states both qualifications nearby, so neither is a release blocker.

**Disposition:** numerical and claim-scope signoff for this bounded prototype/evaluation report, subject to the separately performed visual/package QA. This is not a publication-readiness or scientific-novelty certification.

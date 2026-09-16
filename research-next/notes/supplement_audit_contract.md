# Supplemental audit and summary contract

The independent standard-library script experiments/audit_supplement.py does not import the supplemental runner, primary runner, engine, or primary analyzer. Run it only after the supplemental experiment has completed:

    research-next/.venv/bin/python research-next/experiments/audit_supplement.py \
      --run research-next/runs/SUPPLEMENT_FOLDER \
      --primary research-next/runs/primary-v1 \
      --config research-next/experiments/config_primary.json \
      --manifest research-next/experiments/model_manifest.json

It writes audit.json and, only after validation passes, a descriptive summary.json. It does not modify raw records or primary results. Failure exits nonzero and does not create a new summary; any summary from an earlier invocation must not be treated as current without matching the audit's hashes.

## Checks

The complete run must contain exactly 192 distinct perturbed-case recovery records and 216 distinct detection records, including 24 noops, with the expected two immutable model revisions. Metadata, actual config and manifest files, primary raw/metadata hashes, source files, and numeric environment are bound explicitly. Each recovery row must exactly match its corresponding detection row after excluding the recovery timing array.

All controlled fault coordinates, original/requested/changed vector data, input prefixes, initial tokens, reference/faulty outputs, layer choices and model configuration are checked against the primary evidence. The old-prefix detector must identify precisely the changed layer when the recorded mutation actually changed values; rounded-to-zero mutations and noops must produce no alarm. The no-alarm path must select cut zero. Five positive finite detector timings, logical read bytes equal to twice the checkpoint payload, and both immutability flags are checked for every case.

Each recovery case must retain five methods times five rotated repetitions in the recorded order, with no warmup row. Every state, logits, output token and two-step continuation flag must pass, along with checkpoint and initial-faulty immutability flags. First divergence is recomputed from reference/faulty tokens; layer-start traces must switch irreversibly to zero after that divergence and match the layer-token work formula. Native HF parity requires all 142 positions per model. The excluded warmup is declared in source/metadata, but its execution is not independently recoverable from saved rows.

## Summary interface

models[model].comparisons[baseline][method].speedup contains min/q25/median/q75/max and mean of **paired per-case median timing ratios**. Baselines are full, layer_view_full, and checkpoint_alias_full; methods are those three plus sparse and all. A ratio above one means the method had a lower observed recovery median. timing_ms[method] summarizes per-case timing medians. Correctness counts are in correctness[method]; prefix-length/window/layer cells are under strata.

c0_controls separately reports cases with zero selected sparse cut and zero faulty layer. These are perturbed cases, not noops. Detector summaries are in detection[model].all_cases, .perturbed, and .noops, with observed alarms, routes, timing medians and logical read bytes.

## Evidence limits

No supplemental tensor snapshots exist for independent loading, and the auditor does not rerun model inference. Numerical equality, continuation and immutability remain assertions recorded by the runner; this audit verifies internal consistency and provenance, not an independent implementation or hardware replication. SHA-bound snapshot directories do not authenticate every weight byte.

Detector measurements are separate from recovery timings. They cover only persistent post-checkpoint modifications of the old K/V prefix. Suffix-only and transient/reverted errors, pre-checkpoint errors, and corrupt trusted checkpoints are not covered. The source uses one consolidated layer-decision transfer, but no independent traffic profiler is run. Logical read bytes are not physical memory traffic.

No new full normal-decoding overhead measurements, deployment fault rates, IID assumptions, end-to-end break-even, statistical dominance, or production-detector claims are justified. Sparse/all retain conservative restoration; supplemental timings must not be pooled with the frozen primary measurements.

## Software validation before real data

A complete synthetic schema was constructed **only in memory**, exercising 192 recovery records and 216 detector records. All 98,997 audit predicates passed, summary strata/correctness counts were checked, and a tampered detector record was rejected. No synthetic data were written into any experiment folder. This is a parser/accounting smoke test, not experimental evidence; the real completed supplemental run still requires auditing.

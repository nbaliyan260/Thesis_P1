# RECUT stage-3 artifact: start here

This is a bounded single-GPU, single-sequence autoregressive inference recovery study. It is not model-training recovery, a production server, general GPU fault tolerance, or a claim of publication readiness. The final measured interpretation is in `RESULTS.md` and `../../output/pdf/RECUT_04_Complete_Prototype_and_Validation.pdf`. The original prospective `PLAN.md` and `README.md` remain frozen; this delivery guide is separate.

## What is supplied

- Source, fixed input configurations and model revisions, unit tests, Slurm scripts and the 23-file pre-measurement freeze.
- Complete stage-3 debug and main JSON records, separate warmup/profile records, the predeclared full-state tensor archives, HPC audits and local re-audits.
- A separately declared, post-hoc fallback validation replay with its own three state archives; never pooled into main timing or sample totals.
- Independent raw-result recomputation, contract and prior-art reviews, final report source/PDF, execution logs and a delivery file-hash manifest.
- Required earlier-stage engine/test dependencies. Earlier-stage measured results are not silently pooled into stage 3.

Model weights, Python environments, authentication material and SSH sockets are deliberately absent. Download the public weights at the recorded immutable revisions under their own licenses. Lab usernames/paths in provenance are not credentials. There is no automatic GitHub push or conference submission.

## Audit existing evidence without a GPU or model weights

Use Python with PyTorch and NumPy installed. The audit safely loads only tensor/basic-data archives using `weights_only=True`; it does not execute a saved model. From `research-next`:

```bash
python stage3/freeze_study.py verify
python stage3/audit_study.py runs/stage3-main-v1/smol135m --output runs/stage3-main-v1/smol135m/audit_reader.json
python stage3/audit_study.py runs/stage3-main-v1/qwen05b --output runs/stage3-main-v1/qwen05b/audit_reader.json
python stage3/audit_study.py runs/stage3-main-v1/qwen7b --output runs/stage3-main-v1/qwen7b/audit_reader.json
python notes/independent_stage3_report.py runs/stage3-main-v1/smol135m runs/stage3-main-v1/qwen05b runs/stage3-main-v1/qwen7b --output notes/stage3_reader_recomputed.json --compare-summary --summary-name summary_corrected.json
```

Choose new output filenames rather than overwriting supplied audit reports. Use `--config stage3/config_debug.json` when auditing the debug directories. Do not use `--records-only` when claiming a tensor reload. A record/hash audit is not model re-execution or external replication; only the specified snapshot subset contains independently reloadable full states.

The ZIP's `stage3/reporting/delivery_manifest.json` lists SHA256 and byte length for every included member except itself. Packaging verifies each member after compression. The separately supplied `package_summary.json` records the complete ZIP hash. Hash consistency establishes provenance consistency, not scientific validity or malicious-tampering resistance against an attacker who can replace the manifest too.

### Selected fallback diagnostic

The main Smol code/prefix-128/W8 multi-prefix case (seed 7431, repetition 0) naturally switched from a nonzero replay cut to layer zero. It was not part of the predeclared main snapshot subset. `notes/stage3_fallback_diagnostic_amendment.md` declares the post-hoc validation-only replay, binds the original case and fixes the three-policy snapshot selection before that replay. Its complete evidence is under `runs/stage3-fallback-diagnostic-v1/smol135m`; do not add its timings or counts to the main study.

```bash
python stage3/audit_study.py runs/stage3-fallback-diagnostic-v1/smol135m --config stage3/reporting/fallback_config.json --output runs/stage3-fallback-diagnostic-v1/smol135m/audit_reader.json
python stage3/reporting/verify_fallback_diagnostic.py --output runs/stage3-fallback-diagnostic-v1/smol135m/validation_reader.json
```

The second command verifies the supplied HPC/local audit bindings and exact original-case replay records; the first performs a new full CPU tensor reload. The replay is an additional check of one selected existing fault, not an independent fault observation. A reader reexecution with different trajectories must be reported as such, not forced through the verifier by changing expected hashes.

## Run correctness tests

The combined command is:

```bash
python -m pytest -q stage3/test_stage3_controller.py stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
python -m pytest -q stage3/reporting/test_reporting.py stage3/reporting/test_report_tables.py
```

Explicitly name `tests_independent.py`; default discovery may not include it. Local CPU tests skip CUDA-only cases. Do not describe those skips as GPU validation. The allocated study environment is Python 3.13.9, Torch 2.6.0+cu124 and Transformers 4.51.3; `experiments/environment_freeze.txt` records the original dependency freeze. Local development used a different PyTorch version and is not the performance environment.

## Re-execute the study

Follow the frozen `README.md` and `PLAN.md` for the exact measurement contract and CLI. An authorized BF16-capable CUDA GPU is required for the measured protocol. Public weights must be cached at the pinned revisions. Native/bare clean baselines are unprotected references; full/sparse/all-cuts share the protected contract. Warmups and synchronized profiles are excluded from headline timings.

Original Slurm paths are account-specific. On another account, work in a new copy, localize `model_manifest.json` and lab scripts, preserve a record of the changes, and create a separately named freeze with `freeze_study.py write --file stage3/localized_protocol_freeze.json`. Verify that new freeze explicitly; do not overwrite the supplied original freeze or call the new run an exact reproduction of the original execution environment. The stage-3 fetcher requires the declared cache paths to agree with the download location; it intentionally refuses silently rewriting provenance. The older `experiments/fetch_pinned_models.py` can create a new localized manifest from pinned revisions, but it must not substitute HEAD or overwrite the measured inputs.

Run a fresh debug directory for each model first, and proceed to a new main directory only after all debugging checks pass. The runner rejects nonempty output directories. Preserve failures and declared amendments. Do not run model inference on a shared login node or bypass scheduler limits. The supplied `submit_study.sh` has original run names and must not be resubmitted into their existing directories.

## Interpret the numbers correctly

Headline timings cover BEGIN through COMMIT/refusal. Session construction is measured separately and included in initialization-inclusive session totals. Common input copying, prefill and experimental native-reference verification remain outside both boundaries. Injection overhead is included; subtraction is only a sensitivity diagnostic. Memory peaks include resident experimental reference state and are not production-server minimum requirements.

Case medians first collapse timing repetitions; ratios compare matched case medians. The fixed synthetic prompts, fault strata, policies and repeated timing runs are not independent field-fault samples. Full-state numeric equality does not certify truthful model answers or general raw-bit identity. The detector is numerically blind to signed-zero-only changes; seals preserve bytes. Every potentially corrupting layer must retain numerically unequal old-prefix evidence until detection. Read the complete trust/fault exclusions in `../notes/stage3_contract_review.md`.

## Rebuild the PDF

From the workspace root, with ReportLab and pypdf available:

```bash
python research-next/artifacts/build_pdf.py research-next/artifacts/RECUT_04_Complete_Prototype_and_Validation.md
```

This rebuilds the fourth report only. Render and visually inspect every page after rebuilding, including tables and page transitions. Font fallback may alter pagination. Earlier report PDFs preserve their original chronology and must not be overwritten or described as stage-3 evidence.

All code, reviews and writing here are AI-assisted. Internal agent checks are not human peer review. Human authors and the supervisor must verify the scientific claims and applicable authorship/AI-use requirements before any submission.

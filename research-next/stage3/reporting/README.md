# Post-measurement reporting checks

These tools are outside the frozen stage-3 experiment source set. They do not change the experiment, raw evidence, original `summary.json`, or frozen integrity gates.

The frozen analyzer displayed refused-window seal counters as zero because those counters reside in `rejection.controller_metadata`, not top-level `controller_metadata`. `correct_refusal_counters.py` writes `summary_corrected.json` and `analysis_corrections.json` beside each original summary. The sidecar binds the raw/original/corrected hashes and lists every changed JSON leaf. Only refused-window counter medians and their grouped counter distributions change; timing, correctness and performance ratios do not.

Run from the project root:

```bash
research-next/.venv/bin/python -m pytest -q research-next/stage3/reporting/test_reporting.py research-next/stage3/reporting/test_report_tables.py
research-next/.venv/bin/python research-next/stage3/reporting/correct_refusal_counters.py RUN_DIRECTORY
research-next/.venv/bin/python research-next/notes/independent_stage3_report.py RUN_DIRECTORY --output NEW_REPORT.json --compare-summary --summary-name summary_corrected.json
```

Both command-line tools accept multiple run directories. Correction artifacts and independent reports refuse overwriting; use a new output path for another report. The independent calculation reads raw sessions first and only then compares the selected summary. Selecting a corrected summary also requires its sidecar and original/raw source hashes to match.

The regression tests use synthetic scalar records in temporary directories. They need pytest, but no model weights, CUDA, HPC, or existing study results. They cover input immutability, allowed-only correction paths, stale/tampered inputs, unexpected refusal causes, overwrite refusal, correction binding, the ratio-of-case-medians hierarchy, treatment consistency, and the conditional-window break-even formula.

Timing repetitions are summarized within each case before ratios are formed. Distributions remain separated by model, prefix length and window size. These checks do not establish independent fault samples, physical fault rates, publication novelty, or production serving performance.

## Final artifact pipeline

`report_tables.py` consumes only the three completed main runs, their full HPC/local audits, corrected summaries and `notes/stage3_main_recomputed.json`. `interpretation.json` is manually reviewed prose bound to the final table hash. `compose_report.py` checks those inputs, runtime records, regression gates and the separately audited selected diagnostic before assembling the manuscript. No selected diagnostic timings enter the tables.

The fallback configuration, verifier, submission script and amendment are separately frozen in `fallback_protocol_freeze.json` before that diagnostic job. They remain unchanged after execution. The ordinal clarification in `notes/stage3_fallback_diagnostic_clarification.md` corrects ambiguous prose, not any source or acceptance criterion.

`package_stage3.py` requires a PDF with the matching source hash and a completed all-page visual review. It writes a new manifest and ZIP, verifies every archived member's hash and CRC, then records the whole-archive hash in `package_summary.json`. It refuses overwriting completed delivery outputs. See `../DELIVERY.md` for reproduction and audit commands; do not rerun the original named Slurm jobs into existing evidence directories.

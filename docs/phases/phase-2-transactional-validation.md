# Phase 2 — Transactional recovery and expanded validation

[Repository overview](../../README.md) · [Phase 1](phase-1-pilot.md) · [Release](https://github.com/nbaliyan260/Thesis_P1/releases/tag/phase-2-validated-prototype)

Phase 2 contains **two internal engineering stages**: Stage 2 integrates guarded multi-window recovery; Stage 3 expands validation and complete-path cost accounting. Both remain separately named on disk and separately reported. The bounded implementation/evaluation milestone is complete; publication readiness and deployment usefulness are not established.

Start with [RECUT 04: Complete Prototype and Validation (PDF)](../../output/pdf/RECUT_04_Complete_Prototype_and_Validation.pdf), its [Markdown source](../../research-next/artifacts/RECUT_04_Complete_Prototype_and_Validation.md), and the [Stage 3 delivery guide](../../research-next/stage3/DELIVERY.md).

## Stage 2: guarded consecutive windows

The controller preserves previously committed state, keeps speculative output private, detects changed old-prefix layers without receiving injection coordinates, and selects an eligible saved cut. If corrected token history diverges, all subsequent steps replay from layer zero. A damaged journal triggers full replay; a damaged checkpoint refuses the window without releasing more tokens and poisons the session.

The main matrix used SmolLM2-135M and Qwen2.5-0.5B, prefixes 32/128, windows 8/16, one repeated prompt, one fault seed, six scenarios, two policies and two timing repetitions.

| Stage 2 main check | Result |
| --- | ---: |
| Sessions / workloads | 192 / 8 |
| Attempted windows | 544 |
| Exact-reference commits | 512 |
| Specified checkpoint refusals | 32 |
| Recovered-window executions | 192 |
| Checked committed tokens / native parity positions | 6,144 / 944 |
| Sparse partial-to-full executions | 10 |
| Independent record audit | 85,379 checks on HPC and again locally |

The 107-test software gate passed locally and on the HPC. The separate 24-session debug run had 64 commits and four refusals; it is excluded from main totals. The ten sparse partial-to-full executions are repeated configured cases, not ten independent faults.

Stage 2 compared complete guarded windows, but did **not** measure ordinary unprotected inference as a cost baseline. Its recorded native comparisons are live verdicts and hashes, **not full persisted recovered/golden tensors**. See [Stage 2 results](../../research-next/stage2/RESULTS.md), [execution record](../../research-next/stage2/EXECUTION.md), [source and protocol](../../research-next/stage2/) and [main raw records](../../research-next/runs/stage2-sessions-v1/).

## Stage 3: broader validation and honest protection costs

Stage 3 retains the transaction contract and adds byte-preserving batched sealing, legacy/batched ablations, native and bare clean baselines, a denser all-cuts comparator, a pinned Qwen2.5-7B model, three prompts, two seeds, initial prefixes 128/2,048, windows 8/16, K/V and layer fault strata, output-availability measurements and predeclared full-state archives.

| Stage 3 main check | Result |
| --- | ---: |
| Models / model-workload combinations | 3 / 18 |
| Measured sessions | 2,358 |
| Attempted windows | 7,020 |
| Exact-checked commits / specified refusals | 6,966 / 54 |
| Recovered-window executions | 2,358 |
| Native/custom post-prefill parity positions | 684 |
| Main archives / checked tensor pairs | 21 / 2,338 |
| Local independent audit checks | 1,018,993 |
| Compared independent report quantities | 81,309 |

Warmups (216 sessions), synchronized profiles (72 sessions) and separate diagnostics are excluded. Commit totals include clean native/bare baselines; recovery totals include guard challenges. None of these execution counts is a count of independent real-world faults.

### What improved—and what remains costly

For late K/V faults, complete affected-window full/sparse ratios range from approximately **1.38–1.53×** across the six model/shape strata. Early-layer faults have no consistent benefit. Across the equal-case eight-stratum mixture, full/sparse initialization-inclusive session ratios are about 1.09–1.13×. These comparisons are against equally protected full replay.

Clean sparse protection costs approximately **5–9% over the bare engine at prefix 128** and **14–49% at prefix 2,048**. The long-shape 7B case costs approximately 50% more than native inference. Outputs remain held until commitment. These costs prevent presenting selective recovery as a general inference speedup.

Ratios first reduce timing repetitions within each case, then compare matched case medians. Prefix and window length change together; fault strata also couple layer, K/V, severity and injection time. The synthetic batch-one eager study is not optimized production serving, semantic accuracy or a field-fault incidence measurement.

### Divergence and the separate post-hoc replay

The main study observed 12 divergent replay-window executions and four partial-to-full executions, all in Smol. The four partial-to-full executions arise from one configured case, two policies and two repetitions. Main Qwen runs had no token divergence. Debug and forced regression tests provide additional branch evidence, not additional main fault samples.

That Smol case lay outside the predeclared main archive subset. A declared post-hoc validation-only replay produced 13 sessions, 33 commits, three expected refusals and three archives containing 366 checked tensor pairs. It is stored under `runs/stage3-fallback-diagnostic-v1/` and excluded from every main timing/count above. Its purpose is additional state validation of the selected existing case, not increased primary sample size or new novelty evidence.

## Complete evidence bundle

Download [RECUT_Stage3_Validated_Artifact.zip](https://github.com/nbaliyan260/Thesis_P1/releases/download/phase-2-validated-prototype/RECUT_Stage3_Validated_Artifact.zip) from the [Phase 2 release](https://github.com/nbaliyan260/Thesis_P1/releases/tag/phase-2-validated-prototype).

- Exact size: **990,799,407 bytes**.
- SHA256: `535241cdc8e12f1a1a850ef1efc763b004e8975c235d2624a0c3e2003457e253`.
- Includes complete Stage 3 main/debug/selected-diagnostic tensor archives, raw records, frozen dependencies, audits, report and delivery provenance.
- Does not include pretrained weights, environments, credentials or every earlier-stage result. Earlier material and Stage 2 records remain in the repository.

Stage 3 `.pt` files are intentionally absent from Git; some exceed ordinary GitHub file limits. JSON/JSONL records and audit reports remain browsable in the repository. A clone alone is sufficient for source inspection and record integrity checks, but **not** for the full Stage 3 snapshot audit. Preserve filenames and bytes when downloading evidence; do not replace unavailable tensors with fabricated placeholders.

## Verification and tests

From the repository root, with the Phase 1 commit available locally:

```bash
python3 scripts/verify_phases.py
python3 scripts/verify_phases.py --artifact /absolute/path/RECUT_Stage3_Validated_Artifact.zip
```

The combined verifier checks historical Phase 1 and current Phase 2 bindings; the artifact option also checks the release ZIP. These are integrity checks, not model reruns.

For the full CPU evidence audit, extract the verified ZIP into a new separate directory and follow its `research-next/stage3/DELIVERY.md`. From the **extracted** `research-next` directory, with PyTorch and NumPy available, representative main commands are:

```bash
python stage3/audit_study.py runs/stage3-main-v1/smol135m --output runs/stage3-main-v1/smol135m/audit_reader.json
python stage3/audit_study.py runs/stage3-main-v1/qwen05b --output runs/stage3-main-v1/qwen05b/audit_reader.json
python stage3/audit_study.py runs/stage3-main-v1/qwen7b --output runs/stage3-main-v1/qwen7b/audit_reader.json
```

Use fresh output names if these exist. The delivery guide also covers debug and selected-diagnostic audits, independent raw-result recomputation, immutable model revisions and new-run provenance. CPU snapshot checks load tensors/basic data with `weights_only=True`; they do not require model weights or a GPU, and they do not independently regenerate every unarchived model execution.

From the checkout's `research-next` directory, the offline software tests are:

```bash
python -m pytest -q stage3/test_stage3_controller.py stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
python -m pytest -q stage3/reporting/test_reporting.py stage3/reporting/test_report_tables.py
```

Use an isolated environment with the documented dependency versions. CUDA-only skips are not GPU validation. The recorded final local combined suite passed 296 tests and skipped 17 CUDA-only tests, including 27 reporting tests; the 286-test core/controller suite passed on each allocated debug GPU.

Never overwrite frozen runs or supplied audit reports. GPU reexecution belongs on an authorized allocated compute node, not an HPC login node. Localize account-specific paths into new manifests/freezes and clearly label changed environments.

## Preserved reporting correction and historical statements

The frozen Stage 3 analyzer displayed zero seal counters for refused windows because those counters reside inside rejection metadata. A separate reporting-only tool produced `summary_corrected.json` and `analysis_corrections.json`. Original summaries and every raw record remain intact; no measured timing, correctness outcome or headline ratio was changed. Independent recomputation uses the corrected summaries only for comparison after calculating from raw records.

Historical reports state that no GitHub update occurred during their experiment stages. Those statements remain true for the recorded work and are preserved. This subsequent repository publication is recorded separately in the [changelog](../../CHANGELOG.md).

## Research judgment

This is a successful bounded engineering/evaluation artifact under a narrow trusted-checkpoint fault contract. It is not general hardware resilience, a production service, a proof of minimum recomputation or an established A/A* contribution. Publication-strength work still needs a credible deployment setting, stronger adapted published-method comparisons, optimized-serving evidence and a general insight beyond established reuse/rollback mechanisms. All implementation and review are AI-assisted and require human scientific review.

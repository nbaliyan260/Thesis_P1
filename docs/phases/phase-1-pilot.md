# Phase 1 — Initial RECUT pilot

[Repository overview](../../README.md) · [Phase 2](phase-2-transactional-validation.md)

Phase 1 is the original isolated-recovery feasibility study and its explicitly labeled supplements. Its historical status is **bounded pilot complete**, not a completed thesis or publication-ready system. Phase 2 extends the engineering and evaluation; it does not retroactively change the Phase 1 measurements.

## Immutable historical version

- Tag: [`phase-1-pilot`](https://github.com/nbaliyan260/Thesis_P1/tree/phase-1-pilot).
- Commit: [`c542f51d4a8711da4d14628f47f81e998afdba3d`](https://github.com/nbaliyan260/Thesis_P1/tree/c542f51d4a8711da4d14628f47f81e998afdba3d).
- Original file-hash manifest: [`research-next/artifacts/artifact_manifest.json`](../../research-next/artifacts/artifact_manifest.json), preserved unchanged.

The current repository contains later documentation and shared PDF-builder improvements. Consequently, the historical manifest should be checked against Phase 1 history, not silently rewritten to match the current tree. The combined-phase verifier does this explicitly.

One pre-existing provenance limitation is preserved: the skipped `boundary-selection-v1/metadata.json` records the earlier selector's `capture_boundary_case.py` hash (`e6c3e4f0…`), whereas the saved pilot commit contains the later extension-capable script (`23735533…`). Artifact integrity verification checks archived bytes, not every historical runtime/source binding. The skipped selector is not a positive experiment or added validation result.

## What was implemented and measured

The pilot tests whether retained layer-input activations can reduce repeated computation after a localized KV-cache fault. It compares complete layer replay with selective replay, checks restored state against clean native execution, and explores the history-divergence boundary.

The primary study used pinned SmolLM2-135M and Qwen2.5-0.5B models: 96 actually changed finite perturbations and 12 no-op controls per model, totaling **192 perturbations and 24 controls**. Full, sparse and all-layer recovery each passed the declared checks on all cases. These are not additional independent samples for each recovery policy.

The expanded software suite passed 76 tests locally and on the HPC. A stronger-baseline supplement reused the same 192 faulty cases and measured median paired **recovery-only** full/sparse ratios of 1.29× for SmolLM2 and 1.30× for Qwen. The primary runner uses oracle fault localization; a separate restricted old-prefix comparator localized the changed prefix cases against trusted checkpoints. It was not a general detector integrated into a production service.

Two primary faulty trajectories diverged within the measured token window, both at the final transition. A separately declared, selected W14 diagnostic exercised later full-layer replay after divergence. It is not another independent fault sample or a performance result. Its amendment and audit are part of the historical evidence.

## Documents and evidence

| Material | Purpose |
| --- | --- |
| [Report 01: idea and novelty (PDF)](../../output/pdf/RECUT_01_Idea_and_Novelty.pdf) | Original prospective proposal; later findings supersede optimistic novelty assessments. |
| [Report 02: implementation plan (PDF)](../../output/pdf/RECUT_02_Implementation_Plan.pdf) | Original pre-implementation scope and tests. |
| [Report 03: professor draft and results (PDF)](../../output/pdf/RECUT_03_Professor_Draft_and_Results.pdf) | Final Phase 1 interpretation, measured results and limitations. |
| [`experiments/`](../../research-next/experiments/) | Original engine, runners, protocols, tests and audit code; also dependencies of later stages. |
| [`runs/primary-v1/`](../../research-next/runs/primary-v1/) | Primary isolated-recovery evidence. |
| [`runs/sensitivity-v1/`](../../research-next/runs/sensitivity-v1/) | Stronger-baseline and restricted-detector supplement, reusing primary faults. |
| [`runs/boundary-selection-v1/`](../../research-next/runs/boundary-selection-v1/) and [`boundary-extension-v1/`](../../research-next/runs/boundary-extension-v1/) | Preserved selector outcome and separately selected branch diagnostic. |
| [Final claims review](../../research-next/notes/final_claims_review.md) | Historical scope and claim checks. |

The [100-paper review](../../research-next/literature/100_Paper_Review.md) records screening depth. It is not a claim that every paper was read and independently verified cover to cover.

## Verify or reproduce Phase 1

At the current checkout, use:

```bash
python3 scripts/verify_phases.py
```

To inspect or verify the exact historical tree without replacing the current checkout, create a separate worktree from a clone containing the original commit:

```bash
git worktree add --detach ../RECUT-phase-1-pilot c542f51d4a8711da4d14628f47f81e998afdba3d
cd ../RECUT-phase-1-pilot
python3 scripts/verify_evidence.py
```

Choose a new destination if that directory already exists. The original [reproduction guide at the immutable commit](https://github.com/nbaliyan260/Thesis_P1/blob/c542f51d4a8711da4d14628f47f81e998afdba3d/research-next/README.md) documents pinned weights, runtime, fresh output directories and audit commands. Verifying recorded hashes does not re-execute the models or prove scientific correctness.

## Why there is a second phase

Phase 1's recovery-only timings, compact models and isolated trials did not establish the cost of a complete protected decoding path. [Phase 2](phase-2-transactional-validation.md) adds detector-routed consecutive transactions, guarded commit/refusal, clean unprotected cost references, a 7B model, longer shapes and independently reloadable full-state evidence.

Do not compare headline ratios across phases as a longitudinal speedup: workloads, fault populations and timing boundaries differ. Neither phase establishes general hardware resilience, production benefit, unique novelty or A/A* readiness.

# RECUT: recovering an LLM with less repeated work

RECUT is an implemented research prototype for recovering an LLM's key/value (KV) cache after a supported memory corruption. It saves selected intermediate activations and repeats upper-layer computation only while the corrected token history permits reuse. The latest controller keeps tokens private until a complete inference window is accepted.

**Current status: the bounded prototype and three-model validation are complete. This is not a completed thesis, production system or A/A*-ready paper.** Selective replay saves work in some fault cases, but protection adds normal-operation cost and output delay. Novelty and a useful deployment setting remain research questions.

## Two phases, preserved separately

| Phase | What it contains | Start here |
| --- | --- | --- |
| **Phase 1 — Initial pilot** | Isolated recovery trials, stronger replay baselines, restricted detection supplement and a selected divergence diagnostic; reports 01–03. | [Phase 1 guide](docs/phases/phase-1-pilot.md) · [Historical tag](https://github.com/nbaliyan260/Thesis_P1/tree/phase-1-pilot) |
| **Phase 2 — Transactional recovery and expanded validation** | Internal Stage 2 adds consecutive guarded windows; internal Stage 3 adds larger-model coverage, clean cost baselines, broader fault strata and full-state evidence; report 04. | [Phase 2 guide](docs/phases/phase-2-transactional-validation.md) · [Latest report PDF](output/pdf/RECUT_04_Complete_Prototype_and_Validation.pdf) |

The two public-facing phases are navigation labels, not renamed experiments. Original `experiments/`, `stage2/` and `stage3/` paths remain intact because imports and evidence manifests depend on them. Results from different stages are not pooled.

## Latest result: useful recovery, substantial protection cost

The expanded Stage 3 study used pinned SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-7B base models on allocated NVIDIA RTX 5000 Ada GPUs. It used BF16, eager attention, one sequence, three synthetic prompts, initial prefixes of 128/2,048 tokens, windows of 8/16 steps respectively, and three consecutive windows per session.

| Main Stage 3 evidence | Result and interpretation |
| --- | --- |
| Measured sessions | 2,358; separate warmups, profiles and diagnostics excluded |
| Window outcomes | 6,966 exact-checked commits and 54 specified checkpoint-integrity refusals |
| Late-K/V affected-window recovery | Full/sparse time ratios of approximately **1.38–1.53×**, relative to equally protected full replay—not ordinary inference |
| Clean sparse cost versus the bare engine | Approximately **5–9% extra** at prefix 128 and **14–49% extra** at prefix 2,048, including session initialization |
| Long-shape Qwen-7B cost versus native inference | Approximately **50% extra**; recovery savings do not eliminate shared protection cost |
| Main state archives | 21 archives; 2,338 recovered/reference tensor pairs passed numeric and bytewise CPU checks |

Early-layer faults had no consistent selective-replay benefit. Protected tokens are held until window commitment. Prefix and window length change together between shapes; these are not isolated context-scaling or production-throughput measurements. Timing repetitions, policies and repeated-window executions are not independent field-fault samples.

Read the [complete measured report](research-next/stage3/RESULTS.md) for timing boundaries, matched-case aggregation, memory costs, divergence evidence, the separately selected post-hoc fallback replay, prior work and publication limitations.

## Read or download

- [Latest professor-facing report: RECUT 04 (PDF)](output/pdf/RECUT_04_Complete_Prototype_and_Validation.pdf) and [Markdown source](research-next/artifacts/RECUT_04_Complete_Prototype_and_Validation.md).
- [Phase 2 release](https://github.com/nbaliyan260/Thesis_P1/releases/tag/phase-2-validated-prototype), including the [complete Stage 3 evidence ZIP](https://github.com/nbaliyan260/Thesis_P1/releases/download/phase-2-validated-prototype/RECUT_Stage3_Validated_Artifact.zip).
- [Stage 3 delivery and reproduction guide](research-next/stage3/DELIVERY.md), [prespecified plan](research-next/stage3/PLAN.md) and [execution record](research-next/stage3/EXECUTION.md).
- [Stage 2 results](research-next/stage2/RESULTS.md) and [original pilot report](output/pdf/RECUT_03_Professor_Draft_and_Results.pdf).
- [100-paper review](research-next/literature/100_Paper_Review.md) and [reading-depth statement](research-next/notes/literature_coverage.md): 100 papers screened, including 18 focused deeper reviews—not 100 cover-to-cover verifications.

### Repository records versus full-state bundle

The repository contains the source, PDFs, raw JSON/JSONL measurements, audit reports, provenance, Stage 2 records and reporting tools. The new Stage 3 `.pt` snapshot directories are supplied in the release ZIP, **not in the Git checkout**. Existing small Phase 1 snapshots remain in the repository. A checkout-only record check must not be described as a Stage 3 tensor reload.

The Stage 3 ZIP is exactly **990,799,407 bytes**. Its SHA256 is:

```text
535241cdc8e12f1a1a850ef1efc763b004e8975c235d2624a0c3e2003457e253
```

Extract it into a **new, separate directory**, not over this checkout. It contains the complete Stage 3 evidence and dependencies for the audit commands in its `research-next/stage3/DELIVERY.md`. It is not an archive of every earlier RECUT stage; this repository retains the earlier history and Stage 2 records. Model weights, environments, authentication material and third-party paper PDFs are not included.

## Verify evidence

From a normal Git clone with Phase 1 history available:

```bash
python3 scripts/verify_phases.py
python3 scripts/verify_phases.py --artifact /absolute/path/RECUT_Stage3_Validated_Artifact.zip
```

The first command checks Phase 1 against its immutable Git history and Phase 2 repository files against their recorded bindings. The second also verifies the downloaded full-state bundle. It does not perform model execution or replace the independent scientific audits. See [Phase 2 verification instructions](docs/phases/phase-2-transactional-validation.md#verification-and-tests) for tensor audits and tests.

The original `scripts/verify_evidence.py` belongs to the **Phase 1 tag**. Its historical manifest is preserved byte-for-byte. Later README and PDF-builder changes are legitimate Phase 2 changes, so the old verifier is not the current combined-phase entry point. Do not regenerate an old manifest to make a modified historical artifact appear unchanged.

## Repository layout

```text
docs/phases/                   Phase 1 and Phase 2 navigation and scope
research-next/
  experiments/                 Original engine, pilot protocol and dependencies
  stage2/                      Guarded multi-window controller and its study
  stage3/                      Expanded validation, audits and delivery guide
  runs/                        Separate per-stage raw records and audit reports
  notes/                       Contracts, amendments, reviews and recomputation
  literature/                  Paper reviews and related-work ledgers
  artifacts/                   Report sources and builders
output/pdf/                    Reports 01–04, preserving their chronology
scripts/                       Repository/evidence verification helpers
CHANGELOG.md                   Phase mapping and publication history
```

## Scope and research integrity

The tested contract concerns persistent, finite, numerically unequal old-prefix KV changes during speculation after a trusted checkpoint is captured. Model weights, initial state, control/seal storage and guard/recovery/commit execution are trusted. Physical GPU faults, binary instrumentation, reverted or suffix-only changes, computation-origin faults, distributed serving and crash durability are outside the validated guarantee. This is inference-state recovery, not recovery of model training or a guarantee of truthful/safe model answers.

Prior mechanisms such as intermediate-state reconstruction and history-conditioned reuse limit the novelty claim. No adapted published-system baseline or representative optimized-serving evaluation establishes an A/A* contribution here. Successful audits establish bounded evidence consistency, not guaranteed correctness in every environment or conference acceptance.

All code, experiments, reviews and writing are AI-assisted. Internal separate-agent checks are not human peer review or external replication. Human authors and the supervisor must verify scientific claims, attribution and venue-specific AI-use requirements before submission. Historical account paths and job identifiers are retained as provenance; credentials are excluded.

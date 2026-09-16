# RECUT: selective replay after LLM KV-cache corruption

This repository contains an implemented, tested research prototype and its evidence package. RECUT studies how to recover an LLM's persistent key/value (KV) state after a delayed, localized finite corruption, using selective layer replay instead of always replaying every layer.

**Status: bounded pilot complete; not a completed thesis or A/A*-ready paper.** The final assessment advises against committing to this direction yet. Mechanism-level overlap with [LayerSkip](https://aclanthology.org/2024.acl-long.681/) and other prior work narrows the novelty claim, and realistic end-to-end serving benefit remains unproven. Successful prototype tests do not establish publication readiness.

## Start here

- [Final professor-facing draft and results (17-page PDF)](output/pdf/RECUT_03_Professor_Draft_and_Results.pdf): explanation, related work, methods, measurements, limitations and the final publication assessment.
- [Original idea and novelty proposal (PDF)](output/pdf/RECUT_01_Idea_and_Novelty.pdf) and [pre-implementation plan (PDF)](output/pdf/RECUT_02_Implementation_Plan.pdf): preserved prospective documents. **The final report supersedes their earlier novelty assessment.**
- [Implementation and reproduction guide](research-next/README.md).
- [100-paper review](research-next/literature/100_Paper_Review.md) and [coverage statement](research-next/notes/literature_coverage.md): 100 papers screened, including 18 focused deeper reviews, not 100 cover-to-cover verifications.
- [Final numerical review](research-next/notes/final_filled_draft_review.md) and [PDF quality review](research-next/artifacts/QA_REVIEW.md).

## Measured results

The primary experiment used SmolLM2-135M and Qwen2.5-0.5B, with 96 actually changed finite perturbations and 12 no-op controls per model: **192 perturbations and 24 controls in total**.

| Check | Result | Scope |
| --- | --- | --- |
| Expanded software tests | 76 passed locally and on the HPC | Unit/integration checks, not a deployment guarantee |
| Full, sparse and all-layer recovery | Each passed 192/192 perturbed cases and 24/24 controls | Declared joint cache/logit/token/short-continuation checks |
| Sparse replay vs stronger checkpoint-alias full replay | Median paired recovery-only speedup: 1.29x (SmolLM2), 1.30x (Qwen) | Supplemental measurements of the same 192 faults, not new independent samples |
| Restricted old-prefix comparator | Localized 192/192 changed prefix cases; no alarms on 24 controls | Trusted-checkpoint comparison, not a general fault detector |

Only two primary faulty trajectories diverged within the measured token window, both at its final transition. Five cases differed during the checked two-step continuation, including those two. A separately declared, selected W14 diagnostic exercised later full-layer replay after divergence; it is not another independent fault sample or a performance result.

Results come from compact models, BF16 eager execution, batch size one and short contexts. Recovery timings exclude the separate detector experiment and do not establish overall inference-throughput gains. Physical hardware faults, NVBit instrumentation, production serving, general fault detection and checkpoint certification are outside this pilot's validated scope. Primary localization is an oracle. See the final report for complete assumptions and blind spots.

## Repository layout

```text
research-next/
  experiments/    Frozen protocol, engine, runners, tests and independent audit tools
  runs/           Primary, supplemental and selected diagnostic records/snapshots
  logs/           HPC job logs and local test evidence
  literature/     Per-paper review ledgers and related-work notes
  notes/          Contracts, amendments, execution history and limitations
  artifacts/      Report sources, builders, figures and original package manifest
output/pdf/       Three final PDFs
Research Papers PDFs/download_manifest.csv
                  Original paper-source manifest; not copies of the 100 papers
scripts/          Standard-library evidence-integrity checker
```

The `.pt` files are intentionally included small, generated audit snapshots, not pretrained model weights. Model weights, Python environments, SSH sockets, credentials and rendered-page intermediates are excluded. Some historical logs and manifests retain original machine paths and job identifiers as provenance; adapt paths rather than treating those as portable instructions.

## Verify and reproduce

From the repository root, validate every file covered by the original research-package manifest:

```bash
python3 scripts/verify_evidence.py
```

For software tests, first create a dedicated environment with the versions described in the [reproduction guide](research-next/README.md), then run:

```bash
cd research-next
python -m pytest -q experiments/test_recut_engine.py experiments/tests_independent.py
```

The guide provides pinned-model download, debug, primary, supplemental and audit commands. Run GPU experiments only on an authorized allocated compute node, not an HPC login node. Never overwrite a frozen run to obtain a better result. The stored manifests bind the original evidence; separate reruns should use new directories and appropriate new provenance. Original absolute Slurm paths require adaptation.

The manifest validates the original research artifacts, not this repository's added landing page, ignore rules or verification helper. Byte hashes establish integrity, not scientific correctness. Independent audit code, separate-agent reviews and visual checks are not external replication or human peer review.

## Research integrity

Code and reports were AI-assisted. Human authors must verify scientific claims, attribution, references and venue-specific AI-use requirements before submission. No claim of guaranteed novelty, acceptance or general hardware fault resilience is made. No third-party paper PDFs or pretrained model weights are redistributed here.

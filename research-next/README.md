# RECUT research package

A bounded study of exact elementwise recovery after delayed, localized finite KV-cache perturbations. This is not a completed thesis, deployed detector, physical hardware-fault study, novelty certificate or A/A*-ready paper.

## Latest completed stage: protection cost and broader validation

Start with `stage3/DELIVERY.md`, `stage3/RESULTS.md`, and `../output/pdf/RECUT_04_Complete_Prototype_and_Validation.pdf`. The expanded controller and bounded HPC study cover pinned 135M, 0.5B and 7B models, longer contexts, sequential private-output windows, broader old-prefix K/V faults, integrity refusals, native/bare cost baselines, and canonical-hash-preserving batched transfers. The new portable artifact is `../output/RECUT_Stage3_Validated_Artifact.zip`.

All 2,358 main sessions completed: 6,966 checked commits and 54 specified refusals. All three main HPC/local audits passed; the local audits account for 1,018,993 checks and 2,338 exact/bytewise-equal tensor pairs across 21 predeclared archives. A separately declared selected-case fallback replay is not pooled into those totals. Final local tests: 296 passed, 17 CUDA-only skipped; the allocated core/controller suite passed all 286 cases on each GPU task. See the report for denominators, reference-baseline contributions and limitations; these counts are not independent field-fault samples.

**Current judgment:** this is a completed expanded prototype/evaluation artifact, not a publication-readiness certificate. Recovery benefits must be weighed against normal-operation cost, buffered-output delay, the restricted fault/trust contract and substantial overlap with earlier reuse/rollback mechanisms. Supervisor review and a stronger research contribution remain necessary for an A/A* submission. Earlier stages below retain their own chronology and are not combined with stage-3 measurements.

**Earlier pilot decision (retained):** do not commit to RECUT as an A/A*-targeted thesis on the pilot evidence. The late LayerSkip comparison further narrows the novelty opening: history-conditioned intermediate-state reuse and rollback are not new general principles. The pilot results draft supersedes the earlier prospective novelty assessment; those first two PDFs remain unchanged to preserve the research chronology.

## Earlier pilot reading order

1. `../output/pdf/RECUT_01_Idea_and_Novelty.pdf`: pre-implementation question, professor fit and prior-art constraints.
2. `../output/pdf/RECUT_02_Implementation_Plan.pdf`: prospective protocol and go/no-go criteria.
3. `../output/pdf/RECUT_03_Professor_Draft_and_Results.pdf`: final professor-facing draft and measured results, generated after the primary audit.
4. `literature/100_Paper_Review.md`: all 100 papers, individually screened with honest depth labels. Eighteen focused deeper reviews; 82 selected-section screens. Not 100 cover-to-cover verifications.
5. `notes/literature_coverage.md`: exact coverage, metadata warnings and mapping of all 30 requested sites plus SPAR/MATS.

The package also preserves the unchanged supplied collection's `../Research Papers PDFs/download_manifest.csv` for source-URL resolution; it does not bundle another copy of the 100 original PDFs. The combined readable review includes per-paper source links, including the 33 safety-ledger entries whose original JSON schema lacks a URL field.

## Evidence and scope

The bounded pilot is complete: the expanded 76-test suite passed locally and on the HPC, all three valid recovery methods passed all 216 primary scenarios, and the stronger-baseline supplement reused all 192 perturbed cases. Sparse replay's median paired recovery-only speedup over checkpoint-alias full replay was 1.29x for SmolLM2-135M and 1.30x for Qwen2.5-0.5B. These are not end-to-end serving throughput gains. A separately labeled, selected W14 diagnostic covered the missing post-divergence replay branch; it is not another independent fault sample. Both-runtime audits passed. No HPC jobs remained at the final check, and the SSH connection was closed.

`runs/debug-v1/` is diagnostic and excluded from primary claims. `runs/primary-v1/` is the frozen main matrix: 108 scenarios per model, including 96 perturbations and 12 no-op controls. The models are SmolLM2-135M and Qwen2.5-0.5B with immutable revisions in `experiments/model_manifest.json`.

The primary runner uses actual Hugging Face Llama/Qwen2 layers, one unpadded sequence, BF16 eager single-token operations, deterministic greedy decoding, ordinary DynamicCache, trusted clean checkpoints and oracle fault localization. Reference comparison is experimental validation outside recovery timing, not an implemented online verifier. Private output buffering is modeled, not a network service. No hardware fault induction, NVBit, exploit reproduction or AWS-cluster changes.

The separate post-hoc `runs/sensitivity-v1/` study reuses all 192 perturbed pairs with five recovery methods, including two stronger copy-avoiding full-replay baselines. It also tests exact old-prefix comparison on all 216 primary pairs and routes repair using the unique detected layer. That detector cannot see suffix-only, restored-transient, compute-origin, weight, journal or corrupted-checkpoint faults. This is a narrow feasibility check, not a general fault detector or 192 additional independent samples. See the two amendment notes for the fixed supplemental scope.

Prior RAVEN and ProofOps files are preserved and are not evidence for RECUT.

The frozen primary has only two within-window token divergences, both at its final transition. `runs/boundary-selection-v1/` preserves the strict selector's honest skipped result: no original case exercised a later repair step after divergence with a nonzero cut. A separately declared one-case `runs/boundary-extension-v1/` diagnostic extends the same Qwen primary-063 fault from W=12 to W=14. It is selected post hoc, not another independent fault, and none of its timings enter performance claims. Its amendment, captured full-state archive and separate CPU audit must accompany any branch-coverage statement; continuation tensors are not archived.

## Local correctness tests

Use an isolated environment with PyTorch and Transformers 4.51.3. The Mac development environment used Torch 2.8; the HPC experiment used Torch 2.6.0+cu124. Tests are not themselves GPU performance results.

```bash
python -m pytest -q experiments/test_recut_engine.py experiments/tests_independent.py
```

Explicitly name the independent test file: its filename is not matched by every default pytest discovery pattern. The original primary gate had 58 tests. The final expanded suite has 76, adding alias/view baseline checks, source immutability, no-alarm fallback and explicit detector blind spots.

## Reproduce on an authorized allocated CUDA GPU

Run from this project directory, in a dedicated environment, not an HPC login node. The original environment is Python 3.13.9, Torch 2.6.0+cu124 and Transformers 4.51.3. `experiments/environment_freeze.txt` records transitive packages; `experiments/requirements.txt` contains direct pins. The GPU must support BF16. No account password is needed by the code. Run each command only after the previous command succeeds; a failed debug run is a stop condition, not permission to start the primary run. For strict dependency replay, use the complete freeze with the matching Python/CUDA platform; direct pins alone do not freeze every transitive dependency.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r experiments/requirements.txt
python experiments/fetch_pinned_models.py --manifest experiments/model_manifest.json --output experiments/model_manifest_local.json
python experiments/run_recut.py --manifest experiments/model_manifest_local.json --config experiments/config_debug.json --output runs/my-debug
python experiments/run_recut.py --manifest experiments/model_manifest_local.json --config experiments/config_primary.json --output runs/my-primary
python experiments/analyze_recut.py runs/my-primary
python experiments/audit_results.py runs/my-primary --config experiments/config_primary.json --manifest experiments/model_manifest_local.json --require-summary
```

The runner refuses to overwrite a nonempty run directory. Always use a new run name. `prepare_models.py` is the original discovery/setup script; **do not use it to reproduce exact revisions**, because it initially resolves public model HEAD. Use `fetch_pinned_models.py` above. Checkpoints from `audit_*.pt` are locally generated tensor data; audit loads use `weights_only=True` on CPU.

After the complete 216-case primary run and its audit succeed, the separately labeled sensitivity study can be reproduced with the same localized manifest. Do not continue if either command fails. These commands reuse primary scenarios; they do not create new independent fault evidence.

```bash
python experiments/run_optimized_replay.py --manifest experiments/model_manifest_local.json --config experiments/config_primary.json --primary-run runs/my-primary --output runs/my-sensitivity
python experiments/audit_supplement.py --run runs/my-sensitivity --primary runs/my-primary --config experiments/config_primary.json --manifest experiments/model_manifest_local.json
python experiments/frontier_analysis.py runs/my-primary
```

Fetching public weights needs network access unless the pinned revisions are already cached. Do not inherit `HF_HUB_OFFLINE=1` from the run script during a first download. The original Lustre paths in the supplied manifest are not portable; the pinned fetcher rewrites paths into a new manifest. Different GPUs, drivers or package versions may change floating-point behavior or timings even when the protocol is identical.

`run_hpc.sbatch` records the actual lab resource request and paths. Its absolute task-owned paths require adaptation on another account/cluster. Slurm job IDs and failures are in `notes/execution_log.md`; credentials are deliberately absent. Do not copy or reuse any authentication material from the conversation.

The optional targeted branch diagnostic applies only if the reproduced primary has the exact original primary-063 fault and observed trajectories. The original amendment binds the original raw digest and intentionally rejects unrelated reruns. Do not bypass that binding or silently choose another case; write a new, honestly labeled amendment if the primary evidence changes. The archived original can be independently tensor-audited with `audit_boundary_snapshot.py` and its declared manifest/config/amendment.

## Rebuild the reports

Report building additionally needs ReportLab, pypdf and Matplotlib. The scientific runs do not depend on the PDF libraries. From the workspace root, run the supplemental block builder and figure builder, then `compose_final.py --supplemental-prose research-next/artifacts/supplement_results.md --detector-prose research-next/artifacts/detector_results.md`, and finally `build_pdf.py research-next/artifacts/RECUT_03_Professor_Draft_and_Results.md`. These commands require completed, passed, hash-bound audits; they are not templates for fabricated results. The first two PDFs preserve the pre-implementation artifacts and should not be overwritten during result updates.

`artifacts/pdf_layout.py` is self-contained and does not import earlier RAVEN code or evidence. It uses system Arial/Courier New where available and built-in Helvetica/Courier fallbacks elsewhere. Font fallback and different ReportLab versions may change pagination; re-render and visually review every page after rebuilding. Generated block/figure provenance records bind the final report inputs to the audited supplemental summary.

## Inspect and extend

- `experiments/recut_engine.py`: native layer stepping and strict cache/journal operations.
- `experiments/run_recut.py`: fixed trial harness, native parity gates, restoration, state/continuation checks and timing.
- `experiments/config_primary.json` and `experiments/protocol_freeze.json`: prospective matrix and hash. Do not overwrite to tune results.
- `experiments/analyze_recut.py`: descriptive paired statistics; no deployment failure-rate inference.
- `experiments/audit_results.py`: separate evidence/summary and tensor-snapshot audit, without importing runner or analyzer.
- `run_optimized_replay.py`, `prefix_detector.py` and `audit_supplement.py` (all under `experiments/`): the separately labeled stronger-baseline and restricted-detection study and its independent record audit. Supplemental timing rows contain no independently saved full-state tensors; do not confuse their recorded checks with the primary snapshot audit.
- `notes/recovery_contract.md`: proof sketch, failure boundaries and cost equation.
- `notes/independent_design_review.md` and `notes/publication_assessment.md`: skeptical review and limitations.

Model weights, environments and SSH sockets are excluded from the portable ZIP. Reports and code were AI-assisted; separate-agent checks are not human peer review or an external replication. Human authors must verify scientific claims, references and applicable AI-use rules before submission.

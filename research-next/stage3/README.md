# RECUT stage 3

This directory extends the frozen stage-2 controller and completes a bounded research artifact with stronger cost baselines, broader coverage, seal-transfer optimization and a replayable tensor-audit subset. It is not a general hardware fault detector, production server or publication-readiness certificate. See `PLAN.md` for the pre-measurement protocol; final outcomes belong in the later results report.

## Implementation

`recut_controller.py` preserves the public `TransactionalSession`/`WindowRejected` interface, adding `seal_mode="batched"|"legacy"` and diagnostic-only `profile=True`. Batched seals preserve the exact canonical SHA-256 stream of stage 2 while combining device-to-host transfers. Host seals and controller metadata remain trusted; temporary device staging increases peak memory. Full replay skips meaningless empty-journal hashes in both modes, so the legacy ablation isolates transfer batching rather than every stage-2-to-stage-3 change.

Only `committed_tokens` is the public transcript. Transaction internals exist for the controlled injection harness, not for use as released output. A refused/poisoned session needs a trusted restart outside this prototype. The code is sequential single-owner, with no network or thread-safety/crash-durability guarantee. A whole window's tokens are delayed until commitment.

## Files and provenance

- `run_study.py`: native batched prefill; exact post-prefill native/custom parity; measured, warmup and profiling streams; fixed faults and full-tensor snapshots.
- `analyze_study.py`: descriptive matched costs and sensitivity, not inferred deployment incident rates.
- `audit_study.py`: independent record/arithmetic/coverage checks and separate CPU tensor-archive comparisons.
- `test_stage3_controller.py`: canonical seal equivalence, mutations, controller regression and explicitly conditional CUDA checks.
- `config_study.json`, `config_debug.json`, `model_manifest.json`, `protocol_freeze.json`: immutable selected inputs and source hash binding.
- `fetch_models.py`, `setup_hpc.sbatch`, `run_hpc.sbatch`: pinned public model download and bounded allocated execution. Lab paths/usernames are configuration, not credentials.

The original engine and stage-2 files are imported/read as dependencies but never modified. All stages must remain together under the `research-next` directory. Do not substitute a newer Transformers or alternate attention kernel without a new protocol and native numerical equivalence gate.

## Reproduction

Run from `research-next` in an isolated Python environment. The study runtime uses Python 3.13.9, Torch 2.6.0+cu124, Transformers 4.51.3, NumPy 2.2.4 and pytest 8.3.5; the earlier complete dependency freeze remains in `experiments/environment_freeze.txt`. Local CPU development uses Python 3.9.6/Torch 2.8.0; it is not the performance environment. Public model revisions are fixed, but the supplied local paths are account-specific. Other accounts must create a localized manifest and clearly labeled new freeze, not overwrite the original provenance.

For a fresh authorized environment, install the pinned Torch wheel from the official CUDA 12.4 index, then the existing `experiments/requirements.txt`. Fetch model weights only with the fixed revisions. Do not treat public models' HEAD revisions as equivalent. Model weights are not bundled with this artifact.

```bash
python stage3/freeze_study.py verify
python -m pytest -q stage3/test_stage3_controller.py stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
python stage3/run_study.py --config stage3/config_debug.json --manifest stage3/model_manifest.json --model Qwen/Qwen2.5-0.5B --output runs/my-stage3-debug
python stage3/analyze_study.py runs/my-stage3-debug
python stage3/audit_study.py runs/my-stage3-debug --config stage3/config_debug.json --manifest stage3/model_manifest.json
python stage3/run_study.py --config stage3/config_study.json --manifest stage3/model_manifest.json --model Qwen/Qwen2.5-0.5B --output runs/my-stage3-main
python stage3/analyze_study.py runs/my-stage3-main
python stage3/audit_study.py runs/my-stage3-main --config stage3/config_study.json --manifest stage3/model_manifest.json
```

Each command requires the previous command to pass. Use an allocated BF16-capable CUDA GPU, not a login node. The exact three-model lab command is `bash stage3/submit_study.sh`, after pinned downloads, complete-source freeze and transfer verification. It submits separate debug and main arrays; the entire three-model debug array must succeed before any main task starts. It caps each model task at one GPU/four CPUs/48 GB host memory/two hours and allows at most three model tasks concurrently. It is not a recurring job. If a debug task fails, cancel the pending dependent main array before making a declared amendment. For CPU unit testing no weights/network are needed; the CUDA-only tests must be reported as skipped when no CUDA device exists.

Nonempty run paths and existing snapshot files are refused. Keep failed runs, source amendments and scheduler logs. Warmups/profiles are separate files. Plain/native decoding is clean-only and does not offer resilience equivalent to protected methods. Timing repetitions do not create independent fault samples. CPU snapshot audits use `torch.load(..., weights_only=True)` and verify saved actual/reference tensors; other rows retain live comparison verdicts, not independent full replay.

Window timing excludes initialization; its separately measured cost is also included in `total_with_initialization_seconds` for session comparisons. This prevents the protected constructor's validation from being hidden when comparing total sessions against the plain/native cost baselines. Neither boundary includes shared input forking, prefill or experimental verification.

All implementation and separate-agent checks are AI-assisted, not human peer review. No automatic external publication or GitHub push is part of this stage. Never store account passwords or access tokens in scripts, manifests, logs or artifacts.

# RECUT stage 2: guarded multi-window sessions

This is the next engineering stage after the original bounded RECUT pilot. It adds an implemented single-owner transaction controller, rather than just replaying a previously isolated fault case. The original `../experiments/` and stage-1 evidence remain unchanged. Read `PLAN.md` for the prespecified study; measured outcomes belong in `RESULTS.md` after the GPU run and audits complete.

## What the controller does

1. Begin a window with separate working and recovery copies of the last committed KV cache. Keep the previously committed public state unchanged.
2. Generate privately, saving a few intermediate-layer activations. No token is returned by `step()`.
3. Validate the recovery checkpoint, compare the old prefix at every layer, and choose a saved cut at or below the earliest changed layer. The controller is never told the injected fault location.
4. Replay only from that cut while token history remains valid. After the first corrected token differs, use full-layer replay for all later tokens, even if token IDs reconverge.
5. If saved activations are damaged, use complete replay. If the recovery checkpoint is damaged, reject the window and poison the session without changing committed output or state.
6. Commit a complete validated window, then begin the next one from that state.

This costs extra copies, integrity checks and delayed output. Reducing layer replay does not automatically reduce complete-window latency.

## Contract, not a general safety guarantee

The tested cache faults are finite, persistent old-prefix changes introduced **after checkpoint creation, during speculative execution**. They can affect several layers at different times. Initial state, weights, host control metadata/digest storage, guards, recovery and commit are trusted. Integrity seals detect later storage changes, not errors already present when state was captured. Faults between windows, transient changes restored before detection, suffix-only corruption and computation-origin errors are not covered. Negative-control tests document two of these blind spots.

This is not a concurrent inference service, crash-durable transaction, physical fault-injection study or production security boundary. `WindowTransaction` internals are exposed for the experiment harness only; callers must not treat them as released output. A failed session must be replaced using a trusted source; automatic trusted restart is not implemented.

## Files

- `controller.py`: transaction state, detection-driven routing, integrity guards and fallback.
- `test_transactional.py`: 31 additional offline tests, including explicit synthetic divergence/rollback tests and blind-spot controls.
- `run_sessions.py`: native Hugging Face references, fixed software injections, complete-window timing and memory records.
- `analyze_sessions.py`: descriptive paired counts/timings; no independent-fault statistical inference.
- `audit_sessions.py`: independently implemented record and arithmetic checks. Golden GPU tensors are compared live, not archived for full tensor re-execution.
- `config_sessions.json`, `config_debug.json`, `protocol_freeze.json`: main/preflight configurations and pre-measurement source hashes.
- `run_hpc.sbatch`: bounded single-GPU lab job; contains account-specific paths but no credentials.

## Run from the research-next directory

Use the original isolated pinned HPC environment (Python 3.13.9, Torch 2.6.0+cu124, Transformers 4.51.3), a BF16-capable allocated GPU, and the existing immutable offline model manifest. Do not run model experiments on an HPC login node. A different account must localize the manifest paths and explicitly create a new freeze/run record rather than pretending to match this account's original protocol.

```bash
python stage2/freeze_stage2.py verify
python -m pytest -q stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
python stage2/run_sessions.py --manifest experiments/model_manifest.json --config stage2/config_debug.json --output runs/new-stage2-debug
python stage2/analyze_sessions.py runs/new-stage2-debug
python stage2/audit_sessions.py runs/new-stage2-debug --config stage2/config_debug.json
python stage2/run_sessions.py --manifest experiments/model_manifest.json --config stage2/config_sessions.json --output runs/new-stage2-main
python stage2/analyze_sessions.py runs/new-stage2-main
python stage2/audit_sessions.py runs/new-stage2-main
```

Run each command only if the previous command passed. Nonempty output directories are refused. Debug evidence is separate from the main matrix; failures must be preserved. Raw measurements include artificial fault injection, separately timed. Subtracted timings are diagnostics, not deployment estimates. Native references, session initialization and prefill are outside the complete-window timing boundary.

The timing pilot has only two repetitions. Its excluded warmup is one clean window per model/policy at the first shape, not every shape/recovery branch. GPU allocation peaks include resident gold-reference fixtures; host hashing-buffer peaks are not measured. Full replay retains the same empty-journal bookkeeping interface, so it is a strong copy-avoiding baseline within this implementation, not the fastest theoretically possible implementation.

All code and separate-agent reviews are AI-assisted. This engineering extension alone does not demonstrate a new publishable research contribution or change the earlier A/A* readiness assessment.

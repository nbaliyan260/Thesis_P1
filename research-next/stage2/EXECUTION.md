# Stage-2 execution record

All times refer to 16 September 2026. This log concerns the new stage-2 integration only. Earlier RECUT pilot source and evidence are preserved.

## Before pretrained measurements

- Implemented detector-routed, private-output, multi-window transactions; native-reference session harness; descriptive analyzer; independently implemented standard-library record auditor.
- Local correctness gate: **107 tests passed** (31 stage-2 + 76 original), after fixing public-state aliasing at BEGIN, releasing discarded full-replay cache storage, and preparing transcript allocation before committing public fields. Local environment: Python 3.9.6, Torch 2.8.0, Transformers 4.51.3. One known urllib3/LibreSSL warning, no test failure in the final gate.
- An independent source reviewer requested a stricter checkpoint-refusal acceptance condition. The final harness and auditor require the exact checkpoint-integrity reason and the prescribed checkpoint injection, not an arbitrary exception.
- Separate-agent runner QA: all 12 synthetic tiny-Llama scenario/policy combinations matched native reference, plus timing-distribution arithmetic checks.
- Separate-agent auditor QA: 4,000 float32/BF16 addition/store comparisons, 12 tiny-model session-record checks, then a complete 24-session synthetic fixture with 7,672 record checks. Five deliberately corrupted-record controls were rejected. These are software tests, not pretrained measurements or external replication.
- Source/config/protocol frozen in `protocol_freeze.json` before any stage-2 pretrained GPU run; all 17 bound files verified after upload. Original engine, runner and detector hashes match the earlier pilot.

## HPC submission

Slurm job **195896**, partition `ws-ia`, node `ws-l1-014`: one GPU, four CPUs, 24 GB host-memory limit and 45-minute wall-time limit. The partition had idle nodes at submission. The existing authorized account/runtime and cached pinned weights were reused; no new environment or model download was required.

Ordered gates: freeze verification → 107 unit tests → 24-session pretrained diagnostic → diagnostic analysis/audit → 192-session main run → main analysis/audit → final freeze verification. Any failed command stops the job. Debug data do not enter the main measurements.

The task uses only normal user-space PyTorch software injection. No hardware fault induction, GPU binary instrumentation, privileged operation, existing cloud-cluster change, credential file or GitHub push is part of this step.

## Completion and retrieval

- Job started 16:13:14 and ended 16:19:53 Dubai time: **COMPLETED, ExitCode 0:0**, runtime 00:06:39, confirmed with `scontrol show job 195896`. `squeue -u nazish.baliyan` showed no remaining jobs. The `sacct` accounting backend returned connection-refused, so no accounting-derived host maximum RSS is claimed.
- GPU/runtime tests: **107 passed**. Preflight: **24 sessions**, 64 commits, four expected checkpoint refusals, 24 recovered windows, 60 native parity positions; **15,007 independent record checks passed** on HPC and locally.
- Main: **192 sessions**, 544 attempted windows, 512 exact-reference commits, 32 expected checkpoint refusals, 192 recovered windows across both policies, 944 native parity positions and 6,144 committed tokens checked. **85,379 independent record checks passed** on HPC and locally. No amendment or rerun was needed for pretrained measurements.
- Sources/configs remained unchanged from the pre-measurement freeze, verified on HPC after completion and locally after retrieval. Original engine/runner/detector hashes remain unchanged. Debug and main records were downloaded without replacing original HPC audit files; local audit results use separate `audit_local.json` filenames.
- Saved stdout/stderr are `stage2-195896.out` and `stage2-195896.err`. Two standard Qwen eager/sliding-window warning lines were emitted (debug and main). Recorded Qwen `use_sliding_window` was false; all tested context lengths were below its configured 32,768 window. Exact native/custom parity passed.
- Main results were independently recalculated from raw sessions by a separate agent and agree with the analyzer: upper-layer-fault complete-window ratios around 1.53x, staggered multi-layer ratios 1.10–1.12x; slight clean-window overhead relative to protected full replay. Detailed interpretation and exclusions are in `RESULTS.md`.

No additional GPU run is required for this bounded milestone. Publication readiness, broad fault coverage and production-serving feasibility remain unestablished.

## Final handoff

The professor-readable results and execution record were also copied to the task-owned HPC stage-2 directory. At 16:24:43 Dubai time, the final scheduler check showed no jobs under the account, and the SSH control connection was explicitly closed. No background compute or recurring monitor was left running. Final report numbers and caveats were independently reviewed against raw records; the requested minor stage-1 mean/median terminology correction was already applied. All report links resolve locally. No GitHub push occurred.

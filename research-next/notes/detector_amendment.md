# Snapshot-difference detection feasibility amendment

Specified at approximately 23:41 UTC, 15 September 2026, while the primary run was still in progress and before supplemental measurements. The frozen primary experiment remains unchanged and retains oracle localization. This is a separate post-hoc feasibility check, not a new general-purpose silent-error detector.

## Narrow question

The pilot already assumes a clean resident-GPU checkpoint containing every old prefix K/V tensor. Under append-only execution, those old entries should not change. At the end of the private window, compare each current old-prefix slice with its checkpoint counterpart. Collect differing layer indices using GPU reductions followed by one host transfer. No injected-coordinate or injected-layer information is an input to this comparison.

For exactly one differing layer, that detected layer may route the supplemental repair. For zero differences, do not invent a location: use conservative full replay if a recovery timing is requested. For multiple differing layers, fail closed because this pilot's repair contract has not been extended to that case. Expected injection metadata is used only to evaluate detection after the policy has made its decision.

## Measurements fixed before execution

- All 216 primary model/case pairs: 192 intended perturbations and 24 no-op controls. No-op cases have detection measurements only; the supplemental recovery matrix remains the 192 original perturbed pairs.
- Five measured repetitions plus an excluded warmup. Detection is timed separately from the five recovery methods, with synchronized boundaries. Recovery-only speedup must not silently include or omit detection differently across policies.
- Record detected layer indices, whether actual injected elements changed after BF16 rounding, expected classification, latency repetitions, logical checkpoint payload and at least twice that payload read by exact snapshot comparison. Logical reads are a lower-level payload accounting quantity, not measured HBM traffic.
- Preserve a separate detection JSONL and bind it to primary/source hashes. Evaluate no-op false alarms and detection/localization of actually changed prefix faults. An intended perturbation that rounds away should produce no alarm, not count as a detector miss.
- Add unit controls for suffix-only corruption and a transient fault restored before the check. These should illustrate blind spots, not be relabeled successful detections.

## Claims this cannot support

The detector is ordinary exact snapshot comparison. It is not novel, checksum-based, fault independent, tolerant of a bad checkpoint, or a production verification service. It does not detect suffix-only corruption, compute-only errors that leave the compared prefix unchanged, a transient error that disappears before verification, model-weight errors, corrupted journals, or matching corruption of both compared copies. It does not certify semantic correctness, factuality or alignment.

The full checkpoint's storage, protection and update/copy cost remain required. The comparison scans old-prefix state and has linear read cost; no free or constant-time localization is claimed. Trustworthy recovery execution and output holdback are still assumptions. GPU payload allocations and synchronization must be disclosed. Success on this exact injected family must not be generalized to real hardware fault coverage.

This bounded test can replace injection-oracle routing inside the supplemental experiment only. It does not retroactively change the primary study or prove that detector economics are practical at large contexts and serving scale.

# Supplemental baseline sensitivity study

Decision recorded at approximately 23:34 UTC on 15 September 2026, while the frozen primary run was still executing. Primary raw results, source, configuration and analysis are not changed. At this point progress logs showed correctness outcomes, not aggregate performance conclusions.

An independent claim review identified a potentially avoidable cost: the conservative full replay restores every layer's prefix by copying checkpoint tensors. Under the single-known-faulty-layer contract, all other prefix layers are already clean. A stronger implementation may truncate all layer suffixes using views, restore only the known faulty layer's K/V prefix, and then perform ordinary full-layer replay. The comparison is necessary even if the original copying cost later proves small.

## Scope fixed before supplemental measurements

- Reuse every one of the 192 perturbed primary model/case pairs; no selecting favorable layers or cases. No-op controls remain in the primary study.
- Use exactly the pinned primary model weights, prompts, coordinates/seeds, precision, eager numerical path and output windows. Compare reproduced fault metadata with the primary record.
- Compare conservative full replay, prefix-view/one-layer-restore full replay, checkpoint-alias full replay, sparse RECUT and all-layer RECUT. Use five rotated timing repetitions plus excluded warmup, with the same independent working-cache materialization excluded for every method.
- The stronger baseline gets the same known faulty layer as RECUT, not the exact corrupted coordinate. It restores both complete K/V prefix tensors at that layer. It does not use the experimental clean reference for its repair decision.
- Use prefix views only on the method's independent working state. Subsequent HF DynamicCache appends allocate new tensors. Check that immutable checkpoint and original faulty inputs remain unchanged; verify complete output state, tokens, final logits and two-step continuation against clean replay.
- Before any supplemental measurements, a further code review identified a simpler full-replay optimization: construct a fresh cache whose layers alias the same trusted resident-GPU checkpoint. HF append remains out-of-place. Include this fifth method rather than treating easily avoidable checkpoint copying as essential baseline cost. It uses no fault-coordinate privilege. The common working-cache materialization is excluded and unused by this method; constructing the alias-cache metadata is timed. Prefix views/aliases retain backing storage; do not claim they release that allocation.
- Keep source, raw data, hashes, run identifier and descriptive analysis separate. Report failures, timing losses and all-layer comparisons. This is post-hoc baseline sensitivity, not another 192 independent fault samples or a replacement for the primary protocol.

## Interpretation limits

The new full-replay baseline improves restoration, not decoder kernels or production scheduling. Sparse/all-layer restoration remains the original implementation. This tests whether the original recovery-speedup signal depends on unnecessarily copying clean prefixes; it does not establish superiority over all optimized activation-restoration systems. Timing differences across the primary and supplemental runs are not causal comparisons; all claimed supplemental speedups must be paired within the supplemental run.

No extra fault families, models or intervention strengths are added. No detector, field failure rate or serving-throughput claim follows. If the stronger baseline erases the gain, the final paper must say so.

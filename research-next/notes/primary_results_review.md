# Independent interpretation of the completed primary pilot

## Verdict

The frozen primary run supports **bounded exact-state repair and a depth-dependent replay-work reduction**, not a demonstrated production resilience system, novel checkpointing primitive, or exercised mid-window divergence fallback. Performance conclusions remain conditional on the primary restoration implementation; the stronger supplemental baselines must be reported separately.

I inspected primary-v1 raw rows, summary, both audit reports, metadata and exploratory outputs. This is an independent interpretation from those saved artifacts, not another model execution or an independent implementation replication. Raw episode SHA-256: 6c2d4a577eb57bd7de523c21d67f97d223d034521105e31f7f3fad1404799159.

## What actually diverged

There are 192 injected-fault cases and 24 noops, across two models and four prompts. All 192 nominal injections changed stored values. Only two **within-replay-window** greedy token sequences differ from their clean references; both are Qwen cases using the same structured-record prompt and both first differ at output 12 of a 12-output window:

| Case | Fault | Faulty layer | Sparse cut | First differing output | Actual sparse start trace |
|---|---|---:|---:|---:|---|
| primary-052 | K scalar | 0 | 0 | 12 of 12 | zero for all 12 steps |
| primary-063 | V vector | 11 | 6 | 12 of 12 | six for all 12 steps |

The all-layer trace for primary-063 is eleven for all 12 steps. Therefore **no primary repair executes a later replay step after discovering a token divergence**. Case 063 corrects a final output using a nonzero cut, but does not test the subsequent within-repair all-layer fallback or future-journal invalidation. Two-step clean continuation checks pass after repaired states; this is useful continuation evidence, but is not the same branch coverage.

Do not write that only two token differences were ever observed. The uncorrected negative control also differs during the two-step continuation in five cases: SmolLM2 primary-015, Qwen primary-013 and primary-015, plus Qwen primary-052 and primary-063. Three of these had no difference during their original window. No population frequency can be inferred.

The proposed W=12 to W=14 extension of the same primary-063 fault is a sensible **explicitly post-hoc targeted branch test**. Preserve the original prefix, model revision, coordinates, mutation and first 12 outputs, then inspect full state plus the later zero-cut starts. Do not add it to the 192-case denominator, call it preregistered, or use a deliberately selected boundary case to estimate a divergence rate.

## Correctness and negative controls

Full, sparse and all-layer repair each pass the recorded exact KV/logit/token and two-step continuation criteria in all 216 cases. Every method, including both negative controls, passes all 24 noops. Uncorrected KV is wrong in all 192 injected cases, even though replay-window tokens match in 190.

Slot-only repair passes the complete criterion in only 1/192 injected cases, but its failures have two distinct causes:

- In 128/192 cases, repaired prefix slots leave unequal descendant KV state. Of these, 126 still match the replay-window tokens. This directly supports checking state rather than just selected tokens.
- In 64/192 cases, all at the final decoder layer, slot-only repair restores exact KV and exact continuation state. In 63 of them the saved terminal logits remain stale, so the complete criterion fails; one SmolLM2 case, primary-090, passes everything.

Thus “191 slot-only failures prove 191 cases of descendant KV corruption” would be wrong. The 128 KV mismatches are the stronger evidence. Slot-only does not recompute terminal logits, and its last-layer failures must not be presented as all being persistent-cache failures.

Both saved audits pass 135,764 checks, with 142 native-parity positions per model and three archived tensor snapshots. The local CPU audit warns of cross-runtime Gaussian-RNG mismatch; the same-environment HPC audit has no warning. Both still validate recorded deltas and BF16 application. Snapshot cases are SmolLM2-000, Qwen-000 and Qwen-052; **primary-063 is not among the frozen archived snapshots**, and archived snapshots do not contain continuation tensors. Record checks and limited snapshot verification are not independent tensor verification of all 216 cases.

## Performance interpretation

On the RTX 5000 Ada, BF16 eager single-token implementation, median paired full-replay/sparse-replay ratios are 1.297 for SmolLM2 and 1.308 for Qwen; full/all-layer ratios are 1.822 and 1.774. These are medians of paired case ratios, not ratios of aggregate median times. Each model contributes 32 cases at each of three fault depths. At layer zero the sparse ratio is approximately one; at the final layer it is about 3.51. The largest all-layer ratios arise from replaying only the last layer, not evidence of similarly large end-to-end gains.

Sparse payload is exactly 3/29 of all-layer payload for SmolLM2 and 3/23 for Qwen in this fixed-cut experiment. Sparse recovery medians are higher than all-layer in 82/96 and 81/96 cases respectively; it buys lower payload and measured normal capture cost, not uniformly faster recovery. Timing ranges overlap in 32 and 31 cases. No categorical Pareto result follows.

Normal-overhead medians are about 0.35%/0.32% for sparse and 1.91%/2.05% for all-layer. But sparse observed normal costs are negative in 34/96 and 33/96 cases; differences this small are noise-sensitive. Five normal rotations are not complete cycles of the three policy positions. Do not equate negative measured overhead with free journaling or turn a small median into a guaranteed overhead bound.

The original artificial-mixture cost/saving ratio is approximately 1.08% for sparse in both models. Clipping negative per-case observed cost changes it to 1.93% and 2.06%. This **nonnegative-cost sensitivity** is upward-biased and not a confidence bound or real fault probability. The experiment omits or oracle-provides important detection, checkpoint/journal protection, holdback and integration costs. Primary timings also retain a conservative restoration baseline. The supplemental alias/view baselines and separately timed bounded prefix detector are necessary sensitivity evidence, not retroactive alterations to these results.

## Claims to avoid

Avoid claims of tested primary mid-window fallback, deployment fault rates, statistical independence of repeated prompts/timings, general detector coverage, optimized serving performance, all-case independent tensor reproduction, or novelty clearance. The present primary result is a coherent bounded implementation pilot with an informative state-versus-token negative control; the thesis-level contribution remains to be established against prior art and realistic operational costs.

# Filled final draft: numerical and claim-coherence review

Reviewed 16 September 2026. Outcome: **no numerical correction required for the claims checked below.** The final recommendation remains appropriately conditional: the pilot is complete, but novelty, deployment economics and A/A*-level thesis readiness are not established.

## Scope and method

Read the complete 333-line `artifacts/RECUT_03_Professor_Draft_and_Results.md`. Recomputed counts, per-case timing medians, paired ratios, linearly interpolated quartiles, overhead fractions, means and payload ranges directly from the primary and supplemental JSONL, without importing the experiment runners, their analysis scripts or their audit helpers. Compared those values with the saved summaries and the manuscript. Read both HPC and local audit records and checked their current-file hash bindings. Verified primary snapshot file hashes and sizes, but did not repeat tensor loading or GPU execution in this pass. The boundary archive is independently tensor-audited by the existing separate wrapper; this review checked its hashes, audit status, reconstructed episode and reported traces.

This reviewer previously authored the supplemental measurement module. Thus this is a fresh numerical/prose cross-check, not an independent implementation authorship claim, external replication or human peer review. No manuscript, PDF, experiment source or raw evidence was edited. PDF visual QA belongs to the separate final rendering pass.

## Headline ratios and timing estimands

All values below are ratios within a case of the methods' measured median times, followed by the median across 96 perturbed cases per model. They are not ratios of pooled medians. A baseline/method ratio above one means the named method is faster.

| Quantity | SmolLM2-135M, recomputed | Qwen2.5-0.5B, recomputed | Draft rounding |
| --- | --- | --- | --- |
| Primary conservative-full / sparse | 1.297471436 | 1.307881567 | 1.30x / 1.31x |
| Supplemental checkpoint-alias / sparse | 1.288331910 | 1.302472630 | 1.29x / 1.30x |
| Supplemental checkpoint-alias / all-layer | 1.827903057 | 1.763652795 | 1.83x / 1.76x |
| Primary sparse normal overhead | 0.349131862% | 0.323755419% | 0.35% / 0.32% |

The IQRs in both performance tables match raw recomputation, including all-layer and copy-avoiding full-replay rows. Conservative-full median times of 243.55/206.26 ms and supplemental alias-full times of 239.42/199.71 ms are correct and are not divided across runs to form a performance claim. Same-algorithm layer-zero control medians are 0.99738/0.99878, correctly rounding to 1.00x; the separately labeled alias-full/sparse restoration contrasts are 0.99583/0.99224, correctly rounding to 1.00x/0.99x.

Primary sparse mean H/S is 0.01081677/0.01077218, correctly reported as 1.08%/1.08% of windows under the stated toy expected-cost model. Means H=0.816648/0.688676 ms and S=75.498369/63.931020 ms match the reported rounding. The draft correctly confines these thresholds to the conservative primary comparison, not the separate alias experiment or an observed deployment fault rate. Journal and checkpoint payload ranges in KiB also match the raw records.

## Correctness counts and downstream decisions

- The primary contains exactly 216 model/case pairs: 192 perturbations and 24 no-ops, or 96+12 per model. Every perturbation actually changed at least one BF16 element; before/delta/after records are finite.
- Full, sparse and all-layer recovery pass 192/192 perturbed and 24/24 no-op cases. No-recovery joint success is 0/192; original-slot restoration joint success is 1/192. Both negative controls retain the same W-token sequence in 190/192 cases.
- Original-slot restoration matches cache in 64/192 and final logits in 1/192. The two explicitly distinguished failure patterns are correct: 126 cases have matching W tokens but different cache, and 63 have matching cache but different final logits. No faulty/no-repair cache, logits or checked continuation is non-finite.
- Exactly two primary trajectories diverge within the W-token window: Qwen `primary-052` at layer 0/cut 0 and `primary-063` at layer 11/sparse cut 6. Both first diverge at one-based j=W=12. Thus primary repair has no subsequent full-layer replay transition inside the measured window.
- Exactly five no-repair cases differ during the two further continuation decisions: SmolLM2 `primary-015`; Qwen `primary-013`, `primary-015`, `primary-052` and `primary-063`. **Both within-window divergent cases are included in these five.** These are not five additional cases on top of the two. The draft's current distinction is correct; adding the short parenthesis "including both within-window cases" would be optional clarification, not a necessary correction.

## Separate supplemental and detector evidence

The 192 supplemental model/case identities exactly equal the primary perturbation identities. Each reconstructed case, complete fault record, input prefix, initial token, reference/faulty decisions, fault layer, layer count and model revision matches the primary. There are 4,800 measured recovery trials (192 x 5 methods x 5 repetitions); every saved trial records exact joint recovery and unchanged checkpoint/original-faulty payload guards. These are repeated measurements of the same 192 faults, not 4,800 independent faults or 192 additional fault samples.

The detector file contains exactly all 216 original model/case identities, including 24 no-ops. Unique-layer localization is 96/96 actually changed cases for each model, with zero alarms among each model's 12 no-ops. All 192 intended perturbations changed the compared prefix, so no rounded-away case is omitted or called a miss. There are 1,080 recorded detector timing samples, distinct from all recovery timings. Recomputed median comparison times are 0.929859/0.727071 ms, correctly rounded to 0.930/0.727 ms; their IQRs and logical-read ranges match the table.

The final text retains the necessary limits: detection is ordinary trusted-snapshot difference checking, not a new detector; it does not cover suffix-only/transient-restored/compute/weight/journal/checkpoint faults; it does not certify a new checkpoint or safe external commitment. Primary localization remains an oracle. No supplemental tensors were archived, so its consistency audit is not misrepresented as independent tensor reloading.

## Selected W14 boundary diagnostic

The diagnostic metadata marks `performance_evidence=false` and `new_fault_samples=false`. It reuses Qwen `primary-063` with the same prefix, initial token and complete fault record. The first 12 clean/faulty decisions match the primary; only the diagnostic window extends from 12 to 14.

First divergence remains 12. Sparse starts are `[6]*12+[0]*2`; all-layer starts are `[11]*12+[0]*2`; full starts at zero throughout. With N=24, the resulting layer-step counts are exactly sparse 264, all-layer 204, full 336. All three recorded joint checks, including live continuation, pass. The archive is 8,474,502 bytes = 8.081915 MiB, correctly reported as 8.08 MiB. Both local and HPC boundary audits pass 1,194 checks. The manuscript correctly describes this as one selected post-hoc branch diagnostic, not additional independent fault evidence or timing evidence, and correctly notes that archived tensors exclude continuation.

## Audit accounting and binding

Primary: both audits pass 135,764 checks; 216 rows; 1,944 repair timing samples; 3,240 normal-path samples; three declared and hash-verified snapshot files totaling 27,677,162 bytes (26.39 MiB); 142 native-parity positions per model. The HPC audit has no advisory warnings. The macOS audit has the disclosed cross-runtime Gaussian-RNG advisory. Runtime fields and native-parity coverage in the draft match metadata.

Supplement: both audit records pass 98,997 checks with no failures. Current recovery and detection hashes match metadata and both audits; current summary content hash matches both audit records. Primary raw and summary hashes also match both primary audits. Boundary row/archive/metadata hashes match their metadata and both boundary audit records.

| Reviewed input | SHA-256 |
| --- | --- |
| Filled draft Markdown | `6ba43b0d6758686f306c41a0efcf9c0c49d3b6d98fe9e86a89a1424dbc9a5574` |
| Primary episodes | `6c2d4a577eb57bd7de523c21d67f97d223d034521105e31f7f3fad1404799159` |
| Supplemental recovery episodes | `21fce35699c89381dcb44957874ba7cf32842917f9b77e0c2bfdb21457f5eafe` |
| Detector episodes | `0a4688934acbe0886450f53b09aa0ad441b7536bd41c75f44e328fd43e1e30ba` |
| Boundary diagnostic row | `8863815b998ce602679c6ff590be897e047cad1da629e6a8f67d3f2b9647ef1d` |

The stronger-baseline result preserves the pilot's median recovery-only signal, but does not reverse the late novelty finding or establish end-to-end deployment value. The draft's final "do not commit as an A/A*-targeted thesis yet" recommendation remains consistent with the actual evidence.

# Independent final-claims review

Date: 16 September 2026. Scope: the complete professor-draft template and publication assessment, plus the current descriptive-analysis code. This is a scientific-claims review, not a visual PDF review or external replication. Primary job 195057 was still running when this review began; results placeholders were not treated as measured evidence. No draft, engine, primary runner, frozen config or raw primary data was edited.

## Overall assessment

The documents are unusually candid about oracle localization, synthetic faults, established activation caching, an elementary induction, and uncertain publication value. The publication assessment's skeptical verdict is justified. A successful compact-model pilot would establish a bounded implementation result, not automatically a new thesis contribution.

The strongest unresolved issue is the baseline and attribution of timing gains. The draft already mentions copy-on-write as a future limitation, but that caveat must also constrain the headline numerical interpretation. The full replay implementation copies every clean prefix layer; sparse replay copies fewer layers. Therefore measured recovery savings combine **less copying and less forward recomputation**. They cannot all be attributed to the layer-work identity.

## Concrete corrections and decision gates

### C1 — Do not call the run completed until completion and coverage are verified

The template introduction says “completed bounded pilot.” Keep this only after metadata says complete, the raw-file hash matches, both intended model revisions are present, the exact frozen case-ID set is covered, and the independent audit succeeds. Expected primary coverage is 216 unique model/case pairs: 192 perturbed cases and 24 no-op controls. A scheduler log or partial row count alone is insufficient.

If execution is interrupted or a supported-case mismatch remains unresolved, describe it as an incomplete pilot with its preserved failure, not a completed successful study.

### C2 — State the timing comparison's actual implementation boundary

Recommended result wording:

> “Within the tested eager, single-token copy-and-replay implementation, sparse recovery changed recovery time by [measured statistic]. The measurement includes both prefix-state restoration and forward replay; it does not isolate a pure recomputation speedup.”

Do not write that Q = NW - ck explains all wall-time gain. Full restoration copies N prefix layers; cut-c restoration copies N-c. Here T=64/128 and W=4/12, so restoration traffic can be material. The separately measured checkpoint-copy time is a diagnostic proxy, not a valid quantity to subtract from recovery times as if measured in the same allocator/context.

### C3 — The decisive missing baseline is cheap valid-prefix restoration

Under this exact fault model, only prefix KV at known layer L is corrupted. All other prefix layers remain clean. A strong full-layer baseline can keep those layers, discard the speculative suffix using views/copy-on-write, restore only layer L from the trusted checkpoint, and then replay W transitions from embeddings. It needs the same known layer, not the exact damaged coordinate.

The existing `truncate_cache` helper clones every retained prefix. Using it followed by another layer-L restoration would copy N+1 prefixes, not implement the desired copy-avoidance baseline. A supplemental implementation needs read-only prefix views on an independently owned working cache; ordinary DynamicCache append creates new storage through concatenation. It must test storage ownership, checkpoint immutability and unchanged input snapshots.

An even simpler full-replay baseline can construct a fresh DynamicCache with independent Python lists aliasing all trusted checkpoint tensors. Ordinary DynamicCache append concatenates into new storage rather than overwriting these tensors. Under trusted recovery, it needs neither the fault layer nor the coordinate. This uses the same resident immutable GPU checkpoint, not a free substitute for establishing/protecting that checkpoint.

Root authorized this **post-hoc baseline-sensitivity experiment** after this finding, including both layer-view and checkpoint-alias variants. It must remain separate from the frozen primary evidence. Do not silently replace primary numbers or represent the additional baseline as preregistered. If the optimized baseline matches or beats sparse RECUT, report that as a challenge to the current implementation's performance claim—not proof that every optimized RECUT adaptation is dominated. If it does not, this closes one confound but still does not establish superiority over optimized published systems.

### C4 — Avoid population claims from the balanced synthetic case mixture

The overall average/median gives equal weight to preselected early, middle and final layers, two windows, K/V and scalar/vector perturbations. It is not an estimate for real fault locations or incident frequency. Particularly strong late-layer results must be displayed alongside early-layer cases where c=0 and no layer work is saved.

Report per-model and at least per-layer-family results before any pooled headline. Do not count timing repetitions or three repaired policies as additional fault samples. The draft correctly rejects an IID zero-failure probability claim; keep that restriction.

### C5 — Distinguish the sufficiency boundary from a minimal recovery frontier

The protocol uses a conservative sufficient cut. It is not a proof that every skipped/retained state is necessary, nor that no token-specific, operator-level, sparse-attention or alternative restoration method can do better. Prefer “conservative validity boundary” or “measured tradeoff among tested policies” over language that could imply a globally optimal frontier.

The publication assessment's table says the all-layer baseline “minimizes the chosen start cut.” Numerically this is reversed: it **maximizes the eligible cut / minimizes the number of replayed layers within this fixed-cut family**. Its elapsed time is not a universal lower bound. This is a concrete wording correction.

### C6 — Separate actual examples from conceptual and analytic examples

The last-layer example with correct restored cache but wrong saved logits is established by the independent tiny-model test. The primary matrix has W=4/12, not W=1. Do not imply that every conceptual boundary case appears in the primary pretrained-model dataset. When selecting a recorded example, cite its model/case ID and actual checks.

Similarly, token-preserving but state-corrupt examples demonstrate a mismatch against fault-free numerical execution. They do not by themselves demonstrate harmful content, a task failure, or compromised factual accuracy. The draft already draws that distinction; maintain it in abstract/results inserts.

### C7 — Report actual mutation and non-finite outcomes explicitly

An intended perturbation can round away; `changed_elements==0` must be counted separately. Distinguish 192 intended perturbed cases from the number with an actual numerical mutation. Likewise report whether faulty cache/logits/continuations were finite, rather than allowing non-finite cases to blend into finite-perturbation summaries.

The source perturbation is finite, but its downstream arithmetic need not remain finite in general. If no such outcomes occur, state the observed zero count, not an implicit assumption.

### C8 — Keep the exactness statement narrow and complete

“Exact elementwise equality under the recorded execution path” is correct. “Bitwise,” “all future behavior proved,” and “production-safe” would be overclaims. The two-step continuation is an integration check, not a proof for arbitrary future inputs. The sufficiency argument relies on all relevant state and a deterministic transition function.

The current draft accurately specifies ordinary append-only, unquantized, layer-local KV and a pending final output. Keep these qualifiers near the main claim rather than relegating them entirely to an appendix.

### C9 — Distinguish recovery cost, steady-state overhead and holdback

A recovery-only ratio does not establish service throughput benefit. H/S is a case-mixture sensitivity calculation, not a measured event probability. If S<=0, label the configuration as no positive recovery saving. If H/S>1, it cannot break even within that simple per-window one-incident model even when every window fails. Negative H should be reported as timing noise/unresolved effects, not converted into free protection.

Window decode duration is not a measured network commitment SLO. The current prototype has private Python data, not live clients. The template's explicit commitment limitation is appropriate.

### C10 — Publication and review wording

The draft's “not submission-ready” assessment is supported. A/A* rank is not a scientific success criterion; keep venue details secondary to the fault model, adapted baselines and novelty objection.

“Eighteen closest papers” could imply an objective closeness ranking that the heterogeneous ledger process did not establish. Prefer “18 selected papers received deeper focused review.” Existing coverage metadata supports 100 entries, 18 deeper reviews and 82 selected-section screens; it does not authenticate all papers or establish exhaustive novelty clearance.

This review does not independently reverify venue ranks or deadlines. The separate venue-verification note should remain their evidence source, and the final bibliography still requires human verification.

## Bounded analyses available from existing primary rows

The following require no new model inference or fault selection. Label them descriptive/post-hoc wherever not already prespecified. They can be computed once the primary run is complete and hashed.

### 1. Separate the mechanism into observable failure categories

For original-slot restoration, tabulate mutually informative combinations:

- all W tokens match, but full cache differs;
- cache matches, but final logits and/or pending token differ;
- tokens differ;
- all checked state/output/continuation fields match.

Include continuation equality as an additional diagnostic, not as a substitute for the full-state result. Provide one representative case ID per observed category, selected by a fixed rule such as first lexicographic ID, not maximum effect. Report absence of a category honestly.

**Why useful:** demonstrates exactly what the negative control establishes without conflating visible-token disagreement with state corruption.

### 2. Condition savings on cut depth and divergence

For each model, display recovery ratios and layer-step fractions by early/middle/late fault layer, W, and first-divergence class (none, first transition, later). Plot or tabulate c*k/(N*W), with k=min(first divergence,W), against measured saving.

Report how much total saving comes from no-divergence and late-layer cases. A benefit confined to these favorable strata limits the general claim; a negative result here can falsify the intended practical story without additional runs.

**Important:** first divergence is a measured post-treatment outcome, so these are explanatory strata, not randomized causal subgroup estimates.

### 3. Use c=0 as a same-algorithm timing control

At layer L=0, full, sparse and all-layer policies all invoke the same cut-0 recovery logic. Their ratios should fluctuate near one. Use their dispersion to contextualize any small reported speedups elsewhere. With only three repetitions per method, do not present finely resolved tail-latency claims or strong significance.

Do not subtract a hand-chosen “noise correction” from other results. Report the control and raw paired distributions.

### 4. Diagnose copying versus forward work

Stratify by T/W and compare total recovery time with recorded checkpoint-copy duration and layer-step work. State clearly that checkpoint-copy timing is a different context and only a proxy. The post-hoc optimized baseline is the stronger test of this confound.

If the highest ratios track prefix-copy volume more closely than replay work, describe the result as an implementation restoration effect rather than a novel causal-replay efficiency result.

### 5. Compute dominance and break-even per useful stratum

Present the observed no-journal, sparse and all-layer choices with journal payload, normal-window overhead and recovery cost. Report the number of cases with S<=0, 0<H/S<=1 and H/S>1, while retaining noisy/negative-H cases explicitly rather than forcing a favorable ratio.

Only an observed frontier among these three tested policies is justified; do not call it the attainable frontier of all recovery methods. A positive pooled mean can conceal dominated early-layer or small-window regimes.

### 6. Validate accounting and dependence

Verify every timing repetition satisfies the start-trace work equation, and every method has the intended repetitions/warmup exclusion. Keep model/case as the unit of correctness evidence. Check observed-versus-requested perturbation changes, finite-state counts, exact frozen case coverage, and model revision/source hashes.

For uncertainty, use the recorded paired repetitions and descriptive ranges. Four prompts do not support a convincing task-population confidence interval; a bootstrap over hundreds of dependent model/method/repetition rows would create false precision.

## Suggested short final conclusion

> “The pilot validates a conservative recovery implementation on the recorded cases and exposes when activation journals can reduce work in this reference path. It does not yet establish a materially novel or deployable advantage. The decisive next question is whether the benefit survives an optimized, equally informed valid-prefix baseline and realizable detection/commitment costs.”

If the supplemental stronger baseline removes the speed advantage, replace “benefit survives” with the measured negative finding. Do not soften an adverse result by switching to a weaker comparator or changing the objective after seeing it.

## Authorized supplemental implementation, before GPU execution

This reviewer subsequently authored `experiments/run_optimized_replay.py` and `prefix_detector.py`, at root's explicit request. Consequently this note is independent of the primary engine/runner authorship, but is **not** an independent review of the supplemental implementation; separate agents are reviewing that source and writing an independent output auditor.

The supplemental experiment is bounded to the same 192 intended perturbed primary cases, without fault/prompt retuning or overwriting primary evidence. It compares conservative full replay, known-layer prefix-view full replay, checkpoint-alias full replay, original sparse RECUT, and original all-layer journaling. Each has one excluded warmup and five rotated repetitions. Common working-cache cloning remains excluded for all methods and is unused by the checkpoint-alias variant. All policy-specific restoration/construction and replay are timed. Source hashes, numeric-backend settings, model manifest, config, input IDs, perturbation records and token trajectories are bound to the primary evidence.

Views retain backing allocations until append replaces them; no early memory release is claimed. Checkpoint and original faulty snapshots are checked for raw-byte immutability after recovery and continuation. Complete numerical state/output checks remain the same as the primary. Sparse/all-layer restoration is deliberately not optimized in this sensitivity, so the experiment asks whether the original advantage survives a stronger simple full baseline; it does not chart the best possible optimized RECUT frontier.

### Separate bounded detector feasibility

At root's request, `prefix_detector.py` compares every old-prefix K/V tensor with the trusted checkpoint, aggregates per-layer differences on-device, and transfers a single layer-decision vector. Five timed repetitions plus warmup are outside all recovery timers. Logical input bytes are reported as twice checkpoint payload; comparison temporaries, instructions and physical memory traffic can exceed that input count.

All 216 primary cases receive detection records, including 24 no-ops; only 192 perturbed cases receive the five recovery methods. A unique observed changed layer routes recovery. The injection's known layer is used only to validate the controlled experiment. Multiple changed layers fail closed. No actual change correctly produces no alarm, not a miss; in a rounded-away perturbed case, the sensitivity conservatively uses cut-zero repair rather than inventing a localized layer.

This is **not** a general detector or proof that localization is cheap in deployment. It observes persistent post-checkpoint changes to old-prefix state only. Tests demonstrate no alarm for suffix-only corruption and for an old-prefix error restored to its original value after contaminating descendants. Errors before checkpoint creation, corrupted checkpoint storage, compute-only faults and nonpersistent errors are not covered. Do not rewrite the primary experiment as detector-driven; detector routing belongs only to this separately labeled supplement.

Detection cost is useful new feasibility evidence within this restricted model. It must be shown separately from journal overhead and recovery savings. If the detector is common to compared protected policies, its cost may cancel in their relative difference; it still contributes to absolute normal-path cost and does not make checkpoint trust or holdback free.

### CPU validation status

The combined local offline test suite reached **76 passing tests** (36 original engine tests and 40 independent tests) before any supplemental GPU execution. New tests cover Llama/Qwen2, first/middle/last layers, W=1/4, source-storage aliasing, checkpoint preservation through continuation, the five-way counterbalance, exact primary reconstruction, no-op/zero-change routing, multiple-layer rejection, and the detector's suffix/transient blind spots. These tests do not provide GPU timing evidence. The final source hashes will be retained in supplemental metadata and the independent reviewer/auditor records.

## Source freeze and updated final-composer review

Final local combined validation: **76 passed in 4.81 seconds**, with only the existing urllib3/LibreSSL warning. No supplemental GPU run was performed by this reviewer. Source is frozen absent a critical defect:

| Source | SHA-256 |
| --- | --- |
| `run_optimized_replay.py` | `6081ddd9c7d8572e0e0e9ac67b8a2017e2865034115cc75207642a0995076ac1` |
| `prefix_detector.py` | `0324ad63f6e63c573afd6ee94c978629b03b197e9fe072bc70e999191df93418` |
| `tests_independent.py` | `7aecdb9d929a7b806c1031e79a4245995162e13605fac8113a45dbc0bdfa7ae3` |

Read the updated professor template's stronger-baseline/detector sections and the full `artifacts/compose_final.py`. The detector limitations are now appropriately explicit: persistent old-prefix changes only, no suffix/transient/weight/journal coverage, no population-rate claim, and importantly no ability to certify that a newly created checkpoint is clean.

Remaining concrete final-composition corrections (root files were not edited):

1. **Abstract must include the strongest comparison.** The current composer prints only original conservative-baseline speedups, then merely says a stronger “view-based” study challenges them. It loads `supplemental_summary` but does not use it in the abstract or decision. Once results exist, name checkpoint-alias full replay and include the sparse/all-layer comparison against it, including a slowdown or lost advantage. Do not make a reader reach page eight to discover that a simple stronger comparator invalidated the headline speedup.
2. **Make the final interpretation result-aware.** If sparse recovery loses to checkpoint-alias full replay, state that the current implementation's speed advantage does not survive the stronger baseline. Do not generalize that to all possible optimized layer-cut recovery. If it wins, state the surviving numerical margin while retaining the unresolved optimized-serving/detector economics caveats.
3. **Avoid denying the actual narrow detector measurement.** Replace “No realizable detector economics ... has been demonstrated” with language such as “Only the restricted snapshot-comparator cost has been measured; broader end-to-end detector, checkpoint-validation and commitment economics remain unestablished.” This is more accurate without claiming a general detector.
4. **Bind supplemental results at composition time.** In addition to passed status and the primary binding, recompute supplemental `episodes.jsonl` and `detection.jsonl` hashes and compare them with metadata and `audit.input_sha256`; hash `summary.json` against `audit.summary_sha256`, and require no audit failures. Otherwise a stale passed audit could be paired with modified summary/prose. The primary side already performs stronger raw-data binding.
5. **Use exact temporal/test labels.** Say “the then-current 58-test CPU suite passed before the primary run,” not a potentially ambiguous final “complete suite.” The later 76-test suite includes supplemental tests and was run locally; do not label it an HPC result without its corresponding HPC evidence.
6. **Keep no-alarm semantics experimental.** No-op controls receive detector-only measurements, whereas a zero-change case inside the 192-case recovery matrix receives the documented conservative fallback. Do not imply every no-alarm service window was replayed or safely committed by an implemented deployment policy.
7. **Update the residual-baseline caveat.** The template still lists copy-on-write rollback purely among things that “may change the frontier.” Distinguish the two COW/view variants actually measured from further unimplemented fused/production methods.

These changes affect scientific interpretation and evidence binding, not the frozen experiment source or primary protocol. Any final inserted supplemental/detector prose still needs review against the completed audited numbers; at this stage it is not measured evidence.

## Second pre-results composer/table/figure review

Read the updated 13-section template, all of `compose_final.py`, `compose_supplement_blocks.py`, and `make_results_figure.py`, and cross-checked their field interpretation against `audit_supplement.py`. No data-dependent result was inferred and no root-owned source was edited.

### Checks that are now consistent

- The supplemental ratio is baseline median time divided by method median time, paired within each case. Above one means the named method is faster; table and figure use the correct orientation.
- The reported primary counts are 216 scenarios = 192 intended perturbed + 24 no-op; per model 108 = 96 + 12. Supplemental recovery uses 192/96, while detection uses all 216. Actually changed prefix denominators are used for localization, so rounded-away perturbations are not called detector misses.
- The strongest checkpoint-alias sparse ratio is now included in the abstract, and the decision has a branch for an adverse strongest-baseline result. The detector is no longer described as entirely unmeasured; its restricted comparator scope is explicit.
- Primary normal-path overhead is not silently combined with supplemental timings into a new measured break-even probability. The figure reads only supplemental results and labels IQRs as descriptive, not confidence intervals.
- Supplemental raw recovery, raw detection and summary files are now bound to the passed audit at final composition. The template makes clear that no supplemental tensors were independently reloaded, and reused scenarios are not additional independent trials.

### Remaining corrections sent to root

1. **Same-algorithm control versus restoration contrast.** `compose_supplement_blocks.py` calls layer-zero alias-full/sparse a context for “timing variability.” At c=0, sparse is the same algorithm as conservative full replay, but checkpoint-alias deliberately avoids different restoration work. Therefore alias-full/sparse includes a systematic copying difference; it is not a pure noise floor. Use conservative-full/sparse at L=0 for the same-algorithm variability control. Alias-full/sparse can remain as a clearly named restoration-only contrast. This does not affect the plotted ratio orientation.
2. **Primary summary content hash.** `compose_final.py` still checks the summary's embedded episode hash without hashing the primary summary file itself. `audit_results.py` already records `input_sha256['summary']` (and `paired_analysis`). Require the current summary digest to match both primary audit records before using its values. An embedded raw hash alone cannot exclude a stale or modified numerical summary.
3. **Generated prose and figure provenance.** Regenerate the two Markdown blocks and figure immediately from the same audited supplemental summary used by the final composer. Record their content hashes/input-summary identity in the final artifact provenance if practical. The final composer accepts arbitrary prose paths, so passed data gates alone do not establish that a manually supplied block or an older image came from those values.
4. **Nominal versus actual figure cut.** The figure derives the nominal configured sparse cut from layer labels. A zero-changed case routes to actual cut zero. If any such cases occur in a nonzero-layer stratum, label the displayed cut as nominal and disclose the fallback cases, rather than implying every plotted trial used that cut. This is conditional advice, not a claim such cases occurred.

Minor communication refinements: describe plotted points as the median across per-case ratios of median times (bars are their IQR); use “observed median” for tiny advantages near one; call the pre-primary suite the “then-current 58-test suite”; and distinguish tested COW variants from remaining unimplemented optimizations in the general-baseline caveat. None of these refinements establishes a result before the audited data arrive.

## Final novelty and recommendation coherence check

Read the latest professor template in full and `final_novelty_falsification.md`, with a narrow check of the final composer's decision paragraph. This is a coherence review of the supplied source-derived assessment, not an additional independent full-text survey or numerical result audit.

The recommendation is now consistent: discuss the completed bounded pilot, but do not commit to RECUT as an A/A*-targeted thesis yet. The template and falsification note both acknowledge that activation-assisted reconstruction, checkpointing, history-conditioned reuse and rollback are established. Neither claims that the token-divergence gate alone is a new general principle. Both qualify the six-record negative search as insufficient to prove priority. GhostServe is no longer described as fail-stop-only: its partial recomputation and discussion of soft memory errors are explicit. The fail-stop label remaining in the note refers to TARRAGON's stated model, not GhostServe.

Concise remaining wording/packaging suggestions sent to root:

1. Add near the opening: "This final assessment supersedes the novelty and readiness recommendations in the two earlier prospective PDFs, retained unchanged for provenance." The current statement that the earlier documents are preserved does not express precedence as directly.
2. Replace "A publishable contribution needs a distinct fault-specific result and a useful improvement ..." with "The proposed performance-thesis claim would need a distinct fault-specific result and a useful improvement ..." The narrower statement leaves room for a genuinely novel theoretical or negative result, consistent with the later rejection/redirection discussion.
3. Replace "The only remaining candidate contribution" with "The remaining candidate contribution in this study" to avoid suggesting the bounded review exhausted all possible research contributions.
4. Fix the repeated section numbers 6 and 11 before final rendering. This is a navigation issue, not a scientific defect.
5. Optionally label LayerSkip "ACL 2024" in the final reference list, matching the verified publisher status already recorded in the falsification note.

No root-owned source, prospective PDFs, experiment code or evidence was changed. The filled final draft still requires its separate audit against actual primary, supplemental and targeted-boundary data.

# RECUT: Complete Prototype and Validation

## Recovering an LLM inference window with less repeated work

Stage 3 research artifact | 16 September 2026

Prepared for Nazish Baliyan and discussion with Prof. Abdulrahman Mahmoud. This report completes a bounded implementation and evaluation stage. It is not a completed thesis, deployment certification or conference-acceptance prediction. Earlier RECUT studies remain unchanged and are not pooled into these results.

{{EXECUTIVE}}

### What the project actually does

An autoregressive language model keeps a memory of previous tokens, called the KV cache. If part of that memory is corrupted, later internal states can become wrong too. Merely fixing the original damaged value does not undo those consequences. Our protected full-replay baseline repeats a whole uncommitted window from a trusted checkpoint. RECUT saves a few intermediate layer inputs, detects changed old-prefix cache layers, and tries to repeat only the necessary upper-layer work.

The important safety rule is about history: reuse is permitted only while the recovered token history matches the speculative history. After the first corrected token differs, every later step is recomputed from the first layer, even if token IDs later match again. Tokens remain private until the complete window is accepted.

### What was completed in this stage

The controller now has canonical SHA256-preserving batched transfers, full/sparse/all-cuts policies, complete-window cost measurement, native and bare clean baselines, a larger 7B model, 2,048-token contexts, K/V fault strata, integrity challenges, recorded output holdback, and independently reloadable full-state snapshots. Debugging, main execution, independent checks and the delivery artifact are distinct evidence layers.

> Completing the experiment means accounting for every prespecified outcome and reporting unfavorable findings too. It does not mean that the mechanism is novel, efficient enough for deployment, or ready for an A/A* submission.

<!-- PAGE -->
## 1. Mechanism and conditional correctness

### One protected window

| Phase | Implemented behavior |
| --- | --- |
| Begin | Preserve the public committed state; make separate working and checkpoint copies; seal the checkpoint. |
| Speculate | Decode W steps privately. Sparse saves three layer-input cuts; all-cuts saves every nonzero layer input. Seal captured activations with owner, position, token and cut metadata. |
| Detect | Verify checkpoint integrity; compare each working old-prefix K/V tensor numerically against the checkpoint. The controller receives no injection coordinates. |
| Recover | Select the deepest retained cut no higher than the earliest changed layer. Replay from that cut while history agrees. A damaged required journal forces full replay. |
| Diverge | At the first corrected-token mismatch, truncate to the last valid processed position and permanently switch subsequent steps to full-layer replay. |
| Commit or refuse | Check finite state and checkpoint integrity again. Release the complete window or reject without adding output. A damaged checkpoint poisons the session. |

### Why selective reuse can be valid

A saved cut is the hidden input before that decoder layer executes. Under the declared fault model, layers below the earliest directly corrupted cache layer received no direct fault. While their complete token history agrees with recovery, their saved inputs are still valid. Rebuilding the affected upper layers then restores the clean output for that step. If the next-token decision changes, later lower-layer states no longer have a valid common history and cannot be reused. Full replay from that point restores the remaining trajectory.

This is a conditional code-grounded argument, not a mechanized proof or a minimum-recomputation theorem. The same argument extends across sequential committed windows only if state remains trustworthy between them.

### The trust boundary is essential

The supported contract is persistent, finite, **numerically unequal** changes to old-prefix K/V during speculation after checkpoint capture. Every potentially corrupting layer must retain such evidence until detection. Initial state, model weights, host metadata/seal storage and guard/recovery/commit execution are trusted. There are no concurrent writers or faults during those trusted phases.

Signed-zero-only working-cache changes are invisible to numeric comparison; byte seals distinguish them in checkpoints/journals. Suffix-only, reverted, pre-checkpoint, between-window and computation-origin faults are outside the guarantee. A reverted low-layer fault combined with a surviving high-layer fault is also outside it: the higher alarm alone cannot justify reusing the damaged lower activation.

Checkpoint refusal preserves previous committed output; it does not repair that checkpoint or automatically restart the session. This is a sequential local API, not crash durability, network transactions or rollback of external actions. Clean-state equality does not make a model's answer truthful or safe.

<!-- PAGE -->
## 2. Experimental design and measurement

### Prespecified matrix

| Dimension | Values and scope |
| --- | --- |
| Models | Pinned SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-7B base models; BF16; greedy decoding; one sequence; eager attention. |
| Workloads | Three authored synthetic passages: cloud, code and reasoning. Numbered repetitions form prefixes of 128 or 2,048 tokens; windows contain 8 or 16 steps respectively. Three consecutive windows per session. |
| Clean comparators | Native HF, bare custom engine, full/sparse/all-cuts batched, full/sparse legacy. Native/bare are unprotected cost references, not resilience-equivalent alternatives. |
| Fault treatments | Early, middle and late K/V changes; two staggered old-prefix faults; faults in every window. Finite fraction-bit-3 XOR or 8-RMS vector changes, fixed before measurement. |
| Guards | Required saved-journal corruption and checkpoint corruption. Journal treatment differs by policy and is excluded from fair paired timing. |
| Repetitions | Two fault seeds and two timing repetitions; three clean timing repetitions; separate warmups and synchronized profiles. Repetitions are not independent fault samples. |
| Denser-journal comparator | All nonzero layer cuts, under the same checkpoint/detector/commit contract. This is not a reproduction of HCache, HybridServe, LayerSkip or VeriCache. |

All ordinary fault cases affect the second window; repeated-prefix faults affect every window. Strata couple layer, K/V, severity and injection time, so these factors cannot be given separate causal interpretations. Initial context and window length also change together between the two shapes. The inputs are not a representative accuracy or production-serving benchmark.

### What the clocks include

The primary clock covers BEGIN through COMMIT/refusal: copies, seals, activation capture, speculative work, injection harness, detection, replay, final guards and synchronization. Session initialization is separately timed and added to initialization-inclusive session comparisons. Common input copying, native batched prefill and experimental native-reference checking remain excluded from both boundaries. Injection-subtracted costs are a diagnostic, not a deployment measurement.

Native/custom equality is checked at every post-prefill decode position from the same native batched-prefill checkpoint. It does not establish equivalence between different prefill execution shapes. Every committed window is compared with native-reference cache, logits, tokens and a two-step continuation.

For timing comparisons, first take the median of repetitions for each workload/scenario/seed/variant. Then form matched ratios and summarize across the stated cases. No population confidence interval, field fault rate or significance claim is inferred. A ratio above one favors the denominator; protected/unprotected cost ratios above one instead mean added protection cost.

Method order rotates, but three clean repetitions do not fully counterbalance seven variants. Small differences near 1x may reflect timing variation. Models can occupy different lab nodes of the same GPU type; across-model timing differences are not an isolated scaling law.

<!-- PAGE -->
## 3. Normal-operation cost

{{CLEAN_COST}}

### Why this comparison matters

Recovery savings are paid only when a supported fault is detected. Copies, guards, capture and delayed release are paid on ordinary protected windows too. A speedup against protected full replay therefore does not imply faster ordinary inference. Both comparisons are necessary: a same-contract recovery comparison and an honest unprotected cost reference.

The execution path is a batch-one eager research engine. It does not include an optimized serving scheduler, concurrent requests, FlashAttention/SDPA, quantized caches or distributed execution. These numbers cannot be presented as production throughput, end-user TTFT or general model efficiency.

<!-- PAGE -->
## 4. Recovery benefit and its limits

{{RECOVERY_COST}}

{{DIVERGENCE}}

<!-- PAGE -->
## 5. Whole-session costs and sensitivity

{{SESSION_COST}}

### Supported-incidence sensitivity

For matched complete windows, let H be sparse-clean cost minus full-clean cost and R be full-fault cost minus sparse-fault cost. Let p be a hypothetical fraction of windows affected by a supported fault. If H and R are both positive, sparse becomes cheaper in the illustrative mixture only above p* = H / (H + R). Other sign combinations must be reported separately. This compares two protected methods; it does not price their shared cost against unprotected inference or include one-time initialization.

{{BREAK_EVEN}}

The deliberately injected fault fraction is not a measured deployment incident rate. A favorable threshold does not establish that a real system reaches it, and an observed negative H may be a small timing difference rather than a dependable zero-cost result.

<!-- PAGE -->
## 6. Sealing, memory and delayed output

### The implemented optimization

Batched sealing packs canonical tensor bytes by device, makes fewer device-to-host transfers, and hashes the same ordered metadata/byte stream on the CPU. It preserves stage-2 SHA256 digests; it is not a new cryptographic method or GPU hashing kernel. The legacy ablation uses the same controller with per-tensor copies. Both modes skip irrelevant empty-journal hashes in full replay, so the ablation isolates batching rather than every earlier-stage change.

{{SEALING}}

<!-- PAGE -->
## 6. Memory and delayed output (continued)

{{MEMORY}}

{{AVAILABILITY}}

GPU allocator peaks include the resident model and native-reference fixtures; they are not a production minimum. Logical checkpoint/journal bytes omit allocator and protection overhead. Batched transfer staging adds temporary device and host storage; host peak memory was not measured. Seal-transfer counters do not count every transfer in the controller. Synchronized component profiles are separate diagnostics, not an exhaustive partition of unprofiled wall time.

<!-- PAGE -->
## 7. Validation and evidence strength

{{VALIDATION}}

### What is independently checked

The separate auditor checks complete matrices, source/input/output hash bindings, fault placement and arithmetic, same-treatment pairing, route and divergence rules, commitment/refusal records, latency arithmetic and transfer accounting. It then reloads the predeclared tensor archives on CPU using restricted tensor/basic-data loading and compares recovered/native tensors. Other trials retain recorded live equality checks; they were not all independently re-executed from saved tensors.

A second calculation reconstructs the report's principal numerical results directly from raw sessions before comparing them with the analyzer. It does not treat summary.json as the numerical source of truth. These are internal AI-assisted checks, not human peer review or external replication.

{{REPORTING_CORRECTION}}

<!-- PAGE -->
## 7. Additional verification and limits (continued)

{{SELECTED_DIAGNOSTIC}}

### Important evidence qualifications

Exact state equality is numeric elementwise equality; signed-zero distinctions are separately byte-sensitive. The saved subset's byte-comparison outcome is reported without expanding the general detector contract. Model-file digest records bind the runtime record, but local audits do not reload remote model weights or independently regenerate tokenization and Gaussian draws.

Qwen runtime warnings about eager sliding-window attention are retained where emitted; the actual model configurations disable sliding-window execution, and native parity remains the operational gate. No physical GPU faults, binary instrumentation, malicious attacks, real failure incidence, semantic accuracy or model alignment were measured.

<!-- PAGE -->
## 8. Prior work, novelty and publication judgment

### Closest established mechanisms

| Prior work | Overlap and the remaining distinction |
| --- | --- |
| LayerSkip, ACL 2024 | Reuses lower-layer KV/exit activations during self-speculation and manages the accepted prefix after mismatch. Layer-cut reuse and history-valid rollback are not new general principles. [Published paper](https://aclanthology.org/2024.acl-long.681/) |
| VeriCache, 2026 preprint | Full-KV verification corrects compressed-KV drafting at the first mismatch. Its deliberate compression setting differs from persistent-memory corruption, but generic exact-output verification/reuse claims overlap. [Versioned paper](https://arxiv.org/abs/2605.17613v1) |
| HCache | Reconstructs KV using stored layer-input hidden states and projection work. RECUT's all-cuts replay is not this full serving/restoration system. [Versioned paper](https://arxiv.org/abs/2410.05004v1) |
| HybridServe | Mixes activation and KV storage for recomputation/transfer tradeoffs. Fixed sparse cuts do not establish a new optimal placement policy. [Versioned paper](https://arxiv.org/abs/2501.01792v1) |

This focused recheck examined the four primary sources and selected mechanism sections. The earlier 100-paper screen is preserved separately with its original reading-depth labels; this stage is not a fresh cover-to-cover review of 100 papers. LayerSkip coauthor Anas Mahmoud must not be confused with Prof. Abdulrahman Mahmoud.

### Professor fit

Dependable AI systems, application-visible error consequences and selective protection fit Prof. Abdulrahman Mahmoud's research background. RECUT's fault/recovery evaluation is therefore a defensible supervision topic. That fit does not establish that he endorses the mechanism, accepts its thesis scope or considers it novel. [Official profile](https://mbzuai.ac.ae/academics/faculty-directory/abdulrahman-mahmoud) and [publication list](https://ma3mool.github.io/publications/).

{{PUBLICATION}}

The honest claim is a bounded detector-routed transactional KV-recovery study with measured protection costs, not "the first activation-based repair method," "provably optimal recovery," "general hardware resilience" or "A*-ready." Conference ranking or topical fit cannot substitute for a demonstrated contribution.

<!-- PAGE -->
## 9. Reproducibility, completion and supervisor handoff

{{REPRODUCTION}}

### Research decision

The stage is complete when the configured experiment and its audits finish, not when its conclusion becomes favorable. The remaining publication requirements are different research questions: a credible deployment fault/commit contract, strong adapted published-method comparisons, representative optimized serving, and a general contribution beyond familiar checkpointing/reuse mechanisms. These cannot be certified by adding more repetitions to the same matrix.

If normal-case protection costs or stronger alternatives erase the useful operating region, the performance-oriented thesis claim must be revised or rejected. A negative finding would need a broadly useful explanation or bound to become a strong paper; a slow prototype alone is not enough. No threshold, fault severity or headline sample was selected after observing the final results. The explicitly separated fallback diagnostic was selected afterward for additional state validation and never enters primary performance or sample totals.

### A concise explanation for your professor

{{PROFESSOR_SUMMARY}}

### Delivery boundaries

Source, raw evidence, preserved failures, scheduler logs, audits, result recomputation, this report and a hash-verified archive are provided. Model weights, environments and credentials are not bundled. Earlier stages and other thesis projects are preserved. No paper was submitted and no GitHub repository or production cloud cluster was modified as part of this stage.

All implementation, reviews and writing are AI-assisted. Human authors must verify claims, attribution and applicable AI-use policies before any submission. This document is supervisor-review material, not a substitute for that review.

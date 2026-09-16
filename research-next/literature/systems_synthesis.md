# Systems literature screen, agenda audit, and novelty challenges

Review date: 2026-09-16. Assigned corpus: supplied papers 001-034. This is a research-screening report, not a declaration of established novelty.

## What was actually reviewed

The companion [per-paper ledger](systems_001_034.json) records a contribution, evidence limitation, possible connection, source and page references for all 34 papers. Every paper was screened from its extracted body text, not only its title or abstract. Closest neighbors 015, 017, 023, 024, 027, 028 and 030 received focused main-text review. Other papers received selected-section screening. Equations, appendices, figures, citations and experiments were not comprehensively audited or independently reproduced. Existing page-delimited text was used; no experiment code or HPC jobs were changed. External papers below have individually stated review depth.

The central methodological lesson is to separate three different claims:

1. Storage integrity: the bytes have not changed since a trusted write.
2. Computation integrity: the bytes were correctly produced in the first place.
3. Trajectory integrity: subsequent state and uncommitted output are consistent with the trusted computation.

A checksum created after an erroneous computation proves none of (2). Restoring one old cache block after delayed detection does not automatically repair (3). These observations are not, by themselves, new research contributions.

## Directions already substantially occupied

| Proposed broad claim | Closest primary-source challenge | What remains, if anything |
| --- | --- | --- |
| Better tests for generated GPU kernels | [KernelBench-Verified](https://arxiv.org/abs/2607.16241), [The Correctness Illusion](https://arxiv.org/abs/2606.20128), and [Contract-Grade Verifier](https://arxiv.org/abs/2608.12700) | Generic hidden inputs, shape fuzzing, accurate references, tolerance-free contracts, and training checks are insufficient differentiators. The last paper was abstract-screened, not independently validated. |
| More representative kernel benchmark | [KernelBench-X](https://arxiv.org/abs/2605.04956), [Atrex-Bench](https://arxiv.org/abs/2607.14541) | Production shapes, operator categories and weighted performance already have direct precedents. Abstract-level screens only. |
| A runtime bound for quantized attention | [Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/abs/2605.20868), [WitCert](https://arxiv.org/abs/2607.28699) | Root-agent supplied direct prior-art findings; this subagent did not independently audit their full methods. Do not claim a generic local error gate as novel. |
| Protect important KV layers | [Layer-Selective ECC Allocation for BF16 KV Cache Reliability](https://epapers2.org/apccas2026/ESR/paper_details.php?paper_id=2206) | Official APCCAS 2026 accepted-poster abstract already studies sensitivity-driven selective ECC. Abstract only; no full-paper review. |
| Demonstrate persistent/shared KV corruption | [Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems](https://arxiv.org/abs/2604.17249) | Persistence, propagation and scheduler-time checksums are already studied. This is defensive relevance, not an operational attack proposal; abstract/main-source screening did not establish field fault rates. |
| Fault-tolerant KV snapshots | [Deja Vu](https://arxiv.org/abs/2403.01876), [GhostServe](https://arxiv.org/abs/2605.00831), [Concordia](https://arxiv.org/abs/2606.23521), [LUMEN](https://arxiv.org/abs/2606.17787) | Snapshot streaming, erasure coding, GPU-resident checkpointing and load-aware recovery are existing directions. External abstract-level screens; no claim of full replication. |
| Selectively recompute reused cache | [CacheBlend](https://arxiv.org/abs/2405.16444), [KVShareArena](https://arxiv.org/abs/2609.10266) | CacheBlend, EuroSys 2025, approximates cross-chunk context with selective recomputation. KVShareArena is a September 2026 preprint evaluating reuse/repair across contexts and checkpoints. Neither generic repair nor recomputation is new. |
| Restore KV using saved intermediate activations | [HybridServe](https://arxiv.org/html/2501.01792v1), [KVPR](https://arxiv.org/html/2411.17089v2) | Direct novelty threats: exact KV reconstruction from stored layer inputs, hybrid activation/KV storage and scheduling are already present. Read abstracts and main methods. HybridServe's arXiv record supplies an ICCD 2025 journal reference; KVPR venue not independently checked here. |
| Recompute only affected requests from a valid prefix | [vLLM KV-load-failure RFC](https://github.com/vllm-project/vllm/issues/19329) | The proposal already detects failed blocks after forward and reschedules affected requests from the longest valid prefix. It is an RFC, not evidence that all proposed functionality was deployed. |
| Add ABFT to attention or improve thresholds | [ATTNChecker](https://arxiv.org/abs/2410.11720), [V-ABFT](https://arxiv.org/abs/2602.08043) | Attention propagation-aware ABFT and mixed-precision adaptive thresholds exist. ATTNChecker was externally screened; V-ABFT main text read. Test finite correlated errors, not only infinities/NaNs. |

Supplied paper 027 shows why arbitrary isolated-bit perturbations cannot represent all GPU arithmetic faults. Supplied paper 024 motivates full-trajectory/application outcomes rather than perplexity alone. Their fault models are still models, not evidence of actual error incidence on the available GPU. Supplied 022, 025, 026 and 029 concern major checkpoint/recovery mechanisms, mostly under detected failures; they must not be conflated with delayed finite SDC.

## Candidate question 1: Exact late-error recovery with a bounded layer-time journal

Question: Given a trusted localization event for a finite KV-state error and a bounded uncommitted decoding window, when can a small, provenance-valid layer-cut activation journal restore the reference trajectory more cheaply than longest-valid-prefix suffix replay?

The candidate contribution would be a correctness characterization of reusable layer-time state plus a measured recovery policy under a fixed memory and commit-latency budget. It would not be the discovery that errors propagate, a new detector, ordinary activation checkpointing, or arbitrary selective KV reconstruction.

Proposed scope and differentiator:

- Establish which journal cells remain clean conditional on a localized original error, token history and detection time. A saved activation downstream of the error cannot be trusted merely because its bytes match a checksum.
- Give the same trusted original-block checkpoint and fault localization to all baselines.
- Replay from a valid layer cut only while conditioning on the same token history. At the first repaired sampling decision that changes a token, discard the affected continuation and regenerate all layers thereafter.
- Treat tokens or tool actions already externally committed as outside exact rollback. Initial tests should buffer outputs; no real external actions should occur.
- Define exactness relative to a deterministic reference execution or explicitly account for implementation nondeterminism and RNG state. Floating-point closeness alone is not the same as identical future sampling decisions.
- Charge journals, metadata, protected checkpoint copies, read/write traffic and verification/commit delay. Grouped-query attention can make full-width hidden activations larger than the KV representation they supplement.

Closest threats are HybridServe/KVPR for activation-assisted reconstruction, vLLM for valid-prefix recovery, CacheBlend for selective recomputation and Concordia for recovery logging. Their accessible methods did not establish this exact late-SDC/journal-provenance/token-commit combination, but that absence is not a novelty proof. Classical rollback, output-commit, dependency tracking and checkpoint-placement literature still needs a dedicated check.

Killer baselines and stop conditions:

- Longest-valid-prefix suffix replay, full recomputation, and activation-assisted reconstruction adapted from HybridServe/KVPR, all given identical trusted snapshots.
- Immediate producer-side ABFT or duplicated computation plus storage checksums: a cheap prevention strategy may eliminate the need for delayed recovery.
- Simple periodic snapshots and uniform layer-cut placement at the same memory cost; the proposed policy must improve on these, not only on full-prefill restart.
- Stop if hidden-state journaling overhead dominates at plausible failure rates, if token divergence removes the savings, or if dense-attention dependencies force almost all useful work to be replayed.
- Report break-even error/detection-delay regimes without asserting that real deployed GPUs experience those rates.

Feasible one-GPU falsification test, before a full system:

Use a small open decoder-only model and a deterministic benign prompt set. First use teacher-forced continuations to isolate state repair; then greedy/free decoding with buffered outputs and saved RNG state. Introduce controlled non-adversarial tensor perturbations with varied layer, token position and detection lag; include isolated-bit and structured finite/nullification families, clearly labeled synthetic if raw RTL profiles are unavailable. Compare cached tensors, logits, first token divergence, complete recovered trajectory, replayed layer-token work, bytes retained, median/P99 recovery time and no-fault overhead. A success would identify a substantial operating region, not claim universal acceleration. No experiment was implemented in this review.

## Candidate question 2: Producer-validated reusable KV blocks under realistic compute errors

Question: Does reuse-aware validation at the creation boundary of persistent KV blocks prevent materially more application failures per unit overhead than existing matrix/attention ABFT plus ordinary storage checksums?

The target is compute-origin finite error, not a storage bit-flip detector. Sharing/amortizing a validation result across many consumers could matter, but checksum reuse itself is not new. A defensible contribution needs a new mathematically specified validation scope or cost allocation rule, and evidence that per-layer ABFT/duplication cannot achieve the same frontier.

Necessary distinctions:

- Storage error versus computation error versus quantization approximation.
- A guarantee conditional on a fault model versus a calibrated probability versus an empirical zero-miss observation.
- Local tensor fidelity versus generated-token agreement versus benign application/task success.
- Reused block identity must include all relevant computation context; changing model, adapters, positional encoding or quantization settings invalidates naive reuse.

Killer baselines: V-ABFT or comparable inline pre-rounding checks, ATTNChecker-style attention protection, duplicated projection computation, per-layer ECC, scheduler-time memory checksums, and the quantization certificate methods above. Count cache-creation cost even when amortized. Stop if conventional producer checks plus a checksum dominate, if worthwhile errors are already caught cheaply, or if a proposed witness is only an unreliable semantic probe.

One-GPU first test: use benign QA and structured offline tool-selection examples with no external actions. Reuse each prefix across controlled fanouts (for example 1, 8 and 64), compare compute-origin versus post-write perturbations, and measure harmful misses, false positives, complete task accuracy, no-fault throughput, tail latency and total protection cost. Hold fault locations/distribution and memory budget equal across protection policies. This is a secondary candidate, not a claim of novelty clearance; it may be better treated as a detector/baseline subproblem of candidate 1.

## Research-site access audit

These were research/project checks, not fellowship or application work. Access date for every row: 2026-09-16. A successful landing-page read is not a complete review of an organization's output.

| Organization | Actual access status | Specific project/source and relevance |
| --- | --- | --- |
| FAR.AI | far.ai/research redirected successfully to www.far.ai/research; publications and specific project page readable. | [Training Reliable Activation Probes With a Handful of Positive Examples](https://www.far.ai/research/training-reliable-activation-probes-with-a-handful-of-positive-examples), listed as NeurIPS 2025 Mechanistic Interpretability workshop. Rare-positive detection is relevant to evaluation design, not a validated hardware-error detector. [Concept Influence](https://arxiv.org/abs/2602.14869) illustrates behavior-level attribution, not causal repair guarantees. |
| Berkeley CHAI | [Research index](https://humancompatible.ai/research/) readable. Index appears concentrated in 2023 and earlier; not treated as a complete 2026 agenda. | [Causal Models with Constraints](https://proceedings.mlr.press/v213/beckers23a/beckers23a.pdf), CLeaR 2023, publisher abstract/introduction accessible. Formal causal relationships and allowable interventions are conceptually relevant; this paper is not a transformer-recovery method. |
| Center for AI Safety | safe.ai/research initially failed. www.safe.ai redirected to safe.ai; canonical [research page](https://safe.ai/work/research) readable. | [MASK](https://www.mask-benchmark.ai/) project page readable. Separating honesty from accuracy reinforces the need for distinct outcome metrics; it does not supply hardware reliability evidence. |
| ARC | www.alignment.org/research initially failed; alignment.org redirected to www.alignment.org successfully. | [Mechanistic estimation for wide random MLPs](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/) readable. Structured mechanistic estimates motivate explicit assumptions; limited theoretical settings should not be treated as practical LLM integrity certificates. |
| Redwood Research | [Research page](https://www.redwoodresearch.org/research) readable. | [Adversarial Training for High-Stakes Reliability](https://arxiv.org/abs/2205.01663), NeurIPS 2022 as listed by the organization. Reliability under challenging evaluation is relevant; intentional agent subversion differs from hardware SDC. No attack procedures were reproduced. |
| METR | [Research page](https://metr.org/research/) and specific article readable. | [Measuring Automated Kernel Engineering](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/) explicitly discusses H100 kernel tasks and evaluation limits. End-to-end model behavior and training stability cannot be inferred from isolated-task success. |
| Apollo Research | [Science page](https://www.apolloresearch.ai/science) readable. | [Chain-of-Thought Monitorability: A New and Fragile Opportunity for AI Safety](https://www.apolloresearch.ai/science/chain-of-thought-monitorability-a-new-and-fragile-opportunity-for-ai-safety) readable. Monitoring assumptions are fragile; semantic oversight should not substitute for computational integrity. |
| UK AISI | [Research page](https://www.aisi.gov.uk/research) returned a JavaScript notice but substantial research text was readable; specific page readable. | [Practical challenges of control monitoring in frontier AI deployments](https://www.aisi.gov.uk/research/practical-challenges-of-control-monitoring-in-frontier-ai-deployments) distinguishes synchronous, semi-synchronous and asynchronous oversight. Latency and recovery trade-offs motivate explicit commit semantics, not a new hardware mechanism by themselves. |
| Anthropic Alignment Science | [Index](https://alignment.anthropic.com/) and specific article readable. | [Would This Change Your Answer?](https://alignment.anthropic.com/2026/chive/) evaluates explanations through counterfactual experiments. Use interventions to test causal claims; do not accept plausible interpretations as validated recovery dependencies. |
| Transformer Circuits | [Index](https://transformer-circuits.pub/) readable. A link to Tracing Attention Computation Through Feature Interactions failed; fallback below succeeded. | [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) readable. Attribution graphs offer mechanistic hypotheses, not an exact numerical dependency certificate; approximate explanations cannot justify silently skipping required recovery work. |

## Recommendation to the main investigation

Advance candidate 1 only as a hypothesis to falsify, with HybridServe/KVPR explicitly in the novelty matrix and valid-prefix replay as the primary baseline. Candidate 2 is a useful alternative or supporting subproblem but presently less differentiated. Do not begin a polished prototype, invent an acronym, promise a publishable result, or call either direction genuinely novel before the remaining classical recovery literature and initial break-even tests are resolved.

The research-site survey supports careful causal evaluation, explicit monitor assumptions and separate outcome metrics. It does not justify forcing an alignment-themed probe into an otherwise hardware-systems thesis.

## Classical recovery follow-up: further novelty constraints

A first targeted follow-up found [On the Combination of Silent Error Detection and Checkpointing](https://arxiv.org/abs/1310.8486), whose author record reports PRDC 2013 acceptance. Its abstract already covers delayed silent-error detection, finite checkpoint retention, irrecoverability risk, verification cost and optimal checkpoint periods. Therefore bounded delayed detection, retaining clean checkpoints and optimizing their frequency are not new. This was an abstract-level screen, not a proof audit.

The publisher-indexed [Multi-level checkpointing and silent error detection for linear workflows](https://www.sciencedirect.com/science/article/abs/pii/S1877750317303599) additionally describes task-level partial/guaranteed verification and memory checkpoints. Direct access returned HTTP 403; only the publisher search excerpt was accessible, so its algorithms remain unread. Author-hosted [Combining Checkpointing and Replication for Reliable Execution of Linear Workflows](https://icl.utk.edu/files/publications/2019/icl-utk-1175-2019.pdf) was found, but only a search excerpt was screened. These are required deeper baselines before presenting layer-cut placement as new.

This further narrows candidate 1: the potential contribution must exploit and prove a transformer-specific two-dimensional layer/token recovery property, then demonstrate a useful measured operating region beyond established checkpoint placement and exact activation-assisted reconstruction. Merely adding a delayed detector, a clean-state flag or output buffering to an ordinary checkpoint system would not justify the thesis claim.

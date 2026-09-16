# 100-paper literature review

Derived from the three paper-review ledgers on 2026-09-16. This document makes the recorded screening readable; it does not add new review claims or change the source ledgers.

## Scope and reading depth

All 100 supplied papers received selected-section screening of their extracted full text. Eighteen closer neighbors received focused deeper review; the other 82 received section-level screening. “Focused” does not mean every line, proof, figure or appendix was verified. Page references use physical PDF/extraction-page markers, which may differ from printed article page numbers. Reported measurements are authors’ claims unless independently reproduced elsewhere.

The 18 focused-review IDs are 015, 017, 023, 024, 027, 028, 030, 052, 058, 065, 068, 069, 073, 075, 078, 080, 082 and 085. No paper’s reading depth is upgraded by its inclusion here.

This screen informs a provisional research assessment, **not novelty clearance**. It is not an exhaustive systematic review, publisher authentication exercise or replication of the 100 papers.

## Source-integrity warnings

Records **036 and 039 require canonical source verification**: their supplied first-page arXiv identifiers encode July 2026, but the displayed dates are in May 2026. Their findings and bibliographic details remain provisional. Other venue/version uncertainties are preserved per paper. Submission, a workshop header and a placeholder proceedings notice do not establish main-conference acceptance.

Sources below use each ledger’s URL, supplemented from the [download manifest](../../Research%20Papers%20PDFs/download_manifest.csv) where absent. Record 056 differs from its manifest only in title capitalization (“DeepServe” versus “DEEPSERVE”). The [coverage audit](../notes/literature_coverage.md) gives the complete integrity checks and website-access mapping.

The original ledgers remain unchanged: [001–034](systems_001_034.json), [035–067](safety_035_067.json), [068–100](reliability_068_100.json).

## Paper-by-paper records

### 001. Don’t Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference

Source: [Paper](https://arxiv.org/pdf/2609.05275)

Publication status: arXiv preprint

- Contribution: Joint layer-dropout scheduling and hyperparameter tuning; 2,400 runs from 271M to 8.2B parameters; reports up to 25% training-FLOP reduction and 1.55× inference speed.
- Limitations and evidence: Full pretraining uses substantial Cerebras compute; no fault-resilience evaluation. Aggregate efficiency does not establish dependable graceful degradation.
- Possible connection: Layer dropping and depth elasticity alone are not a novel resilience mechanism.
- Page references: pp. 1–3 (overview), 5 (tuning), 17 (conclusion)
- Review depth: Section screening; selected extracted-text sections, not full-line review

### 002. CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark

Source: [Paper](https://aclanthology.org/2026.acl-long.1592.pdf)

Publication status: ACL 2026 long paper; official ACL Anthology PDF

- Contribution: Paired CUDA/HIP and SASS/RDNA3 corpus, models and 369-task benchmark for NVIDIA-to-AMD transpilation; 60,694 aligned programs.
- Limitations and evidence: Correctness is relative to the benchmark inputs and supported source/assembly environments, not proof for arbitrary inputs or hardware. Runtime/memory parity claims are task-conditioned.
- Possible connection: Hardware portability is closely aligned with Mahmoud; reliability would require a contribution beyond translation and test-suite passing.
- Page references: pp. 1–2, 5, 8
- Review depth: Section screening

### 003. Beyond Prediction: Tail-Aware Scheduling for LLM Inference

Source: [Paper](https://arxiv.org/pdf/2606.18431)

Publication status: arXiv; supplied PDF declares ICML 2026, venue not independently verified

- Contribution: UNIBOOST uses prediction-free priority shaping, lightweight distribution statistics and KV-aware preemption; reports improved P99 completion latency over SRPT.
- Limitations and evidence: Results depend on arrival distributions, scheduler settings and preemption costs; no state-integrity mechanism.
- Possible connection: Necessary latency/SLO baseline if adding recovery work to an inference server.
- Page references: pp. 1, 5, 9
- Review depth: Section screening

### 004. C2CServe: Leveraging NVLink-C2C for Elastic Serverless LLM Serving on MIG

Source: [Paper](https://arxiv.org/pdf/2605.19481v1)

Publication status: arXiv preprint; placeholder conference text is not venue verification

- Contribution: C2CServe streams CPU-resident weights into GH200 MIG instances using hybrid HBM/C2C execution and hierarchical scheduling.
- Limitations and evidence: Specific NVLink-C2C hardware and shared-link contention matter; cold-start benefits do not establish numerical reliability.
- Possible connection: Shows why resource placement and data movement must be measured in a recovery system, not treated as free.
- Page references: pp. 1, 5, 13
- Review depth: Section screening

### 005. Cascade: Exploiting SLO-Aware Latency Budget for Fair and High Goodput LLM Inference Serving

Source: [Paper](https://arxiv.org/pdf/2608.06557v1)

Publication status: arXiv preprint

- Contribution: Cascade allocates SLO-aware latency budgets across queuing, restoring, prefetching, retaining and recomputing KV state.
- Limitations and evidence: Goodput depends on workload and remaining-cost estimates. No proof of cache-content correctness.
- Possible connection: Budget-aware recomputation scheduling already exists; recovery scheduling alone would be incremental.
- Page references: pp. 1, 5
- Review depth: Section screening

### 006. Efficient Multi-round LLM Inference over Disaggregated Serving

Source: [Paper](https://arxiv.org/pdf/2602.14516)

Publication status: arXiv; supplied PDF declares ICML 2026, venue not independently verified

- Contribution: AMPD routes incremental prefill locally or to a disaggregated prefill pool; combines request reordering and deployment planning.
- Limitations and evidence: Offline planner and workload assumptions limit dynamic generality; no silent-data-corruption treatment.
- Possible connection: Reusing multi-turn state and deciding when to recompute are crowded; include routing/transfer costs in later scaling.
- Page references: pp. 1–2, 5, 9
- Review depth: Section screening

### 007. Geometry-Aware Online Scheduling for LLM Serving: From Theoretical Bound to System Practice

Source: [Paper](https://arxiv.org/pdf/2606.22327)

Publication status: arXiv preprint

- Contribution: Smallest-Volume-First and one-bit variants use request memory–time geometry; supplies competitive analysis and vLLM implementation.
- Limitations and evidence: Competitive bounds are conditional on the formal workload/information model; single-worker evidence is not distributed reliability evidence.
- Possible connection: Useful scheduling theory baseline, not a new state-integrity problem.
- Page references: pp. 1–3, 5, 10
- Review depth: Section screening

### 008. Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control

Source: [Paper](https://arxiv.org/pdf/2602.02987)

Publication status: arXiv preprint

- Contribution: Fluid/queueing control for heterogeneous prefill/decode contention, revenue and SLI fairness using empirical A100 service rates.
- Limitations and evidence: Large-scale evidence is substantially model/calibrated-simulation based; asymptotic optimality requires stated scaling assumptions.
- Possible connection: Recovery work changes queueing load; cannot infer tail guarantees from a per-request repair speedup.
- Page references: pp. 1–3, 5, 31
- Review depth: Section screening of a long paper; not a full mathematical audit

### 009. MoEless: Efficient MoE LLM Serving via Serverless Computing

Source: [Paper](https://arxiv.org/pdf/2603.06350)

Publication status: arXiv preprint

- Contribution: MoEless uses layer-aware expert load prediction and serverless replica placement; reports an eight-GPU prototype.
- Limitations and evidence: Expert-footprint and load-prediction assumptions constrain transferability; not fault tolerance.
- Possible connection: Expert replication is not a new reliability thesis without integrity semantics.
- Page references: pp. 1, 5, 11
- Review depth: Section screening

### 010. OpScale: Operator-level Provisioning and Autoscaling for LLM Serving

Source: [Paper](https://arxiv.org/pdf/2608.13499)

Publication status: arXiv preprint

- Contribution: Operator-level profiling, provisioning, placement and autoscaling; evaluation includes 40 A100 and 24 GB200 GPUs.
- Limitations and evidence: Large-cluster claims cannot be reproduced on one GPU; profile and communication-cost assumptions are material.
- Possible connection: Operator-level resource allocation is prior art; a recovery mechanism needs gains beyond ordinary autoscaling.
- Page references: pp. 1, 6–7 (excerpts), 14 (conclusion excerpt)
- Review depth: Section screening

### 011. Prefill/Decode-Aware Evaluation of LLM Inference on Emerging AI Accelerators

Source: [Paper](https://arxiv.org/pdf/2606.17104)

Publication status: arXiv preprint

- Contribution: Separates prefill and decode when comparing Llama-2-7B on A100 and Groq; highlights single-stream latency versus batched throughput.
- Limitations and evidence: One model and platform-specific batching support; no universal accelerator winner follows.
- Possible connection: A thesis evaluation must separate TTFT, TPOT, batch throughput and hardware assumptions.
- Page references: pp. 1, 5
- Review depth: Section screening

### 012. Scalable Joint Resource Allocation for SLO-Constrained LLM Inference in Heterogeneous GPU Clouds

Source: [Paper](https://arxiv.org/pdf/2604.07472)

Publication status: arXiv preprint

- Contribution: Joint model, GPU, parallelism and routing allocation using MILP and scalable heuristics.
- Limitations and evidence: Near-optimality is relative to a calibrated resource/SLO model and synthetic scenarios; not real-time fault handling.
- Possible connection: Avoid repackaging profile-driven resource optimization as reliability.
- Page references: pp. 1, 5, 13
- Review depth: Section screening

### 013. FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration

Source: [Paper](https://aclanthology.org/2026.findings-acl.1610.pdf)

Publication status: ACL 2026 Findings; official ACL Anthology PDF

- Contribution: FastKV decouples context reduction for prefill from KV compression for decode via token-selective propagation.
- Limitations and evidence: Importance stabilization and aggregate quality are model/task dependent; discarded information is not formally protected.
- Possible connection: Selective propagation is an efficiency comparator; approximate token omission must not be labeled exact repair.
- Page references: pp. 1, 5, 8
- Review depth: Section screening

### 014. From Bits to Chips: An LLM-based Hardware-Aware Quantization Agent for Streamlined Deployment of LLMs

Source: [Paper](https://arxiv.org/pdf/2601.03484)

Publication status: arXiv preprint

- Contribution: HAQA uses an LLM agent to search hardware-aware quantization/deployment configurations across CNN and language-model examples.
- Limitations and evidence: Mixed experimental settings and hardware-specific tuning; no sound end-to-end quality contract. Figure typography was not visually verified.
- Possible connection: An agent wrapper around quantization search is not a sufficient new thesis mechanism.
- Page references: pp. 1, 5 (figure extraction limited), 6–7 (excerpts), 9
- Review depth: Section screening; one extracted figure was not interpretable

### 015. InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models

Source: [Paper](https://arxiv.org/pdf/2602.23200)

Publication status: arXiv preprint

- Contribution: InnerQ groups the inner GEMV dimension (K per token, V per channel), uses fused dequantization, adaptive symmetric/asymmetric groups, and FP16 sink/recent windows.
- Limitations and evidence: Quality experiments and latency evidence use different settings: simulated quantization on task suites versus batch-one Jetson Xavier NX microkernels. Metadata, zero points and preserved windows must be included in effective bit cost; not an end-to-end server speed claim.
- Possible connection: Quantized layout and metadata can affect corruption propagation, but generic KV quantization and kernel fusion are established here.
- Page references: pp. 1, 5–15
- Review depth: Focused main-text review; pp. 2–4 and final references/appendix not fully read

### 016. Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware

Source: [Paper](https://arxiv.org/pdf/2607.17283)

Publication status: arXiv preprint

- Contribution: Studies speculative decoding on consumer Apple hardware; verifier/drafter bottlenecks explain why only some model pairings accelerate.
- Limitations and evidence: Distribution tests and greedy agreement are finite empirical checks, not universal implementation proofs; Metal execution limits are hardware-specific.
- Possible connection: Charge verification and recovery overhead; mathematical losslessness does not imply practical speedup.
- Page references: pp. 1, 5, 12
- Review depth: Section screening

### 017. ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM Agents

Source: [Paper](https://arxiv.org/pdf/2608.19662v1)

Publication status: arXiv preprint

- Contribution: ReCache isolates tool/resource attention, resets local positions, trains reusable resource representations, and prunes using leave-one-resource-out invocation loss.
- Limitations and evidence: Requires fine-tuning and changed attention structure. Reported 7.2 versus 26.3 ms TTFT excludes offline cache build; OOD invocation F1 falls relative to the uncompressed model. General dialogue reuse is not established.
- Possible connection: Persistent reusable tool KV state is already studied; use as a possible workload, not the novelty claim.
- Page references: pp. 1, 3–9
- Review depth: Focused main-text review; p. 2 and appendices not fully read

### 018. The Diminishing Returns of Early-Exit Decoding in Modern LLMs

Source: [Paper](https://arxiv.org/pdf/2603.23701)

Publication status: arXiv preprint

- Contribution: Benchmarks early-exit adaptability and oracle upper bounds; newer architectures/checkpoints often offer less redundant depth.
- Limitations and evidence: Oracle opportunities are not deployable decision policies, and selected checkpoints do not establish universal trends.
- Possible connection: Do not assume a cheap early-exit detector or large spare layer budget.
- Page references: pp. 1, 5, 8
- Review depth: Section screening

### 019. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Source: [Paper](https://arxiv.org/pdf/2504.19874)

Publication status: arXiv first submitted 2025; supplied filename year is not publication verification

- Contribution: TurboQuant combines random rotations, scalar quantization and residual inner-product correction with near-optimal vector-distortion analysis.
- Limitations and evidence: Vector distortion and inner-product properties do not directly imply full-generation or application correctness; metadata and implementation precision matter.
- Possible connection: Any KV-risk certificate must distinguish local numerical distortion from trajectory-level correctness.
- Page references: pp. 1, 5
- Review depth: Section screening; theorem proofs not audited

### 020. Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use

Source: [Paper](https://arxiv.org/pdf/2601.22362)

Publication status: arXiv preprint

- Contribution: Measures precision, batching and arrival-shaping effects on H100/TGI energy; compute-bound prefill and decode respond differently.
- Limitations and evidence: Large energy ratios include utilization/arrival effects rather than a single algorithmic gain; dequantization can hurt decode.
- Possible connection: Measure energy per completed request with latency and load held comparable, not merely kernel FLOPs.
- Page references: pp. 1, 5–6, 9
- Review depth: Section screening

### 021. Coordinating GPU Data Centers and Power Grid Regulation Service for Exogenous Carbon Benefits

Source: [Paper](https://arxiv.org/pdf/2601.22487)

Publication status: arXiv; supplied PDF declares ICS 2026, venue not independently verified

- Contribution: EcoCenter coordinates GPU power modulation with grid-frequency regulation and evaluates exogenous carbon benefits.
- Limitations and evidence: Counterfactual grid/reserve assumptions drive carbon claims; not a study of numerical faults or cache recovery.
- Possible connection: Peripheral motivation only; no reason to force power-grid objectives into an inference-integrity thesis.
- Page references: pp. 1, 5, 11
- Review depth: Section screening

### 022. DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers

Source: [Paper](https://arxiv.org/pdf/2601.16956)

Publication status: arXiv preprint

- Contribution: Composable StateProviders separate state abstraction from movement; nonblocking lazy snapshots exploit weight immutability and coalesced staging.
- Limitations and evidence: Checkpoint throughput assumes capture consistency/overlap; experiments up to 70B and 256 A100s do not validate state content or detect SDC.
- Possible connection: A strong baseline for checkpoint plumbing; copying an already incorrect state does not make it trustworthy.
- Page references: pp. 1, 5–7 (selected excerpts)
- Review depth: Section screening

### 023. KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

Source: [Paper](https://arxiv.org/pdf/2607.16241v1)

Publication status: arXiv preprint

- Contribution: KernelBench-Verified strengthens baseline precision settings, hidden input transformations and memory measurement for seven models on a 250-task-derived suite.
- Limitations and evidence: Fixed shapes remain legitimate specialization; finite transformed inputs and a fixed tolerance do not establish arbitrary-input correctness. Maximum allocation measures space, not memory traffic; no training or cross-GPU compositional guarantee.
- Possible connection: Generic hidden-test kernel verification is already occupied; declare the numerical baseline contract and include whole-model costs.
- Page references: pp. 1–8
- Review depth: Focused full main-text review; appendices not fully read

### 024. LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training

Source: [Paper](https://arxiv.org/pdf/2604.10390)

Publication status: arXiv preprint

- Contribution: LLM-PRISM maps permanent GPU fault signatures into structured tensor-level injections and studies full-training trajectories, including silent downstream degradation despite benign perplexity.
- Limitations and evidence: GPT-2-scale models/data and a single-defect activation model limit transfer. Proprietary/raw RTL signature availability was not established; causal datatype explanations should not be generalized beyond measured evidence.
- Possible connection: Realistic finite, correlated faults and application-level outcomes are essential for a dependable-inference thesis; random single-bit tests alone are inadequate.
- Page references: pp. 1, 3–10
- Review depth: Focused main-text review; p. 2 and final pages not fully read

### 025. PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint

Source: [Paper](https://arxiv.org/pdf/2607.01646)

Publication status: arXiv preprint

- Contribution: PHOENIX combines remote-memory checkpoints, hot spares, communicator reconstruction and overlapped checkpointing for rapid training recovery.
- Limitations and evidence: Detected fail-stop failures differ from finite SDC. Zero-overhead claims depend on overlap and reserve-resource accounting.
- Possible connection: Generic fast checkpoint/restart is prior art; same checkpoint resources must be given to a recovery baseline.
- Page references: pp. 1, 5
- Review depth: Section screening

### 026. ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload

Source: [Paper](https://arxiv.org/pdf/2605.11215)

Publication status: arXiv preprint

- Contribution: ReCoVer combines fault-tolerant collectives, in-step gradient recovery and workload redistribution over surviving GPUs.
- Limitations and evidence: Gradient/statistical equivalence is not necessarily bit-identical floating-point execution; tested failures are not general undetected corruption.
- Possible connection: Failure recovery needs a precise correctness target and explicitly scoped fault model.
- Page references: pp. 1, 5, 9
- Review depth: Section screening

### 027. The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance

Source: [Paper](https://arxiv.org/pdf/2605.04213)

Publication status: arXiv preprint

- Contribution: Gate-level study of a reduced production GPU yields structured finite SDC patterns: many multibit corruptions, nullification, unit-dependent bit positions and warp periodicity.
- Limitations and evidence: Single stuck-at gate faults, selected microbenchmarks and conditioning on corruption events are not field failure rates. The unnamed reduced GPU and protected memory hierarchy constrain extrapolation.
- Possible connection: Use published structured fault families as clearly labeled models; do not call synthetic tensor perturbations physically faithful without validated mapping.
- Page references: pp. 1–7
- Review depth: Focused full main-text review

### 028. The Correctness Illusion in LLM-Generated GPU Kernels

Source: [Paper](https://arxiv.org/pdf/2606.20128v2)

Publication status: arXiv preprint

- Contribution: Schema-driven operator testing uses boundary shapes, high-precision references, per-operator/dtype tolerances and replay across five GPUs.
- Limitations and evidence: Many errors are seeded and some tests use stand-ins; this is not a measured natural error prevalence for all generated kernels. Finite tests and cross-hardware agreement are not proofs.
- Possible connection: An improved generated-kernel fuzzer alone would be a weak novelty claim.
- Page references: pp. 1–8
- Review depth: Focused full main-text review; references/appendix not fully read

### 029. Training LLMs with Fault Tolerant HSDP on 100,000 GPUs

Source: [Paper](https://arxiv.org/pdf/2602.00277)

Publication status: arXiv preprint

- Contribution: Fault-tolerant HSDP uses data-replica failure units, dynamic membership and nonblocking recovery/catch-up at reported 100,000-GPU scale.
- Limitations and evidence: Scale and failure rates are deployment-specific; smaller-scale accuracy evidence and detected failure handling do not establish SDC protection.
- Possible connection: Useful context for major fault-tolerance problems, not a feasible first experimental system for one GPU.
- Page references: pp. 1, 5, 13
- Review depth: Section screening

### 030. V-ABFT: Variance-Based Adaptive Threshold for Fault-Tolerant Matrix Multiplication in Mixed-Precision Deep Learning

Source: [Paper](https://arxiv.org/pdf/2602.08043)

Publication status: arXiv preprint

- Contribution: V-ABFT models checksum-difference variance directly, calibrates hardware error, and computes checksums before low-precision output rounding to improve mixed-precision fault detection.
- Limitations and evidence: Practical thresholds use independence/approximately Gaussian assumptions, a 2.5-sigma multiplier and empirical hardware calibration. Zero observed false positives is not distribution-free soundness. Low-bit faults remain difficult; exact correction assumes at most one error per row.
- Possible connection: Mandatory producer-boundary baseline for compute-origin KV errors. A tighter threshold or ordinary ABFT wrapper is not sufficient novelty.
- Page references: pp. 1–7
- Review depth: Focused full main-text review

### 031. Beyond Linear Probes: Dynamic Safety Monitoring for Language Models

Source: [Paper](https://arxiv.org/pdf/2509.26238)

Publication status: Supplied PDF states ICLR 2026; external venue independently unverified

- Contribution: TPC dynamically evaluates progressively higher-order polynomial activation probes, with early exits for easy cases.
- Limitations and evidence: Harmful-prompt classification is not response or hardware-error monitoring; finite benchmark evidence does not settle shifted/rare-positive generalization.
- Possible connection: Adaptive semantic probes already exist, and should not be presented as reliable hardware integrity witnesses without validation.
- Page references: pp. 1, 5, 10
- Review depth: Section screening

### 032. Online Safety Monitoring for LLMs

Source: [Paper](https://arxiv.org/pdf/2607.02510)

Publication status: Supplied PDF states ICML 2026 Hypothesis Testing workshop

- Contribution: Online threshold monitoring with calibrated external verification signals; compares simple constant thresholds with sequential alternatives.
- Limitations and evidence: Calibration and signal quality are essential; token likelihood can be weak, and expensive external verifiers must be charged. Workshop status is not a main-conference acceptance.
- Possible connection: A risk monitor or threshold gate alone is a crowded direction; hardware recovery needs a distinct mechanism.
- Page references: pp. 1, 5 (results/conclusion)
- Review depth: Section screening

### 033. Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection

Source: [Paper](https://arxiv.org/pdf/2601.12033)

Publication status: arXiv preprint

- Contribution: Protects critical weights at higher precision using contrastive/gradient sensitivity, studying multilingual fairness and safety under quantization.
- Limitations and evidence: Weight-focused, finite benchmark and judge-based evaluation; runtime hardware cost and broad distribution-shift protection need separate evidence.
- Possible connection: Critical-weight precision protection is prior art; a KV-state study must show a fundamentally different persistent-state problem.
- Page references: pp. 1, 5 (tables), 9
- Review depth: Section screening

### 034. ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions

Source: [Paper](https://arxiv.org/pdf/2608.10621)

Publication status: arXiv preprint

- Contribution: ProbGuard estimates calibrated risk from probability-weighted early-token representations and Monte Carlo continuation labels.
- Limitations and evidence: Offline labeling can be costly; finite prefix calibration does not imply universal shift robustness. No hardware-fault or state-repair semantics.
- Possible connection: A calibrated early warning wrapper is not the desired dependable-systems thesis.
- Page references: pp. 1, 5, 7
- Review depth: Section screening

### 035. Pruning Unsafe Tickets: A Resource-Efficient Framework for Safer and More Robust LLMs

Source: [Paper](https://aclanthology.org/2026.acl-long.1209.pdf)

Publication status: ACL 2026 long-paper header in supplied PDF

- Contribution: Gradient-free, model-specific attribution plus iterative pruning removes parameters associated with unsafe behavior; works with quantized variants.
- Limitations and evidence: Authors: beam search costs more than greedy/one-pass; static token attribution may miss semantics/context.
- Possible connection: Already occupies lightweight safety pruning; not a persistent-state repair method.
- Page references: 1, 3, 9
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 036. RAIL Guard: Closing the Evaluation-to-Remediation Gap in Responsible AI for LLM Agents

Source: [Paper](https://arxiv.org/pdf/2607.16215)

Publication status: Preprint; supplied first-page arXiv identifier/date appear inconsistent and need publisher verification

Metadata warning: identifier/date inconsistency remains unresolved; verify the canonical publisher/arXiv record before relying on this citation.

- Contribution: Eight-dimension evaluate-rewrite-reevaluate pipeline; reports 96.9% convergence vs 49.1% block/retry, with a 22.3% utility penalty for its strongest convergence method.
- Limitations and evidence: Authors: model-judge dependence, 12.4% regression among remediation failures, English scope, small domain samples.
- Possible connection: Output remediation and pre-action checks overlap generic ProofOps framing; no numerical-state recovery mechanism.
- Page references: 1, 3, 12
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 037. Robust and Efficient Guardrails with Latent Reasoning

Source: [Paper](https://arxiv.org/pdf/2605.29068)

Publication status: arXiv:2605.29068v2, 3 Sep 2026; no acceptance established

- Contribution: CoLaGuard replaces explicit reasoning with recurrent latent states learned through staged supervision; reports matched GuardReasoner macro-F1 and 12.9x speedup.
- Limitations and evidence: Authors: text-only harmfulness detection; no broader policy, multilingual, multimodal, adaptive or long-horizon evaluation; inherited supervision bias; more causal interventions needed.
- Possible connection: Evidence that efficient moderation is crowded; latent-state execution fidelity is an adjacent systems question, not yet established here.
- Page references: 1, 3, 9
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 038. Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets

Source: [Paper](https://aclanthology.org/2026.acl-long.756.pdf)

Publication status: ACL 2026 long-paper header

- Contribution: Relates upstream alignment/downstream fine-tuning representation similarity to safety degradation; lower similarity improves robustness.
- Limitations and evidence: Reviewer inference: concerns dataset selection and weight adaptation, not fixed-weight execution transformations; observational correlation alone is not a numerical fault mechanism.
- Possible connection: Prevents treating benign adaptation-induced safety drift as new; possible methodological control for weight changes.
- Page references: 1, 3, 9
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 039. Benchmarking Confidential GPU Inference on NVIDIA H100 under Intel TDX

Source: [Paper](https://arxiv.org/pdf/2607.19353)

Publication status: Preprint; supplied identifier/date inconsistency requires verification

Metadata warning: identifier/date inconsistency remains unresolved; verify the canonical publisher/arXiv record before relying on this citation.

- Contribution: Compares confidential/nonconfidential execution for Mistral-7B and Qwen3-30B-A3B on one H100; measures saturation and latency/throughput penalty.
- Limitations and evidence: Authors: two models/one setup, Secure Boot disabled, no low-level profiling to attribute overhead.
- Possible connection: Useful end-to-end cost-accounting example; insufficient novelty foundation for dependable semantic execution.
- Page references: 1, 3, 8
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 040. CloakLM: Obfuscating GPU Memory Layout to Mitigate Model Ex-filtration for Serving

Source: [Paper](https://arxiv.org/pdf/2606.18400)

Publication status: Preprint 2026 header

- Contribution: Software memory-obfuscation across transfer layout, weight placement and physical HBM pages while preserving authorized logical views.
- Limitations and evidence: Authors' scope: trusts serving OS/runtime/metadata and does not defend full serving-stack compromise. Reviewer: obfuscation is not a cryptographic guarantee.
- Possible connection: Privacy defense, not benign numerical reliability; do not convert its threat model into an offensive reproduction.
- Page references: 1, 3
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 041. DP-Fusion: Token-Level Differentially Private Inference for Large Language Models

Source: [Paper](https://arxiv.org/pdf/2507.04531)

Publication status: ICLR 2026 conference-paper header

- Contribution: Blends private-context and redacted-context output distributions to bound tagged tokens' influence, with token-group privacy accounting.
- Limitations and evidence: Authors: guarantees cover tagged spans only; m groups need m+1 distributions per step; single-group implementation trades strength for smoother/cheaper behavior.
- Possible connection: Strong example of stating a precise distributional contract and compute cost; formal privacy is different from restoring exact numerical execution.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 042. EncFormer: Secure and Efficient Transformer Inference over Encrypted Data

Source: [Paper](https://arxiv.org/pdf/2604.09975)

Publication status: Submitted to IEEE TDSC, arXiv:2604.09975v1; acceptance not established

- Contribution: Co-designs compatible ciphertext layouts, CKKS/MPC conversions and nonlinear kernels; reports communication and latency gains on GPT2/BERT.
- Limitations and evidence: Scope: semi-honest two-party setting and selected GLUE accuracy; submitted manuscript rather than established accepted venue.
- Possible connection: Cross-stage representation contracts are prior art; approximate cryptographic arithmetic motivates distinguishing functional quality from exactness.
- Page references: 1, 3, 13
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 043. Gecko: Fast Private Inference via Secure Public Encoder Offloading

Source: [Paper](https://arxiv.org/pdf/2608.02378)

Publication status: arXiv:2608.02378v1 preprint

- Contribution: Public frozen encoder plus projections and a private gated predictor reduces encrypted computation; assesses extra extraction risk from reusing the public encoder.
- Limitations and evidence: Authors: evaluated CNN image/audio classifiers; transformer/generative extension introduces richer output and predictor challenges.
- Possible connection: Low relevance to KV fault recovery; valuable example of narrowing a guarantee to precisely evaluated leakage.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 044. Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference

Source: [Paper](https://arxiv.org/pdf/2608.09225)

Publication status: Preprint; venue not established

- Contribution: KVGov uses principal-scoped cache identities and discusses different cache matching mechanisms; isolation is the central defense.
- Limitations and evidence: Authors: defense and attack-rate experiments mainly simulation; boundary-salt efficiency extrapolated; semantic caches require separate isolation mechanism.
- Possible connection: State-sharing isolation is already addressed; separate namespace/integrity problems from numerical state repair. No attack reproduction proposed.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 045. Privacy from Symmetry: Orthogonally Equivariant Transformers for LLM Inference

Source: [Paper](https://arxiv.org/pdf/2606.16461)

Publication status: Preprint; venue not established

- Contribution: ConjFormer enables inference in a rotated hidden-state basis using scalar RMSNorm and weight conjugation; fine-tuned small-model experiments.
- Limitations and evidence: Authors explicitly disclaim formal cryptographic guarantees; purpose is to remove direct nearest-neighbor inversion in original coordinates.
- Possible connection: Relevant to semantics-preserving transformations but not a general security or finite-precision exactness guarantee.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 046. Scaling up FHE-based Privacy-Preserving ML: Higher Throughput, Longer Inputs for LLama-3-8B

Source: [Paper](https://arxiv.org/pdf/2601.18511)

Publication status: Preprint; venue not established

- Contribution: Sylph co-designs outlier control, polynomial nonlinear evaluation and mixed public/private context processing for encrypted Llama inference.
- Limitations and evidence: Scope: long-context setting encrypts only last prompt portion; 8 RTX PRO 6000 GPUs; ciphertext approximation/domain assumptions matter.
- Possible connection: Useful contrast: practical confidentiality requires operator and precision co-design; hardware need makes this a poor first one-GPU thesis direction.
- Page references: 1, 3, 13
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 047. Your Inference Request Will Become a Black Box: Confidential Inference for Cloud-based Large Language Models

Source: [Paper](https://aclanthology.org/2026.acl-long.4.pdf)

Publication status: ACL 2026 long-paper header

- Contribution: Talaria partitions inference between a client-verified confidential VM and cloud, using reversible masking; reports identical outputs and lower token recovery.
- Limitations and evidence: Authors: honest-but-curious cloud that executes protocol correctly. Reviewer: bitwise claims apply to evaluated protocol/setup, not arbitrary faults.
- Possible connection: Exact output preservation is already a systems claim in confidentiality; does not address delayed internal-state corruption.
- Page references: 1, 3, 9
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 048. AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis

Source: [Paper](https://arxiv.org/pdf/2603.03378)

Publication status: Preprint dated Mar 16 2026

- Contribution: Read-write separated multi-agent diagnostic runtime, GRPO-trained observer, failure-trajectory correction; AIOpsLab evaluation.
- Limitations and evidence: Authors: production validation and broader benchmarks remain future work. Distinguish best@5 from avg@1/avg@5.
- Possible connection: Strong overlap with generic operational-agent repair; avoids assuming wrapper-level permission controls are a new scientific contribution.
- Page references: 1, 3, 15
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 049. Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization

Source: [Paper](https://arxiv.org/pdf/2602.04900)

Publication status: arXiv:2602.04900v3 preprint

- Contribution: Industry integration/evaluation of Kueue, accelerator slicing and inference-aware routing for Whisper and summarization.
- Limitations and evidence: Scope: illustrative pipeline and component workloads. Reviewer: a performance-integration study, no semantic correctness or numerical fault assurance.
- Possible connection: Use realistic scheduling context only; reproducing orchestration improvements alone is weak novelty.
- Page references: 1, 3, 6, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 050. Stalled, Biased, and Confused: Uncovering Reasoning Failures in LLMs for Cloud-Based Root Cause Analysis

Source: [Paper](https://arxiv.org/pdf/2601.22208)

Publication status: FORGE 2026, DOI 10.1145/3793655.3793732 in PDF

- Contribution: Controlled RCA evaluation isolates reasoning failures across six models, two agent workflows and two cases; 16-category taxonomy.
- Limitations and evidence: Authors: one known fault at a time; noisy alerts; sensitivity to phrasing/representation; LLM-judge limitations.
- Possible connection: Supports controlled mechanism experiments and confound isolation; not a GPU-state recovery system.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 051. TopoEvo: A Topology-Aware Self-Evolving Multi-Agent Framework for Root Cause Analysis in Microservices

Source: [Paper](https://arxiv.org/pdf/2605.15611)

Publication status: Preprint; venue not established

- Contribution: Combines multimodal alignment, topology embeddings/symptom tokens, hypothesis-evidence-test reasoning and incident-memory adaptation.
- Limitations and evidence: Authors: stronger intervention-aware causal validation, safer cost-aware adaptation and broader platforms left future work.
- Possible connection: Topology-aware symptom repair is not new at the application level; inferential topology constraints should not be called causal proof.
- Page references: 1, 3, 10
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 052. Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees

Source: [Paper](https://aclanthology.org/2025.findings-emnlp.1330.pdf)

Publication status: Findings of EMNLP 2025 header

- Contribution: Mahmoud coauthored system combines specialized small models, tokenizer changes, context extension and test validation for ISA transpilation.
- Limitations and evidence: Authors: tests/coverage do not guarantee full semantic equivalence; O2 hurts accuracy sharply (HumanEval ARMv8 99.39% O0 vs 45.12% O2); real BringUpBench 49.23% O0.
- Possible connection: Direct adviser alignment: model capability paired with explicit hardware/software correctness checks. Inspiration is contract clarity, not claiming tested outputs are formally proved.
- Page references: 1, 3, 4, 5, 6, 9
- Review depth: Focused full-text section review (method, results, limitations; not full line-by-line appendix review)

### 053. Wafer-Scale Systems: A Carbon Perspective

Source: [Paper](https://hotcarbon.org/assets/2025/paper-160.pdf)

Publication status: Supplied paper; venue not established by selected text

- Contribution: Mahmoud coauthored comparison of operational and embodied carbon for wafer-scale vs GPU LLM systems using total carbon-delay product.
- Limitations and evidence: Authors: yield assumptions; flash excluded for comparison (including maximum CS-3 flash changes embodied carbon greatly); utilization reporting uncertainty.
- Possible connection: Adviser alignment in cross-layer measurement and cost accounting. Recovery energy/cost could be secondary metric, not substitute for correctness mechanism.
- Page references: 1, 3, 6
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 054. BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching

Source: [Paper](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf)

Publication status: OSDI 2025 proceedings cover

- Contribution: Network-aware parameter multicast and layer-granular remote execution make autoscaling fast/live without many host copies.
- Limitations and evidence: Authors: scaling policy and elastic parallel configurations left future work. Reviewer: cached-model availability is distinct from request KV semantic recovery.
- Possible connection: Distributed load/live execution baseline; generic fast restart or layer-level scheduling alone not novel.
- Page references: 1, 2, 4, 15
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 055. Colocating ML Inference and Training with Fast GPU Memory Handover

Source: [Paper](https://www.usenix.org/system/files/atc25-wang-jiali.pdf)

Publication status: USENIX ATC 2025 proceedings cover

- Contribution: Sirius resizes training memory at millisecond scale, safely reclaims memory and reallocates under inference SLOs.
- Limitations and evidence: Reviewer scope limit: colocation resource safety and latency; not protection against silent numerical corruption. Hardware/workload effects need replication.
- Possible connection: Useful for journal memory and recovery interference costs; no proof that adding metadata journals is inexpensive.
- Page references: 1, 2, 4, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 056. DeepServe: Serverless Large Language Model Serving at Scale

Source: [Paper](https://www.usenix.org/system/files/atc25-hu-junhao.pdf)

Publication status: USENIX ATC 2025 proceedings cover

- Contribution: Production Ascend platform with request-job-task abstraction, FlowServe, mixed PD scheduling and rapid prewarmed scaling.
- Limitations and evidence: Scope: industrial architecture and deployment-specific components; first GPU experiment cannot reproduce full production scale.
- Possible connection: Important evidence inference is stateful/distributed; avoid promising a whole new serving platform as thesis novelty.
- Page references: 1, 2, 4, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 057. Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow

Source: [Paper](https://arxiv.org/pdf/2406.01566v2)

Publication status: ASPLOS 2025, DOI 10.1145/3669940.3707215

- Contribution: Joint model placement/request scheduling formulated as max-flow and solved via MILP; evaluates 24-42 heterogeneous GPU nodes.
- Limitations and evidence: Reviewer: optimality is with respect to formulated capacities/objective, not universal runtime or fault model.
- Possible connection: Hardware heterogeneity alone and topology-aware scheduling already mature; recovery needs a distinct correctness/cost objective.
- Page references: 1, 4, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 058. Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot

Source: [Paper](https://www.usenix.org/system/files/fast25-qin.pdf)

Publication status: FAST 2025 proceedings

- Contribution: Global distributed prefix KV pool uses CPU DRAM/SSD/network with disaggregated prefill/decode and cache-aware scheduling.
- Limitations and evidence: Authors implement transfer completion/error status and alternate paths for temporary NIC failures; this is not detection of semantically corrupted yet successfully transferred state. Public evaluation uses dummy architecture in some trace experiments.
- Possible connection: Closest systems motivation for persistent/reused KV. Prefix blocks are keyed by prefix content, may have replicas, and feed incremental prefill; repair descendants require separate semantics.
- Page references: 1, 2, 4, 5, 6, 7, 14
- Review depth: Focused full-text section review (method, results, limitations; not full line-by-line appendix review)

### 059. NanoFlow: Towards Optimal Large Language Model Serving Throughput

Source: [Paper](https://www.usenix.org/system/files/osdi25-zhu-kan.pdf)

Publication status: OSDI 2025 proceedings cover

- Contribution: Splits into nano-batches and overlaps compute, memory and communication within devices; automated search balances interference.
- Limitations and evidence: Reviewer: throughput-optimal model and common workloads do not establish deterministic outputs under changed reduction order or fault protection.
- Possible connection: Sets realistic high-utilization baseline; journals and replay must be costed against overlap already present.
- Page references: 1, 2, 4, 15
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 060. Resource Multiplexing in Tuning and Serving Large Language Models

Source: [Paper](https://www.usenix.org/system/files/atc25-he-yongjun.pdf)

Publication status: USENIX ATC 2025 proceedings cover

- Contribution: LLMStation couples iteration scheduling, suspendable autograd and batching tuning/inference to use spare resources while meeting latency SLOs.
- Limitations and evidence: Reviewer: shared-resource throughput gains do not establish numerical equivalence or state recovery after corruption.
- Possible connection: Shows coexistence/resource overhead to include; not a direct competing reliability mechanism.
- Page references: 1, 2, 4, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 061. Torpor: GPU-Enabled Serverless Computing for Low-Latency, Resource-Efficient Inference

Source: [Paper](https://www.usenix.org/system/files/atc25-yu.pdf)

Publication status: USENIX ATC 2025 proceedings cover in supplied PDF

- Contribution: Late binds models from main memory to GPUs; overlaps loading/execution and schedules with interference awareness.
- Limitations and evidence: Scope: model-swapping and latency/resource cost, not persistent KV consistency or delayed-error repair.
- Possible connection: Supports performance model for replay/journal transfer, not a novel safety claim.
- Page references: 1, 2, 4, 14
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 062. WaferLLM: Large Language Model Inference at Wafer Scale

Source: [Paper](https://www.usenix.org/system/files/osdi25-he.pdf)

Publication status: OSDI 2025 proceedings cover

- Contribution: PLMR hardware model guides wafer-scale parallelism and MeshGEMM/MeshGEMV on Cerebras WSE2.
- Limitations and evidence: Authors: current wafer-scale software limits; reviewer: raw comparisons span very different accelerator sizes and budgets.
- Possible connection: Useful adviser-adjacent architecture background; cannot claim feasibility from GPU-only resources.
- Page references: 1, 2, 4, 15
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 063. Weaver: Efficient Multi-LLM Serving with Attention Offloading

Source: [Paper](https://www.usenix.org/system/files/atc25-gao.pdf)

Publication status: USENIX ATC 2025 proceedings

- Contribution: Offloads hot-model attention to spare cold-model GPUs; GPU-driven control and operator splitting limit head-of-line blocking.
- Limitations and evidence: Reviewer: performance sharing and modest cold-model latency overhead, not integrity of remote KV or recovery semantics.
- Possible connection: Important reminder KV-only placement is possible because attention has no weights; remote-state fault recovery adds distinct requirements.
- Page references: 1, 2, 4, 8
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 064. Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching

Source: [Paper](https://arxiv.org/pdf/2504.06319)

Publication status: arXiv:2504.06319v2 preprint

- Contribution: Prefetches future KV blocks into GPU L2 to overlap loads and attention compute; H20 results vs XFormers/FA3.
- Limitations and evidence: Authors show diminishing I/O optimization scope as tensor parallelism shards heads; reviewer: H20/kernel-specific measurements.
- Possible connection: Any reliability scanning/replay competes with optimized memory path; asynchronous reads create lifetime boundaries that a protocol must specify.
- Page references: 1, 4, 7
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 065. BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

Source: [Paper](https://arxiv.org/pdf/2503.18773)

Publication status: arXiv:2503.18773v3 (5 Jan 2026); source filename year is initial version, acceptance not established

- Contribution: Layout induction, warp cooperation, online residual-block quantization and pipelined dequantization use Tensor/CUDA cores; multiple GPU generations.
- Limitations and evidence: Average LongBench quality is evaluated, not rare semantic or fault outcomes. Scale/zero metadata and packed KV are distinct state. Correct cooperative softmax is essential, with invalid output without it in ablation.
- Possible connection: Closest low-bit systems neighbor. Half-precision residual windows roll into low-bit packed state, so repair must restore data and quantizer metadata and respect quantization group boundaries.
- Page references: 1, 7, 8, 9, 11, 12
- Review depth: Focused full-text section review (method, results, limitations; not full line-by-line appendix review)

### 066. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

Source: [Paper](https://arxiv.org/pdf/2503.01840)

Publication status: arXiv:2503.01840v3 preprint in supplied copy

- Contribution: Token prediction from fused multi-layer target features and training-time tests improves speculative draft scaling and accepted length.
- Limitations and evidence: Exact target-distribution property comes from verifier/rejection sampling assumptions; paper does not prove robustness of a faulty/approximate verifier.
- Possible connection: Key contract precedent: approximate proposing plus trusted verification. A recovery proposal must not assume current faulty state is a trusted target.
- Page references: 1, 4, 9
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 067. Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits

Source: [Paper](https://arxiv.org/pdf/2505.21594)

Publication status: arXiv:2505.21594v1 preprint

- Contribution: Early target exits allow the edge to pre-draft future branches while awaiting final verification; tested on Jetson/A100 and robot VLM task.
- Limitations and evidence: Scope: edge/cloud communication, small draft/target pairs and evaluated workloads; early predictions are not final output until verified.
- Possible connection: Pre-draft buffering overlaps delayed commitment/replay; any new holdback protocol must distinguish itself from speculative verification.
- Page references: 1, 4, 12
- Review depth: Targeted full-text screen (abstract/introduction, selected body excerpt, conclusion/limitations when identified); not a full line-by-line read

### 068. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

Source: [Paper](https://arxiv.org/pdf/2501.01005)

Publication status: MLSys 2025, identified in supplied PDF

- Contribution: Composable sparse attention formats, just-in-time customization and load-balanced scheduling compatible with CUDA Graphs. Appendix discusses pinned planning metadata, FP8 KV with higher-precision queries/output, and specialized reductions.
- Limitations and evidence: Authors' evaluation shows kernel gains do not uniformly translate to end-to-end serving gains; appendix notes a BF16 vLLM regression from Python overhead. Reviewer: numerical quality is empirical, not a state-integrity or next-token preservation contract.
- Possible connection: Realistic attention backend and instrumentation target. A thesis must report full serving overhead and cannot call an attention-kernel speedup a reliability improvement. Useful control for replay arithmetic/backend consistency.
- Page references: 1, 4, 7, 8, 9, 10, 19, 20
- Review depth: Focused deep read: selected design/evaluation pp. 4, 7-10 and complete targeted appendix passages pp. 19-20; remaining pages screened selectively, not cover-to-cover.

### 069. Inference-Time Hyper-Scaling with KV Cache Compression

Source: [Paper](https://arxiv.org/pdf/2506.05345)

Publication status: NeurIPS 2025, identified in supplied PDF

- Contribution: Dynamic Memory Sparsification learns per-head eviction with delayed removal, letting later tokens absorb information before deletion; distillation and a compression budget retrofit models so reduced KV cost buys additional inference-time reasoning.
- Limitations and evidence: Authors limit experiments to models up to 32B, contexts up to 32K and compression ratios up to 8. Reviewer: more KV-efficient reasoning does not certify state fidelity. KV-read accounting and real latency are distinct; Quest retrieval and true memory compression must not be compared as identical operations.
- Possible connection: Directly defeats generic 'compress KV to buy more reasoning' novelty. Delayed eviction also establishes intentional downstream information persistence. Killer baseline for any shared-state versus extra-sampling budget study.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 22, 23, 24
- Review depth: Focused deep read: methods pp. 3-5 and limitations p. 16; selected evaluation pp. 6-10 and appendix pp. 20-24. Not all 33 pages read in full.

### 070. LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

Source: [Paper](https://arxiv.org/pdf/2509.01229)

Publication status: arXiv preprint; conference acceptance not verified

- Contribution: Packed overflow-safe low-bit dequantization and pipelined load/dequantize/matrix-multiply execution support W4A8 serving; LiquidServe integrates the optimized kernels.
- Limitations and evidence: Reviewer: hardware- and workload-dependent throughput maxima are not universal gains. Avoid conflating overflow-free dequantization arithmetic with end-to-end decision correctness; persistent-state errors and repair are not evaluated.
- Possible connection: Relevant numerical implementation baseline, but writing another fast low-bit kernel or adding a generic quality check is a weak thesis distinction.
- Page references: 1, 3, 4, 5, 6, 8, 9, 10
- Review depth: Structured full-text screen: abstract and selected method/evaluation passages across pp. 2-10; not all 12 pages read line-by-line.

### 071. PPSD: Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding

Source: [Paper](https://arxiv.org/pdf/2509.19368)

Publication status: arXiv preprint; venue not independently verified

- Contribution: Pipelines early-exit drafting and remaining-layer verification using per-token verification while drafting; analyzes exit depth and acceptance effects on throughput.
- Limitations and evidence: Reviewer: correctness assumes a functioning target verifier and its state. Shared faulty KV can invalidate both draft and verifier; multi-GPU pipeline gains are not a one-GPU comparison.
- Possible connection: A verification-looking execution path is not automatically an independent integrity check. Useful example when defining common-cause state and checking assumptions.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen: abstract plus selected design, theory and experiments pp. 2-10; remaining appendix not deeply reviewed.

### 072. QuantX: A Framework for Hardware-Aware Quantization of Generative AI Workloads

Source: [Paper](https://arxiv.org/pdf/2505.07531)

Publication status: arXiv preprint; venue not verified

- Contribution: Hardware-aware post-training quantization selects recipes using model/layer distribution properties, including aggressive low-bit generative and multimodal configurations and deployment integration.
- Limitations and evidence: Reviewer: empirical quality across selected model/task combinations does not imply preservation of particular decisions or runtime robustness; platform and quantizer dependence remain.
- Possible connection: Hardware-aware quantization alone is already crowded. A dependable-systems contribution would need an explicit failure/recovery contract and appropriate quality baselines.
- Page references: 1, 2, 3, 4, 5, 6
- Review depth: Structured screen of all six pages using abstract, method, results and concluding passages; not every line read.

### 073. Speculative Decoding for Multi-Sample Inference

Source: [Paper](https://arxiv.org/pdf/2503.05330)

Publication status: arXiv preprint, March 2025; later venue not verified

- Contribution: Uses consensus among parallel samples to construct speculative drafts through suffix matching and a DAG; target-model sampling verifies matches. Evaluates reasoning workloads on one A100.
- Limitations and evidence: Authors evaluate acceptance length rather than demonstrate the proposed method's end-to-end speedup, citing implementation limitations; aggregation overhead and generalization beyond reasoning remain open. Appendix shows speculative baselines can lose speed at larger batch sizes.
- Possible connection: Important anti-overclaiming example: theoretical avoided decoding is not realized latency. Target verification still assumes intact target state; multi-sample branches may share a compromised prefix.
- Page references: 1, 2, 3, 4, 5, 8
- Review depth: Focused deep read: pp. 1-3, 5 and 8, with selected p. 4 evaluation and other passages. Eight-page document, not claimed fully line-by-line.

### 074. SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation

Source: [Paper](https://aclanthology.org/2025.emnlp-main.1306.pdf)

Publication status: EMNLP 2025, pp. 25734-25753, identified in supplied PDF

- Contribution: Transforms prompt processing so later-layer KV is derived from earlier representations, with cross-layer sharing and focused distillation of attention projection parameters, reducing prefill work.
- Limitations and evidence: Authors observe quality degradation when skipping too much of prefill and note further data/long-context adaptation work. Distillation uses multi-GPU resources. Reviewer: knowledge preservation is empirical, not bitwise equivalence or fault recovery.
- Possible connection: Shows cross-layer dependencies can differ from vanilla transformers. Start any recovery proof with a standard architecture; do not assume it automatically covers transformed cache layouts.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15
- Review depth: Structured full-text screen: selected methods/evaluation pp. 1-9 and appendix pp. 13-15; not all 20 pages read deeply.

### 075. KernelBench: Can LLMs Write Efficient GPU Kernels?

Source: [Paper](https://arxiv.org/pdf/2502.10517)

Publication status: Supplied source is an arXiv preprint; subsequent venue not verified here

- Contribution: Benchmark of 250 GPU tasks spanning primitives, operator sequences and architectures; fast_p jointly requires correctness and a specified speed advantage over a PyTorch reference.
- Limitations and evidence: The paper's correctness protocol uses five random inputs per task, not a universal proof. Authors note limited language/hardware scope and future expansion. Reviewer: isolated kernel checks omit persistent autoregressive state transitions.
- Possible connection: A thesis cannot merely rebrand kernel testing as proof. The possible distinction is state/sequence restoration under a specified fault model, with independent executable oracles.
- Page references: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Focused deep read of correctness protocol p. 4 and limitations p. 10; selected pp. 1-10 screened. Extensive remaining appendix not fully read.

### 076. Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs

Source: [Paper](https://arxiv.org/pdf/2503.11901)

Publication status: SC 2025 author-accepted version, DOI 10.1145/3712285.3759821

- Contribution: Longitudinal A100/H100 fleet study spanning 1,056 GPUs and 11.7 million GPU-hours, comparing logged errors and modeling availability under recovery/spare policies.
- Limitations and evidence: Observation durations differ substantially by GPU generation; logged errors are not all physical faults or silent corruptions. Job-type categorization is inferred. Availability estimates depend on explicit simulated job/recovery assumptions.
- Possible connection: Strong motivation for recovery cost, but does not justify treating synthetic injection frequency as a measured silent-error rate. Report fault-free overhead and conditional recovery benefit separately.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Review depth: Structured full-text screen of selected pp. 1-11, with closer reading of recovery/availability discussion p. 11.

### 077. Understanding and Detecting Fail-Slow Hardware Failure Bugs in Cloud Systems

Source: [Paper](https://www.usenix.org/system/files/atc25-dong.pdf)

Publication status: USENIX ATC 2025, proceedings cover in supplied PDF

- Contribution: Studies 48 fail-slow incidents and develops Sieve to expose high-level concurrency/timeout defects through targeted fault conditions in distributed cloud software.
- Limitations and evidence: Reviewer: its observed failure mechanisms and tested systems concern fail-slow distributed control, not numerical KV corruption. Discovered/confirmed bugs do not establish deployment-wide incidence.
- Possible connection: Useful methodology for connecting fault, propagation, symptom and recovery. Borrow controlled causal experiments, not the claim that generic fault injection itself is novel.
- Page references: 2, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured screen: actual paper abstract p. 2 and selected methods/results pp. 3-10; not all 17 pages read deeply.

### 078. Understanding Silent Data Corruption in LLM Training

Source: [Paper](https://arxiv.org/pdf/2502.12340)

Publication status: arXiv preprint; venue not verified

- Contribution: Paired deterministic executions on healthy and unhealthy production nodes isolate real silent errors; submodule-level lockstep replacement prevents prior errors contaminating later measurements. Shows losses can conceal state errors.
- Limitations and evidence: Authors discuss a small node sample, single-node tensor-parallel scope, limited horizons and measurement synchronization reducing utilization and observed error rates. Checksum-style approaches did not provide adequate precision/recall in the examined setting.
- Possible connection: Closest empirical reliability neighbor. Distinguish isolated faults from propagated state and do not infer physical rates from random perturbations. A checker can perturb the phenomenon it measures; recovery evaluation needs both isolated and natural propagation modes.
- Page references: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Focused deep read of pp. 3-4 and 8-9, plus selected pp. 1-10; remaining appendix not fully reviewed.

### 079. Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails

Source: [Paper](https://arxiv.org/pdf/2501.09004)

Publication status: arXiv source; later venue not verified here

- Contribution: Human/LLM-assisted safety dataset with 34,248 interactions and an expanded risk taxonomy, used to train adaptable guardrails and study policy-following generalization.
- Limitations and evidence: Reviewer: labels depend on taxonomy and annotation/judging procedures; dataset success does not establish hardware reliability or universal policy truth.
- Possible connection: Possible downstream benign policy-classification evaluation, not a foundation for runtime state-integrity novelty. No harmful prompt construction is needed for the proposed numerical-reliability pilots.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Structured full-text screen of selected abstract, dataset, training and evaluation passages pp. 1-9; not the entire 33-page document.

### 080. Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability

Source: [Paper](https://arxiv.org/pdf/2509.23666)

Publication status: arXiv preprint; venue not verified

- Contribution: UAT combines a learned reliability estimator with online UCB threshold selection to trade predicted correctness against early-exit computation; evaluates multiple model/task families.
- Limitations and evidence: The risk theorem assumes the learned estimator approximates conditional correctness with a specified probability, plus threshold/horizon conditions. Reviewer: it is not an unconditional guarantee under arbitrary distribution shift or faulty internal state.
- Possible connection: Defeats generic online risk-adaptive compute novelty. Any new contract must state whether it protects execution fidelity, task accuracy or learned risk prediction; these are different objects.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Focused deep read of methods/theorem pp. 4-6; selected abstract and evaluation pp. 1-9. Not all 25 pages read in full.

### 081. GuardReasoner: Towards Reasoning-based LLM Safeguards

Source: [Paper](https://arxiv.org/pdf/2501.18492)

Publication status: arXiv source; later venue not verified

- Contribution: Reasoning-supervised safeguards trained from a large reasoning dataset followed by hard-sample preference optimization; evaluates several model sizes and safety tasks.
- Limitations and evidence: Authors discuss judging/label and reasoning-cost concerns. Reviewer: explanatory output is not automatically faithful causal evidence, and benchmark classification accuracy says nothing about state integrity.
- Possible connection: Guard model baseline if needed; adding reasoning to a checker is not a new systems contribution. Training resource needs exceed the simplest one-GPU inference pilot.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Structured full-text screen of selected pp. 1-9, including training, experiments and limitations; remaining appendix not deeply reviewed.

### 082. Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

Source: [Paper](https://arxiv.org/pdf/2506.20251)

Publication status: ICML 2025, PMLR 267, identified in supplied PDF

- Contribution: Evaluates quantization-induced safety degradation and patches it using preference optimization against the original model, low-rank updates, requantization and salient-weight masking.
- Limitations and evidence: Reviewer: teacher preferences can inherit teacher errors; results are specific to model/quantizer/calibration choices. This is static weight-level behavioral restoration, not detection and recovery of a damaged runtime cache.
- Possible connection: Direct prior art against generic quantization-aware safety repair. The possible thesis must identify a distinct state semantics/fault model rather than rename this patching objective.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Focused methods/evaluation read pp. 5-8 and selected pp. 1-9; not all appendices read.

### 083. Safe Pruning LoRA: Robust Distance-Guided Pruning for Safety Alignment in Adaptation of LLMs

Source: [Paper](https://arxiv.org/pdf/2506.18931)

Publication status: arXiv preprint; venue not verified

- Contribution: Prunes LoRA components using a dimension-robust distance from an aligned reference subspace to retain safety during model adaptation.
- Limitations and evidence: Reviewer: depends on an aligned reference and the adapter/fine-tuning setting; empirical safety/utility metrics do not certify execution fidelity under transient or persistent state errors.
- Possible connection: Further evidence that selective safety-critical parameter protection is crowded. Cache integrity should not be justified as merely another critical-layer safety patch.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Structured full-text screen of selected pp. 1-9, not all 13 pages.

### 084. SafeQuant: LLM Safety Analysis via Quantized Gradient Inspection

Source: [Paper](https://aclanthology.org/2025.naacl-long.127.pdf)

Publication status: NAACL 2025, pp. 2522-2536, identified in supplied PDF

- Contribution: Inspects gradients associated with an affirmative response signal, compresses selected gradient features and classifies safety from these white-box representations.
- Limitations and evidence: Authors note the need for white-box weights/gradients, limiting black-box deployment. Reviewer: backward-pass cost and the chosen response signal matter; the method does not protect a quantized model's KV state.
- Possible connection: Important disambiguation: 'quantized' here describes gradient features, not a runtime KV quantization guarantee. Possible probe baseline only, not a numerical integrity oracle.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen of selected pp. 1-10, including limitations p. 9; remaining appendix not deeply reviewed.

### 085. SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models

Source: [Paper](https://aclanthology.org/2025.findings-acl.105.pdf)

Publication status: Findings of ACL 2025, pp. 2053-2069

- Contribution: Routes using the small safeguard's representation, learning when the large model is correct and the small model is wrong, with data augmentation and an oracle-relative risk argument.
- Limitations and evidence: The theorem bounds excess risk relative to an oracle using routing error and a finite-moment assumption, not absolute safety. Authors note absent large-model features and training-data representativeness.
- Possible connection: Killer baseline against generic small/large safety routing. Shared corrupted inputs/state can invalidate the independence implicitly hoped for from escalation.
- Page references: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Focused deep read of router/theorem pp. 3-4 and limitations p. 9, plus selected pp. 1-10; not full appendix.

### 086. Safety Layers in Aligned Large Language Models: The Key to LLM Security

Source: [Paper](https://arxiv.org/pdf/2408.17003)

Publication status: ICLR 2025, identified in supplied PDF; original arXiv identifier is from 2024

- Contribution: Identifies middle-layer safety-related behavior through representation analysis and parameter interventions, then preserves those layers during fine-tuning.
- Limitations and evidence: Reviewer: localization is model/task/intervention dependent and does not prove a universal safety module. Preserving selected weights does not ensure intact arithmetic or cache state.
- Possible connection: Distinguish a structurally clean replay boundary from a behaviorally 'safe layer'; they are not interchangeable. Avoid novelty claims based solely on selectively protecting layers.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen of selected pp. 1-10; not all 27 pages.

### 087. Unified Multi-Task Learning & Model Fusion for Efficient Language Model Guardrailing

Source: [Paper](https://arxiv.org/pdf/2504.19333)

Publication status: arXiv preprint; venue not verified

- Contribution: Combines synthetic custom-policy training, multitask guardrails and searched model merging to obtain small unified guard classifiers.
- Limitations and evidence: Reviewer: synthetic policy data and model-merge search determine transfer quality; compact classifier F1 does not establish numerical resilience or universal policy compliance.
- Possible connection: Another efficient-guard baseline; a thesis about runtime recovery should not be sold as a new guardrail just because it can eventually protect a guard's execution.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen of selected pp. 1-10; remaining appendix not deeply reviewed.

### 088. A Scalable Multi-GPU Framework for Encrypted Large-Model Inference

Source: [Paper](https://arxiv.org/pdf/2512.11269)

Publication status: arXiv preprint initially December 2025; venue not verified

- Contribution: Cerium supplies a compiler/runtime for multi-GPU fully homomorphic encrypted inference, optimizing arithmetic representations, fusion, memory layout and communication.
- Limitations and evidence: Reviewer: very large memory and specialist cryptographic systems requirements; substantial encrypted inference latency remains. Privacy under the protocol does not imply correctness when hardware state is corrupted.
- Possible connection: Cryptographic privacy is orthogonal to the proposed accidental-error recovery problem. Full encrypted-LLM implementation is a poor first one-GPU thesis pilot.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
- Review depth: Structured screen using selected passages across all 13 pages; not every line read.

### 089. Breaking the Layer Barrier: Remodeling Private Transformer Inference with Hybrid CKKS and MPC

Source: [Paper](https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf)

Publication status: USENIX Security 2025, proceedings cover in supplied PDF

- Contribution: Reorganizes private transformer inference across layer/operator boundaries, combining CKKS and MPC conversions and fused matrix operations to reduce communication and latency.
- Limitations and evidence: Approximate nonlinear computation is assessed empirically. Authors note limitations to batching across users with different keys. Reviewer: selected BERT/GPT-style evaluations are not a universal autoregressive-state correctness claim.
- Possible connection: Useful separation of protocol privacy, approximate numerical quality and hardware integrity. Not a direct solution to delayed state contamination.
- Page references: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15
- Review depth: Structured full-text screen: abstract p. 2 and selected design/evaluation/discussion pp. 3-15; not all 21 pages read deeply.

### 090. BumbleBee: Secure Two-party Inference Framework for Large Transformers

Source: [Paper](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)

Publication status: NDSS 2025 primary proceedings source

- Contribution: Optimizes secure two-party transformer inference with low-communication linear computation and approximate activation protocols.
- Limitations and evidence: Reported full-model examples still require minutes for a generated token; selected setup costs are excluded from online evaluation. Reviewer: security follows the protocol's stated adversary assumptions, not arbitrary GPU fault tolerance.
- Possible connection: Not a feasible direct baseline for a lightweight one-GPU recovery system; include only if discussing why confidentiality and numerical reliability are distinct.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13
- Review depth: Structured full-text screen of selected pp. 1-9 and 11-13; not all 18 pages.

### 091. Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs

Source: [Paper](https://arxiv.org/pdf/2509.18886)

Publication status: arXiv preprint; venue not verified

- Contribution: Compares confidential CPU/GPU LLM serving performance and cost, including AMX-enabled CPU execution and confidential H100 cloud configurations.
- Limitations and evidence: Authors' cloud setup lacks a matching bare-metal comparison and excludes some GPU modes. Costs and overhead depend on context, batch size, sockets and virtualization. Reviewer: TEE isolation does not make a checker physically independent or arithmetic fault-free.
- Possible connection: Useful overhead-accounting example and deployment constraint, not evidence that a protected checker resolves common-cause state errors.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen of selected pp. 1-10; remaining appendix not fully reviewed.

### 092. ENSI: Efficient Non-Interactive Secure Inference for Large Language Models

Source: [Paper](https://arxiv.org/pdf/2509.09424)

Publication status: arXiv preprint; venue not verified

- Contribution: Optimizes CKKS-based noninteractive inference for low-bit models using specialized encoding, attention-function substitution and normalization/bootstrapping integration.
- Limitations and evidence: Reviewer: replacing nonlinear functions changes the executed model; microbenchmark speedups and approximation-error experiments do not prove long-horizon answer equivalence. Requires cryptographic expertise beyond the initial recovery prototype.
- Possible connection: Illustrates why execution equivalence must name the reference model and arithmetic rather than treating every approximation as the same model.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10
- Review depth: Structured full-text screen of selected pp. 1-10; not all 12 pages.

### 093. Inside Job: Defending Kubernetes Clusters Against Network Misconfigurations

Source: [Paper](https://arxiv.org/pdf/2506.21134)

Publication status: arXiv preprint; supplied placeholder ACM metadata does not establish final acceptance

- Contribution: Studies application network misconfigurations across organizations and relates configuration analysis to runtime evidence to identify and validate deployment problems.
- Limitations and evidence: Authors describe context-dependent false positives and intentionally overlapping services. Reviewer: configuration anomalies are not equivalent to exploitable failures or to numerical state corruption.
- Possible connection: Methodological lesson: validate suspected faults with independent evidence. No offensive reproduction is needed or included; direct technical relevance to KV repair is low.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 15, 16
- Review depth: Structured full-text screen of selected pp. 1-9 and validation/limitations pp. 15-16; not all 26 pages.

### 094. MettEagle: Costs and Benefits of Implementing Containers on Microkernels

Source: [Paper](https://www.usenix.org/system/files/osdi25-miemietz.pdf)

Publication status: OSDI 2025, proceedings cover in supplied PDF

- Contribution: Implements container-like environments over the L4Re capability-based microkernel and evaluates isolation-related design benefits and execution costs.
- Limitations and evidence: Authors note difficulty quantitatively measuring trustworthiness and substantial work remaining for OCI image compatibility. Isolation does not eliminate all shared-state/timing issues.
- Possible connection: Supports explicit trusted-computing-base reasoning. Building a microkernel/container platform would be a different, much larger thesis than dependable LLM state recovery.
- Page references: 2, 3, 4, 5, 6, 7, 8, 9, 10, 14
- Review depth: Structured full-text screen: abstract p. 2, selected pp. 3-10, and scope discussion p. 14; not all 19 pages.

### 095. Performance of Confidential Computing GPUs

Source: [Paper](https://arxiv.org/pdf/2505.16501)

Publication status: arXiv source; exact venue not established from copyright notice alone

- Contribution: Measures confidential H100 serving under model swapping, traffic and scheduling conditions, highlighting transfer-driven latency and throughput costs.
- Limitations and evidence: Reviewer: one VM/GPU setup and workload mix; overhead is not directly comparable with resident-model TEE results. Latency measurements do not establish resistance to silent errors.
- Possible connection: Demonstrates why a repair design must include checkpoint movement, admission/capacity effects and workload mix, not just isolated recomputation time.
- Page references: 1, 2, 3, 4, 5, 6
- Review depth: Structured screen of all six pages through selected abstract/method/results passages, not every line.

### 096. Securing Transformer-based AI Execution via Unified TEEs and Crypto-protected Accelerators

Source: [Paper](https://arxiv.org/pdf/2507.03278)

Publication status: arXiv preprint; venue not verified

- Contribution: TwinShield divides trusted CPU and untrusted accelerator work, using protected offloading and verification mechanisms for transformer matrix/attention computations.
- Limitations and evidence: Reviewer: guarantees depend on the specific trusted execution and cryptographic threat model. Verification overhead is measurable; these checks do not automatically reconstruct temporally contaminated cache descendants.
- Possible connection: Prior art against 'first verified attention/GEMM' claims. A potential baseline for placement of independent checks; recovery semantics would need to be a separate contribution.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Review depth: Structured full-text screen of selected pp. 1-11, including verification and overhead passages; not all 15 pages.

### 097. Towards Confidential and Efficient LLM Inference with Dual Privacy Protection

Source: [Paper](https://arxiv.org/pdf/2509.09091)

Publication status: arXiv preprint; venue not verified

- Contribution: Combines client-side trusted embedding work with accelerator execution and input-privacy perturbations to protect model/input information while controlling latency.
- Limitations and evidence: Reviewer: privacy/utility tradeoffs and the protected deployment assumptions are central; deliberately randomized inputs are not accidental arithmetic faults. Reported setup uses server hardware and multiple GPUs.
- Possible connection: Low direct relevance. Do not equate successful recovery of deliberate privacy noise with restoration after a corrupted state transition.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9
- Review depth: Structured full-text screen of selected pp. 1-9; not every line of the ten-page document.

### 098. AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds

Source: [Paper](https://arxiv.org/pdf/2501.06706)

Publication status: arXiv source; venue not independently verified here

- Contribution: Orchestrates fault scenarios, workloads, telemetry, agent interfaces and task oracles for cloud detection, localization, root-cause analysis and mitigation evaluation.
- Limitations and evidence: Authors identify need for finer task oracles and show additional agent steps can increase cost without better outcomes. Reviewer: general cloud success metrics do not measure numerical execution fidelity.
- Possible connection: Borrow explicit oracles and cost accounting, but an observability wrapper or repair-agent workflow would be too close to the rejected broad ProofOps/RAVEN framing.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Review depth: Structured full-text screen of selected pp. 1-11; remaining appendix not deeply reviewed.

### 099. Automated Dynamic AI Inference Scaling on HPC-Infrastructure: Integrating Kubernetes, Slurm and vLLM

Source: [Paper](https://arxiv.org/pdf/2511.21413)

Publication status: MIND 2025 workshop author version indicated; placeholder metadata does not independently establish final publication details

- Contribution: RAMSES integrates vLLM, Kubernetes and Slurm to allocate and scale inference resources across HPC-style infrastructure, with preliminary request-load measurements.
- Limitations and evidence: Reviewer: limited evaluation and orchestration-focused latency; no numerical correctness or persistent-state recovery claim. Integration alone is not evidence of a new dependable-inference mechanism.
- Possible connection: Possible future deployment context, not a reason to require HPC for the first falsification experiment. No HPC access was used.
- Page references: 1, 2, 3, 4, 5, 6
- Review depth: Structured screen across all six pages using selected architecture/evaluation/conclusion passages; not every line.

### 100. LogSage: An LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation

Source: [Paper](https://arxiv.org/pdf/2506.03691)

Publication status: arXiv preprint; venue not independently verified

- Contribution: Combines log pruning, structured diagnosis, retrieval of historical fixes and remediation tools, evaluated on public failures and large-scale industrial CI/CD executions.
- Limitations and evidence: Industrial diagnosis accuracy is human-sampled over an early period; remediation tool coverage is limited and rerun success is conditional on applicable cases. Reviewer: diagnosis precision does not imply end-to-end autonomous repair success.
- Possible connection: Useful warning against misleading recovery denominators. Numerical-state work must report all detected faults, unrepairable cases and committed-output limitations, not only easy successfully repaired examples.
- Page references: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Review depth: Structured full-text screen of selected pp. 1-11, with closer attention to industrial validation and limitations p. 10; not every line.



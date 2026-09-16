# Reliability/stateful-inference synthesis — papers 068–100

Reviewed 2026-09-16. The companion JSON records actual reading depth and physical PDF pages for all 33 papers. This is a critical screening and targeted deep review, not a claim that every appendix was read. No proposed system was implemented or experimentally validated, and no novelty claim is cleared by this search alone.

## Bottom line

The strongest dependable-AI-systems opening in this subset is **correct recovery of persistent inference state**, not another guardrail, quantization-aware safety patch, confidence router or kernel checker. But even this opening is narrow: saved hidden activations, KV restoration, prefix rewind and checkpoint logs already exist. A defensible contribution would have to establish when a recovered cache and its uncommitted token sequence are actually consistent with clean execution, and demonstrate that exploiting that structure is worth its steady-state cost.

The proposed recovery question below is a falsification candidate, not a recommendation to commit to a thesis immediately. Its credible payoff is an explicit execution contract and a measured recovery cost frontier. Its main risk is that ordinary verified-prefix replay or prevention is already cheaper.

## What the supplied subset already rules out

- DMS (069) already spends KV savings on additional reasoning; SwiftKV (074) already transforms cross-layer cache computation. A new compression/reasoning budget wrapper is insufficient.
- Q-resafe (082), SPLoRA (083), safety-layer preservation (086), and related guardrails already address efficient behavioral safety interventions. These do not solve numerical state restoration, but defeat broad claims that efficiency-induced safety loss or selective protection is new.
- UAT (080) and SafeRoute (085) already supply risk-aware compute selection with conditional mathematical arguments. Their assumptions must not be confused with unconditional correctness.
- KernelBench (075) makes correctness part of its performance metric; TwinShield (096) verifies transformer offloads. A generic kernel-checking contract needs much stronger differentiation.
- Real training SDC measurements (078) show both masked losses and contamination-sensitive measurement. Fleet error logs (076) motivate recovery but do not supply a realistic inference SDC rate. Controlled perturbations can establish mechanism, not incidence.
- TEEs/private inference (088–097) mostly concern a different threat model. They do not automatically make a numerical checker independent of shared bad state.
- AIOpsLab/LogSage (098/100) show that tool-assisted diagnosis and remediation are already substantial fields; a monitoring/repair-agent integration would risk repeating the rejected broad framing.

## Fresh primary-source novelty attack

These are additional sources accessed during this review, separate from the 33-paper assignment. Except where indicated, they are preprints or public artifacts, not assumed peer-reviewed results. Reading depth is explicit; performance claims were not reproduced.

| Source and status | Actual access/read depth | Consequence for the candidate |
|---|---|---|
| [HCache](https://arxiv.org/html/2410.05004v1), EuroSys 2025, DOI visible in author HTML | Abstract, introduction and §3.1–3.2 read closely; selected remaining text | Caches layer-input hidden states and reconstructs KV with projections. Hidden-state restoration itself is not new. The analyzed storage advantage assumes conventional multi-head attention; it must be recalculated for GQA. |
| [HybridServe](https://arxiv.org/html/2501.01792v1), 2025 preprint | Abstract, §3.3 and §4.1–4.3 focused reading | Already combines per-layer activation and KV cache blocks and schedules activation-to-KV recomputation. A journal is not by itself the contribution. |
| [Concordia](https://arxiv.org/html/2606.23521v1), June 2026 preprint | Fault model, design/checkpoint paths, evaluation and recovery sections examined | GPU-resident delta checkpoints and committed logs already support LLM recovery. Its explicit fail-stop model and trusted worker differ from delayed silent corruption that contaminates derived state. Do not claim first KV logging or fault-tolerant serving. |
| [vLLM Ascend KV-load failure RFC](https://github.com/vllm-project/vllm-ascend/issues/10561), public engineering proposal | Issue text read | Last-valid-prefix rewind and recomputation are an essential baseline. A recovery paper must beat this, not only full prompt replay or local slot replacement. |
| [Models Take Notes at Prefill](https://arxiv.org/abs/2606.17107), June 2026 preprint | Primary abstract and HTML opened; no full methodological replication | Already describes source information being memoized downstream so source-KV edits fail to remove its influence. First observation of downstream persistence is not available as novelty. |
| [Shared-prefix KV corruption study](https://arxiv.org/abs/2604.17249), 2026 preprint; source claims SECRYPT acceptance | Primary abstract read | Shared-prefix failures and checksum scheduling are already studied. Prevention before use is a potentially fatal baseline; no offensive procedure is needed for our study. |
| [LineageKV](https://github.com/cbun/lineagekv), public draft/artifact | Repository README reviewed | Semantic stale-memory repair by modifying source-span cache is adjacent, but does not establish exact numerical descendant repair. Must distinguish semantic rewriting from restoring execution state. |
| [TwinKV](https://arxiv.org/abs/2608.27128), August 2026 preprint | Primary abstract read | Repairs eviction selections under a fixed cache budget, not accidental corrupted-state dependencies. Avoid unqualified “first KV repair” language. |
| [HEAL](https://arxiv.org/abs/2606.21023), June 2026 preprint | Primary abstract and HTML opened; selected description inspected | Numerical reproducibility already explicitly addresses horizontal KV and vertical-layer error propagation using higher-fidelity arithmetic. Generic numerically stable inference is not a fresh gap. |
| [Does Accuracy Equal Evidence?](https://arxiv.org/abs/2608.01631), August 2026 preprint | Primary abstract opened, supplementary indexed description inspected | Fixed-trace replay and answer-versus-evidence discrepancies under cache compression are already being evaluated. Fixed-token replay is an experimental control, not the new contribution. |
| [X-Cache](https://arxiv.org/abs/2604.20289), April 2026 preprint | Primary abstract and HTML opened | Uses full computation at KV-update chunks to avoid accumulating approximation contamination in autoregressive video. This is a different model family but a relevant preventative scheduling analogy. |
| [CacheBlend](https://arxiv.org/abs/2405.16444), EuroSys 2025 paper | Primary author search result/abstract-level review only | Selective recomputation for cached RAG context is already established. Requires a deeper comparison before claiming a novel minimal-recomputation mechanism. |

Parent review separately identified Runtime-Certified Bounded-Error Quantized Attention (2605.20868), WitCert (2607.28699), and a kernel contract verifier (2608.12700). Those are cross-team leads, not full reads by this reviewer; runtime KV bounds should not be selected as novel without that comparison.

## Candidate 1: contamination-aware recovery with bounded output commitment

**Question.** Given a trusted localization of an accidental finite KV error detected within a bounded uncommitted decoding window, can a small number of saved layer-input boundaries restore the correct cache and greedy token sequence more cheaply than replaying from the last verified prefix?

**Precise distinction.** HCache/HybridServe restore from valid activation representations. A delayed error can make some later saved activations invalid. The possible contribution is identifying which saved boundaries remain valid, recomputing the causal descendants, and handling the first token divergence without leaving stale descendants behind. Generic checkpointing, activation caching, rollback and the observation that errors propagate are not claimed new.

Begin with an ordinary causal decoder, fixed weights, a specified deterministic backend and greedy generation. Keep one clean prefix checkpoint available equally to every baseline. Retain token IDs and a bounded rolling journal at a few layer cuts. Correct the localized source state, then replay only from a demonstrably clean boundary while recorded tokens remain equal to corrected decisions. At the first decision divergence, discard subsequent state and regenerate from the corrected history. The outputs in the window must not have been externally committed. A later extension to sampling needs explicitly coupled/saved RNG state; equal distributions and the same realized sequence are different contracts.

**Required assumptions and nonclaims.** The localization signal, clean checkpoint, journal metadata, model weights and reference arithmetic are trusted. Corrupted journals or unknown earlier faults invalidate the cheap path and require conservative fallback. No already emitted text or external action is undone. Agreement with fault-free model execution is not proof that the model's answer is true or safe. Cross-kernel floating-point equivalence cannot simply be assumed; first use the identical deterministic execution path, then measure a stated tolerance/decision-equivalence contract separately.

**Killer baselines.** (1) No protection, to quantify effect rather than assume every perturbation matters. (2) Correct original slot only. (3) Full prefill from the clean committed history. (4) Last-verified-prefix restore plus all-layer suffix replay, with the same checkpoint. (5) HCache/HybridServe-style full-layer activation restoration, with equal extra-memory budgets and validity checks. (6) Full duplicate window state or more frequent ordinary checkpoints at equal bytes. (7) Check-before-use or cheap checksum protection appropriate to the fault model. A method does not win by giving its competitors older checkpoints or charging only their metadata overhead.

**One-GPU first experiment.** Use a 0.5–3B standard decoder with explicit cache tensors and benign synthetic key/value retrieval, variable-update tracking and exact arithmetic tasks. Start with fixed-token replay to test state restoration independently from changed text, then greedy closed-loop decoding. Apply controlled non-adversarial numerical perturbations only inside the local test harness, varying affected layer and detection delay (for example 0, 1, 8, 32 steps). Compare local repair, full suffix replay, and a few cut placements before engineering a serving backend. Measure cache/logit error, next-token and full-window agreement, repair milliseconds, fault-free journal write cost, checkpoint/journal memory and output holdback latency. Injection counts are experiment conditions, not estimates of physical error frequency.

**Fast rejection criteria.** Reject or substantially narrow if valid-cut recovery is almost always equivalent to full replay, sequence divergence happens too early to exploit reused work, journals cost more than their occasional savings, or affordable before-use checking removes the motivating late-error regime. A fault model requiring a perfect free detector/localizer while ignoring its cost is not an acceptable empirical claim. One microbenchmark speedup without serving-capacity and output-delay accounting is insufficient.

**Assessment.** Worth a small falsification pilot; not yet a cleared thesis. It becomes substantive only if the validity/sequence contract reveals a genuinely missing recovery case and its resource-aware policy beats the strong baselines. Otherwise it is a combination of existing checkpoint primitives.

## Candidate 2: when shared-state sampling ceases to provide independent evidence

**Question.** Under a fixed measured inference budget, when does rebuilding independent prefix state reduce final-answer error more than drawing additional suffix samples from one shared compressed or perturbed prefix?

The proposed object is a budget-allocation frontier conditioned on common state, not the old observation that correlated errors weaken majority voting. DMS, multi-sample speculation and existing shared-prefix corruption work make broad versions of this idea unoriginal. Deterministically reconstructing the same lossy quantization does not remove systematic bias; distinguish accidental corrupted memory from approximation bias, and use a higher-fidelity reference or genuinely varied state construction where justified.

**Killer baselines.** DMS-like compression plus additional reasoning, ordinary self-consistency at equal latency, independently recomputed full-precision prefixes, diversified compression seeds, no-eviction/higher-precision caches, and a fault-free oracle. Also compare any existing adaptive sampling/pruning policy fairly rather than only a fixed sample count. If an integrity check cheaply detects and repairs the common-state fault, it is a baseline, not an inconvenient exclusion.

**One-GPU first experiment.** A 1–3B model on synthetic reasoning with exact ground truth; sample 1/4/8/16 branches and divide them across 1/2/4 independently constructed cache groups. Run paired question/seed sets under clean, approximate and benign perturbed-state conditions. Measure answer error, within/between-group error dependence, confidence calibration, actual GPU time and memory, and p95/p99 latency. Give rebuilding its full prefill charge. A hierarchical uncertainty estimate must not report branch count as independent sample size.

**Fast rejection criteria.** Reject if fresh state does not reduce conditional error at equal cost, prevention dominates, the effect is entirely generic ensemble correlation, or no practically identifiable allocation signal exists. This is lower-confidence than Candidate 1 and should remain a reserve direction, not an already-novel proposal.

## Positioning

For a dependable-AI-hardware/systems advisor, Candidate 1 has the cleaner fit: an explicit fault model, trusted state, recovery semantics, memory/latency accounting and a reproducible systems artifact. The AI-safety connection is that monitors and deployed agents depend on their execution substrate, not a claim that numerical fidelity solves alignment. Program/organization websites support audience and training choices only; they do not validate either candidate's novelty or guarantee funding/admission.

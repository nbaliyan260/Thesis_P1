# Safety/reliability literature synthesis (papers 035–067 + targeted primary-source challenge)

Date: 16 September 2026. This is a screening and falsification memo, not a declaration of novelty or a completed systematic review.

## Review integrity

All 33 assigned extracted texts were opened; the per-paper ledger records physical PDF pages, contribution, scope/limitations, connection, publication status, and actual review depth. Most received selected-section full-text screening; 052 (Guaranteed Guess), 058 (Mooncake), and 065 (BitDecoding) received deeper method/results/limitations reviews. This is **not** 33 line-by-line full-paper reads. No experiments, fault reproductions, HPC sessions, or prototype changes were performed. Paper numbers below map to `safety_035_067.json`.

The first 11 requested program websites were inspected separately in `program_agendas_01_11.json`. A page being accessible is not evidence that its project catalog was accessible. That ledger preserves failed initial opens, successful www alternatives, and unverified project details.

## What the supplied papers establish

- 035–038 and 037 in particular show that lightweight safety pruning, alignment-drift diagnosis, and efficient latent guardrails are occupied areas. 036 and 048–051 also make generic evaluate-remediate loops, diagnostic agents, and topology-aware application repair weak novelty anchors.
- 041 (DP-Fusion) is a good example of an explicit distributional guarantee with accounting costs. 045 explicitly does not claim cryptographic security; 047's exact-output claim is protocol/setup scoped. A thesis should not collapse statistical accuracy, decision agreement, bitwise state equality, and formal safety into one word such as “correct.”
- 052 is directly adviser-aligned: specialized models plus hardware/software correctness testing. Its limitations explicitly distinguish test coverage from semantic equivalence, and difficult compilation settings substantially lower success. This supports rigorous contracts and falsification, not renaming testing as proof.
- 058 makes persistent/distributed KV a real systems object. Transfer completion/failure handling is not assurance that successfully transferred bytes represent the right semantic state.
- 065 makes quantizer state a first-class recovery obligation: packed values, scales/zero points, layout, and the half-precision residual window that later becomes a quantized block. Restoring one numerical element may not restore a group-consistent state.
- 054–064 show that live scaling, overlap, memory handover, distributed placement, and prefetching already offer strong performance baselines. Reliability instrumentation competes with these optimizations; it cannot be assumed free.
- 066–067 supply an important precedent for speculative computation and delayed commitment. Approximation is safe only relative to the stated trusted verifier; a verifier reading compromised state is not automatically trusted.

## Closest primary-source collisions found online

| Work and verified status | Occupied contribution | Consequence for a proposed thesis |
|---|---|---|
| [HybridServe: Efficient LLM Inference with Activation Checkpointing and Hybrid Caching](https://arxiv.org/abs/2501.01792), arXiv primary record lists IEEE ICCD 2025 and DOI | Stores decoder-input activations to reconstruct layer KV without recomputing preceding layers; balances KV/activation representations for host offloading. Method §§3.3–4 inspected in HTML. | Hidden-state checkpoints and partial KV reconstruction are established primitives. The novelty cannot merely be “store activations to replay fewer layers.” |
| [Models Take Notes at Prefill](https://arxiv.org/abs/2606.17107), June 2026 preprint | Explains stale downstream conclusions after local field edits; full affected-suffix recomputation is an explicit reliable baseline, while selective repair is less reliable. Relevant main-text mechanism and related work inspected. | Local restoration failing to remove descendants is already known. Distinguish exact accidental-error recovery from semantic editing/composition. |
| [KVShareArena](https://arxiv.org/abs/2609.10266), September 2026 preprint | Cross-context/checkpoint cache reuse and repair benchmark; selective re-encoding for assembled contexts, with compute/memory costs. | Reuse quality and selective repair alone are not novel; compare the recovery objective and strong repair baselines. |
| [Aborted but Not Forgotten](https://arxiv.org/abs/2608.15939), August 2026 preprint; abstract only reviewed for defensive contract | Formalizes rollback consistency: attended cache state must agree with committed application state; transaction-local restoration. | Sequence/state closure is also occupied as a security contract. No reproduction or exploitation is proposed; only the defensive scope distinction matters here. |
| [Stage-Replay Divergence Follows the KV Cache](https://arxiv.org/abs/2607.28495), July 2026 preprint | Same tokens can give different retained-versus-reprefilled BF16 states; incremental replay controls can recover equality in the studied setup. | Exact replay requires an execution-path contract, not just token IDs. Fresh full-prefill is not automatically a bitwise oracle for live decode. |
| [GhostServe](https://arxiv.org/abs/2605.00831), primary record states MLSys 2026 | Low-overhead KV protection via host-side streaming parity/checkpointing. | Device loss/restart already has strong protection baselines. Delayed silent contamination is a narrower fault model, not an excuse to omit checkpoint comparisons. |
| [Concordia](https://arxiv.org/abs/2606.23521), June 2026 preprint | Device-resident persistent checkpoint hooks, kernel-level integration, registered state and committed delta logs. Abstract inspected. | Cheap state logging and recovery are not themselves novel. Full technical comparison remains necessary before proposal commitment. |
| [Belayer](https://arxiv.org/abs/2608.14635), primary record reports July submission/August revision; preprint | Reuses independently owned GPU allocations/weights, rebuilds request KV from logged token prefixes, coordinates environment restoration with model context. Abstract inspected. | Prefix-consistent recovery and avoiding full cold start already exist; do not promise broad agent rollback novelty. |
| [kvpack](https://github.com/High-Performance-AI-Lab/kvpack), primary software README, not peer-reviewed research | Exact-byte state restoration, compatibility namespace, complete-state inventory and atomic installation; engine owns semantic correctness. README inspected. | Byte integrity versus semantic provenance is a useful explicit boundary. Checkpoint storage plumbing alone is weak research novelty. |
| [Alignment Collapse Under KV Cache Quantization](https://arxiv.org/abs/2606.09864), August-revised preprint | Measures safety degradation beyond perplexity, diagnoses sensitive channels and proposes protection. | “KV quantization can harm safety” and generic safety-aware channel protection are already direct prior art. |
| [Do Activation Monitors Survive Model Updates?](https://arxiv.org/abs/2606.15980), June-revised preprint | Benchmarks, predicts and repairs activation-monitor staleness under updates, including quantization. | A general monitor-portability benchmark needs a more specific mechanism or joint-system result. |
| [HEAL](https://arxiv.org/abs/2606.21023), June 2026 preprint | Studies kernel-boundary numerical divergence and compensation. | Cross-device numerical drift and a generic compensated execution path are occupied. |
| [Reliability Scaling Laws for Quantized LLMs](https://arxiv.org/abs/2607.10855), primary record states TMLR 2026 (not independently verified on venue site) | Studies calibration, uncertainty, and natural input robustness; effects can be non-monotonic. | Do not assume more precision universally gives more reliable behavior. |

Additional adjacent primary records: [QuantiBias](https://arxiv.org/abs/2607.21063) (generative/multilingual bias under quantization), [X-Cache](https://arxiv.org/abs/2604.20289) (autoregressive video caching with full-compute KV updates to avoid persistent contamination), [KVBoost](https://arxiv.org/abs/2608.21362) (selective cache recompute; displayed submission date/identifier inconsistency should be resolved before bibliography), and [The Illusion of Equivalence](https://arxiv.org/abs/2604.15409) (cache/no-cache finite-precision divergence). These were abstract-level checks, not full critical replications.

## Candidate 1: clean-cut recovery after delayed numerical error detection

**Scientific question:** Given a localized finite error in persistent KV, a known-clean checkpoint, and detection within a bounded uncommitted window, when can a small set of trusted layer-time cuts recover the *same state and generated continuation* as canonical fault-free execution at lower total cost than valid-prefix suffix replay?

**Conditional mechanism worth testing, not yet demonstrated novel:** Maintain rolling hidden-state journals at selected layer boundaries; use the causal dependency graph to identify which cuts remain clean. Restore the original damaged block from the same clean source available to every baseline. Recompute only descendants reachable from the error until the corrected next-token choice differs; then invalidate all later token-dependent state and regenerate from the earliest divergence. Quantization metadata/grouping are part of state. This combines known primitives; a new result would have to be the clean-cut theorem, the optimal/near-optimal recovery boundary under stated constraints, or an empirically meaningful cost frontier unavailable from existing checkpoint policies.

**Closest prior art:** HybridServe for activation reconstruction; Models Take Notes and KVShareArena for descendant staleness/re-encoding; Aborted but Not Forgotten for committed-state consistency; GhostServe/Concordia for protected checkpoints; ordinary checkpoint/rematerialization theory still needs a dedicated review. No blanket novelty claim is justified.

**Required scope and controls:**

1. Detection/localization and clean-source availability are explicit assumptions, not contributions. Detection delay is varied and charged. Faults are accidental numerical disturbances in an isolated local experiment.
2. Define exactness: bitwise full state, greedy token agreement, or distributional equality are different contracts. Canonical numerical execution must be replay-compatible; same token history alone does not establish it.
3. A hidden activation generated after exposure to the error may be contaminated. A cut is trusted because it is outside the causal descendants, not because it is merely “earlier in the layer stack.” After token divergence, even lower-layer activations for the discarded suffix are stale.
4. Holdback protects only outputs not yet committed. Nothing can un-send an emitted token. Initial study excludes external tool actions and adversaries.
5. Compare at least original-slot-only restore, valid-prefix suffix replay, ordinary periodic clean checkpoints, activation-based recovery, and the proposed cut selection, all with identical clean source and output contract.
6. Report failure-free journal bandwidth, capacity, latency and energy separately from recovery savings. Journal storage scales with number of cuts × pending tokens × hidden width × bytes/value, plus metadata. GQA can make hidden journals relatively expensive compared with KV.
7. Do not invent a realistic fault rate. Report measured cost curves and break-even hazard/delay sensitivity. Rare failures may make continuous journaling a net loss.

**First one-GPU falsification test:** A 1–3B open-weight decoder on benign structured tasks, fixed tokenizer/kernel path/seed, with an oracle marking a synthetic local numerical disturbance. Use teacher-forced identical token sequences first to isolate state propagation, then free greedy continuation to test the token-divergence boundary. Sweep a few disturbance layers and detection delays (e.g., 1, 4, 16 token steps), comparing the baselines above. Record every mismatch of canonical KV plus quantizer metadata, not only final answer quality. No existing prototype or HPC environment was changed.

**Kill conditions:** (a) a supposedly clean cut is not sufficient for exact recovery; (b) canonical replay costs erase layer-saving gains; (c) ordinary checkpoint/suffix replay dominates the latency-memory-overhead frontier; (d) earliest token divergence usually forces almost complete replay; (e) existing work already implements the same clean-cut contract. A “restoring the original slot sometimes fails” plot alone would not support a thesis.

**Assessment:** Best of the two for adviser alignment, but high novelty risk and still a research question—not a selected thesis.

## Candidate 2: common-mode reliability of a model and its runtime monitor

**Scientific question:** Do benign numerical execution changes create correlated errors between a model and a monitor derived from its own activations, and can deliberate precision/execution diversity reduce *joint missed failures* at a fixed resource budget?

**Possible distinct mechanism:** Optimize the joint system rather than the model or monitor independently: correlate task errors with monitor misses conditional on execution precision, layer activation scaling, and benign kernel changes. A small precision-diverse monitor or independent high-precision readout could break common-mode numerical failures. This is an hypothesis; independence and benefits must be measured, not assumed.

**Closest prior art:** Activation-monitor staleness explicitly covers model updates/quantization; Alignment Collapse covers KV safety sensitivity; CoLaGuard (037) covers efficient latent moderation; Reliability Scaling Laws covers natural robustness/calibration. A common-mode diversity result must be separated from these and from established diverse redundancy in dependable computing. No universal safety claim is defensible.

**First one-GPU test:** Use one small open model and a lightweight activation monitor for benign, automatically labeled constraints (e.g., structured-output validity, arithmetic consistency, or permitted schema choices). Build matched runs varying only numerical execution; hold weights/data fixed. Compare independent small verifier, same-model high-precision readout, same-model matched precision, and precision-diverse monitor at equal measured cost. Estimate conditional miss rates and joint failure correlation with confidence intervals, including clean controls. Do not generate harmful outputs or reproduce guardrail bypasses.

**Falsification:** No reproducible common-mode correlation beyond task difficulty; quantization does not hurt the monitor (as some prior findings suggest); cheap independent verification dominates; apparent benefit disappears with threshold/cost matching; or prior art already shows the proposed diversity intervention. A catalog of accuracy drops would not be enough.

**Assessment:** Easier first measurement and closer to several program agendas, but weaker hardware-method novelty than Candidate 1 and highly crowded. Keep as a fallback question, not an assured thesis direction.

## Program-agenda implications

The actual projects support three distinct impact arguments, not a single generic “AI safety fellowship fit”:

- Technical assurance/evaluation-to-deployment: Pivotal's Silent Updates; ERA's hardware and evaluation agenda; GovAI technical-governance problems.
- Causal/mechanistic scientific method: Anthropic circuit tracing; PIBBSS/PIRAMID; Iliad's applied-mathematics orientation.
- Oversight under efficiency pressure: LASR probe generalization, IAPS monitorability work, Apart's perturbation diagnostics.

OpenAI's inspected announcement is broad enough to include robustness/evaluation, but no concrete hardware project roster was verified. AI Safety Camp's inspected page describes research incubation, not a verified hardware project. No program's existence establishes novelty, publication potential, admission probability, or an endorsement of either candidate.

## Bottom line

Do not pitch another general safety wrapper, generic KV integrity detector, or undifferentiated quantization-safety benchmark. A potentially stronger thesis is an explicit, narrow, falsifiable dependable-execution question with a defensible recovery/assurance contract and strong existing baselines. Candidate 1 is worth a tightly bounded pilot only after the additional activation-checkpointing and rollback prior art is integrated; a negative cost/novelty result is a useful early stop.

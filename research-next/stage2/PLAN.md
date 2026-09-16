# RECUT stage 2: transactional multi-window integration

Prepared 16 September 2026, before this stage's pretrained-model measurements. This extends engineering scope; it does not supersede the initial pilot's negative publication-readiness assessment or establish novelty over LayerSkip/VeriCache.

## Decision and scope

Build a single-owner, in-process transactional session over the existing frozen incremental Hugging Face engine. Run multiple consecutive decode windows. Hold outputs private until each window finishes its checks or recovery. Route repair from observed changed prefix layers, never from injected-location metadata. Compare with checkpoint-alias full replay under the same checkpoint, detector and commit assumptions.

Supported cache faults are persistent finite numerical changes to one or more old-prefix KV layers, occurring only during speculative execution after checkpoint creation. Multiple layers and staggered injection times are included. Their earliest changed layer supplies a conservative clean cut. A change that reverts before inspection is not covered. Initial state, weights, control metadata and digest storage are trusted; guards, recovery and commit are fault-free. This is not a general detector, concurrent server, crash-durable transaction or production deployment.

Checkpoint and journal SHA-256 seals guard post-capture storage mutation. They do not establish semantic cleanliness of captured state. Checkpoint damage poisons the session and releases no new outputs. Damaged/missing/misbound saved activations cause complete replay from a valid checkpoint. After a corrected token diverges, later journals are never reused, even if token IDs later reconverge. Recovery restores all layers at/above the selected cut, not only layers with direct faults.

## Prespecified measurement matrix

Two existing pinned models; prefix lengths 32/128; window lengths 8/16; three consecutive windows per session; seed 7431; two measurement repetitions per method. Six scenarios: clean, one persistent prefix fault, two staggered prefix faults, a repeated fault in every window, prefix fault plus journal damage, and checkpoint damage. Full replay and sparse replay use the same cache faults and clean reference. Journal damage is meaningful only for sparse replay and is labelled as a method-specific guard challenge, not a fair performance workload by itself. Checkpoint-damage trials are availability/refusal checks, not successful recovery-speed samples.

The unpadded batch-one greedy engine, eager attention and original model revisions remain fixed. All cases are declared before pretrained measurements; freeze source/config hashes after local synthetic-model tests and review. Original stage-1 code/data are not modified. A debug/preflight failure blocks the main benchmark; failed evidence is retained and any engineering amendment is recorded before a fresh run.

## Correctness and cost

Native Hugging Face incremental forward supplies the independent clean reference. Check every committed window's token sequence, complete logical KV state and logits, plus continuation outside timed work. Test no premature output, consecutive-window state transfer, earliest-layer selection, journal failure fallback, checkpoint refusal, malformed cache and exceptions. Synthetic control-flow tests cover divergence before the end and reconvergence; suffix/reverted faults are explicit negative controls, not successful protections.

Charge complete BEGIN-to-COMMIT wall time, including checkpoint creation/sealing, journal storage/sealing, prefix comparison, recovery, guards and CUDA synchronization. Setup/prefill and reference validation are outside this boundary. Artificial injection is included in raw window wall time and separately measured; any subtraction is a labelled diagnostic, not a deployment measurement. Report clean and fault windows separately, replay-only timing separately, held-output latency, memory payloads/peak allocation and paired ratios. The deliberate fault frequency is not a hardware fault rate and does not justify overall production speedup.

## Boundaries and exit criteria

Retain all failures. No supported-case wrong committed state/token, missed intended changed-prefix fault, new token committed after checkpoint rejection, or altered stage-1 artifact is acceptable. No positive speedup is required to call the bounded experiment complete. A slowdown is an actionable result about this integration, not a reason to tune thresholds or omit cases. No architectural binary instrumentation, existing AWS changes, credential storage, new thesis-publication promise or automatic GitHub push is in this step.

Prior-work anchors rechecked for this stage: [LayerSkip, ACL 2024](https://aclanthology.org/2024.acl-long.681/), [VeriCache methods](https://arxiv.org/html/2605.17613v1). The testable advance is a measured, guarded multi-window integration, not a claim that activation reuse or rollback has become new.

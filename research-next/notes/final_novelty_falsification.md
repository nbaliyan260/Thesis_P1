# Final bounded novelty-falsification pass

Accessed 16 September 2026. Scope: test whether prior LLM KV-cache numerical-fault recovery already saves clean hidden inputs, reuses them only while corrected token history agrees, then abandons old journals and regenerates the remaining private window through all layers. Six primary records received focused method reading; this is not an exhaustive survey, full implementation audit or independent reproduction. No implementation/HPC actions or exploit reproduction were performed. The first two prospective PDFs were not changed.

## Outcome

No inspected method was confirmed to contain that entire **delayed numerical-fault recovery** package. This negative search result does not establish novelty. More importantly, a published self-speculation method substantially overlaps its central reuse-and-disagreement logic. The conditional assessment becomes more skeptical: the divergence boundary itself should not be presented as a new general algorithm. The current pilot is evidence about a narrowly instantiated recovery contract, not yet a demonstrated A/A*-level thesis contribution.

## Search and access log

Queries actually issued, grouped in order:

1. `"KV cache" "silent" "recovery" activations divergence`; `"KV cache" "fault" "replay" activation recovery`; `"LLM" "corruption" "token divergence" recovery`.
2. `"GhostServe" checkpointing KV cache arxiv`; `"MemGhost" KV checkpoint recovery arxiv`; `"Characterization-Guided GPU Fault Resilience in NVIDIA MPS" arxiv`; `"KV" "activation" "divergence" "recovery" LLM fault`.
3. `"VeriCache" KV cache arxiv verify compressed`; `"LayerSkip" "rollback" "cache"`.
4. `site.aclanthology.org "LayerSkip: Enabling Early Exit Inference"`; `site.proceedings.mlsys.org "GhostServe"` for publisher status.

Search snippets and secondary pages were discovery leads only. The apparent “MemGhost KV recovery” lead was not accepted: the search also returned an unrelated persistent-agent memory-injection paper. No attack payloads were inspected or reproduced. AnchorTP appeared in discovery but was not selected for another method read once the six-record budget was filled. HCache, HybridServe, KVPR, classical delayed-error checkpointing and the previously reviewed vLLM RFC were not re-surveyed. GhostServe is the one previously abstract-screened record deepened here because its recovery method was directly relevant.

All six selected method sources were accessible in browser-readable primary HTML/author repository text. No PDF extraction or visual inspection was needed. Some combined browser returns were truncated; the relevant LayerSkip pseudocode and GhostServe recovery paragraphs were reopened individually. Depth below refers only to material actually inspected, not cover-to-cover reading. HTML line numbers are retrieval locators, not PDF page numbers.

## 1. LayerSkip — strongest algorithmic overlap

**Status/access/depth:** Published ACL 2024, verified in the [publisher record](https://aclanthology.org/2024.acl-long.681/). Focused reading of [author full text](https://arxiv.org/html/2404.16710v1), §§4.3.1–4.3.3 and Appendix A.4; HTML lines 208–232 and 672–786.

**Source-derived mechanism:** Early layers draft tokens and save KV plus an exit-layer query cache. Verification computes the remaining layers using those saved values. The algorithm accepts the matching token prefix and a verified correction, crops the shared cache to the accepted frontier, and continues from the pending token. This is already activation reuse coupled to token-agreement checks and cache rollback.

**Our inference:** This materially undermines novelty claims about the reuse gate or disagreement handling alone. Its stated task is self-speculative acceleration, not restoring an identified corrupted old-prefix layer from a checkpoint. It resumes fresh drafting rather than permanently executing every layer for the rest of an existing held window. That difference is a narrower recovery policy, not automatically a substantial new contribution.

## 2. VeriCache — lossy KV with exact-output correction

**Status/access/depth:** Retrieved 2026 arXiv preprint; no publisher venue was verified in this pass. [Accessible full methods](https://arxiv.org/html/2605.17613v1), especially §4.1, HTML lines 186–208; runtime scheduling context in §§4.2 and 5.

**Mechanism and overlap:** Compressed KV drafts tokens; full KV verifies the drafted positions. At the first mismatch, it accepts the earlier matching tokens plus the verifier correction and discards the remaining draft. Full KV can reside outside GPU memory, with transfer/verification scheduling central to the system. Thus correcting numerically altered KV trajectories while preserving exact output is not an unoccupied problem. The inspected method does not save a provably clean lower-layer journal for late single-layer corruption or prescribe RECUT's permanent full-layer suffix replay. Its verification is a full-model speculative pass, not the pilot's incremental upper-layer recovery path. Performance claims were not independently checked.

## 3. ExactKV — author-hosted diagnostic framework

**Status/access/depth:** [Author repository technical report](https://github.com/utkarshg20/ExactKV/blob/main/paper/ExactKV_Technical_Report.md), mutable `main`, accessed on the date above; peer-reviewed publication status not established. Focused reading of §§2.1–2.2, 3.1–3.4, 10–12 and limitations; HTML lines 226–308 and the named later sections, not every experiment table.

**Mechanism and overlap:** Its documented loop drafts with compressed KV, verifies against authoritative full KV, commits a matching prefix/correction, then realigns compressed state. It explicitly credits VeriCache for the algorithmic pattern and positions itself as measurement infrastructure. First-divergence indexing and recovery diagnostics therefore also have close prior examples. The inspected method does not add clean layer-cut activation replay. This is primary author evidence, not equivalent in evidentiary status to an accepted paper; its large reported experiment counts were not audited or imported into RECUT claims.

## 4. Characterization-Guided GPU Fault Resilience in NVIDIA MPS

**Status/access/depth:** [arXiv full text](https://arxiv.org/html/2605.26461v1), retrieved preprint; no publisher venue verified here. Focused reading of §6.1–6.2 and §7.2, HTML lines 239–258 and 302–318.

**Mechanism and overlap:** A standby process shares physical GPU pages containing weights/KV while maintaining separate process state. Periodic forward-state metadata identifies request progress; takeover replays the gap after the last published state. The inspected design relies on preserved GPU-resident contents, not correcting a silently contaminated speculative suffix. It gives a concrete earlier example of bounded token replay and recovery/normal-overhead trade-offs, but no observed activation-cut/first-disagreement repair policy. Driver/fault experiments were neither executed nor operationally reproduced.

## 5. GhostServe — existing record, deeper recovery check

**Status/access/depth:** [MLSys 2026 publisher record](https://proceedings.mlsys.org/paper_files/paper/2026/hash/d99e8e80a6c41e148db686918dd7eab3-Abstract-Conference.html). Focused reading of [full text](https://arxiv.org/html/2605.00831v1), §4.1, §4.2's partial-recomputation paragraph and §8; HTML lines 127–148, 191–192 and 318–323. Not a full artifact audit.

**Mechanism and overlap:** Streaming KV is protected with parity; recovery reconstructs unavailable cache data. The scheduler can overlap recomputing an initial cache portion with parity/data transfer for the remainder. This is stronger evidence than the earlier abstract-only screen that partial recomputation itself is established. The limitations discuss soft memory errors and static runtime topology, so describing the paper only as generic node fail-stop recovery would be too coarse. Nevertheless, the inspected recovery passages do not specify correcting a delayed contaminated token trajectory by reusing lower-layer hidden journals until token divergence.

## 6. TARRAGON — request-level fail-stop restoration

**Status/access/depth:** [arXiv full text](https://arxiv.org/html/2601.01310v1), retrieved preprint; no publisher venue verified here. Focused reading of failure model §3.3, worker replay §5.1 and cache-management §§6.1–6.2; HTML lines 203–206, 223–226 and 249–263.

**Mechanism and overlap:** Attention workers checkpoint token/layer KV increments and commit request progress; a replacement restores the affected request and resumes from its committed token. Stateless expert work can replay on healthy workers. The declared failure model is fail-stop, including unreachable workers/links, with Byzantine behavior out of scope. The method reinforces that request-local checkpoint/replay and layer-granular retained work are established. The inspected protocol does not recompute a corrected token history after delayed numerical contamination or condition reuse of a lower-layer activation journal on that history.

## Consequences for claims and next decision

These readings do not justify “first,” “novel recovery primitive,” “new token-divergence rule,” or “numerical KV corruption has no exact-output recovery prior art.” A defensible description is: an auditable, controlled instantiation of a fault-specific recovery contract, adapting existing checkpoint/reconstruction and speculative-reuse ideas, with explicit trust assumptions and full-state checks. The claim requires qualification because similarity of outputs alone is weaker than state equivalence, but stronger testing by itself is not a sufficient novelty argument.

The remaining thesis question is whether realistic delayed-fault localization, trusted-state maintenance, output commitment and optimized recovery can yield a nontrivial systems or theoretical advance over adaptations of established mechanisms. Current measured correctness or recovery-only speedups cannot answer that by themselves. Do not immediately add more baselines or change this completed pilot in response: document the overlap, preserve the evidence, and use it as a supervisor decision gate. No exact overlap found in six focused readings is a bounded search outcome, not a patentability judgment, absence proof, or publication recommendation.

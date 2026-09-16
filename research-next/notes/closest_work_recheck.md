# Closest-work mechanism recheck

Root checking pass on 16 September 2026 while the frozen GPU experiment ran. Selected mechanism and evaluation sections were revisited; this is not a claim to reproduce either published system.

- [HCache](https://arxiv.org/html/2410.05004v1), sections 3.1-3.2 and selected evaluation/ablation sections: per-layer inputs support KV reconstruction without redoing the complete decoder, coupled with a storage/compute schedule. Its purpose and implementation include offloaded historical-state restoration. The reported MHA storage argument must not be copied as a universal GQA ratio. This verifies that activation-assisted restoration is prior art, not RECUT novelty.
- [HybridServe](https://arxiv.org/html/2501.01792v1), sections 3.3-4.3 and implementation methodology: hybrid activation/KV representation and resource-aware allocation are already explicit. The published system's optimized serving integration is not reproduced by this pilot. Storing activations or choosing a compute/storage tradeoff alone cannot substantiate a new contribution.
- [KVPR](https://arxiv.org/abs/2411.17089), author record revisited: partial recomputation is another incumbent approach; the present study must compare the required recovery contract rather than imply all alternatives perform only naive full-token replay.

RECUT's conditional distinction is not a better general activation cache. It concerns validity of speculative intermediate state after a localized numerical error has propagated and possibly changed input history. The fixed-cut rule is sufficient under a narrow model, not a minimal-replay theorem. Clean activation-restoration techniques still need a provenance/trust decision before speculative activations can be reused; a missing exact combination in the accessed sources is not proof of priority or of a material advance.

The follow-up GPU study therefore challenges the original conservative baseline with view-based restoration and a no-prefix-copy checkpoint-alias replay. This closes a concrete local confound only; it does not replace an optimized adaptation of the complete prior systems, or demonstrate superiority over them.

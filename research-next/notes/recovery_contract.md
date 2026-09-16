# RECUT: conditional recovery contract and cost test

16 September 2026. Read against `RECUT_01_Idea_and_Novelty.md` and `RECUT_02_Implementation_Plan.md`. This is a mathematical specification and proof sketch, not machine-checked verification, a detector guarantee, or a demonstrated performance result. No experiment code or HPC state was modified.

## 1. Model and indexing

Let the decoder contain **N layers indexed 0,...,N−1**, and let **L** be the layer whose existing prefix KV is perturbed after a trusted prefix checkpoint of length **T**. Transition **t**, for t=1,...,W, consumes input token x_t, appends that token's KV at each layer, and chooses x_(t+1) by a deterministic greedy rule. The initial x_1 is selected before the perturbation from clean prefix logits. Therefore, after t transitions, the processed cache has length T+t; the last newly selected token x_(t+1) has not yet been inserted.

A cut **c** denotes the input to layer c: c layers are skipped and N−c layers are replayed. Cut 0 means start from the token embedding, with no activation journal required. A journal records immutable copies of layer inputs, together with request/model identity, token position, cut, input-token history identity, and positional/cache metadata. Token identity alone is not a sufficient validity condition.

Assume:

1. **Layer-causal architecture.** Layer ℓ reads only its own previous KV and the current input produced by lower layers. No higher-layer recurrent state, cross-layer cache aliasing or shared mutable state can influence lower layers except through selected token IDs. Ordinary decoder attention with independent logical per-layer KV satisfies this abstraction; transformed/shared-layer and hybrid recurrent architectures need separate proofs.
2. **Specified fault.** The only direct perturbation is in layer L's prefix KV. The checkpoint predates it and is clean. Other caches, weights, journals, metadata and the recovery execution suffer no additional faults. Localization is trusted; recording the faulty execution is not itself verification.
3. **Deterministic execution equivalence.** Clean reference and replay implement identical per-layer numerical functions and greedy tie-breaking, including precision, reduction ordering, attention masks, position IDs, logical cache lengths and relevant kernel selection. Merely pinning weights or setting evaluation mode does not establish this assumption.
4. **No premature commitment.** The W outputs and any actions derived from them remain private until verification/recovery completes. The retained checkpoint and journals cover the whole affected interval.

Write hats for the speculative, perturbed run. Let j be the first transition with clean output x_(j+1) different from speculative output x̂_(j+1); set j=∞ if none occurs within W. Let k=min(j,W).

## 2. Conditional proposition

**Proposition (conservative clean-cut recovery).** Under assumptions 1–4, select a saved cut 0≤c≤L, or fall back to c=0. Restore clean prefix KV for replayed layers c,...,N−1, discard their speculative suffix, and replay transitions in order from the saved layer-c inputs while corrected and speculative output histories agree. Include transition j in this cut-based replay. If its output differs, discard every speculative journal entry after j and truncate **all retained lower-layer caches to length T+j** before regenerating transitions j+1,...,W through layers 0,...,N−1 using corrected tokens. Then the resulting W outputs, every logical KV entry, final logits and next input token equal the clean deterministic W-transition execution.

This equality covers the active logical state and metadata, not unused bytes in overallocated GPU buffers. Numerical bitwise equality is conditional on assumption 3; no tolerance claim is silently substituted for it. For a fault-free continuation under the same assumptions, equality persists by induction.

### Proof sketch

**Clean lower-region lemma.** Before the first differing output, the two runs have identical input-token histories. Induct on transitions and, within each transition, on layers below c. Their initial prefix states are unchanged by the layer-L fault; their current inputs and layer-local prior KV are equal. Determinism gives equal outputs and appended KV. Consequently ĥ_t^c=h_t^c for every t≤k. This includes c=L because the saved value is the input to layer L, before its faulty attention read. Saved inputs above L have no such structural guarantee.

**Replayed upper region.** Its prefix is restored from the trusted checkpoint. Induct on t≤k. The journal supplies the clean layer-c input; upper-layer cache entries from earlier replayed transitions are already clean. Determinism yields the clean upper-layer outputs, appended KV, logits and greedy decision. If that decision matches the old one, the induction continues. If it differs at j, transition j itself is still valid because its input history has not yet diverged.

**Divergence boundary.** At the end of transition j, lower-layer KV through T+j is clean by the lemma and upper-layer KV through T+j is clean by replay. Later speculative entries are not justified by that lemma and must become unreachable. With all caches trimmed to this clean boundary and the corrected x_(j+1), ordinary full-layer replay reconstructs the remaining clean execution. If no divergence occurs, the retained lower layers and replayed upper layers are already clean through T+W. This proves the claim.

Two operational details are necessary: do not expose future speculative entries through a mask or incorrect cache-length counter during replay; and do not reuse a later journal simply because its individual token ID happens to match again after an earlier divergence. Its preceding history can still differ.

## 3. Limits and counterexamples

- **Already-emitted output:** if a committed token is b while the clean run would have emitted a≠b, no later cache transformation can make the already observed transcript equal the clean transcript. State repair can support future service, but transcript repair requires an application-level retraction protocol. External effects are not undone by this theorem.
- **Wrong localization or dirty checkpoint:** if the actual fault is below c, the saved layer-c input may already be contaminated. Restoring the reported layer and replaying from that input can preserve the wrong result. If the trusted checkpoint itself is bad, even c=0 replay from that checkpoint has no clean-reference guarantee. An uncertain layer can be handled conservatively only if a trustworthy lower bound on its earliest possible location and a clean time boundary exist.
- **Corrupted or misbound journal:** a corrupted hidden value, stale request mapping or incorrect position can drive a deterministic replay to the wrong state. Token comparison does not independently certify that value. Fall back to a separately trusted checkpoint/token history when journal integrity is uncertain.
- **Changed kernels:** different accumulation order can change low bits and flip an arbitrarily small greedy margin. Equal mathematical real-valued formulas therefore do not prove bitwise state or sequence equivalence. A relaxed numerical contract needs separate error bounds and margin assumptions; it is not this proposition.
- **Unsupported state:** cross-layer/recurrent state, concurrent mutation, hidden cache compression state, or faulty model weights can violate the dependency structure. Multiple errors can be covered only after extending the fault set and proving a clean boundary below every possible source.
- **Sampling:** without a specified coupling and saved random state, the same logits do not imply the same realized token sequence. Start with greedy decoding as planned.

The proposition is **sufficient, not minimal or optimal**. A particular error may have no effect, special attention structure may permit smaller repair, and a deeper cut may happen to be valid. None is certified solely by the layer index. Observing agreement on several outputs also does not prove complete cache equality.

## 4. Work, time and memory

For a fixed selected c≤L and first divergence j, count only executed layer-token steps:

\[
Q_0=NW,\qquad Q_c=(N-c)k+N(W-k)=NW-ck,\quad k=\min(j,W).
\]

Thus the saved work is ck. With no divergence, it is cW; with immediate divergence at transition 1, it is only c. c=0 gives no saved layer work. An all-layer journal can choose c=L, giving the largest structural cut in this conservative scheme, but not a universally optimal recovery algorithm. The arithmetic-only speedup proxy NW/(NW−ck) omits journal/checkpoint handling, heads, launch costs and memory effects; it is not a measured latency prediction.

Let a_(ℓ,t) be measured cost of executing layer ℓ at transition t using the prescribed path; B_0 and B_c are restoration costs; G is common output-head/decision work; and E_c includes journal reads, comparisons and truncation. A first-order accounting model is

\[
R_0=B_0+G+\sum_{t=1}^{W}\sum_{\ell=0}^{N-1}a_{\ell,t},
\]
\[
R_c=B_c+G+E_c+
\sum_{t=1}^{k}\sum_{\ell=c}^{N-1}a_{\ell,t}+
\sum_{t=k+1}^{W}\sum_{\ell=0}^{N-1}a_{\ell,t}.
\]

Kernel fusion and overlap can make additive profiling inaccurate; use measured complete repair times for the decision. Both methods must receive the same clean checkpoint and reference semantics. A faster batched-prefill baseline may be added, but report its changed numerical contract separately.

For batch size one, activation element size b and selected nonzero cuts S, journal payload is approximately M_S=bW∑_(c∈S)d_c, plus metadata, alignment and buffers. The full prefix KV checkpoint payload is P=2bT∑_(ℓ=0)^(N−1)h_KV,ℓ d_head,ℓ when both K and V use b bytes. Count it for every method. With uniform width d=h_Q d_head, one activation divided by its same-layer KV is h_Q/(2h_KV): it can exceed one for GQA. Do not inherit a blanket “half the KV bytes” claim from an MHA setting.

## 5. Break-even and Pareto test

For a fixed detection/commit protocol, let p be the probability of a supported incident requiring recovery per window, H_S the additional steady-state cost of collecting/maintaining journals per window, and ΔR_S=R_0−R_S the conditional recovery saving. If journal overhead is approximately the same in clean and affected windows, the expected incremental time is

\[
\Delta C_S=H_S-p\,\mathbb{E}[\Delta R_S\mid\text{incident}].
\]

An expected-time advantage needs a positive conditional saving and

\[
p>H_S/\mathbb{E}[\Delta R_S\mid\text{incident}].
\]

If overhead differs, use (1−p)H_clean+p(H_fault+R_S−R_0), not the simplified expression. Charge additional verification, checkpoint protection or metadata work in H or R as appropriate. Do not estimate p from the number of deliberately injected test episodes. When p=0, positive journal overhead cannot improve average processing time. If p or the fault mix is unknown, report the threshold and sensitivity rather than a deployment-speedup claim.

Compare cut policies using a common correctness/fault coverage, detector, checkpoint and commit contract. Policy A dominates B only if its steady-state overhead, memory/peak-capacity cost, recovery latency distribution and output holdback are all no worse, with at least one strictly better. Otherwise report a Pareto tradeoff. A fixed W-step holdback adds user-visible latency even when repair is never needed; a favorable throughput break-even does not cancel this latency. Methods that detect before use have different costs/coverage and must be compared end-to-end, not forced into the same p assumption.

## 6. Honest novelty interpretation

The proof is a straightforward deterministic dependency/rollback induction. That is useful for specifying and testing software, but is not by itself a deep new correctness theorem. Delayed detection, checkpoint retention and verification-overhead optimization are classical topics; [Aupy et al., PRDC 2013](https://arxiv.org/abs/1310.8486) explicitly analyze these issues. Checkpointing/holdback and the break-even arithmetic above should not be claimed as inventions.

[HCache §3.1](https://arxiv.org/html/2410.05004v1) already restores KV from saved layer inputs, and [HybridServe §3.3–4.2](https://arxiv.org/html/2501.01792v1) combines activation and KV representations. Reconstructing KV from activations and choosing a storage/compute tradeoff are established. A provenance-aware all-layer comparator here is an adaptation of that idea, not a claim to have reproduced either complete system.

The possible contribution is narrower: a carefully scoped validity frontier after delayed numerical corruption and token divergence, plus convincing evidence that sparse cut selection produces useful matched-resource recovery points. It remains vulnerable to the objection that this is an obvious application of existing checkpointing to a known dependency graph. A thesis/paper must earn its distinction through a missing practical recovery case, stronger system integration or a consequential general result—not the acronym, the induction, or a recovery-only speedup against an unfair baseline.

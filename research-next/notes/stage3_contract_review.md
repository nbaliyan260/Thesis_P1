# RECUT stage 3: conditional correctness and contract review

Review date: 2026-09-16. This is a code-grounded internal argument, not a machine-checked proof, an independent external replication, or a certification. The reviewer also implemented the stage-3 sealing change; the separate test and runner work provides additional internal checks, not external independence. GPU study results were pending when this note was written.

## Bottom line

I found no contradiction between the implemented replay algorithm and its **narrow conditional contract**. The critical condition is stronger than “some persistent fault is eventually detected”: every directly corrupted cache layer that could have affected speculative execution must remain numerically distinguishable in the old prefix at detection, unless the controller takes an unconditional full replay. Otherwise the earliest detected layer need not be a safe dependency boundary.

The implementation does not establish general silent-data-corruption detection. It recovers a guarded, single-owner autoregressive inference window under specific data-mutation and execution assumptions. It does not recover model training, damaged weights, crashed processes, or arbitrary transient GPU faults.

“Clean” means the fault-free execution of the same fixed finite-precision model/configuration and greedy decoding process. It does not mean that the model's answer is factually correct, safe, aligned, or useful for its task. State equality is finite elementwise equality, not universal raw-bit identity.

Code reviewed:

| File | SHA256 |
| --- | --- |
| stage3/recut_controller.py | 4ebca316ee5c4a745cf7b7535acef499de21b0d249cd8fafe53fa1815c54b46a |
| experiments/recut_engine.py | c0e0949ccf415853c3c047dbbba727df81539531a53704bb3479856bf8b7cac9 |
| experiments/prefix_detector.py | 0324ad63f6e63c573afd6ee94c978629b03b197e9fe072bc70e999191df93418 |

Line references below refer to these frozen versions.

## 1. Assumptions required by the argument

1. The initial prefix KV cache and pending input token are semantically correct for the intended token history. Constructor layout and finiteness checks do **not** establish this. Model weights, decoding policy, control metadata, seals and executable code are trusted.
2. The model satisfies the supported engine contract: pinned Transformers 4.51.3, native Llama or Qwen2 causal decoder, eager attention, evaluation mode, ordinary DynamicCache, default RoPE, no sliding-window/quantized/offloaded cache, one unpadded sequence, and one-token decode steps. Stage-3 measurements use the declared FP32 or BF16 setting.
3. Legitimate attention updates append new K/V entries without rewriting old entries. Decoder dependencies flow from lower to higher layers within a step; feedback to lower layers in later steps occurs through the selected next token.
4. Direct working-cache faults occur only in a transaction's old prefix during speculation, after independent working/checkpoint copies and the initial checkpoint seal exist. Every potentially corrupting layer retains at least one numerically changed old-prefix element until detection. Multiple and staggered faults are allowed within this condition.
5. Journal/checkpoint challenges are post-capture mutations. Guard, recovery, copying, seal computation, final validation, and commit execution are reliable. There are no concurrent writers, other-stream races, new faults during those trusted periods, or mutations between transactions.
6. SHA256 comparison is treated as a reliable integrity check for these accidental mutations, with no digest collision in the evaluated states and trusted stored digests. This is a computational integrity assumption, not an unconditional mathematical guarantee or authenticated defense against an attacker who can replace both data and seals.
7. The caller respects single ownership. It does not mutate exposed public cache/logit objects or treat private transaction fields as authorized output. Python attribute accessibility is not an information-flow security boundary.

The controller's constructor checks supported cuts, a nonempty prefix, tensor layout, and finiteness (controller lines 191–218). Engine construction rejects unsupported model/cache settings (engine lines 184–217). These checks enforce some assumptions but do not prove semantic cleanliness, fault-free execution, or single ownership.

## 2. What the detector actually establishes

The detector compares only each working K/V tensor's first P positions against the checkpoint, where P is the transaction's prefix length. It computes elementwise inequality, consolidates one Boolean per layer, and returns every changed layer (detector lines 15–33). The controller validates the checkpoint's layout and seal before this comparison (controller lines 357–361).

For trusted finite checkpoint values and reliable comparison execution:

- A numerically changed old-prefix value is observable in its layer.
- An unchanged old prefix has no layer alarm.
- The detector does not identify every incorrectly computed suffix entry. Those are repaired by dependency reasoning and replay, not individually detected.
- Its output is not an externally supplied injection location. The harness knows injected layers for verification, but the public controller API does not receive that information.

**Signed zero matters.** The implementation uses tensor inequality, so +0 and -0 compare equal. “Persistent finite change” must therefore mean a persistent **numerically unequal** old-prefix value, not merely a different raw bit pattern. A sign-bit-only change from +0 to -0 is outside the detector's positive-detection contract. Conversely, checkpoint and journal SHA256 seals preserve raw bytes and can detect a signed-zero mutation in those guarded objects. Those are different contracts; they must not be conflated.

A finite wrong value already present before checkpoint creation is copied into both states and can remain invisible. A suffix-only fault, a fault that disappears before detection, or incorrect arithmetic with no surviving old-prefix mutation may also be invisible. A finiteness guard can reject some additional failures, but that is not a general semantic detector.

Particularly important counterexample outside the contract: a low-layer mutation corrupts a captured activation and later reverts; a separate high-layer mutation remains. The detector then reports only the high layer. A cut below that high layer but above the reverted low-layer fault may reuse a bad activation. The contract must exclude this combination, not just exclude windows with no alarm.

## 3. Conditional replay argument

Let P be the committed prefix length, W the speculative window length, F the set of all directly corrupted cache layers during speculation, and D the detected set at the end. Under the persistence assumptions, D includes every layer in F. In the controlled injector, D equals the directly changed layer set; equality is an empirical verification, not a semantic oracle inside the controller.

If D is nonempty, the controller chooses the largest saved cut c with c <= min(D), or c = 0 when none exists (controller lines 365–373). Hidden state saved at cut c is captured **before executing layer c** (engine lines 295–300). A fault in layer c's historical K/V therefore cannot directly corrupt that same step's incoming hidden state at c. Choosing c equal to the earliest affected layer is safe under the stated directional dependencies.

### Same-history prefix of replay

At the beginning of replay, every layer at or above c is restored to the trusted checkpoint. Layers below c keep their speculative suffix, and the saved activation at c is reused. For an input token whose entire preceding token history still matches speculation:

1. Layers below c had no direct in-scope fault.
2. They processed the same token history as the clean execution.
3. Their current input activation and saved hidden state at c are therefore the clean ones.
4. Executing layers c through the final layer from restored/rebuilt state yields the clean output and corresponding upper-layer cache entries.

The engine additionally checks journal position, token, cut, engine identity and tensor layout (engine lines 262–280). These local checks help catch misuse, but equality of a single current token is not enough to establish equality of its entire history. That is why the controller's persistent divergence state is essential.

Lower layers may temporarily contain future speculative K/V entries while skipped during replay. The engine derives the attention-mask length from the explicit current position rather than the longer layer-zero cache (engine lines 284–292); executed layers must be append-ready. This handling is specific to the supported eager DynamicCache path and should not be generalized to other cache/mask implementations without new validation.

### First token divergence

Suppose the first corrected next-token output differs at replay step d, counting from one. Up to processing that step, its input history was still shared, so the cache through length P + d is valid. The predicted differing token is the input to the **next** step; it is not yet a newly appended cache entry.

The controller records divergence permanently, truncates all layers to P + d, and runs every later step from layer zero (controller lines 328–349; engine truncation lines 75–97). It does not resume journal reuse if later token IDs happen to reconverge. Consequently it never reuses a saved activation derived from an incompatible earlier token history.

When no divergence occurs, the lower-layer speculative suffix is valid for the same complete token history; upper layers have been rebuilt. When divergence occurs, the prefix through d is valid and all subsequent layers/positions are regenerated. If c = 0, the full current window is rebuilt from the checkpoint and saved activations are not consulted.

### Clean route and induction across windows

If D is empty, the no-alarm commit is conditionally correct only because the contract excludes other sources of bad speculative state and requires in-scope cache mutations to remain observable. A no-alarm result alone does not establish those assumptions.

After either valid route, finiteness/layout checks and a final checkpoint seal check precede publication (controller lines 385–390). A committed window supplies the trusted state for the next transaction, assuming no intervening fault. Thus the one-window argument extends by induction to sequential windows under the same assumptions. This is a reasoning argument about the code, not a mechanized proof.

## 4. Integrity challenges and already committed output

BEGIN makes independent working and checkpoint copies while preserving the previous public cache (controller lines 252–259). Speculative step() returns None and does not append public committed tokens (lines 295–315). A successful finish prepares its result/transcript before publishing the new fields (lines 390–403).

| Challenge | Implemented behavior | What remains valid |
| --- | --- | --- |
| Post-capture journal mutation or misbinding on a required partial-replay path | Journal seal/structure check fails; replay starts from layer zero using the trusted checkpoint | Previously committed output is unchanged; the current window can still be regenerated |
| Damaged checkpoint detected before replay | WindowRejected; session poisoned; no new tokens committed | Previous public cache, logits, pending token and transcript remain untouched under the ownership assumptions |
| Replay/validation exception or final checkpoint guard failure | Reject and poison, with no successful finish result | The old committed public state remains separate from speculative/recovery state |
| Journal damage when no replay is needed | Unused journals need not be validated | Journal tensors are cloned capture records, not live forward inputs, so unused damage does not by itself change the committed output |

Checkpoint rejection does **not** repair the checkpoint, automatically restart the session, or retract earlier committed windows. Previously correct public state is preserved; the API deliberately stops the poisoned session.

Full-policy execution captures no saved activations and cannot use journal replay. Stage 3 consequently omits hashing empty journal rows. Its private seal list uses None placeholders; this does not change the full-policy public path. Full and sparse/all-cuts journal-corruption runs are not identical interventions, and their challenge timings must remain excluded from fair same-treatment policy comparisons.

“Atomic commit” here means the caller observes a completed method under single ownership. Public attributes are assigned sequentially, without a lock, write-ahead log or durable transaction. Thread races, process interruption between assignments, network delivery and external side effects are not covered.

## 5. Why batched seals preserve the old digest

Stage-3 cache/journal seals retain the same metadata serialization, ordering and byte representation as stage 2 (controller lines 62–154):

1. Cache header, K-before-V collection labels, tensor layout metadata and canonical contiguous tensor bytes have the same order.
2. Journal owner/index metadata and sorted cut entries have the same serialization and tensor layout/payload order.
3. Viewing a contiguous tensor as uint8 reinterprets its representation; it does not numerically convert or normalize signed zero.
4. The batched implementation concatenates byte views by device, copies that byte buffer to CPU, then restores each original tensor's byte slice at its original place in the digest stream.
5. CPU memoryviews retain their backing NumPy/tensor storage for hashing. Legacy mode copies, hashes and releases tensors individually.

Therefore the digest streams match when the input tensors and metadata are stable throughout the guard and transfers are correct. Grouping tensors by device does not change digest order. Including device strings in metadata is intentional: equivalent values on different devices need not have equal seals; equivalence is to stage 2 on the same original layout/device.

This is **not an atomic snapshot of concurrently changing tensors**. Neither the original per-tensor schedule nor the new packing kernel establishes a race-free snapshot against arbitrary concurrent writes. The trusted guard/single-owner assumptions are necessary. A CPU batched memoryview also reads the stable source storage directly; the same assumption applies.

Batching trades fewer device-to-host copy operations for a transient device staging buffer and a host copy. Seal counters report payload transfers only, not all controller traffic, allocator peaks, host peak memory, or an exhaustive timing decomposition. Optional component profiling adds synchronizations and remains separate from unprofiled measured rows.

## 6. Evidence boundaries and remaining tests

The local regression suite and CPU integration fixture provide executable checks, including native-model comparisons, mutation guards, multistep windows, synthetic divergence paths, canonical digest equivalence and saved-tensor reloading. They do not prove the assumptions above. The main pretrained GPU study was still pending at this review; its actual failure/success outcomes and divergence coverage must be reported separately.

Important limitations or unestablished behaviors:

- Persistence is imposed by the software injector, not inferred from field hardware behavior. Fraction-bit flips and additive changes do not reproduce every register, arithmetic, cache-controller or architecture-level failure.
- Suffix-only faults, reverted faults, signed-zero-only working-prefix changes, faulty computation, bad initial/prefill state, mutated weights, correlated control/seal corruption and faults during guards/recovery remain outside the positive correctness claim.
- A synthetic forced-divergence unit test establishes a control-flow case, not the frequency or generality of natural pretrained-model divergence. Report whether the real runs exercise partial-to-full fallback.
- Fixed quarter-layer cuts and an all-cuts comparison do not establish optimal cut placement or a workload-adaptive policy.
- Only the prespecified tensor archives are independently reloadable. Other trials retain recorded live native comparisons; hashes and internal review are not external replication.
- Common native batched prefill is trusted setup. Post-prefill custom/native parity does not demonstrate interchangeable prefill kernels or correctness of a corrupt prefill.
- No multi-request batching, tensor/pipeline parallelism, asynchronous service, FlashAttention/SDPA, quantized KV, sliding-window cache, sampling RNG rollback, distributed recovery, concurrent writer or process-crash semantics are established.
- Native/bare clean baselines quantify extra protection cost but are not resilience-equivalent alternatives. Timing repetitions are not independent physical fault samples. Break-even calculations are sensitivity models, not estimates of real fault incidence.

Suggested precise claim: “For supported single-sequence eager autoregressive decoding, RECUT preserves previously committed output and reconstructs the clean finite state of each released window under trusted guards and persistent, numerically observable post-checkpoint old-prefix KV mutations. It reuses saved lower-layer work only while replayed token history agrees, and otherwise falls back irreversibly to full-layer replay. The experiments test this bounded claim; they do not establish general GPU fault tolerance.”

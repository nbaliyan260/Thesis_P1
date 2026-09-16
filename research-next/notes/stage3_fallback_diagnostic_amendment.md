# Stage 3 amendment: selected natural-fallback replay, validation only

Date: 2026-09-16. This amendment is written **after observing the original main study outcome and before submitting the diagnostic replay**. It is not a preregistration of that outcome. The main protocol, frozen implementation, source manifest, main records, and their analysis remain unchanged.

## Selection and purpose

The selected case is `HuggingFaceTB/SmolLM2-135M`, prompt `code`, initial prefix 128 tokens, window 8 tokens, three windows, `multi_prefix`, seed 7431, timing repetition 0, and the full/sparse/all-cuts batched policies. The original affected window (zero-based index 1) exhibits a corrected-token divergence at index 7. Sparse and all-cuts replay consequently return permanently to layer zero for the remaining step. This replay adds saved recovered/reference tensors for that already-observed case and checks them independently on CPU. It tests reproducibility and the saved-state correctness of this selected trace; it does not discover a new case or new mechanism.

**Every diagnostic session, timing, refusal, committed window, parity check, and snapshot is excluded from all main-study performance aggregates and counts, all novelty claims, and all claims about independent fault samples or fault frequency.** The clean and checkpoint-challenge sessions are required protocol controls, not extra main evidence. No performance improvement is claimed from this replay.

## Original evidence binding

Original directory: `runs/stage3-main-v1/smol135m`. Original workload ID: `HuggingFaceTB--SmolLM2-135M--code--p128--w8`.

| Binding | SHA-256 |
| --- | --- |
| Original `metadata.json` bytes | `8a1695ba05c046af03fb7e1a6a7910c53b53efc594a94ab9cfe596f3220633e4` |
| Original `sessions.jsonl` bytes | `fe8b1c0c4d785590c66d8582bfff6b8bc23f1115e35255ab55923d1fc41fd403` |
| Original `references.jsonl` bytes | `0f1f36d1d0ea802d49e0a32e46d6d11362e56f3818018edc2cd8086fa5a2d7f5` |
| Canonical original source map | `b0726201ade8978a20daeb7edaaec192f5eadea864ede656e4ad40c83bfdb51d` |
| Canonical selected native reference | `4940d983e78f347cf57ac5133d4d07db61fafac6d647816da6a0a0d427bea149` |
| Canonical affected-window actual faults | `1f95e81b06757c50fedc9f2f6fdc23ee320a547fff52fceb0f955766ffde4e35` |
| Canonical full-policy three-window trace | `0eb0f62ed2c415964bcd43ed6ad1f9343f9928fa9c1075442a9624bfca2bfe4e` |
| Canonical sparse-policy three-window trace | `4b2733420fc00bfc2b6da799b37430d27029cc30d41d27117a91770a92957c6f` |
| Canonical all-cuts three-window trace | `f9a82a731e53bab958581e0c55896be031048e74ebd273109204e37f55592942` |

Canonical hashing means SHA-256 of UTF-8 `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)`. The source map is the complete original `source_sha256` mapping. The native reference is the complete matching record, including native KV/logit fingerprints and parity checks. The fault payload is the complete actual `faults` list in window 1, including coordinates and applied values. Each trace is the ordered three-window list projected onto `window_index`, `status`, `faults`, `speculative_tokens`, `tokens`, `detected_layers`, `selected_cut`, `first_divergence`, `replay_starts`, `fallback_reason`, and `recovered`. The verifier additionally compares these projected records directly, not only their hashes. Original source paths resolve relative to `stage3/`, including the recorded `../experiments/` entries.

## Declared diagnostic matrix and resources

`reporting/fallback_config.json` is an exact reduction of the original main configuration: only this Smol model, only its authored code prompt, shape 128/8, three windows, seed 7431, one repetition and one clean repetition, fault case `multi_prefix`, guard case `checkpoint_corruption`, legacy case `clean`, no warmups, and no profiling. The frozen validator requires a nonempty guard list. Snapshot selection uses prompt `code`, short prefix 128, case `multi_prefix`, and long case `multi_prefix`; the frozen OR selection yields one affected-window archive for each of the three batched policies.

Expected coverage: **13 measured sessions; 36 attempted windows; 33 committed windows; three specified checkpoint refusals; one reference with 26 post-prefill native/custom parity positions; three saved tensor archives; zero warmups; zero profile records.** Seven clean variants, three faulted batched variants, and three checkpoint-challenge batched variants constitute the 13 sessions. Each checkpoint refusal terminates its session after the clean first window and rejected second window.

Output is a new directory, `runs/stage3-fallback-diagnostic-v1/smol135m`. The existing original manifest and frozen runner, analyzer, and auditor are used without changes. Maximum allocation is partition `ws-ia`, one GPU, four CPUs, 8 GB host memory, and ten minutes. Source-freeze verification and original-case/config verification run before any model execution; freeze verification runs again afterward. There is no unit-test rerun and no modification of the original main output. Root inspects and binds this amendment and the new helper files before submission.

## Declared trace and acceptance criteria

In the affected window, all three policies must detect layers `[14, 22]` and have first corrected-token divergence 7. Expected replay starts are:

| Policy | Eight replay starts |
| --- | --- |
| Full | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| Sparse | `[7, 7, 7, 7, 7, 7, 7, 0]` |
| All-cuts | `[14, 14, 14, 14, 14, 14, 14, 0]` |

The frozen independent auditor must pass on HPC and pass again locally with tensor reloading enabled, producing separately bound `audit.json` and `audit_local.json`. The local audit uses `torch.load(..., map_location="cpu", weights_only=True)` and checks each entire saved recovered/reference cache, logits, tokens, and continuation state. All three archives must pass; the 30-layer model gives 122 paired tensors per archive, 366 total. Archive file hashes, byte lengths, identities, model revision, native references, all raw-record hashes, source/config/manifest hashes, and the auditor source hash must bind correctly. These are checks of recorded reference agreement under the stated fault contract, not a mathematical proof or independent hardware-fault validation.

The post-hoc verifier requires the exact original actual fault values and coordinates; exact speculative and corrected token traces; expected detector, cut, divergence, and fallback trace; native KV/logit/reference fingerprint equality; and both completed snapshot audits. It emits only `validation_only.json`, with an explicit exclusion flag and no performance aggregates. Any mismatch, missing archive, incomplete run, or failed audit makes verification fail. Do not silently tune the case, change seeds, substitute another case, weaken acceptance gates, or pool a rerun into main evidence. An unsuccessful replay must be reported as such or receive a separately explicit amendment.

## New helper bindings at preparation

`stage3/reporting/fallback_protocol_freeze.json` binds this amendment, the exact reduced configuration, submission script and validation verifier before diagnostic submission. The submission script invokes the verifier before model execution. That invocation checks all four hashes and prints the freeze/amendment digests into the scheduler log; final validation binds the same digests. The freeze deliberately does not hash itself. These new helpers are postmeasurement delivery/validation tools, not changes to the original source freeze.

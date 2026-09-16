# Independent RECUT design review

Reviewed: 16 September 2026. Reviewer: independent literature/safety agent, not recovery-runner author. Scope: `RECUT_01_Idea_and_Novelty.md` and `RECUT_02_Implementation_Plan.md`, followed by read-only engine review. No engine or runner files modified; with root authorization the reviewer added only `experiments/tests_independent.py` and ran local CPU unit tests. No HPC access or execution.

## Verdict before code

The conservative protocol is logically plausible for a narrowly scoped deterministic causal decoder with append-only, layer-local KV state and a trusted clean checkpoint. The central clean-cut argument is sound under those assumptions. The documents are appropriately conditional about novelty and performance.

However, the stated proof sketch is not enough to certify an implementation. The main failure opportunities are mismatched token/cache frontiers, hidden state outside KV, and accidentally re-admitting journals after divergence. The pilot must not claim a general theorem for compressed/sliding/recurrent caches or stateful generation solely from tests of ordinary DynamicCache.

## Findings and required clarifications

### F1 — Every layer must be truncated at the divergence frontier (high priority)

Plan §Phase 2 steps 3–4 (lines 21–23 in the reviewed version) says lower layers may retain their old suffix while inputs agree, then switches to layer zero. Switching the start layer alone is not sufficient.

Let transition j be zero-based. The prefix contains T processed tokens. The input to transition j is token x_j; that transition appends the KV corresponding to x_j and chooses output x_(j+1). If corrected output x_(j+1) differs, recovery has a clean processed prefix of length T+j+1. Before processing the new input, **all layers must be cropped to exactly that length**. Lower layers can otherwise still contain future entries created from discarded tokens. Reject any mismatch between explicit position, visible cache length, and causal mask.

During partial replay, lower layers may physically retain future entries only if those layers are skipped and the executed layers' attention cannot see that future. Native attention masks and position counters must be based on the replay frontier, not a global cache length accidentally inherited from layer zero.

Engine author was informed and confirmed this boundary; runner still needs independent inspection.

### F2 — Matching current token is weaker than matching complete history (high priority)

Journal provenance containing token index, layer cut, and current token identity does not establish a clean journal. Two histories can diverge and later emit the same token at the same position. A saved hidden input can then match its local metadata but still encode the abandoned earlier prefix.

The conservative implementation must disable old journals irreversibly after the first corrected-output disagreement. A later token reconvergence must not re-enable them. A history digest can diagnose mistakes but does not replace this rule in the present protocol.

### F3 — State must include more than K and V (high priority for any broader claim)

For the supported ordinary dense/GQA model, the executable state also includes the pending next input token, logical cache lengths, positional state/cache_position, attention-mask semantics, model/version/weights, and fixed numerical execution configuration. Generation controls can add EOS/finished flags, stopping criteria, repetition processors and RNG, although the pilot explicitly excludes sampling and ignores EOS for its fixed-length computation window.

Quantized caches add scales, zero points, residual windows, group boundaries and packing state. Sliding or recurrent caches can overwrite old state; merely cropping a suffix need not restore a valid prefix. Shared cross-layer state, adaptive computation, retrieval, mutable model buffers and external actions require separate contracts.

State the initial scope as append-only, layer-local, unquantized KV for explicitly supported model classes. Unsupported cache families should fail closed, not silently reuse the proof.

### F4 — Trust and error-time assumptions need executable checks (medium priority)

The pilot injects one perturbation into a known cached-prefix layer immediately after a verified checkpoint. This is stronger than general delayed detection of an unknown-time fault. The proof requires that the selected journal cut, clean checkpoint, immutable weights and recovery computation were not themselves affected. If localization only names a layer but multiple faults touched earlier layers, c<=L may be unsafe.

A cut c denotes the hidden input **before layer c**, including the appropriate preceding residual/norm conventions. Since faulty KV is read at layer L, c=L can be clean; the output of layer L (input to L+1) can be contaminated. Enforce this indexing and document the finite single-layer fault model. Do not generalize a controlled injection into evidence about hardware fault incidence or detector quality.

Checkpoint and journal tensors must own detached storage, not aliases of live buffers. Snapshot checksums before and after trial execution catch shallow-copy errors.

### F5 — Exact numerical equality is not always raw-bit identity (medium priority)

Typical tensor equality treats +0 and -0 as equal. That is exact elementwise numerical equality, not bytewise identity. Non-finite values must be classified before comparison; NaNs also complicate equality diagnostics. If the report uses “bitwise,” compare dtype/shape and raw bytes, while reporting layout/stride and numerical maximum error separately.

Identical input values and weights do not guarantee bitwise outputs if kernel shape, batching, precision settings, reduction order or layout select different numerical paths. The plan correctly fixes incremental shapes, but must also record dtype, attention implementation, deterministic settings, TF32 policy where applicable, and backend/library/device versions. A shape-compatible path is an assumption to test, not an automatic consequence of single-token input.

### F6 — An oracle equality check is not a low-cost operational verifier (medium priority)

Plan step 5 uses a clean full replay to verify recovery. That is appropriate for scientific validation. If executed on every deployed incident, it removes the advertised work-saving benefit. The current pilot must label it an out-of-band experimental oracle, excluded consistently from recovery timing, not a deployed monitor. Detector/localizer costs remain an explicit unimplemented component.

An independently executed reference that shares the same custom stepping function can reproduce a shared bug. Library-forward comparisons and analytic toy tests are essential. They serve different roles and neither should be described as full formal verification.

### F7 — Negative controls should have guaranteed witnesses, not required universal failure (medium priority)

Slot-only restoration can succeed for W=0, an unused/masked entry, a zero perturbation, or faults that never changed descendants. With W=1 and a last-layer-only fault, restoring the original slot may leave the entire cache correct even though the last output/logits were wrong, because that layer's newly cached K/V were produced before attention read the faulty history. Failure must therefore be evaluated across token, state and logits—not cache alone.

An unsafe c>L cut may accidentally carry the same hidden values in a no-op or insensitive case. The policy must always reject it, but its deliberately forced numerical execution need not fail every random trial. Require analytically chosen counterexample tests and report empirical control failure rates without tuning the final experiment to force them.

### F8 — Cost comparisons need matched protection and complete timing boundaries (medium priority)

Sparse/all-layer methods must pay actual journal acquisition, storage and bandwidth. All methods must receive the same trusted checkpoint and localization assumptions. Time restoration, cropping/metadata operations, recovered forward work, and synchronization consistently; do not charge the full baseline for cloning all layers while giving RECUT unrecorded free setup.

The H/S break-even expression is a simple one-incident model, conditional on equal common costs. Report it as such. Holdback latency, memory pressure, interference and pre-failure error exposure are separate objectives. Small-model layer-step reductions may not translate into GPU-time improvements because launching fewer/smaller operations changes overhead.

## Precise state/frontier invariant

Before replay transition j:

- The pending input is the fault-free x_j.
- Every executed layer has exactly T+j processed-token entries visible.
- Replayed-layer KV matches the canonical fault-free state through that frontier.
- Skipped-layer entries used for final restoration are clean only through the matching input prefix.
- A journal is eligible only if no earlier corrected output has disagreed, c<=L, and all snapshot/provenance checks pass.

After the step, the processed-token frontier is T+j+1 and the pending output is x_(j+1). At disagreement, truncate every layer to T+j+1 and disable all future old journals. Resume from embeddings on x_(j+1). At the final transition, a disagreement still changes the pending next input even if no further recovery step remains.

This invariant makes the usual induction precise. It does not establish optimality, detector reliability, or generality beyond the supported cache semantics.

An independent work-count assertion follows. For N layers, W transitions and selected cut c, full replay executes NW layer-token steps. RECUT with no divergence executes W(N-c). If the first corrected output differs at zero-based transition j, it executes (j+1)(N-c)+(W-j-1)N = NW-c(j+1). A final-transition divergence therefore still has W cut-replayed transitions; an immediate divergence has exactly one. These counts exclude checkpoint/copy work and do not predict elapsed time. Verify them from the executed trace, not only from a formula reported by the runner.

## Do token + complete-KV + continuation checks suffice?

**Conditional answer:** if “complete” includes all executable state listed above and the transition function is deterministic and identical, equality of state plus pending input is sufficient for future equality by induction. Full historical token equality also tests that no incorrect private output will be committed. A continuation test is then a useful integration check but logically redundant.

**Practical answer:** plain K/V tensor comparison plus token list and a short continuation is not sufficient for an unrestricted claim. Omitted position/quantizer/generation metadata can remain dormant for several steps and cause later divergence. A short continuation cannot prove all future behavior, and shared implementation bugs can make both paths agree with the wrong computation. For this pilot, compare full-state metadata, exact per-layer cache tensors, pending next input, final logits and several continuation steps; separately test library equivalence and analytic witnesses.

The final output of W transitions is generally not yet represented in the KV cache. Report both the W processed inputs and W predicted outputs, or make that offset explicit. Comparing a cache to a token list that includes its not-yet-processed final prediction is an off-by-one error.

## Exact counterexample and boundary tests

### A. Guaranteed slot-only and unsafe-cut witness, no real-model search

Use a two-layer scalar causal attention toy with identity value projection, zero query/key scores (uniform causal attention), identity output projection, residual addition, and no MLP. Its arithmetic is simple and exactly checkable.

Start with a single prefix whose layer-0 value is 1 and layer-1 value is 2. The next token embedding is 1. Clean computation gives:

- Layer 0: h1 = 1 + mean(1, 1) = 2; the appended layer-0 value is 1.
- Layer 1: h2 = 2 + mean(2, 2) = 4; the appended layer-1 value is 2.

Perturb only the old prefix layer-0 value from 1 to 3:

- Layer 0 becomes h1 = 1 + mean(3, 1) = 3.
- Layer 1 becomes h2 = 3 + mean(2, 3) = 5.5; its new cached value is 3.

Now restore the original prefix slot from 3 to 1. The derived layer-1 cached value remains 3 instead of 2: slot-only restore fails full-state equality. Teacher-force the same tokens or choose logits that emit the same token for h2=4 and 5.5; then token-only validation passes while state equality fails.

For unsafe cut c=1 with faulty layer L=0, restore all prefix cache correctly but replay from the recorded contaminated h1=3. The result is still 5.5 with descendant value 3. Valid c=0 restores h1=2 and h2=4. The ordinary policy must reject c=1 before execution; a separately labeled test-only forced unsafe path demonstrates the counterexample.

### B. Immediate token divergence

For the same toy, use two output logits [h2, 5] and a fixed tie rule. Clean output chooses the constant-5 token; faulty output chooses the h2 token. Choose distinct input embeddings for those token IDs and W>=2. Valid recovery must retain repaired state for the first input, truncate every layer at T+1, then embed the corrected output. Force an intentionally buggy implementation that reuses the next old journal; it must disagree in state or output.

### C. Delayed divergence and reconvergence

Use an explicit deterministic toy readout with a fixed position-dependent bias so the first corrected output matches and the second differs. This is a white-box algorithm test, not a claim of an observed language-model behavior. Include a later position where both paths emit the same token ID. Verify journals remain disabled after that reconvergence. Do not search the frozen real-model trial set for a prompt producing this pattern.

### D. Boundary matrix

| Case | Required outcome |
| --- | --- |
| c=0 | Same tokens, state and logits as complete suffix replay; equal layer-step count. |
| c=L | Eligible when cut is pre-attention layer input; exact repair in supported cases. |
| c>L | Ordinary API rejects; forced unsafe witness A fails. |
| L=0 | No positive-depth clean cut exists. |
| L=N-1, W=1 | Test logits/pending token as well as KV; cache-only detection can miss the error. |
| Divergence at j=0 | All later steps start at layer zero; no later old journal read. |
| Divergence at j=W-1 | Correct pending final output even though zero further steps are replayed. |
| W=0 if API permits | No descendants; original-slot restore may suffice. Otherwise reject explicitly. |
| No-op perturbation | Every valid method equals clean reference; negative controls may also equal. |
| Teacher-forced identical tokens | Full-state test still detects stale descendants in witness A. |
| Missing/malformed/aliased journal | Reject or documented full-replay fallback without mixing caches. |
| Wrong position or visible suffix | Reject before attention; catch future-cache leakage. |
| EOS inside window | Follow documented fixed-length semantics consistently for every method. |
| Non-finite perturbation/outcome | Classified and excluded from finite-fault guarantee; do not mask as success. |
| Prefix preservation | Original checkpoint bytes remain unchanged throughout all branches. |
| Branch isolation | Running one repair does not mutate another method's input snapshot. |
| Continuation | Continue using final pending output, not replaying or skipping it. |

## Ledger completeness audit

Read-only jq audit over:

- `research-next/literature/systems_001_034.json`
- `research-next/literature/safety_035_067.json`
- `research-next/literature/reliability_068_100.json`

Result: **100 records, 100 distinct IDs, exactly 001–100, no duplicate, missing or unexpected IDs.** Per-file counts: **34, 33, 33**. This verifies identifier coverage, not depth or bibliographic correctness. Individual ledgers preserve the selected-section review limitations.

## Engine review and independent CPU tests

Read all of `research-next/experiments/recut_engine.py` and `test_recut_engine.py`. The engine uses native Hugging Face Llama/Qwen2 decoder modules, ordinary DynamicCache, eager attention and default RoPE. It explicitly restricts runtime version, model type, precision/device and single-token unpadded execution. Its explicit position-sized attention mask correctly avoids deriving an upper-layer replay mask from retained future entries in layer zero. Executed layers must be append-ready before mutation. Checkpoints and journals use cloned tensor storage; cropping copies to release discarded allocations. The engine documents that history cleanliness remains the caller's responsibility.

No critical engine defect was identified in the supported path. Remaining narrow caveats: complete trusted cache state and immutable model/config remain caller assumptions; `compare_caches` is exact elementwise comparison rather than byte identity and does not compare strides; the engine does not independently discover a corrupted journal or globally validate history. These are acceptable only when reported as assumptions, not advertised safeguards. Malformed or partially initialized DynamicCaches are not the intended repair input.

Local offline test results using `research-next/.venv/bin/python` (PyTorch 2.8.0, Transformers 4.51.3):

- Existing `test_recut_engine.py`: **36 passed** in 3.47 seconds.
- Independently written `tests_independent.py`: **8 passed** in 3.63 seconds.
- Both runs emitted a local urllib3/LibreSSL compatibility warning; no network requests or model downloads were made.

The independent tests include exact Fraction-arithmetic witnesses for slot-only and unsafe-cut repair, identical-token/different-state behavior, immediate token divergence, a real tiny-model last-layer case with equal restored KV but unequal saved logits, stale journals accepted by local metadata after token reconvergence, rejection of untruncated future lower-layer state, clone isolation, and signed-zero equality semantics. The reconvergence test deliberately proves the engine's local checks alone cannot certify a history; it is not a successful end-to-end recovery test.

Reviewed SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `recut_engine.py` | `c0e0949ccf415853c3c047dbbba727df81539531a53704bb3479856bf8b7cac9` |
| `test_recut_engine.py` | `a7a72d7b1eab6ce041ae46be3800220a86ef9ae821ff6ce379bb42bd6b2a13ed` |
| `tests_independent.py` | `4b173936ec72476162989696b95508a0574a98f9bc9cd1ef83cb899b0d446617` |

These hashes/counts document the initial engine-review stage. The independent test file was subsequently extended for the runner; its final hash and results follow.

## Runner review and direct helper tests

Read the complete `research-next/experiments/run_recut.py`, then directly tested its imported `decode`, `repair`, `verify`, `native_parity`, and `run_case` functions on tiny randomly initialized native Llama models using CPU only. No pretrained-model outcomes, downloads or GPU measurements were used in this review.

### Findings resolved before the reviewed runner version

The initial native-parity gate checked four decode positions after the longest prefix, although primary windows can run twelve transitions plus two continuation steps. This left some exercised attention lengths outside the independent native-forward gate. The runner author changed the gate to `max(window)+2` at the longest prefix; source inspection confirmed the change. Model configuration is now also retained in metadata. This strengthens shape coverage but does not prove every possible numerical input or architecture.

### Confirmed behavior

- `repair` retains an irreversible one-based first-divergence marker. Once set, every subsequent step starts at layer zero, even after token reconvergence.
- At a changed output with cut>0, it crops all layers to the just-processed input frontier. With cut=0 there is no retained future suffix to crop.
- Its actual `starts` trace agrees with the independent layer-step formula. A mismatch is an assertion failure.
- `decode` stores W predicted outputs while KV covers the initial pending input plus the first W-1 outputs. `verify` correctly starts continuation with the final pending output.
- Each recovery branch receives its own deep cache clone; restoring layers occurs inside the timed function. The working-cache clone is excluded equally for all compared recovery methods. GPU synchronization brackets timing; verification/continuation is outside timing.
- Method order rotates across repetitions. No-fault journal overhead is measured separately with a corresponding excluded warmup. Raw memory peaks explicitly include resident correctness fixtures and are not represented as deployment memory.
- The slot-only control retains faulty derived state and saved outputs/logits. Consequently it correctly exposes both stale-cache failure and output-only failure; it is not secretly recomputing a suffix.
- Missing journals cause an exception rather than silent reuse; this is fail-closed behavior, not automatic fallback. The public plan should not claim automatic missing-journal fallback unless implemented.

### Independent test additions and results

The independent suite now has **22 passing tests** (3.55 seconds for the standalone run). A final combined run of the engine and independent suites produced **58 passed** in 4.41 seconds, with only the previously noted local urllib3/LibreSSL warning. In addition to the eight earlier tests, the independent suite covers:

1. Six branch-control combinations: cuts 1/2 crossed with first output divergence at transitions 1/2/4. A coherent speculative history is constructed by overriding one decision of a constant-logit tiny model, then feeding that changed token into the next step. Later decisions reconverge. This is an explicit white-box control-flow test, not a claim that the model spontaneously experienced a prefix-KV fault.
2. All journals after that divergence are removed before repair. Success therefore proves the runner does not access them after the first disagreement.
3. Native-forward verification of the processed-input/pending-output offset and the two-step continuation.
4. Six direct `run_case` tests across early/middle/late layers, K/V, scalar/vector/no-op disturbances. Each uses two timing repetitions for all three valid methods and two no-fault overhead repetitions. All valid methods exactly match tokens, cache, logits and continuation; no-op controls also match.
5. Explicit missing-journal failure. No result is silently reported as a successful recovery.

No critical supported-path runner defect remained in the reviewed version. These are finite unit/integration tests, not proof of all-state correctness or evidence of GPU performance, practical fault rates, production safety or publication novelty. Unsupported architectures/cache types, missing-state assumptions, and conditional checkpoint/detector trust remain outside the guarantee.

Final reviewed runner SHA-256: `969fa347eed616fce8956809ec612d991e08afa6ac48bcce5342491fb4b5f7c9`.

Extended independent-test SHA-256: `e70013fe2665ddc6c172c3ac3f5c8caf84b23c61e0f909da2aa527bb88dbb12a`.

Run both suites explicitly because `tests_independent.py` is not necessarily matched by pytest's default `test_*.py` pattern:

```sh
research-next/.venv/bin/python -m pytest -q research-next/experiments/test_recut_engine.py research-next/experiments/tests_independent.py
```

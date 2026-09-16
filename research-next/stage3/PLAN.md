# RECUT stage 3: completion protocol for the bounded research artifact

Prepared 16 September 2026 before stage-3 pretrained measurements. This stage completes a broader, reproducible prototype evaluation; it does not promise novelty, thesis approval or conference acceptance. Stages 1 and 2 remain immutable evidence.

## Question and implementation

Does detector-routed selective reconstruction retain a useful cost/memory tradeoff after protection is charged against ordinary inference, the all-layer activation comparator is added, and the model/context/fault coverage grows?

Implement batched GPU-to-host transfers for cache and activation SHA-256 seals, preserving exactly the stage-2 canonical byte stream, layout metadata and owner/position binding. Keep the old per-tensor transfer schedule as a same-code ablation. Full replay may skip unused empty-journal hashes in both modes. No weak checksum substitutes for SHA-256. Device batching adds a temporary allocation; both allocator peaks and logical payloads must be reported. Optional synchronized component profiling is a separate diagnostic, never pooled into headline timings.

The fault contract stays narrow: persistent finite old-prefix KV changes during speculation after checkpoint capture. Initial state, model weights, host metadata/seal storage, and guard/recovery/commit execution are trusted. Checkpoint/journal corruption challenges occur during speculation. Detection is exact numeric prefix comparison, not a general hardware/computation-error detector. Reverted, suffix-only, pre-checkpoint, between-window and computation-origin faults remain outside the guarantee. Single-owner private windows are not concurrent serving or crash durability.

## Fixed workload

Three pinned base models: SmolLM2-135M, Qwen2.5-0.5B and Qwen2.5-7B. The public ungated 7B revision is d149729398750b98c0af14eb82c78cfe92750796, resolved before downloading or testing. No trust_remote_code or quantization. Reuse the existing Python 3.13.9, Torch 2.6.0+cu124, Transformers 4.51.3 environment on authorized Slurm GPUs. Download public weights on a CPU allocation; never run model experiments on a login node.

Three authored synthetic prompt domains (cloud operations, code, reasoning), each extended with numbered repeated passages. These are not a representative external task benchmark or an accuracy evaluation. Two paired shapes: prefix 128/window 8 and prefix 2048/window 16. Three consecutive windows per session. Fault seeds 7431 and 9817; two timing repetitions for fault trials, three for clean trials. Every variant receives identical tokenized input and common checkpoint.

Prefill is one native batched Hugging Face call, outside timing, shared by all variants. Custom/native exact equality is checked at every subsequent decode transition, not against an alternate incremental-prefill numerical path. Each committed window is checked for full logical KV, logits, tokens and two continuation steps. Output buffering conventions and the initial pending-token offset are identical to the earlier prototype.

## Comparators and trial matrix

Protected policies are full checkpoint-alias replay, three-quarter-position sparse cuts, and every nonzero layer cut. All-layer activation storage is a strong member of the same replay family, not a reproduction of an entire published serving system. All use the same checkpoint, detector, integrity and commitment contract.

Clean costs additionally include bare custom-engine decoding and native HF incremental decoding. These are unprotected cost baselines, not resilience-equivalent competitors. Each clean workload uses seven variants: native, bare, full/sparse/allcuts batched, and full/sparse legacy. Legacy full/sparse are also run on the value_late fault case. Rotate method order by repetition. Warm up each workload/variant's clean path and each protected variant's value_late recovery path, excluded from primary data.

Eight fixed fault strata (zero-based layers, one-based after-step times):

| Case | Direct change | Layer | Time |
| --- | --- | --- | --- |
| key_early | K, finite 8-RMS Gaussian vector | 0 | after step 1 |
| value_early | V, one fraction-bit-3 XOR | 0 | after step W/2 |
| key_mid | K, finite 8-RMS Gaussian vector | floor(N/2) | after step 1 |
| value_mid | V, one fraction-bit-3 XOR | floor(N/2) | after step W/2 |
| key_late | K, one fraction-bit-3 XOR | floor(3N/4) | after step 1 |
| value_late | V, finite 8-RMS Gaussian vector | floor(3N/4) | after step W/2 |
| multi_prefix | V vector, then K vector | floor(3N/4), floor(N/2)-1 | after steps 1, 2 |
| repeated_prefix | V vector in every window | floor(3N/4) | after step 1 |

All except repeated_prefix affect only the second window. The fraction-bit perturbation leaves exponent/sign bits unchanged; it is software bit manipulation, not physical fault injection. Strata deliberately couple kind, severity, location and timing, so the data cannot isolate their causal effects independently. No threshold or magnitude is tuned after observing results. The controller receives no injection coordinates.

Two guard challenges use the first seed and one repetition for all three batched policies: (a) a late V-prefix perturbation plus saved-activation corruption where such activations exist; (b) checkpoint corruption only. The journal treatment is policy-specific and excluded from fair timing comparisons. Checkpoint trials must fail with the specific checkpoint-integrity reason, preserving previous committed state, not merely throw any exception.

Per model: six workloads; 131 measured sessions per workload = 786 sessions. Across three models: 2,358 sessions, 7,020 attempted windows, 6,966 expected commits and 54 expected checkpoint refusals. Separate profiles use clean/value_late with full/sparse batched, one repetition per workload. Warmups and profiles do not enter measured-session totals. Repeated deterministic treatments are not independent fault samples.

## Timing, memory and output availability

Primary wall time is full BEGIN-to-COMMIT (or specified refusal), including working/checkpoint copies, seals, activation capture, speculative work, detector, repair and final guards. Setup, session construction, shared prefill and reference validation are excluded. Injection time is included and separately measured; subtraction is diagnostic only. Per-step completion timestamps and window-release time show buffering delay, not network latency or request TTFT. Plain/native outputs are available at their step completion; protected tokens are available only at commitment.

Session construction is additionally measured after the common input fork has completed, for every variant. Report it separately and in session totals including initialization: protected finite-state validation must not disappear from a whole-session cost comparison simply because it lies outside individual windows. Common input copying, prefill and reference checking remain outside that second boundary too.

Report absolute times, native/bare cost, protected-full/bare and protected-sparse/bare overhead, full/sparse/allcuts paired fault-window and session ratios, legacy/batched sealing effects, layer work, logical state/journals, and GPU allocation peaks. Keep clean, successful recovery and refusal outcomes separate. Layer-zero no-saving cases remain visible. Profiling components are neither an exhaustive partition nor part of performance headline data.

For supported-incidence sensitivity, use matched complete-window clean overhead H = sparse_clean - full_clean and conditional advantage R = full_fault - sparse_fault. If both are positive, the illustrative break-even is H/(H+R); other signs require explicit cases, not clipping into a favorable answer. Never estimate real fault incidence from deliberately injected trials or call this production throughput.

## Evidence and termination

Before measurement: all original tests plus stage-3 equivalence/mutation tests pass; source, config and immutable model revisions freeze. A small per-model debug matrix must pass before the main matrix starts. Failures are retained and stop that model's job. Any engineering amendment is declared before a fresh run; no failed directory is overwritten.

Persist full CPU tensor archives for a predeclared subset: cloud prompt, prefix 128, first seed/repetition, second window, key_early/value_late for all three batched policies; additionally cloud prefix 2048, multi_prefix, sparse batched, first seed/repetition/second window. Seven archives per model. Include recovered/native KV, logits, tokens and two-step continuation. A separate process loads only tensor/basic data with weights_only=True and verifies exact finite equality. Other cases retain live comparison verdicts and hashes; do not claim they were tensor-reexecuted independently.

Completion requires all configured outcomes accounted for, correct protected commits/refusals, canonical seal equivalence, independent record and snapshot audits, reproducibility instructions and an evidence-backed final report/PDF. Positive speedup is not an exit criterion. No physical/binary GPU instrumentation, production cloud changes, paper submission or automatic GitHub publication is authorized by this protocol. A measured slowdown or an unfavorable novelty assessment must remain in the final report.

Prior-art concerns remain: activation reconstruction, dependency-aware replay and history-conditioned rollback are established techniques. More engineering and larger experiments do not automatically create a publishable research contribution.

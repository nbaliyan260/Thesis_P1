# RECUT stage 2: implemented and GPU-validated

16 September 2026. **Engineering milestone complete within the declared fault model; not a finished thesis or publication-ready system.**

## What changed

RECUT now runs consecutive guarded decoding windows rather than only isolated repair trials. It preserves the last committed state, keeps speculative tokens private, identifies changed old-prefix layers itself, and repairs from an eligible saved layer activation. No injected fault location is supplied to the controller.

It also handles several persistent prefix faults introduced at different times. Damaged saved activations trigger complete replay. A damaged recovery checkpoint causes a safe refusal: no new tokens are released, previously committed state remains unchanged, and that session cannot continue. When repair changes a token, all later steps use full-layer replay; later token reconvergence does not permit reuse of the old history.

Implemented code: [controller](controller.py), [tests](test_transactional.py), [GPU harness](run_sessions.py), [analyzer](analyze_sessions.py), and [independent record auditor](audit_sessions.py). [The plan](PLAN.md), [source freeze](protocol_freeze.json), [reproduction instructions](README.md), and [execution record](EXECUTION.md) document the protocol. Original stage-1 source and data were not changed.

## What was run

The existing HPC was sufficient. Slurm job **195896** completed successfully on an NVIDIA RTX 5000 Ada Generation with 32 GB device memory, using one GPU, four CPUs and a 24 GB host-memory allocation. Job duration was **6 minutes 39 seconds**, including tests and the separate preflight. The main harness took approximately 336 seconds, including loading, references and checks, not just timed windows.

The fixed main matrix used SmolLM2-135M and Qwen2.5-0.5B at their previously pinned revisions, BF16 eager attention, unpadded batch one, greedy decoding, prefixes 32/128, window lengths 8/16 and three consecutive windows. It used one repeated text prompt, one fault seed, six scenarios, two policies and two timing repetitions. Full replay uses a copy-avoiding checkpoint-alias baseline under the same transaction, checkpoint and detection contract.

There are eight model/prefix/window workloads and 48 workload/scenario combinations. The 192 sessions are **repeated policy trials, not 192 independent fault samples**. Perturbations were controlled finite Gaussian additions to value-cache vectors, not physical hardware faults. Journal and checkpoint damage were explicit additional guard challenges. All settings were frozen before pretrained measurements; no failing GPU case was discarded or tuned away.

## Correctness results

| Check | Main result |
| --- | ---: |
| Local and HPC unit-test gates | 107 tests passed on each runtime |
| Main session trials | 192/192 met their prescribed outcome |
| Attempted windows | 544 |
| Committed windows | 512, all matched the native clean reference |
| Windows repaired after working-prefix corruption | 192 across both policies |
| Damaged-checkpoint windows | 32/32 rejected without new output or altered prior committed state |
| Damaged-journal challenges to sparse replay | 16/16 used full-replay fallback successfully |
| Committed tokens checked | 6,144 |
| Native/custom parity positions | 944, with exact recorded cache/logit agreement |
| Independent main record audit | 85,379 checks passed on HPC and again locally |

Every committed window was checked against native Hugging Face execution for its complete logical KV cache, logits, token sequence and two additional continuation steps. No early token release or supported-case wrong committed state was observed. The four checkpoint-rejection cases and 64 commits from the separate 24-session debug run are **not** added to these main totals; its independent audits passed 15,007 checks each.

The main run also exercised the history-divergence safety branch: **10 sparse measured windows** used a nonzero cut and then switched to full replay before the window ended. These represent five configured Qwen workload/scenario/window instances, each repeated twice; some share the same perturbation/history across scenarios. They are not ten independent faults. For example, Qwen at prefix 32/window 8 repaired from cut 18 for five steps, then used layer zero for the remaining three. SmolLM2 had no token divergence in this main matrix, although its corrupted cache still required repair.

The saved evidence contains hashes and live comparison verdicts, **not full persisted recovered/golden GPU tensors**. The independent auditor checks source/data binding, fault arithmetic, coverage, replay traces, commitment accounting and all reported summary statistics. It is a record audit, not a second model execution or external replication.

## Performance: complete guarded windows

The main result below is **median paired full-policy time divided by sparse-policy time**. A ratio above 1 favors sparse replay. Timed work includes BEGIN, both cache copies, integrity seals, activation capture, speculative decoding, delayed prefix detection, repair, guards and output commitment. It also includes separately recorded artificial injection work. Prefill, session construction, native references and post-commit verification are excluded. This is BEGIN-to-window-release latency, not request time-to-first-token or production throughput.

| Window category | Paired windows per model | SmolLM2 full/sparse | Qwen full/sparse |
| --- | ---: | ---: | ---: |
| Clean scenario | 24 | 0.990x | 0.993x |
| Single upper-layer prefix fault, affected window only | 8 | 1.533x | 1.529x |
| Two staggered prefix faults, affected window only | 8 | 1.120x | 1.102x |
| Upper-layer prefix fault in every window | 24 | 1.537x | 1.520x |

Interpretation: the upper-layer-fault cases had about **34–35% lower complete-window latency** with sparse replay. The multi-layer cases had about **9–11% lower latency**, because the earlier damaged layer requires replaying more of the model. Pure-clean windows had a median paired overhead of **0.98% for SmolLM2 and 0.70% for Qwen relative to the protected full-replay controller**. These are not overhead estimates relative to ordinary unprotected inference; that comparison was not measured here.

For single-fault windows, paired-ratio interquartile ranges were 1.528–1.552 for SmolLM2 and 1.454–1.547 for Qwen. For multi-fault windows they were 1.116–1.132 and 1.093–1.119. These are descriptive spread summaries, not confidence intervals. Only two repetitions were taken, and warmup covered the first clean window per model/policy, not every shape or recovery branch.

Replay-only medians were approximately 3.56x/3.48x for the single upper-layer fault and 1.29x/1.29x for multi-layer faults, for SmolLM2/Qwen respectively. These larger replay-only ratios must not replace complete-window results. Median replay layer-step reductions were 73.33%/75.00% for single upper-layer faults and 23.33%/23.44% for multi-layer faults. Early divergence reduces Qwen's savings in individual cases.

Subtracting measured injection time changes the paired medians only slightly, but those adjusted numbers remain diagnostics. The journal-damage challenge differs by policy and is excluded from matched performance aggregates. Checkpoint rejection is an availability/safety outcome, not a successful recovery-speed sample. No aggregate production speedup is inferred from the deliberately high experimental fault frequency.

Do not compare these ratios directly with stage 1's 1.29x/1.30x recovery-only median ratios: the fault populations, selected layers, prompts and timing boundaries differ. Across entire three-window sessions with just one upper-layer fault, the median paired ratio was about 1.21x for each model; unaffected windows dilute the affected-window advantage. Even that session figure uses an artificial fault schedule, not a deployment failure rate.

## Memory and operating cost

At BEGIN, the controller retains committed state and makes separate working and recovery copies. It is not a zero-copy transaction. The input cache and recovery checkpoint each ranged from 720–3,600 KiB for SmolLM2 and 384–1,920 KiB for Qwen. Sparse activation journals added 27–54 KiB and 42–84 KiB respectively in this small matrix.

Measured GPU allocation increases above each window's baseline were 1.939–8.046 MiB for full versus 1.975–8.116 MiB for sparse on SmolLM2; Qwen was 2.144–5.528 MiB versus 2.190–5.622 MiB. Paired starting allocation baselines matched. Absolute peaks include resident model and native-reference fixtures, and host hashing-buffer peaks were not measured. **This implementation reduces repeated computation, not total memory usage.**

Tokens are deliberately held until a whole window is committed. This delay must be included in any future user-facing latency evaluation; the prototype is not transparent streaming.

## What this establishes—and what it does not

This establishes a working, measured guarded multi-window integration under a narrow persistent-prefix fault contract. It is a meaningful engineering step beyond the isolated pilot: detector-derived routing, multiple faults, state transfer across windows, corruption fallback/refusal and actual pretrained early-divergence replay were tested together.

It does not make activation reuse or rollback a new research principle. The prior-art constraints from [LayerSkip](https://aclanthology.org/2024.acl-long.681/) and [VeriCache](https://arxiv.org/html/2605.17613v1) remain. **The work is not yet A/A*-submission-ready.** These measurements alone do not establish a novel research contribution or performance advantage over production inference engines.

Coverage is limited to two compact models, one prompt, one seed, short contexts, batch one and a finite value-vector corruption family. Initial state, weights, control metadata/digest storage, and guard/recovery/commit execution are trusted. Faults must occur during speculation after checkpoint creation and remain in the old prefix until detection. Suffix-only, restored transient, computation-origin and between-window/pre-checkpoint faults are outside the guarantee. Seals authenticate post-capture storage integrity, not semantic correctness at capture. The controller is single-owner, not concurrent, crash-durable or a general security boundary.

## Recommended next research gate

Continue as a bounded systems feasibility project, not as a promised top-tier paper. The next useful step is to profile and reduce checkpoint/journal checking cost without weakening the same correctness contract, then test a larger model, diverse prompts/seeds, longer contexts and realistic fault frequencies. Include an ordinary unprotected inference cost baseline as well as an optimized protected full-replay baseline, and measure output-delay/service-level effects.

Before expanding into a thesis-scale campaign, identify a defensible contribution beyond existing reuse/rollback: for example, a rigorously bounded low-cost detection/recovery co-design whose complete-path benefit survives realistic serving constraints. That is a research question, not an already demonstrated novelty claim. Physical or binary-level fault experiments remain pending lab authorization; none were performed here.

## Evidence links

- [Main summary](../runs/stage2-sessions-v1/summary.json), [metadata](../runs/stage2-sessions-v1/metadata.json), [raw sessions](../runs/stage2-sessions-v1/sessions.jsonl), [native-reference records](../runs/stage2-sessions-v1/references.jsonl).
- [HPC audit](../runs/stage2-sessions-v1/audit.json) and [local audit](../runs/stage2-sessions-v1/audit_local.json).
- [Separate debug summary](../runs/stage2-debug-v1/summary.json) and [debug local audit](../runs/stage2-debug-v1/audit_local.json).
- [HPC stdout](stage2-195896.out) and [stderr](stage2-195896.err). The only stderr output was the library's Qwen eager/sliding-window warning; recorded `use_sliding_window` was false, tested contexts were well below 32,768, and exact native parity passed. The Slurm accounting database was unavailable afterward; `scontrol` independently reported COMPLETED, ExitCode 0:0.

Implementation, experiments and separate-agent checks were AI-assisted. They are not human peer review. No GitHub update was made in this stage.

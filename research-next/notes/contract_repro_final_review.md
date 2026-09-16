# Final contract and reproducibility review

16 September 2026. Reviewed `notes/recovery_contract.md`, the professor-draft template, `README.md`, `fetch_pinned_models.py`, relevant runner/auditor interfaces, model/environment manifests and the package layout. No engine, runner, config, draft or README edits; no HPC actions. The reviewer previously authored the contract note and runner, so this is a separate checking pass, not independent authorship or external replication. Primary-result completeness was not assumed while the job was running.

## Verdict

No indexing defect found in the stated work identity or conditional proof. The contract is sufficient under its explicit assumptions, not a proof of minimal work, complete implementation correctness or a new checkpointing principle. The reproduction commands and model downloader are structurally consistent with the actual CLI. Before release, tighten command failure gating and qualify environment reproducibility; do not finalize a completed-pilot statement until the raw run and independent audit have completed successfully.

## 1. The work identity is exact for the stated object

Layers are indexed 0,...,N−1. Cut c is the **input** to layer c, hence skips exactly c decoder blocks. Transition t consumes x_t, appends its KV and selects x_(t+1). If the first changed **output decision** occurs at one-based j, transition j still has the old, clean input history and can use the saved cut. Full-layer regeneration begins at j+1.

With j=∞ for no divergence and k=min(j,W):

\[
Q_c=k(N-c)+(W-k)N=NW-ck.
\]

The endpoint cases are consistent:

| Case | Partial transitions | Work | Required final handling |
|---|---:|---:|---|
| c=0 | Irrelevant | NW | Ordinary full replay |
| First output differs at j=1 | 1 | NW−c | Crop to T+1; consume corrected x_2 next |
| First output differs at j=W | W | W(N−c) | Keep corrected pending x_(W+1); it is not in KV yet |
| No output divergence | W | W(N−c) | Lower suffix is still valid; upper suffix is rebuilt |

Q counts decoder-layer invocations only. It excludes checkpoint copies, cache cropping, embedding/norm/output heads, comparisons, journal handling and the two diagnostic continuation steps. The continuation is outside repair timing; including it in a work count would add 2N, not change the W-transition identity. A changing cut policy would require summing the actual skipped c_t values, not reuse the fixed-c formula.

## 2. Guarantee assumptions to retain visibly

The draft accurately includes the principal limits: clean pre-fault checkpoint, known layer, append-only layer-local state, no further faults, identical numerical functions, private outputs, and irreversible fallback after token divergence. Four clarifications matter:

1. **Clean initial pending token.** x_1 is selected from clean prefix logits before injection and is trusted alongside the checkpoint. A checkpoint containing only T tokens cannot justify a next token previously selected from faulty logits. The implementation and current draft satisfy this stronger timing condition.
2. **A journal entry is not a certificate.** Engine identity, position and current token are useful misbinding checks but do not prove integrity or the whole prior token history. Their validity follows from the restricted fault model plus the no-prior-divergence invariant. Keep the distinction between that theorem assumption and an implemented detector/provenance system. The runner currently fails closed on a missing entry; it does not implement the contract note's optional automatic full-replay fallback.
3. **Logical lengths are part of the recovered state.** At divergence, all layers must expose exactly T+j entries before the changed token is processed. Old future backing bytes can remain physically allocated only if unreachable through masks/lengths. A later equal token cannot revalidate an old history-dependent activation.
4. **Measured equality is not bytewise evidence.** The professor draft correctly says exact elementwise equality, excluding signed-zero distinctions. The supporting note's conditional bitwise sentence should not be read as a measured bitwise result. A bit-representation theorem needs deterministic functions of complete bit-represented inputs; ordinary elementwise equality alone is not generally substitutive for every IEEE-754 operation (for example, reciprocal distinguishes positive and negative zero). The existing tests establish the declared elementwise checks and short continuation, not a universal backend proof.

The listed counterexamples cover already-emitted outputs, bad localization, contaminated checkpoints/journals, changed arithmetic, shared/recurrent state and uncoupled sampling. Also keep the fixed-length/no-EOS-stop convention visible: this pilot continues for W transitions even if EOS appears, so a production stopping/commit protocol is not validated. Quantization scales, cache eviction state, allocator mappings and other mutable execution state would have to join the trusted/recovered state if those features were added.

## 3. Stronger baseline: fairness issue identified during this review

Under the pilot's fault model, the original prefix is unchanged in every layer except L. Full replay can therefore crop every speculative suffix and restore only the layer-L prefix, then execute all W transitions. It needs the same layer oracle as RECUT; no extra exact-coordinate information is required.

However, the current engine's `truncate_cache` clones every retained prefix tensor. Calling that helper and then restoring L would still copy all prefix layers plus L again, and would **not** constitute a copy-minimized baseline. The supplemental implementation must use prefix views/copy-on-write, preserve owner/checkpoint integrity, and charge retained backing allocations when reporting memory. This point was communicated to the root and supplemental author, who confirmed a view/COW implementation rather than cloning truncation.

The proposed supplemental comparison is appropriate if separately labeled post-hoc and run over all frozen perturbed cases, with equal working-copy boundaries, paired timing and full-state/continuation checks. It must not replace or silently relabel primary measurements. The root subsequently requested five methods: original full replay, COW full replay, a fresh full-replay cache aliasing trusted checkpoint tensors, sparse RECUT and all-layer RECUT. The installed Transformers 4.51.3 `DynamicCache.update` replaces each existing layer tensor with an out-of-place `torch.cat`; this supports the alias idea provided list containers/counters are independent, all prefix aliases remain read-only, and other state assumptions hold. The alias comparator needs no extra fault-coordinate oracle and avoids even the layer-L copy. Supplemental implementation and timing results were not yet available when this note was first written; they require a separate code/test audit. Current conservative copies should not be called an optimized universal baseline.

The expected-time break-even condition H < p E[S] remains correct only under the note's common detector/commit protocol and compatible fault-conditioned cost distribution. A ratio above one has no feasible probability under that single-incident model. A positive recovery-only gain is not a service benefit, and negative measured normal overhead is not evidence that journaling is free.

## 4. Reproduction audit and recommendations

**Confirmed.** The README's runner, analyzer and auditor argument forms match their actual parsers. A newly localized manifest is correctly used for the new run and its audit, so its changed path/hash is not confused with the original manifest. The package builder retains both `research-next/` and `output/pdf/`, making the README's `../output/pdf/...` links valid after extraction. The original manifest pins both model revisions. The fetcher preserves those revisions, rewrites only new local paths and refuses an existing output manifest. Its file allow-list covers the two selected model/tokenizer formats, and the runner separately disables remote code and requires safetensors.

Read-only mocked execution of the actual fetcher passed three cases: pinned revision forwarded unchanged; existing output rejected before download; unpinned `main` revision rejected before download. Network calls and filesystem writes were mocked out. This does not validate network availability, upstream retention or downloaded weight contents.

**Recommended before packaging:**

- **Gate primary after debug explicitly.** The displayed shell commands are separate lines with no strict-shell failure handling. If pasted as one block into an ordinary shell, a nonzero debug exit does not necessarily stop the following primary command. State “stop and review the debug result before running primary,” or supply a failure-gated shell/script. The original execution's manual gate should remain documented separately.
- **Distinguish protocol replay from exact environment recreation.** The commands install direct pins, while several transitive dependencies remain free to vary. Refer to `experiments/environment_freeze.txt` and `experiments/environment_setup.json` with their real paths, and offer the captured freeze as a strict Linux/CUDA recreation option. Pinning PyTorch/Transformers plus model commits does not promise identical historical outputs on different drivers, GPUs or tokenizers.
- **Label file paths accurately.** Under “Inspect and extend,” the bare code/config filenames actually live in `experiments/`; `publication_assessment.md` lives under `notes/`. This is navigational, not a functional blocker for the displayed commands.
- **Keep offline/network prerequisites clear.** Fetching weights requires network access unless already cached. A shell inheriting `HF_HUB_OFFLINE=1` from the provided experiment Slurm script must not be used to fetch uncached models. The original local manifest's Lustre paths are not portable; the fetch step is required.
- **Preserve limited authentication claims.** Exact commit/path checks do not independently authenticate every local model file after modification. The audit already discloses this. A later hardened artifact could record per-file weight/tokenizer digests; do not claim those were already checked.

No hard blocker was found in the downloader for the two supplied manifests. No remote downloads, environment installation or GPU tests were performed in this review. Completion/results statements must be filled only after the corresponding saved evidence is verified. The README's 58-test count described the original suite; the later supplemental tests expanded the local suite to 74 (see below). Keep the original execution count distinct from the final artifact's expanded suite.

## 5. Supplemental baseline code review and local tests

Subsequently read the complete `experiments/run_optimized_replay.py`. Reviewed revision SHA-256: `6269b0afb23ea396daeb5a20dfb0f3e03f825b774db4aef60765df4de00ee72e`. No critical correctness or timing-boundary defect found for its bounded contract.

- **Checkpoint aliasing:** a new `DynamicCache` receives independent shallow key/value lists and its own `_seen_tokens=T`. The underlying tensors are initially shared read-only. The installed append operation replaces each layer tensor out of place. Checkpoint lists and counters are not shared mutable containers.
- **Layer-view baseline:** suffix cropping uses views, not the copying engine helper. Only full layer-L K/V is restored; no exact-coordinate privilege is used. Backing owners remain alive through the first full step, when every prefix view is replaced, and are then released. The memory note correctly discloses retained storage.
- **Inputs and checks:** input IDs, initial pending token, recorded perturbation and reference/faulty output trajectories must match the primary row. Native parity is rerun. After each method and diagnostic continuation, raw-byte guards check that checkpoint and original faulty cache remain unchanged.
- **Timing:** the common working clone is outside every timer, including the alias method that does not use it. Policy-specific construction/restoration and replay are inside. One warmup and five rotated repetitions place each of five methods at every order position once. Complete-state/logit/output/continuation verification is outside timing. The metadata explicitly says the unoptimized sparse/all-layer restoration remains in this sensitivity study.
- **Scope:** immutable source/config/manifest/raw hashes are checked; all 216 primary records and all 192 perturbed targets are required. The supplement does not overwrite the primary, and marks itself post-hoc. Matching recorded trajectories does not independently authenticate all original model weight bytes or unsaved intermediate states.

Independent reviewer smoke: six tiny-model scenarios across Llama and Qwen2, five methods times two repetitions (60 measured repair checks), all passed complete-state and two-step continuation checks. Separate storage tests confirmed different list objects, shared initial tensor pointers, independent counters, replacement of every tensor pointer after append and unchanged checkpoint contents. The full current CPU suite subsequently passed **74/74** tests in 4.65 seconds, using local Torch 2.8.0/Transformers 4.51.3. This is not CUDA/BF16 performance evidence; that remains the separately executed supplemental job.

Interpretation warning: if checkpoint-alias full replay reduces or removes a reported speedup, conclude that the original prototype comparison was sensitive to restoration copies. Do not conclude that every optimized clean-cut implementation is dominated: sparse/all-layer policies could themselves use carefully validated alias/COW restoration. Conversely, do not omit the stronger full-replay result because it weakens the main narrative.

## 6. Optional snapshot detector assessment

A separately measured exact comparison of each current old-prefix slice against its trusted checkpoint is a useful feasibility demonstration under the **persistent, post-checkpoint, old-prefix-only** fault model. It can derive a single changed layer without injection metadata, provided the model never modifies those old-prefix entries legitimately. It is standard snapshot differencing, not a novel or general SDC detector.

Charge all prefix/checkpoint reads, temporary buffers, reduction/host synchronization, retained snapshot memory and snapshot maintenance. Measure no-op false positives and coverage of actually changed perturbations; a perturbation rounded back to its original value is not a missed numerical change. A suffix-only change, transient corruption repaired before checking, bad journal/weights, or shared corruption of current state and trusted checkpoint can evade this detector. Include a small blind-spot unit control if implementing it. Keep detector records separate from frozen primary and five-method baseline timings; feeding its derived layer into a repair policy is a new post-hoc feasibility result, not evidence that the original experiment already had non-oracle localization.

## 7. Final implemented detector and supplemental review

The subsequent implementation has now been read in full. Reviewed SHA-256 digests:

- `run_optimized_replay.py`: `6081ddd9c7d8572e0e0e9ac67b8a2017e2865034115cc75207642a0995076ac1`.
- `prefix_detector.py`: `0324ad63f6e63c573afd6ee94c978629b03b197e9fe072bc70e999191df93418`.

**Verdict: no critical defect found; suitable for the bounded, separately labeled GPU supplement.** This is code/local-test clearance, not a claim that the GPU run has completed or that the detector is production-ready. No source changes or HPC actions were made during this review.

**Routing and absence of an injection oracle.** `detect_prefix_layers` accepts only current and checkpoint caches; it does not receive a seed, case, injected coordinates or true layer. It compares K and V over positions before T, reduces one Boolean per layer, and transfers the stacked decisions to the host once. The configured layer is used to reconstruct the controlled fault and check ground truth, but only `detected[0]` enters the recovery policies. Sparse chooses the deepest saved cut no greater than that detected layer. More than one changed layer aborts. For no alarm, sparse/all use cut zero and the layer-view method falls back to checkpoint-alias full replay; these are safe full-recovery paths from a clean checkpoint when recovery is requested. The ground-truth agreement check is a benchmark audit outside the policy, not an operational detector capability.

**Important no-alarm qualification.** No alarm does not certify the speculative suffix. A suffix-only error or a transient/reverted old-prefix error can leave a corrupted suffix with unchanged old-prefix bytes. The new tests demonstrate these blind spots. The conservative no-alarm recovery branch is correct if invoked, but the study does not establish a production trigger that would request recovery for such invisible faults. Weights, journals, pre-checkpoint errors and checkpoint/common-cause corruption remain outside coverage. Elementwise comparison also intentionally does not distinguish signed zero. Coverage should be reported against perturbations with `changed_elements > 0`, not classify a rounded-to-no-change injection as a missed numerical fault.

**Integrity, numerical gates and fairness.** Raw-byte guards include tensor shape, dtype, device and cache counters, and are checked after detection and after every recovery plus diagnostic continuation. Thus the detector, views, aliases and append operations must preserve both the checkpoint and original faulty source, including signed-zero/NaN representations. The backend gate compares primary Torch/Transformers/CUDA/GPU identity, BF16/eager selection, deterministic algorithms, TF32 policy, BF16 reduction policy, float32 matmul precision and CUBLAS workspace configuration. Native incremental parity is rerun through the longest prefix and maximum window plus two transitions. The recorded Python version is not part of this equality gate; matching these flags still does not authenticate all model bytes or guarantee identical behavior on arbitrary hardware. Common working-copy exclusion and restoration/replay inclusion remain unchanged for all five methods. Alias safety continues to depend on this ordinary DynamicCache implementation's out-of-place append, not an arbitrary in-place cache backend.

**Counts and cost interpretation.** The supplement reconstructs all 216 primary cases. Its `episodes.jsonl` contains 192 perturbed-case recovery comparisons, five methods and five timed repetitions; `detection.jsonl` contains all 216 detector records. The 24 no-op rows are detection-only, not 24 additional recovery trials. Detector timing has its own warmup and five synchronized repetitions and is outside repair timing. `logical_bytes_read = 2 × checkpoint payload` counts the two input snapshots, not measured HBM traffic or temporary reduction traffic. Do not present detector-plus-repair latency, retained checkpoint/journal memory, or normal-operation overhead as included in a recovery-only speedup. The short-prefix experiment does not establish long-context feasibility.

The reviewer reran the complete local suite after this extension: **76/76 passed in 4.85 seconds**, using local CPU Torch 2.8.0/Transformers 4.51.3. Tests include suffix/transient blind spots, multiple-layer rejection, no-op detection-only behavior, a perturbation producing no numerical change and conservative zero-cut routing. The only warning concerned the local urllib3/LibreSSL combination; no test failed. This final artifact-suite result is distinct from the original frozen primary's earlier 58-test gate and from CUDA/BF16 experiment evidence.

## 8. Primary boundary-coverage gap and declared diagnostic amendment

The completed primary's two changed-token cases are both Qwen, with first divergence at j=W=12: `primary-052` selects sparse cut 0; `primary-063` selects cut 6 at layer 11. Thus no primary trial executes an in-window full-layer transition after a nonzero-cut divergence. The strict selector was run locally and honestly recorded `skipped` at `runs/boundary-selection-v1/metadata.json`, without loading model weights or using a GPU. Its source hash describes the initial selector revision, before the subsequently authorized extension was added; the skipped record is preserved unchanged.

The coordinating agent explicitly authorized a single post-hoc W=12 to W=14 extension of `primary-063`, with no new fault sampling or altered primary/config files. The premeasurement scope is frozen in `notes/boundary_extension_amendment.md` and `experiments/boundary_extension_amendment.json`. This is selected branch-coverage evidence, not another independent fault trial or a performance measurement. Before accepting a result, require identical original prefix, pending input, complete fault record and first 12 clean/faulty decisions, then j=12 and sparse starts `[6]*12+[0]*2`. Complete-state snapshots are limited to 20 MiB and must pass the existing independent tensor auditor through the new CPU wrapper.

Capture source SHA-256: `23735533bdac94f7f4ea1ae134e6bb2142dac7e279b1850f451821d1287611ef`. CPU wrapper SHA-256: `417a4c1bc92b8542868e31f5f785668baefcd9c4869575e6ee1b9b926db08d44`. Amendment SHA-256: `eaeefc0f9e2ac4e8b37019c3e18d2a216664d34f7836b2d09d5208215b646a32`. Local checks covered syntax, five selector/order assertions, the real-data skip, overwrite refusal and the amended command's CPU-only CUDA rejection. A separate reviewer found no indexing/recovery defect. The final wrapper additionally checks all numerical backend fields and complete native-parity coverage against primary metadata. No GPU capture outcome is asserted by this source-review entry.

The draft template's then-current line 124 overstated primary coverage by saying the traces verify irreversible fallback after divergence; the coordinator was notified. Conditional algorithm/proof descriptions and the explicitly conceptual figure are unaffected. Final reporting must distinguish frozen primary coverage from any subsequently completed amended diagnostic.
